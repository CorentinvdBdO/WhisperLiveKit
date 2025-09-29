from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from whisperlivekit import TranscriptionEngine, AudioProcessor, parse_args
from whisperlivekit.silero_vad_iterator import FixedVADIterator
import asyncio
import base64
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine: Optional[TranscriptionEngine] = None

# In-memory session registry
_sessions: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _format_utterance_id(n: int) -> str:
    # Matches the observed format like "00_00000000"
    return f"{n // 100000000:02d}_{n:08d}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v2/live")
async def initiate_live(request: Request, region: Optional[str] = None):
    """Initiate a Gladia-like live session.
    Accepts the client configuration and returns a session token with a WebSocket URL.
    """
    try:
        cfg = await request.json()
    except Exception:
        cfg = {}

    token = str(uuid.uuid4())
    _sessions[token] = {
        "created_at": _now_iso(),
        "config": cfg,
        "region": region or "local",
    }

    url = f"ws://{args.host}:{args.port}/v2/live?token={token}"

    response = {
        "id": token,
        "created_at": _sessions[token]["created_at"],
        "url": url,
    }
    # Match observed envelope in logs
    return JSONResponse(response)


class GladiaSessionState:
    def __init__(self, session_id: str, engine_args):
        self.session_id = session_id
        self.engine_args = engine_args
        self.byte_counter = 0
        self.time_ms_counter = 0.0
        self.prev_partial_text: Optional[str] = None
        self.line_id_map: Dict[int, str] = {}
        self.lines_by_index: Dict[int, Any] = {}
        self.finalized_line_indexes: set[int] = set()
        self.translation_sent_line_indexes: set[int] = set()
        self.utterance_counter = 0
        self.all_finals: Dict[str, Dict[str, Any]] = {}

    def next_utterance_id(self) -> Tuple[str, int]:
        idx = self.utterance_counter
        self.utterance_counter += 1
        return _format_utterance_id(idx), idx


async def send_ack_for_audio_chunk(websocket: WebSocket, state: GladiaSessionState, chunk_len_bytes: int, sample_rate: int = 16000, bytes_per_sample: int = 2):
    start_byte = state.byte_counter
    end_byte = state.byte_counter + chunk_len_bytes
    state.byte_counter = end_byte

    samples = chunk_len_bytes / bytes_per_sample
    duration_ms = int(round(1000.0 * samples / sample_rate))
    start_ms = int(state.time_ms_counter)
    end_ms = start_ms + duration_ms
    state.time_ms_counter = float(end_ms)

    ack = {
        "acknowledged": True,
        "type": "audio_chunk",
        "session_id": state.session_id,
        "created_at": _now_iso(),
        "data": {
            "byte_range": [start_byte, end_byte],
            "time_range": [start_ms, end_ms],
        },
        "error": None,
    }
    await websocket.send_json(ack)


def _collect_words_for_line(tokens: List[Any], line) -> List[Dict[str, Any]]:
    if not tokens or line is None:
        return []
    start = float(line.start or 0.0)
    end = float(line.end if line.end is not None else (line.start or 0.0))
    speaker = getattr(line, 'speaker', None)
    words: List[Dict[str, Any]] = []
    for t in tokens:
        try:
            ts = float(getattr(t, 'start', None) or 0.0)
            te = float(getattr(t, 'end', None) or 0.0)
        except Exception:
            continue
        if ts < start or te > end:
            continue
        if speaker is not None and getattr(t, 'speaker', None) not in (speaker, -1, 0):
            continue
        text = (getattr(t, 'text', '') or '').strip()
        if not text:
            continue
        words.append({
            "word": text,
            "start": ts,
            "end": te,
            "confidence": getattr(t, 'probability', None)
        })
    return words


def make_transcript_message(session_id: str, line, utterance_id: str, is_final: bool, language_fallback: Optional[str], words: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    utterance = {
        "text": (line.text or "").strip(),
        "start": float(line.start or 0.0),
        "end": float(line.end or line.start or 0.0),
        "language": line.detected_language or language_fallback or "auto",
        "confidence": None,
        "channel": 0,
        "words": words or [],
    }
    return {
        "type": "transcript",
        "session_id": session_id,
        "created_at": _now_iso(),
        "data": {
            "id": utterance_id,
            "is_final": bool(is_final),
            "utterance": utterance,
        },
        "error": None,
    }


def make_translation_message(session_id: str, utterance_id_int: int, original_line, target_language: Optional[str]) -> Dict[str, Any]:
    # Build translation payload matching the gist of Gladia's structure
    translated_text = (getattr(original_line, "translation", "") or "").strip()
    if not translated_text:
        return {}

    original_utt = {
        "text": (original_line.text or "").strip(),
        "start": float(original_line.start or 0.0),
        "end": float(original_line.end or original_line.start or 0.0),
        "language": getattr(original_line, "detected_language", None),
        "confidence": None,
        "channel": 0,
        "words": [],
    }
    translated_utt = {
        "text": translated_text,
        "language": target_language or "en",
        "start": float(original_line.start or 0.0),
        "end": float(original_line.end or original_line.start or 0.0),
        "channel": 0,
        "confidence": None,
        "words": [],
    }
    return {
        "type": "translation",
        "session_id": session_id,
        "created_at": _now_iso(),
        "data": {
            "utterance_id": str(utterance_id_int),
            "utterance": original_utt,
            "original_language": original_utt["language"] or "auto",
            "target_language": translated_utt["language"],
            "translated_utterance": translated_utt,
        },
        "error": None,
    }


async def gladia_results_dispatcher(websocket: WebSocket, state: GladiaSessionState, results_generator, engine_args, audio_processor: AudioProcessor) -> None:
    """Consume FrontData objects from the existing pipeline and emit Gladia-like messages."""
    language_fallback = getattr(engine_args, "lan", None)
    target_language = getattr(engine_args, "target_language", None)

    try:
        async for front in results_generator:
            # 'front' is a FrontData object with .lines (List[Line]) and .buffer_transcription
            lines = [ln for ln in (front.lines or []) if (ln and ln.text)]
            if not lines:
                continue

            # Snapshot tokens for word-level timestamps
            try:
                current_state = await audio_processor.get_current_state()
                tokens_snapshot = getattr(current_state, 'tokens', [])
            except Exception:
                tokens_snapshot = []

            # Track latest line objects
            for i, ln in enumerate(lines):
                state.lines_by_index[i] = ln

            # Finalize all but the last line
            for i in range(0, len(lines) - 1):
                if i not in state.line_id_map:
                    utt_id_str, _ = state.next_utterance_id()
                    state.line_id_map[i] = utt_id_str
                if i not in state.finalized_line_indexes:
                    utt_id = state.line_id_map[i]
                    words = _collect_words_for_line(tokens_snapshot, lines[i])
                    msg = make_transcript_message(state.session_id, lines[i], utt_id, True, language_fallback, words)
                    await websocket.send_json(msg)
                    state.finalized_line_indexes.add(i)
                    # Cache full transcript for summary
                    state.all_finals[utt_id] = msg["data"]["utterance"]

            # Handle the last (current) line as partial
            last_idx = len(lines) - 1
            if last_idx >= 0:
                if last_idx not in state.line_id_map:
                    utt_id_str, _ = state.next_utterance_id()
                    state.line_id_map[last_idx] = utt_id_str
                utt_id = state.line_id_map[last_idx]
                partial_text = (lines[last_idx].text or "").strip()
                if partial_text and partial_text != state.prev_partial_text:
                    words = _collect_words_for_line(tokens_snapshot, lines[last_idx])
                    msg = make_transcript_message(state.session_id, lines[last_idx], utt_id, False, language_fallback, words)
                    await websocket.send_json(msg)
                    state.prev_partial_text = partial_text

            # Emit translations for newly finalized lines when available
            if target_language:
                for i in range(0, len(lines) - 1):
                    if i in state.finalized_line_indexes and i not in state.translation_sent_line_indexes:
                        tmsg = make_translation_message(state.session_id, i, lines[i], target_language)
                        if tmsg:
                            await websocket.send_json(tmsg)
                            state.translation_sent_line_indexes.add(i)

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while dispatching results.")
    except Exception as e:
        logger.exception(f"Error in gladia_results_dispatcher: {e}")


def build_session_summary(state: GladiaSessionState) -> Dict[str, Any]:
    # Compose a final transcript summary akin to a closing message
    utterances = []
    # Ensure the last line is marked final if it exists but wasn't finalized
    if state.lines_by_index:
        max_idx = max(state.lines_by_index.keys())
        if max_idx not in state.finalized_line_indexes:
            utt_id = state.line_id_map.get(max_idx)
            line = state.lines_by_index.get(max_idx)
            if utt_id and line:
                state.all_finals[utt_id] = {
                    "text": (line.text or "").strip(),
                    "start": float(line.start or 0.0),
                    "end": float(line.end or line.start or 0.0),
                    "language": getattr(line, "detected_language", None) or getattr(state.engine_args, "lan", None) or "auto",
                    "confidence": None,
                    "channel": 0,
                    "words": [],
                }

    # Preserve order by index
    for i in sorted(state.line_id_map.keys()):
        utt_id = state.line_id_map.get(i)
        if not utt_id:
            continue
        utterances.append({
            "id": utt_id,
            "utterance": state.all_finals.get(utt_id, {}),
        })

    return {
        "type": "session_summary",
        "session_id": state.session_id,
        "created_at": _now_iso(),
        "data": {
            "utterances": utterances,
        },
        "error": None,
    }


@app.websocket("/v2/live")
async def live_websocket(websocket: WebSocket, token: str):
    global transcription_engine

    if token not in _sessions:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    logger.info(f"Gladia-like WebSocket connected. token={token}")

    # Force PCM input mode for this API (client sends raw PCM base64)
    try:
        setattr(transcription_engine.args, "pcm_input", True)
    except Exception:
        pass

    # Create per-connection audio processor using the shared engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )

    # Start processing tasks and transform their output into Gladia messages
    results_generator = await audio_processor.create_tasks()
    session_state = GladiaSessionState(session_id=token, engine_args=transcription_engine.args)

    dispatcher_task = asyncio.create_task(gladia_results_dispatcher(websocket, session_state, results_generator, transcription_engine.args, audio_processor))

    try:
        while True:
            try:
                # Client can send JSON or binary frames
                msg = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client during receive loop.")
                break

            if "bytes" in msg and msg["bytes"] is not None:
                audio_bytes = msg["bytes"]
                await audio_processor.process_audio(audio_bytes)
                await send_ack_for_audio_chunk(websocket, session_state, len(audio_bytes))
                continue

            if "text" in msg and msg["text"]:
                try:
                    message = json.loads(msg["text"])  # type: ignore
                except Exception:
                    message = None
                if not message:
                    continue

                mtype = message.get("type")
                if mtype == "audio_chunk":
                    data = message.get("data", {})
                    chunk_b64 = data.get("chunk")
                    if not chunk_b64:
                        continue
                    try:
                        audio_bytes = base64.b64decode(chunk_b64)
                    except Exception:
                        continue

                    await audio_processor.process_audio(audio_bytes)
                    await send_ack_for_audio_chunk(websocket, session_state, len(audio_bytes))

                elif mtype == "stop_recording":
                    # Signal end of stream and break
                    await audio_processor.process_audio(b"")
                    break

                else:
                    # Unknown message types can be ignored or logged
                    logger.debug(f"Ignoring message type: {mtype}")
            else:
                # No actionable data
                continue

    except Exception as e:
        logger.error(f"Unexpected error in live_websocket loop: {e}", exc_info=True)

    finally:
        logger.info("Cleaning up Gladia-like WebSocket endpoint...")
        # Wait shortly for the dispatcher to flush remaining outputs
        try:
            await asyncio.wait_for(dispatcher_task, timeout=2.0)
        except Exception:
            if not dispatcher_task.done():
                dispatcher_task.cancel()
                try:
                    await dispatcher_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"Exception while awaiting dispatcher_task completion: {e}")

        await audio_processor.cleanup()

        # Send final session summary
        try:
            summary = build_session_summary(session_state)
            await websocket.send_json(summary)
        except Exception:
            # Socket may already be closed
            pass

        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Gladia-like WebSocket endpoint cleaned up.")


def main():
    import uvicorn

    uvicorn_kwargs = {
        "app": "whisperlivekit.gladia_api_server:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main() 