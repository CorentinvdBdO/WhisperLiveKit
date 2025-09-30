try:
    from whisperlivekit.whisper_streaming_custom.whisper_online import backend_factory
    from whisperlivekit.whisper_streaming_custom.online_asr import OnlineASRProcessor
except ImportError:
    from .whisper_streaming_custom.whisper_online import backend_factory
    from .whisper_streaming_custom.online_asr import OnlineASRProcessor
from argparse import Namespace
import sys

def update_with_kwargs(_dict, kwargs):
    _dict.update({
        k: v for k, v in kwargs.items() if k in _dict
    })
    return _dict

class TranscriptionEngine:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, **kwargs):
        if TranscriptionEngine._initialized:
            return

        global_params = {
            "host": "localhost",
            "port": 8000,
            "diarization": False,
            "punctuation_split": False,
            "target_language": "",
            "vac": True,
            "vac_chunk_size": 0.04,
            "log_level": "DEBUG",
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "transcription": True,
            "vad": True,
            "pcm_input": False,
            "disable_punctuation_split" : False,
            "diarization_backend": "sortformer",
        }
        global_params = update_with_kwargs(global_params, kwargs)

        transcription_common_params = {
            "backend": "simulstreaming",
            "warmup_file": None,
            "min_chunk_size": 0.3,  # Reduced from 0.5 for more responsive transcription
            "model_size": "tiny",
            "model_cache_dir": None,
            "model_dir": None,
            "lan": "auto",
            "task": "transcribe",
        }
        transcription_common_params = update_with_kwargs(transcription_common_params, kwargs)                                            

        if 'no_transcription' in kwargs:
            global_params['transcription'] = not global_params['no_transcription']
        if 'no_vad' in kwargs:
            global_params['vad'] = not kwargs['no_vad']
        if 'no_vac' in kwargs:
            global_params['vac'] = not kwargs['no_vac']

        self.args = Namespace(**{**global_params, **transcription_common_params})
        
        self.asr = None
        self.tokenizer = None
        self.diarization = None
        self.vac_model = None
        
        if self.args.vac:
            import torch
            self.vac_model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")            
        
        if self.args.transcription:
            if self.args.backend == "simulstreaming": 
                from whisperlivekit.simul_whisper import SimulStreamingASR
                
                simulstreaming_params = {
                    "disable_fast_encoder": False,
                    "custom_alignment_heads": None,
                    "frame_threshold": 25,
                    "beams": 1,
                    "decoder_type": None,
                    "audio_max_len": 20.0,
                    "audio_min_len": 0.0,
                    "cif_ckpt_path": None,
                    "never_fire": False,
                    "init_prompt": None,
                    "static_init_prompt": None,
                    "max_context_tokens": None,
                    "model_path": './base.pt',
                    "preload_model_count": 1,
                }
                simulstreaming_params = update_with_kwargs(simulstreaming_params, kwargs)
                
                self.tokenizer = None        
                self.asr = SimulStreamingASR(
                    **transcription_common_params, **simulstreaming_params
                )
            else:
                
                whisperstreaming_params = {
                    "buffer_trimming": "segment",
                    "confidence_validation": False,
                    "buffer_trimming_sec": 15,
                }
                whisperstreaming_params = update_with_kwargs(whisperstreaming_params, kwargs)
                
                self.asr = backend_factory(
                    **transcription_common_params, **whisperstreaming_params
                )

        if self.args.diarization:
            if self.args.diarization_backend == "diart":
                from whisperlivekit.diarization.diart_backend import DiartDiarization
                diart_params = {
                    "segmentation_model": "pyannote/segmentation-3.0",
                    "embedding_model": "pyannote/embedding",
                }
                diart_params = update_with_kwargs(diart_params, kwargs)
                self.diarization_model = DiartDiarization(
                    block_duration=self.args.min_chunk_size,
                    **diart_params
                )
            elif self.args.diarization_backend == "sortformer":
                from whisperlivekit.diarization.sortformer_backend import SortformerDiarization
                self.diarization_model = SortformerDiarization()
        
        self.translation_model = None
        if self.args.target_language:
            if self.args.lan == 'auto' and self.args.backend != "simulstreaming":
                raise Exception('Translation cannot be set with language auto when transcription backend is not simulstreaming')
            else:
                # Check if LLM translation is requested
                translation_backend = kwargs.get('translation_backend', 'nllb')
                
                if translation_backend == 'llm':
                    from whisperlivekit.translation.translation import load_llm_model
                    llm_params = {
                        "api_url": kwargs.get('llm_api_url', "http://localhost:1717"),
                        "model_name": kwargs.get('llm_model_name', "openai/gpt-oss-120b"),
                        "context_size": kwargs.get('llm_context_size', 5)
                    }
                    self.translation_model = load_llm_model(**llm_params)
                else:
                    # Default NLLB translation
                    from whisperlivekit.translation.translation import load_model
                    translation_params = { 
                        "nllb_backend": "ctranslate2",
                        "nllb_size": "600M"
                    }
                    translation_params = update_with_kwargs(translation_params, kwargs)
                    self.translation_model = load_model([self.args.lan], **translation_params) #in the future we want to handle different languages for different speakers
        TranscriptionEngine._initialized = True


def online_factory(args, asr):
    if args.backend == "simulstreaming":    
        from whisperlivekit.simul_whisper import SimulStreamingOnlineProcessor
        online = SimulStreamingOnlineProcessor(asr)
    else:
        online = OnlineASRProcessor(asr)
    return online
  
  
def online_diarization_factory(args, diarization_backend):
    if args.diarization_backend == "diart":
        online = diarization_backend
        # Not the best here, since several user/instances will share the same backend, but diart is not SOTA anymore and sortformer is recommended
    
    if args.diarization_backend == "sortformer":
        from whisperlivekit.diarization.sortformer_backend import SortformerDiarizationOnline
        online = SortformerDiarizationOnline(shared_model=diarization_backend)
    return online


def online_translation_factory(args, translation_model):
    #should be at speaker level in the future:
    #one shared nllb model for all speaker
    #one tokenizer per speaker/language
    
    # Check the backend type and use appropriate translation class
    if hasattr(translation_model, 'backend_type') and translation_model.backend_type == 'llm':
        from whisperlivekit.translation.translation import LLMOnlineTranslation
        return LLMOnlineTranslation(translation_model, [args.lan], [args.target_language])
    else:
        from whisperlivekit.translation.translation import OnlineTranslation
        return OnlineTranslation(translation_model, [args.lan], [args.target_language])