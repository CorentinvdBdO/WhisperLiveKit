import logging
import time
import ctranslate2
import torch
import transformers
from dataclasses import dataclass, field
import huggingface_hub
from whisperlivekit.translation.mapping_languages import get_nllb_code
from whisperlivekit.timed_objects import Translation
import requests
import json
from collections import deque

logger = logging.getLogger(__name__)

#In diarization case, we may want to translate just one speaker, or at least start the sentences there

MIN_SILENCE_DURATION_DEL_BUFFER = 3 #After a silence of x seconds, we consider the model should not use the buffer, even if the previous
# sentence is not finished.

@dataclass
class TranslationModel():
    translator: ctranslate2.Translator
    device: str
    tokenizer: dict = field(default_factory=dict)
    backend_type: str = 'ctranslate2'
    nllb_size: str = '600M'
    
    def get_tokenizer(self, input_lang):
        if not self.tokenizer.get(input_lang, False):
            self.tokenizer[input_lang] = transformers.AutoTokenizer.from_pretrained(
                f"facebook/nllb-200-distilled-{self.nllb_size}",
                src_lang=input_lang,
                clean_up_tokenization_spaces=True
            )
        return self.tokenizer[input_lang]


@dataclass 
class LLMTranslationModel():
    api_url: str
    model_name: str
    backend_type: str = 'llm'
    context_size: int = 5  # Number of previous message pairs to include for context
    
    def __post_init__(self):
        # Ensure URL ends with /v1
        if not self.api_url.endswith('/v1'):
            if not self.api_url.endswith('/'):
                self.api_url += '/'
            self.api_url += 'v1'


def load_model(src_langs, nllb_backend='ctranslate2', nllb_size='600M'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = f'nllb-200-distilled-{nllb_size}-ctranslate2'
    if nllb_backend=='ctranslate2':
        MODEL_GUY = 'entai2965'
        huggingface_hub.snapshot_download(MODEL_GUY + '/' + MODEL,local_dir=MODEL)
        translator = ctranslate2.Translator(MODEL,device=device)
    elif nllb_backend=='transformers':
        translator = transformers.AutoModelForSeq2SeqLM.from_pretrained(f"facebook/nllb-200-distilled-{nllb_size}")
    tokenizer = dict()
    for src_lang in src_langs:
        if src_lang != 'auto':
            tokenizer[src_lang] = transformers.AutoTokenizer.from_pretrained(MODEL, src_lang=src_lang, clean_up_tokenization_spaces=True)

    translation_model = TranslationModel(
        translator=translator,
        device=device,
        tokenizer=tokenizer,
        backend_type=nllb_backend,
        nllb_size=nllb_size
    )
    return translation_model


def load_llm_model(api_url: str = "http://localhost:1717", model_name: str = "openai/gpt-oss-120b", context_size: int = 5):
    """Load LLM translation model that calls a VLLM OpenAI-compatible server"""
    return LLMTranslationModel(
        api_url=api_url,
        model_name=model_name,
        context_size=context_size
    )

class LLMOnlineTranslation:
    def __init__(self, translation_model: LLMTranslationModel, input_languages: list, output_languages: list):
        self.translation_model = translation_model
        self.input_languages = input_languages
        self.output_languages = output_languages
        self.context_history = deque(maxlen=translation_model.context_size)  # Store (original, translated) pairs
        
        # Add buffer management like NLLB translation
        self.buffer = []
        self.len_processed_buffer = 0
        self.translation_remaining = Translation()
        self.validated = []
        
        # Test API connectivity
        self._test_api_connectivity()
    
    def _test_api_connectivity(self):
        """Test if the LLM API is accessible"""
        try:
            # Simple test with minimal message
            test_messages = [
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": "Hello"}
            ]
            
            url = f"{self.translation_model.api_url}/chat/completions"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.translation_model.model_name,
                    "messages": test_messages,
                    "max_tokens": 8192,
                    "temperature": 1,
                    "reasoning_effort": "low"
                },
                timeout=50
            )
            
            if response.status_code == 200:
                logger.info(f"LLM API connectivity test successful at {url}")
            else:
                logger.warning(f"LLM API connectivity test failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.warning(f"LLM API connectivity test failed: {e}")
        
    def _build_system_prompt(self, input_lang: str, output_lang: str) -> str:
        """Build system prompt for translation"""
        return f"""You are a professional translator. Translate text from {input_lang} to {output_lang}.

Rules:
- Translate only the last user message
- Maintain the original meaning and tone
- Keep the same level of formality
- Preserve proper nouns when appropriate
- For partial or incomplete sentences, provide the best translation possible
- Respond only with the translation, no explanations"""

    def _build_conversation_messages(self, input_text: str, input_lang: str, output_lang: str) -> list:
        """Build conversation messages with context and the new input"""
        messages = [
            {"role": "system", "content": self._build_system_prompt(input_lang, output_lang)}
        ]
        
        # Add context from previous translations
        for original, translated in self.context_history:
            messages.append({"role": "user", "content": original})
            messages.append({"role": "assistant", "content": translated})
        
        # Add the current text to translate
        messages.append({"role": "user", "content": input_text})
        
        return messages

    def _call_llm_api(self, messages: list) -> str:
        """Call the LLM API with the conversation messages"""
        try:
            url = f"{self.translation_model.api_url}/chat/completions"
            payload = {
                "model": self.translation_model.model_name,
                "messages": messages,
                "max_tokens": 8192,
                "temperature": 1,
                "reasoning_effort": "low"
            }
            
            logger.debug(f"Calling LLM API at {url} with payload: {payload}")
            
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"LLM API response: {result}")
            
            # Handle potential None content
            content = result.get("choices", [{}])[0].get("message", {}).get("content")
            if content is None:
                logger.error(f"LLM API returned None content. Full response: {result}")
                return "[Translation error: API returned no content]"
            
            translation = content.strip()
            return translation
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request failed: {e}")
            return f"[Translation error: {str(e)}]"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"LLM API response parsing failed: {e}")
            return "[Translation error: Invalid response format]"

    def translate(self, input_text: str, input_lang: str, output_lang: str) -> str:
        """Translate text using LLM"""
        if not input_text.strip():
            return ""
            
        logger.debug(f"Translating: '{input_text}' from {input_lang} to {output_lang}")
        
        messages = self._build_conversation_messages(input_text, input_lang, output_lang)
        translation = self._call_llm_api(messages)
        
        logger.debug(f"Translation result: '{translation}'")
        
        # Only add successful translations to context history (not error messages)
        if not translation.startswith("[Translation error:"):
            self.context_history.append((input_text, translation))
        
        return translation

    def translate_tokens(self, tokens):
        """Translate tokens - compatible with existing interface"""
        if not tokens:
            return None
            
        text = ' '.join([token.text for token in tokens])
        start = tokens[0].start
        end = tokens[-1].end
        
        if self.input_languages[0] == 'auto':
            input_lang = getattr(tokens[0], 'detected_language', 'auto')
            if input_lang == 'auto':
                input_lang = 'French'  # Fallback
        else:
            input_lang = self.input_languages[0]
            
        # Convert language codes to full names for better LLM understanding
        lang_mapping = {
            'fr': 'French',
            'en': 'English', 
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'auto': 'French'  # Fallback
        }
        
        input_lang_name = lang_mapping.get(input_lang, input_lang)
        output_lang_name = lang_mapping.get(self.output_languages[0], self.output_languages[0])
        
        translated_text = self.translate(text, input_lang_name, output_lang_name)
        
        translation = Translation(
            text=translated_text,
            start=start,
            end=end,
        )
        return translation

    def insert_tokens(self, tokens):
        """Insert tokens into buffer for processing"""
        self.buffer.extend(tokens)

    def process(self):
        """Process buffered tokens and return translated segments"""
        if len(self.buffer) < self.len_processed_buffer + 3: # nothing new to process
            return self.validated + [self.translation_remaining] if self.translation_remaining else self.validated
            
        i = 0
        while i < len(self.buffer):
            if self.buffer[i].is_punctuation():
                translation_sentence = self.translate_tokens(self.buffer[:i+1])
                if translation_sentence:
                    self.validated.append(translation_sentence)
                self.buffer = self.buffer[i+1:]
                i = 0
            else:
                i += 1
                
        # Translate remaining tokens
        self.translation_remaining = self.translate_tokens(self.buffer)
        self.len_processed_buffer = len(self.buffer)
        
        result = self.validated.copy()
        if self.translation_remaining:
            result.append(self.translation_remaining)
        return result

    def insert_silence(self, silence_duration: float):
        """Handle silence - clear context if too long"""
        if silence_duration > MIN_SILENCE_DURATION_DEL_BUFFER:
            self.context_history.clear()
            self.buffer = []
            if self.translation_remaining:
                self.validated.append(self.translation_remaining)
                self.translation_remaining = Translation()


class OnlineTranslation:
    def __init__(self, translation_model: TranslationModel, input_languages: list, output_languages: list):
        self.buffer = []
        self.len_processed_buffer = 0
        self.translation_remaining = Translation()
        self.validated = []
        self.translation_pending_validation = ''
        self.translation_model = translation_model
        self.input_languages = input_languages
        self.output_languages = output_languages

    def compute_common_prefix(self, results):
        #we dont want want to prune the result for the moment. 
        if not self.buffer:
            self.buffer = results
        else:
            for i in range(min(len(self.buffer), len(results))):
                if self.buffer[i] != results[i]:
                    self.commited.extend(self.buffer[:i])
                    self.buffer = results[i:]

    def translate(self, input, input_lang, output_lang):
        if not input:
            return ""
        nllb_output_lang = get_nllb_code(output_lang)
            
        tokenizer = self.translation_model.get_tokenizer(input_lang)
        tokenizer_output = tokenizer(input, return_tensors="pt").to(self.translation_model.device)
        
        if self.translation_model.backend_type == 'ctranslate2':
            source = tokenizer.convert_ids_to_tokens(tokenizer_output['input_ids'][0])    
            results = self.translation_model.translator.translate_batch([source], target_prefix=[[nllb_output_lang]])
            target = results[0].hypotheses[0][1:]
            result = tokenizer.decode(tokenizer.convert_tokens_to_ids(target))
        else:
            translated_tokens = self.translation_model.translator.generate(**tokenizer_output, forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_output_lang))
            result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return result
    
    def translate_tokens(self, tokens):
        if tokens:
            text = ' '.join([token.text for token in tokens])
            start = tokens[0].start
            end = tokens[-1].end
            if self.input_languages[0] == 'auto':
                input_lang = tokens[0].detected_language
            else:
                input_lang = self.input_languages[0]
                
            translated_text = self.translate(text,
                                            input_lang,
                                            self.output_languages[0]
                                            )
            translation = Translation(
                text=translated_text,
                start=start,
                end=end,
            )
            return translation
        return None
            

    def insert_tokens(self, tokens):
        self.buffer.extend(tokens)
        pass
    
    def process(self):
        i = 0
        if len(self.buffer) < self.len_processed_buffer + 3: #nothing new to process
            return self.validated + [self.translation_remaining]
        while i < len(self.buffer):
            if self.buffer[i].is_punctuation():
                translation_sentence = self.translate_tokens(self.buffer[:i+1])
                self.validated.append(translation_sentence)
                self.buffer = self.buffer[i+1:]
                i = 0
            else:
                i+=1
        self.translation_remaining = self.translate_tokens(self.buffer)
        self.len_processed_buffer = len(self.buffer)
        return self.validated + [self.translation_remaining]

    def insert_silence(self, silence_duration: float):
        if silence_duration >= MIN_SILENCE_DURATION_DEL_BUFFER:
            self.buffer = []
            self.validated += [self.translation_remaining]

if __name__ == '__main__':
    output_lang = 'fr'
    input_lang = "en"
    
    
    test_string = """
    Transcription technology has improved so much in the past few years. Have you noticed how accurate real-time speech-to-text is now?
    """
    test = test_string.split(' ')
    step = len(test) // 3
    
    shared_model = load_model([input_lang], nllb_backend='ctranslate2')
    online_translation = OnlineTranslation(shared_model, input_languages=[input_lang], output_languages=[output_lang])
    
    beg_inference = time.time()    
    for id in range(5):
        val = test[id*step : (id+1)*step]
        val_str = ' '.join(val)
        result = online_translation.translate(val_str)
        print(result)
    print('inference time:', time.time() - beg_inference)