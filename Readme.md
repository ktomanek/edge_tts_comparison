Compare different small TTS models running on edge devices.

# Installation

* ```pip install -r requirements.txt```
* ```pip install -e .```

* ensure espeak (or espeak-ng) is installed on your system and the library path set
    * eg, on Mac install: ```brew install espeak-ng```
    * ensure ```PHONEMIZER_ESPEAK_LIBRARY``` is set, if it isn't follow below instruction to set it:
        * find the library with: ```brew list espeak-ng | grep dylib```
        * set accordingly, eg: ```export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib```

## KittenTTS

* code and dependencies included in this package
* download model files: ```sh download_kittentts_models.sh```

## Kokoro

* install all deps: ```pip install -r requirements.txt```
* download model files: ```sh download_kokoro_models.sh```

## Piper

while there is ```pip install piper-tts``` this doesn't work due to circular dependencies and a conflict on the version on piper-phonemize.

The best way to resolve this seems to be:

* ```pip install onnxruntime piper-phonemize-cross```
* ```git clone https://github.com/rhasspy/piper.git``` 
* delete the requirements file: ```rm piper/src/python_run/requirements.txt```
* then ```cd piper/src/python_run/``` and run run ```pip install .```

* download model files: ```sh download_piper_models.sh```


# Performance


* for Kokoro, running ```kokoro-v1.0.fp16.onnx```
* for Piper, running ```en_US-lessac-low.onnx```
* for Kitten, running ```kitten_tts_nano_v0_1.onnx```
* text: 
    * ```The quick brown fox jumps over the lazy dog. Dr. Smith asked whether it's 3:30 PM today.```
    * (text chosen to covers key phonetic elements, numbers, punctuation, and common pronunciation challenges in a short test case)

inference time after warmup:

environment | Kokoro | Piper | KittenTTS
| -- | -- | -- |
| MacBook Pro M2 | 0.75s | 0.085s| 0.68s
| Raspberry Pi 5 | 8.0s| 0.66s| - 




# Stream LLM output into TTS

Generate text output via an LLM using Ollama and synthesize speech in streaming-fashing.
Using Piper, this works in realtime; Kokoro seems too slow for that even on a Mac M2.

For Kokoro, when increasing the speaking speed, first word is often cut off.
## Installation

* install ollama locally: https://ollama.com/download
* then pull the model you want ot use, eg: 

```ollama pull gemma3:1b```

* then install [ollama python library](https://github.com/ollama/ollama-python) 

```pip install ollama```

* other dependencies

* ```pip install sounddevice nltk```
* download sentence splitter: ```python -c "import nltk; nltk.download('punkt_tab')```

## Run

```
python stream_llm_to_tts.py \
    --ollama-model-name=gemma3:1b \
    --tts_engine piper  \
    --speaking-rate=2.0 \
    --prompt "Explain to me what a cat does all day. Use exactly 3 sentences."
```


```
python stream_llm_to_tts.py \
    --ollama-model-name=gemma3:1b \
    --tts_engine kokoro  \
    --speaking-rate=1.0 \
    --prompt "Explain to me what a cat does all day. Use exactly 3 sentences."
```