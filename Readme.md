Compare different small TTS models running on edge devices.

# Installation

## Kokoro

installs all deps

```pip install -r requirements.txt```

## Piper

while there is ```pip install piper-tts``` this doesn't work due to circular dependencies and a conflict on the version on piper-phonemize.

The best way to resolve this seems to be:

* ```pip install onnxruntime piper-phonemize-cross```
* ``` git clone https://github.com/rhasspy/piper.git``` 
* delete the requirements file: ```rm piper/src/python_run/requirements.txt```
* then run ```pip install .``` in the ```cd piper/src/python_run/``` directory

## Download models

see ```download_kokoro_models.sh``` and ```download_piper_models.sh```

# Performance


* for Kokoro, running ```kokoro-v1.0.fp16.onnx```
* for Piper, running ```en_US-lessac-low.onnx```
* text: Sentence with ~20 words and punctuation, about 6secs when spoken in "normal" speed.

inference time: 

environment | Kokoro | Piper
| -- | -- | -- |
| MacBook Pro M2 | 3.3s | 0.15s|
| Raspberry Pi 5 | 8.0s| 0.66s|




# Stream LLM output into TTS

Generate text output via an LLM using Ollama and generate synthesized speech output via streaming through Piper.

## Installation

* install ollama locally: https://ollama.com/download
* then pull the model you want ot use, eg: 

```ollama pull gemma3:1b```

* then install [ollama python library](https://github.com/ollama/ollama-python) 

```pip install ollama```

* other dependencies

```pip install sounddevice```

## Run

```
python stream_llm_to_tts.py \
    --ollama-model-name=gemma3:1b \
    --piper-model en_US-lessac-low.onnx  \
    --speaking-rate=4.0 \
    --prompt "Explain to me what a cat does all day. Use exactly 3 sentences."
```