Compare different small TTS models running on edge devices.

# Installation

* ```pip install -r requirements.txt```
* ```pip install -e .```

* ensure espeak (or espeak-ng) is installed on your system and the library path set
    * on Mac install: ```brew install espeak-ng```
    * on Linux: ```sudo apt-get install espeak-ng```
    * ensure ```PHONEMIZER_ESPEAK_LIBRARY``` is set, if it isn't follow below instruction to set it:
        * find the library with: ```brew list espeak-ng | grep dylib```
        * set accordingly, eg: ```export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib```

* download models:
    * kokoro: ```sh download_kokoro_models.sh``` (see different models in the script)
    * piper: ```sh download_piper_models.sh``` (many other voices are available!)
    * pocket-tts-onnx: ```sh download_pocket_tts.sh``` (onnx models)

## Included Models

The core installation only includes models with onnx runtime, to keep dependencies to a minimum.

* Piper TTS
* Kokoro TTS
* ONNX-port of Pocket TTS
    * Note: for pocket-tts-onnx we included the core code from https://huggingface.co/KevinAHM/pocket-tts-onnx and minimally adapted it. Models need to downloaded directly from above repo with the provided script.

## Optional TTS models

The dependencies for the following models aren't installed directly, to avoid extensive dependencies (eg, pytorch). To use, please install `pip install -r requirements_optional.txt`.

* [Kitten-TTS](https://github.com/KittenML/KittenTTS)
* original [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)

# Performance

* for Kokoro, running ```kokoro-v1.0.fp16.onnx```
* for Piper, running ```en_US-lessac-low.onnx```
* for Kitten, running ```kitten_tts_nano_v0_1.onnx```
* for PocketTTS
* for PocketTTSOnnx
* text: 
    * ```The quick brown fox jumps over the lazy dog. Dr. Smith asked whether it's 3:30 PM today.```
    * (text chosen to covers key phonetic elements, numbers, punctuation, and common pronunciation challenges in a short test case)



Reporting mean ± stdev over 10 runs, inference time after 3x warmup.

environment | platform | Piper | Kokoro | PocketTTS Onnx | 
| -- | -- | -- | -- | --|
| Raspberry Pi 5 | CPU  0.54s | 4.83s | xx 
| MacBook Pro M2 | CPU | 0.15s ± 0.01s | 1.20s ± 0.02s | 1.25s ± 0.08s 
| Orange Pi 5 pro | CPU |  xxxs | xxxs | xx 


# Licence


## Pocket-TTS-Onnx


- **Models**: CC BY 4.0 (inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))
- **Code**: Apache 2.0

See details (also for prohibited use): https://huggingface.co/KevinAHM/pocket-tts-onnx/blob/main/README.md
