Compare different small TTS models running on edge devices.

# Installation

## System Dependencies

* **PortAudio** (required for audio playback):
    * on Mac: ```brew install portaudio```
    * on Linux (Ubuntu/Debian): ```sudo apt-get install portaudio19-dev```
    * on Linux (Fedora/RHEL): ```sudo dnf install portaudio-devel```

* **espeak-ng** (required for phonemization):
    * on Mac: ```brew install espeak-ng```
    * on Linux: ```sudo apt-get install espeak-ng```
    * ensure ```PHONEMIZER_ESPEAK_LIBRARY``` is set, if it isn't follow below instruction to set it:
        * find the library with: ```brew list espeak-ng | grep dylib```
        * set accordingly, eg: ```export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib```

## Python Dependencies

* ```pip install -r requirements.txt```
* ```pip install -e .```

## Download Models

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

## Non-Streaming

environment | platform | Piper | Kokoro | PocketTTS Onnx | 
| -- | -- | -- | -- | --|
| Raspberry Pi 5 | CPU | 0.53s ± 0.00s | 4.80s ± 0.01s | 6.98s ± 0.65s |
| Orange Pi 5 pro | CPU | 0.54s ± 0.01s | 4.09s ± 0.07s | 5.43s ± 0.36s |
| MacBook Pro M2 | CPU | 0.15s ± 0.01s | 1.20s ± 0.02s | 1.25s ± 0.08s  |

## Streaming

* PocketTTS supports output real streaming
* We're here testing how well that works on different platforms
* For PocketTTS, a parameter to control the trade-off between speed and quality is LSD (Least Significant Digit/Diffusion) steps. Default is 10.
* We're reporting RTF for different LSD params, as well as:
    * Time to First Byte (TTFB)
    * Total synthesis time: total
* In all cases, we pre-buffer 4 chunks before playback starts to prevent audio cutoff and stuttering

--> on RPI, even with LSD steps reduced to 1, we're still facing a RTF > 1.0 and stuttering is audible.

environment     | platform | LSD steps |  TTFB | TTFA | total time | RTF | audio duration |
--- | --- | --- | --- | -- | -- | -- | -- |
| MacBook Pro M2 | CPU  | 10  | 0.080s ± 0.006s | 0.130s ± 0.157s | 1.46s ± 0.09s | 0.21x |6.56s |
| Raspberry Pi 5 | CPU | 10 |0.311s ± 0.008s | 0.347s ± 0.118s | 8.92s ± 0.42s | 1.28x | 6.96s |
| Raspberry Pi 5 | CPU | 5 | 0.277s ± 0.002s | 0.308s ± 0.098s  | 7.57s ± 0.14s | 1.15x | 6.56s |
| Orange Pi 5 pro | CPU | 0.390s ± 0.000s|0.831s ± 0.000s| 12.25s ± 0.00s | 1.76x  | 6.96s|


# Licence


## Pocket-TTS-Onnx


- **Models**: CC BY 4.0 (inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))
- **Code**: Apache 2.0

See details (also for prohibited use): https://huggingface.co/KevinAHM/pocket-tts-onnx/blob/main/README.md
