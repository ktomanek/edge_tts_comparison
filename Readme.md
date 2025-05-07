Compare different small TTS models running on edge devices.

# Installation

## Kokoro

installs all deps

```pip install -r requirements.txt```

## Piper

while there is ```pip install piper-tts``` this doesn't work due to circular dependencies and a conflict on the version on piper-phonemize.

The best way to resolve this seems to be:

* ``` git clone https://github.com/rhasspy/piper.git```
* edit ```cd src/python_run/requirements.txt``` to contain only:
```
onnxruntime
piper-phonemize-cross
```
* then run ```pip install .``` in the ```cd src/python_run/``` directory

## Download models

see ```download_kokoro_models.sh``` and ```download_piper_models.sh```
