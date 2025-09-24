from tts_lib import tts_engines
import time 
import soundfile as sf

# # not needed it espeak is found, but if you encounter problems, consider setting the path
# # this might be different on your system (see Readme for installation instructions)
# import os
# os.environ['PHONEMIZER_ESPEAK_LIBRARY']="/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib"


# text = 'In a dramatic overnight operation, India said it launched missile and air strikes on nine sites across Pakistan.'
text = "The quick brown fox jumps over the lazy dog. Dr Smith asked whether it's 4:30 PM today."
target_sr = 16000

print(f"Running Kitten TTS...")
kitten_tts = tts_engines.TTS_KittenTTS()
t1 = time.time()
audio, sampling_rate = kitten_tts.synthesize(text, target_sr=target_sr)
print('>> kittentts synthesis time:', time.time()-t1)
sf.write('kitten_tts.wav', audio, target_sr)

print(f"Running Piper TTS...")
piper_tts = tts_engines.TTS_Piper()
t1 = time.time()
audio_float32, sampling_rate = piper_tts.synthesize(text, target_sr)
print('>> piper synthesis time:', time.time()-t1)
sf.write('piper_tts.wav', audio_float32, target_sr)

print(f"Running Kokoro TTS...")
kokoro_model = tts_engines.TTS_Kokoro()
t1 = time.time()
audio_float32, sampling_rate = kokoro_model.synthesize(text, target_sr)
print('>> kokoro synthesis time:', time.time()-t1)
sf.write('kokoro_tts.wav', audio_float32, target_sr)

