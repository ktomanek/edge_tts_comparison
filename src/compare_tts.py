from tts_lib import tts_engines
import time 
import soundfile as sf

# # not needed it espeak is found, but if you encounter problems, consider setting the path
# # this might be different on your system (see Readme for installation instructions)
# import os
# os.environ['PHONEMIZER_ESPEAK_LIBRARY']="/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib"

run_warmup = False

text = 'In a dramatic overnight operation, India said it launched missile and air strikes on nine sites across Pakistan.'
# text = "The quick brown fox jumps over the lazy dog. Dr Smith asked whether it's 4:30 PM today."
target_sr = 16000

# Reference text for voice cloning
ref_text = text


print(f">>> Running Piper TTS...")

t1 = time.time()
piper_tts = tts_engines.TTS_Piper(warmup=run_warmup)
print('>> piper model load and warmup time:', time.time()-t1)

t1 = time.time()
audio_float32, sampling_rate = piper_tts.synthesize(text, target_sr)
print('>> piper synthesis time:', time.time()-t1)
sf.write('piper_tts.wav', audio_float32, target_sr)


print(f">>> Running Kokoro TTS...")
t1 = time.time()
kokoro_model = tts_engines.TTS_Kokoro(warmup=run_warmup)
print('>> kokoro model load and warmup time:', time.time()-t1)

t1 = time.time()
audio_float32, sampling_rate = kokoro_model.synthesize(text, target_sr)
print('>> kokoro synthesis time:', time.time()-t1)
sf.write('kokoro_tts.wav', audio_float32, target_sr)

print(f">>> Running PocketTTS ONNX...")
t1 = time.time()
ref_audio = 'kokoro_tts.wav'
pocket_tts_onnx = tts_engines.TTS_PocketTTSOnnx(voice=ref_audio, warmup=run_warmup)
#emb_path = 'pocket_tts_onnx_voice_emb.npy'
# tts_engines.TTS_PocketTTSOnnx.export_voice_embeddings(ref_audio, emb_path)
#emb = tts_engines.TTS_PocketTTSOnnx.load_voice_embeddings(emb_path)
#pocket_tts_onnx = tts_engines.TTS_PocketTTSOnnx(voice=emb, warmup=run_warmup)
print('>> pockettts onnx model load and warmup time:', time.time()-t1)

t1 = time.time()
audio, sampling_rate = pocket_tts_onnx.synthesize(text, target_sr=target_sr)
print('>> pockettts onnx synthesis time:', time.time()-t1)
sf.write('pocket_tts_onnx.wav', audio, target_sr)

print(f">>>Running PocketTTS...")
t1 = time.time()
pocket_tts = tts_engines.TTS_PocketTTS(warmup=run_warmup)
print('>> pockettts model load and warmup time:', time.time()-t1)

t1 = time.time()
audio, sampling_rate = pocket_tts.synthesize(text, target_sr=target_sr)
print('>> pockettts synthesis time:', time.time()-t1)
sf.write('pocket_tts.wav', audio, target_sr)


# print(f"Running Kitten TTS...")
# t1 = time.time()
# kitten_tts = tts_engines.TTS_KittenTTS(warmup=run_warmup)
# print('>> kittentts model load and warmup time:', time.time()-t1)

# t1 = time.time()
# audio, sampling_rate = kitten_tts.synthesize(text, target_sr=target_sr)
# print('>> kittentts synthesis time:', time.time()-t1)
# sf.write('kitten_tts.wav', audio, target_sr)
