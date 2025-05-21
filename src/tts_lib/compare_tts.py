from tts_lib import tts_engines
import time 
import soundfile as sf

text = 'In a dramatic overnight operation, India said it launched missile and air strikes on nine sites across Pakistan.'
target_sr = 16000

onnx_model = 'en_US-lessac-low.onnx'
piper_tts = tts_engines.TTS_Piper(onnx_model)
t1 = time.time()
audio_float32, sampling_rate = piper_tts.synthesize(text, target_sr)
print('piper synthesis time:', time.time()-t1)
sf.write('piper_tts.wav', audio_float32, target_sr)


kokoro_voice = 'am_eric'
model_path = 'kokoro-v1.0.fp16.onnx'
kokoro_model = tts_engines.TTS_Kokoro(model_path, speaker_voice=kokoro_voice)
t1 = time.time()
audio_float32, sampling_rate = kokoro_model.synthesize(text, target_sr)
print('kokoro synthesis time:', time.time()-t1)
sf.write('kokoro_tts.wav', audio_float32, target_sr)


