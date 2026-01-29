from tts_lib import tts_engines
import time
import soundfile as sf
import statistics

# # not needed it espeak is found, but if you encounter problems, consider setting the path
# # this might be different on your system (see Readme for installation instructions)
# import os
# os.environ['PHONEMIZER_ESPEAK_LIBRARY']="/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib"

def benchmark_synthesis(tts_engine, text, target_sr=16000, num_runs=10):
    """Run synthesis multiple times and report statistics.

    Args:
        tts_engine: TTS engine instance
        text: Text to synthesize
        target_sr: Target sample rate
        num_runs: Number of synthesis runs (default: 10)

    Returns:
        Tuple of (audio, sample_rate) from the last run
    """
    times = []
    audio_result = None
    sr_result = None

    print(f">> Running {num_runs} synthesis iterations...")
    for i in range(num_runs):
        t1 = time.time()
        audio, sr = tts_engine.synthesize(text, target_sr=target_sr)
        elapsed = time.time() - t1
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.2f}s")

        # Keep the last result for saving to file
        audio_result = audio
        sr_result = sr

    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f">> Synthesis time: {mean_time:.2f}s ± {stdev_time:.2f}s (mean ± stdev)")

    return audio_result, sr_result

run_warmup = True

# text = 'In a dramatic overnight operation, India said it launched missile and air strikes on nine sites across Pakistan.'
text = "The quick brown fox jumps over the lazy dog. Dr Smith asked whether it's 4:30 PM today."
target_sr = 16000

# Reference text for voice cloning
ref_text = text
num_inf_runs = 10


print(f">>> Running Piper TTS...")
t1 = time.time()
piper_tts = tts_engines.TTS_Piper(warmup=run_warmup)
print(f'>> piper model load and warmup time: {time.time()-t1:.2f}')
audio_float32, sampling_rate = benchmark_synthesis(piper_tts, text, target_sr, num_runs=num_inf_runs)
sf.write('piper_tts.wav', audio_float32, target_sr)


print(f">>> Running Kokoro TTS...")
t1 = time.time()
kokoro_model = tts_engines.TTS_Kokoro(warmup=run_warmup)
print(f'>> kokoro model load and warmup time: {time.time()-t1:.2f}')
audio_float32, sampling_rate = benchmark_synthesis(kokoro_model, text, target_sr, num_runs=num_inf_runs)
sf.write('kokoro_tts.wav', audio_float32, target_sr)

print(f">>> Running PocketTTS ONNX...")
t1 = time.time()
ref_audio = 'kokoro_tts.wav'
pocket_tts_onnx = tts_engines.TTS_PocketTTSOnnx(voice=ref_audio, warmup=run_warmup)
print(f'>> pockettts onnx model load and warmup time: {time.time()-t1:.2f}')
audio, sampling_rate = benchmark_synthesis(pocket_tts_onnx, text, target_sr, num_runs=num_inf_runs)
sf.write('pocket_tts_onnx.wav', audio, target_sr)
