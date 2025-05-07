# convenvience wrapper for several TTS libraries running on edge devices
import numpy as np
import librosa


class TTS:
    def __init__(self):
        pass

    def synthesize(self, text: str):
        raise NotImplementedError("This method should be overridden by subclasses.")


class TTS_Piper(TTS):
    def __init__(self, model_path: str):
        super().__init__()
        import piper

        self.model_path = model_path
        self.piper_voice = piper.voice.PiperVoice.load(model_path)
        self.sampling_rate = self.piper_voice.config.sample_rate        
        print('SAMPLE RATE:', self.sampling_rate)

    def synthesize(self, text: str, target_sr=16000):
        it = self.piper_voice.synthesize_stream_raw(text)
        # 16 bit audio bytes
        audio_bytes = b''.join(it)

        # convert to float32 in [-1, 1] range
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_np.astype(np.float32) / 32767.0

        # possibly resample to 16kHz
        if self.sampling_rate == target_sr:
            return audio_float32, self.sampling_rate
        else:
            print('resampling from', self.sampling_rate, 'to', target_sr)
            return librosa.resample(audio_float32, orig_sr=self.sampling_rate, target_sr=target_sr), target_sr


class TTS_Kokoro(TTS):

    def __init__(self, model_path: str='kokoro-v1.0.fp16.onnx', voice_path: str='kokoro-voices-v1.0.bin',  speaker_voice: str='am_eric'):
        super().__init__()
        from kokoro_onnx import Kokoro
        from kokoro_onnx.tokenizer import Tokenizer

        self.model_path = model_path
        self.voice_path = voice_path
        self.speaker_voice = speaker_voice
        self.tokenizer = Tokenizer()
        self.kokoro = Kokoro(model_path, voice_path)
        self.language = "en-us"
        self.voice = self.kokoro.get_voice_style(speaker_voice)

    def synthesize(self, text: str, target_sr=16000):
        phonemes = self.tokenizer.phonemize(text, lang=self.language)
        samples, sample_rate = self.kokoro.create(
            phonemes, voice=self.voice, speed=1.0, is_phonemes=True
        )        
        if sample_rate == target_sr:
            return samples, sample_rate
        else:
            print('resampling from', sample_rate, 'to', target_sr)
            return librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr), target_sr