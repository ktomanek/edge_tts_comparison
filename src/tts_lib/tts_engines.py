# convenvience wrapper for several TTS libraries running on edge devices
import numpy as np
import librosa

class TTS:
    def __init__(self):
        pass

    def get_sample_rate(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def synthesize(self, text: str, speaking_rate:float, return_as_int16: bool):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def warmup(self):
        print("Warming up model...")
        self.synthesize("This is a warmup text to initialize the TTS engine. Cats are great, I love cats!")

class TTS_KittenTTS(TTS):
    def __init__(self, model_path: str='KittenML/kitten-tts-nano-0.2',
                 voice: str='expr-voice-2-m'):
        """There is also a supposedly better model called mini:
        "KittenML/kitten-tts-mini-0.1"
        It is noticeably slower and I can't really find a big quality gain.
        """
        
        from kittentts import KittenTTS
        print(f"Loading kittentts with model: {model_path}")
        self.kitten_tts = KittenTTS(model_path)
        self.voice_for_synthesis = voice
        self.sample_rate = 24000 # default for KittenTTS
        
        self.warmup()
        print("KittenTTS loaded and warmed up.")


    def get_sample_rate(self):
        return self.sample_rate

    # TODO select voice
    def synthesize(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
        samples = self.kitten_tts.generate(text, speed=speaking_rate, voice=self.voice_for_synthesis)
        sample_rate = self.sample_rate
        if sample_rate != target_sr:
            # print('resampling from', sample_rate, 'to', target_sr)
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        if return_as_int16:
            samples = (samples * 32767).astype(np.int16)

        return samples, sample_rate        

class TTS_Piper(TTS):
    def __init__(self, model_path: str='models/piper/en_US-lessac-low.onnx'):
        super().__init__()
        from piper.voice import PiperVoice

        self.model_path = model_path
        self.piper_voice = PiperVoice.load(model_path)
        self.sampling_rate = self.piper_voice.config.sample_rate      

        self.warmup()
        print("PiperTTS loaded and warmed up.")

    def get_sample_rate(self):
        return self.sampling_rate

    def synthesize(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
            # Create synthesis configuration
            from piper.voice import SynthesisConfig
            synthesis_config = SynthesisConfig(
                length_scale=1.0 / speaking_rate,  # Higher values = slower speech
                # You can add other parameters here like:
                # noise_scale=0.667,   # Controls variability in voice
                # noise_w=0.8          # Controls phoneme width variation
            )
            
            # Generate audio chunks - this is now the new API
            audio_chunks = list(self.piper_voice.synthesize(text, synthesis_config))
            
            # Combine all audio data from chunks
            audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
            
            # Convert to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            
            if return_as_int16 and self.sampling_rate == target_sr:
                return audio_np, self.sampling_rate
            else:
                # Convert to float32 in [-1, 1] range
                samples = audio_np.astype(np.float32) / 32767.0
                
                # Resample if needed
                if self.sampling_rate != target_sr:
                    # print(f'Resampling from {self.sampling_rate} to {target_sr}')
                    samples = librosa.resample(samples, orig_sr=self.sampling_rate, target_sr=target_sr)
                
                if return_as_int16:
                    samples = (samples * 32767).astype(np.int16)
                
                return samples, target_sr


class TTS_Kokoro(TTS):

    def __init__(self,
                 model_path: str='models/kokoro/kokoro-v1.0.fp16.onnx', 
                 voice_path: str='models/kokoro/kokoro-voices-v1.0.bin',  
                 speaker_voice: str='am_eric',
                 language: str="en-us"):
        super().__init__()
        from kokoro_onnx import Kokoro
        from kokoro_onnx.tokenizer import Tokenizer

        self.model_path = model_path
        self.voice_path = voice_path
        self.speaker_voice = speaker_voice
        print(f"Using Kokoro model: {model_path} with voice: {speaker_voice}")
        self.tokenizer = Tokenizer()
        self.kokoro = Kokoro(model_path, voice_path)
        self.language = language
        self.voice = self.kokoro.get_voice_style(speaker_voice)
        self.sample_rate = 24000 # default for Kokoro

        self.warmup()
        print("KokoroTTS loaded and warmed up.")


    def get_sample_rate(self):
        return self.sample_rate

    def synthesize(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
        phonemes = self.tokenizer.phonemize(text, lang=self.language)
        samples, sample_rate = self.kokoro.create(
            phonemes, voice=self.voice, speed=speaking_rate, is_phonemes=True
        )        
        if sample_rate != target_sr:
            # print('resampling from', sample_rate, 'to', target_sr)
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        if return_as_int16:
            samples = (samples * 32767).astype(np.int16)

        return samples, sample_rate