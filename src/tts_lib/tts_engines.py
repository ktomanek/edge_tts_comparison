# convenvience wrapper for several TTS libraries running on edge devices
import numpy as np
import librosa
import os
import time

class TTS:
    def __init__(self, warmup: bool=True):
        pass

    def get_sample_rate(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def synthesize(self, text: str, speaking_rate:float, return_as_int16: bool):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def synthesize_stream(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
        """Stream audio synthesis in chunks. Yields (audio_chunk, sample_rate) tuples.
        Only supported by engines with native streaming capabilities.
        """
        raise NotImplementedError("Streaming is not supported by this TTS engine.")

    def warmup(self):
        print("Warming up model...")
        self.synthesize("This is a warmup text to initialize the TTS engine. Cats are great, I love cats!")

class TTS_KittenTTS(TTS):
    def __init__(self, model_path: str='KittenML/kitten-tts-nano-0.2',
                 voice: str='expr-voice-2-m',
                 warmup: bool=True):
        """There is also a supposedly better model called mini:
        "KittenML/kitten-tts-mini-0.1"
        It is noticeably slower and I can't really find a big quality gain.
        """
        
        from kittentts import KittenTTS
        print(f"Loading kittentts with model: {model_path}")
        self.kitten_tts = KittenTTS(model_path)
        self.voice_for_synthesis = voice
        self.sample_rate = 24000 # default for KittenTTS
        
        if warmup:
            self.warmup()
            print("KittenTTS loaded and warmed up.")
        else:
            print("KittenTTS loaded (no warmup).")


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
    def __init__(self, model_path: str='models/piper/en_US-lessac-low.onnx', warmup: bool=True):
        super().__init__()
        from piper.voice import PiperVoice

        self.model_path = model_path
        self.piper_voice = PiperVoice.load(model_path)
        self.sampling_rate = self.piper_voice.config.sample_rate      

        if warmup:
            self.warmup()
            print("PiperTTS loaded and warmed up.")
        else:
            print("PiperTTS loaded (no warmup).")

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
                 language: str="en-us",
                 warmup: bool=True):
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

        if warmup:
            self.warmup()
            print("KokoroTTS loaded and warmed up.")
        else:
            print("KokoroTTS loaded (no warmup).")


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

class TTS_PocketTTS(TTS):
    """100m param TTS model from Kyutai that can run on CPU in real-time.
    Supports streaming as well as voice cloning from a short audio prompt.
    """
    def __init__(self, voice: str='alba', warmup: bool=True):
        """Initialize PocketTTS with a voice.
        Available voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
        Can also provide a path to a wav file for voice cloning.
        """
        super().__init__()
        from pocket_tts import TTSModel

        print(f"Loading PocketTTS with voice: {voice}")
        self.tts_model = TTSModel.load_model()
        self.voice_state = self.tts_model.get_state_for_audio_prompt(voice)
        self.sample_rate = self.tts_model.sample_rate

        if warmup:
            self.warmup()
            print("PocketTTS loaded and warmed up.")
        else:
            print("PocketTTS loaded (no warmup).")

    def get_sample_rate(self):
        return self.sample_rate

    def synthesize(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
        audio_tensor = self.tts_model.generate_audio(self.voice_state, text)

        samples = audio_tensor.numpy()
        sample_rate = self.sample_rate

        if sample_rate != target_sr:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr

        if return_as_int16:
            samples = (samples * 32767).astype(np.int16)

        return samples, sample_rate

    def synthesize_stream(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
        """Stream audio synthesis in chunks. Yields (audio_chunk, sample_rate) tuples.

        Args:
            text: Text to synthesize
            target_sr: Target sample rate (default: 16000)
            speaking_rate: Speaking rate (not supported by PocketTTS, parameter ignored)
            return_as_int16: If True, return int16 audio, otherwise float32

        Yields:
            Tuple of (audio_chunk, sample_rate) where audio_chunk is a numpy array
        """
        for chunk_tensor in self.tts_model.generate_audio_stream(self.voice_state, text):
            chunk = chunk_tensor.numpy()
            sample_rate = self.sample_rate

            if sample_rate != target_sr:
                chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr

            if return_as_int16:
                chunk = (chunk * 32767).astype(np.int16)

            yield chunk, sample_rate

class TTS_PocketTTSOnnx(TTS):
    """ONNX version of PocketTTS - 100m param TTS model from Kyutai.
    Uses ONNX Runtime instead of PyTorch for significantly reduced overhead.
    Ideal for edge devices like Raspberry Pi.
    Supports streaming and voice cloning from a short audio prompt.
    """

    def _get_tts_model_instance(temperature: float=0.3, lsd_steps: int=10):
        from .pocket_tts_onnx import PocketTTSOnnx
        from pathlib import Path

        # Use ONNX models from models/pockettts_onnx, tokenizer from src/tts_lib/
        module_dir = Path(__file__).parent
        models_dir = Path(__file__).parent.parent.parent / "models" / "pockettts_onnx"
        tokenizer_path = module_dir / "tokenizer.model"

        print(f"Models directory: {models_dir}")
        tts_model = PocketTTSOnnx(
            models_dir=str(models_dir),
            tokenizer_path=str(tokenizer_path),
            temperature=temperature,
            lsd_steps=lsd_steps
        )
        return tts_model


    def __init__(self, voice='alba', temperature: float=0.3, lsd_steps: int=10, warmup: bool=True):
        """Initialize PocketTTS ONNX with a voice.

        Args:
            voice: Can be:
                   - Path to a wav file for voice cloning (e.g., 'my_voice.wav')
                   - Pre-loaded embeddings from load_voice_embeddings() (numpy array)
            temperature: Generation diversity (0.3=deterministic/default, 0.7=balanced, 1.0=expressive)
            lsd_steps: Quality/speed tradeoff (1=faster/lower quality, 10=default)
            warmup: Whether to run warmup synthesis
        """
        super().__init__()
        self.tts_model = TTS_PocketTTSOnnx._get_tts_model_instance(temperature=temperature, lsd_steps=lsd_steps)
        self.sample_rate = 24000  # Default for PocketTTS

        # Pre-encode voice during initialization
        if isinstance(voice, np.ndarray):
            print(f">> Using PocketTTS ONNX with pre-loaded voice embeddings")
            self.voice_embeddings = voice
        else:
            print(f">> Using PocketTTS ONNX with reference voice audio file from: {voice}")
            voice_str = str(voice)
            if not os.path.exists(voice_str):
                raise FileNotFoundError(f"Voice reference audio file not found: {voice_str}")

            print(f"> Encoding voice audio...")
            t1 = time.time()
            self.voice_embeddings = self.tts_model._get_voice_embeddings(voice)
            print(f"> Voice encoded in: {time.time() - t1:.2f} seconds")

        if warmup:
            self.warmup()
            print("PocketTTS ONNX loaded and warmed up.")
        else:
            print("PocketTTS ONNX loaded (no warmup).")

    def get_sample_rate(self):
        return self.sample_rate

    def synthesize(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
        """Synthesize audio from text.

        Note: speaking_rate is not supported by PocketTTS ONNX and will be ignored.
        """
        audio = self.tts_model.generate(text, voice=self.voice_embeddings)

        samples = audio  # Already a numpy array
        sample_rate = self.sample_rate

        if sample_rate != target_sr:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr

        if return_as_int16:
            samples = (samples * 32767).astype(np.int16)

        return samples, sample_rate

    def synthesize_stream(self, text: str, target_sr=16000, speaking_rate=1.0, return_as_int16=False):
        """Stream audio synthesis in chunks. Yields (audio_chunk, sample_rate) tuples.

        Args:
            text: Text to synthesize
            target_sr: Target sample rate (default: 16000)
            speaking_rate: Speaking rate (not supported by PocketTTS ONNX, parameter ignored)
            return_as_int16: If True, return int16 audio, otherwise float32

        Yields:
            Tuple of (audio_chunk, sample_rate) where audio_chunk is a numpy array
        """
        for chunk in self.tts_model.stream(text, voice=self.voice_embeddings):
            sample_rate = self.sample_rate

            if sample_rate != target_sr:
                chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr

            if return_as_int16:
                chunk = (chunk * 32767).astype(np.int16)

            yield chunk, sample_rate

    def export_voice_embeddings(audio_path: str, output_path: str):
        """Export voice embeddings from an audio file for faster loading later.

        Args:
            audio_path: Path to the audio file to encode (wav, mp3, etc.)
            output_path: Path where to save the embeddings (.npy file)

        Example:
            tts = TTS_PocketTTSOnnx()
            tts.export_voice_embeddings('my_voice.wav', 'my_voice_embeddings.npy')
        """
        tts_model = TTS_PocketTTSOnnx._get_tts_model_instance()
        embeddings = tts_model.encode_voice(audio_path)
        np.save(output_path, embeddings)
        print(f"✓ Voice embeddings exported to: {output_path}")
        print(f"  Shape: {embeddings.shape}, Size: {embeddings.nbytes / 1024:.1f} KB")

    @staticmethod
    def load_voice_embeddings(embeddings_path: str):
        """Load pre-computed voice embeddings from a file.

        Args:
            embeddings_path: Path to the .npy embeddings file

        Returns:
            Numpy array of embeddings that can be passed as 'voice' parameter

        Example:
            embeddings = TTS_PocketTTSOnnx.load_voice_embeddings('my_voice_embeddings.npy')
            tts = TTS_PocketTTSOnnx(voice=embeddings)
            # Or use directly in synthesis:
            audio, sr = tts.synthesize("Hello", target_sr=16000)
        """
        embeddings = np.load(embeddings_path)
        print(f"✓ Voice embeddings loaded from: {embeddings_path}")
        print(f"  Shape: {embeddings.shape}, Size: {embeddings.nbytes / 1024:.1f} KB")
        return embeddings