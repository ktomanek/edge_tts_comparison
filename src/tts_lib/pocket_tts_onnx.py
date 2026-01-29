"""
PocketTTS ONNX - Pure ONNX inference for Pocket TTS

A standalone, production-ready class for text-to-speech with voice cloning.
Supports both offline (batch) and streaming modes with adaptive chunking.

This is copied from https://huggingface.co/KevinAHM/pocket-tts-onnx and 
minimally adapted.

pocket-tts-onnx's licence as per original code:
- **Models**: CC BY 4.0 (inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))
- **Code**: Apache 2.0


"""

import os
import time
from pathlib import Path
from typing import Generator, Optional, Union
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

import soundfile as sf
import scipy.signal

class PocketTTSOnnx:
    """
    Pure ONNX inference engine for Pocket TTS.

    Supports:
        - Offline (batch) generation
        - Streaming generation with adaptive chunking
        - INT8 and FP32 models
        - Voice cloning from audio files
        - Auto GPU/CPU detection
        - Temperature control for generation diversity

    Args:
        models_dir: Directory containing ONNX models
        tokenizer_path: Path to sentencepiece tokenizer.model
        precision: Model precision - "int8" (CPU optimized, fastest) or "fp32" (full precision)
        device: "auto", "cpu", or "cuda"
        temperature: Sampling temperature (0.0 = deterministic, 0.7 = default, 1.0 = more diverse)
        lsd_steps: Number of flow matching steps (default 10, lower = faster but lower quality)
    """

    SAMPLE_RATE = 24000
    SAMPLES_PER_FRAME = 1920
    FRAME_DURATION = SAMPLES_PER_FRAME / SAMPLE_RATE  # 0.08s per frame

    VALID_PRECISIONS = ("int8", "fp32")

    def __init__(
        self,
        models_dir: str = "onnx",
        tokenizer_path: str = "tokenizer.model",
        precision: str = "int8",
        device: str = "auto",
        temperature: float = 0.7,
        lsd_steps: int = 10,
        num_threads: int = 0,  # 0 = auto-detect
    ):
        self.models_dir = Path(models_dir)

        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {self.VALID_PRECISIONS}, got '{precision}'")

        self.precision = precision
        self.temperature = temperature
        self.lsd_steps = lsd_steps
        self.num_threads = num_threads

        # Setup execution providers
        self.providers = self._get_providers(device)

        # Setup session options for optimal performance
        self.session_options = self._create_session_options()

        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))

        # Load models
        self._load_models()

        # Pre-compute s/t buffers for flow matching
        self._precompute_flow_buffers()

        # Cache for voice embeddings
        self._voice_cache = {}

    def _get_providers(self, device: str) -> list:
        """Get ONNX execution providers based on device setting."""
        if device == "cpu":
            return ["CPUExecutionProvider"]
        elif device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # auto
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    def _create_session_options(self) -> ort.SessionOptions:
        """Create optimized ONNX Runtime session options.

        Returns:
            SessionOptions configured for optimal performance
        """
        sess_options = ort.SessionOptions()

        # Set thread count (0 = auto-detect, otherwise use specified value)
        if self.num_threads > 0:
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = 1  # Single model execution, avoid overhead
        else:
            # Auto-detect: use all available cores
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            sess_options.intra_op_num_threads = num_cores
            sess_options.inter_op_num_threads = 1

        # Enable all graph optimizations
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use sequential execution mode (simpler, often faster for single-threaded scenarios)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        return sess_options

    def _load_models(self):
        """Load ONNX models (dual model architecture)."""
        # Select model files based on precision
        suffix = "_int8" if self.precision == "int8" else ""
        flow_main_file = f"flow_lm_main{suffix}.onnx"
        flow_flow_file = f"flow_lm_flow{suffix}.onnx"
        mimi_file = f"mimi_decoder{suffix}.onnx"

        self.mimi_encoder = ort.InferenceSession(
            str(self.models_dir / "mimi_encoder.onnx"),
            sess_options=self.session_options,
            providers=self.providers
        )
        self.text_conditioner = ort.InferenceSession(
            str(self.models_dir / "text_conditioner.onnx"),
            sess_options=self.session_options,
            providers=self.providers
        )
        # Dual model split: main (transformer) + flow (flow network)
        self.flow_lm_main = ort.InferenceSession(
            str(self.models_dir / flow_main_file),
            sess_options=self.session_options,
            providers=self.providers
        )
        self.flow_lm_flow = ort.InferenceSession(
            str(self.models_dir / flow_flow_file),
            sess_options=self.session_options,
            providers=self.providers
        )
        self.mimi_decoder = ort.InferenceSession(
            str(self.models_dir / mimi_file),
            sess_options=self.session_options,
            providers=self.providers
        )

    def _precompute_flow_buffers(self):
        """Pre-compute s/t time step buffers for flow matching."""
        dt = 1.0 / self.lsd_steps
        self._st_buffers = []
        for j in range(self.lsd_steps):
            s = j / self.lsd_steps
            t = s + dt
            self._st_buffers.append((
                np.array([[s]], dtype=np.float32),
                np.array([[t]], dtype=np.float32)
            ))

    def _init_state(self, session: ort.InferenceSession) -> dict:
        """Initialize state tensors for a stateful model."""
        state = {}
        type_map = {
            "tensor(float)": np.float32,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }
        for inp in session.get_inputs():
            if inp.name.startswith("state_"):
                shape = [s if isinstance(s, int) else 0 for s in inp.shape]
                dtype = type_map.get(inp.type, np.float32)
                state[inp.name] = np.zeros(shape, dtype=dtype)
        return state

    def _increment_step(self, state: dict, n: int):
        """Increment step counters in state dict."""
        for k in state:
            if "step" in k:
                state[k] = (state[k] + n).astype(np.int64)

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess audio file for voice cloning."""

        audio, sr = sf.read(str(path))

        # Convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 24kHz if needed
        if sr != self.SAMPLE_RATE:
            num_samples = int(len(audio) * self.SAMPLE_RATE / sr)
            audio = scipy.signal.resample(audio, num_samples)

        # Normalize and reshape
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        return audio.reshape(1, 1, -1)

    def encode_voice(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Encode an audio file into voice embeddings for cloning.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            Voice embeddings array [1, N, 1024]
        """
        audio = self._load_audio(audio_path)
        embeddings = self.mimi_encoder.run(None, {"audio": audio})[0]
        
        # Normalize dimensions to [1, N, 1024]
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings[None]
            
        return embeddings

    def _get_voice_embeddings(self, voice: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Get voice embeddings from various input types."""
        # Already embeddings
        if isinstance(voice, np.ndarray):
            return voice

        voice_str = str(voice)

        # Check cache
        if voice_str in self._voice_cache:
            return self._voice_cache[voice_str]

        # Audio file
        if os.path.exists(voice_str):
            embeddings = self.encode_voice(voice_str)
        else:
            raise ValueError(f"Voice file '{voice_str}' not found.")

        # Cache and return
        self._voice_cache[voice_str] = embeddings
        return embeddings

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize text for the model."""
        # Prepare text
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")

        # Ensure proper punctuation
        if text[-1].isalnum():
            text = text + "."
        if not text[0].isupper():
            text = text[0].upper() + text[1:]

        # Tokenize
        token_ids = self.tokenizer.Encode(text)
        return np.array(token_ids, dtype=np.int64).reshape(1, -1)

    def _update_state_from_outputs(self, state: dict, result: list, session: ort.InferenceSession):
        """Update state dict from model outputs."""
        for i in range(2, len(session.get_outputs())):
            name = session.get_outputs()[i].name
            if name.startswith("out_state_"):
                idx = int(name.replace("out_state_", ""))
                state[f"state_{idx}"] = result[i]

    def _run_flow_lm(
        self,
        voice_embeddings: np.ndarray,
        text_ids: np.ndarray,
        max_frames: int = 500,
        frames_after_eos: int = 3,
    ) -> Generator[np.ndarray, None, None]:
        """
        Run flow LM autoregressive generation, yielding latents.

        Uses dual model architecture:
        - flow_lm_main: transformer/conditioner (produces conditioning vector)
        - flow_lm_flow: flow network (Euler integration for latent sampling)

        Yields individual latent frames as they're generated.
        """
        # Text conditioning
        text_emb = self.text_conditioner.run(None, {"token_ids": text_ids})[0]
        if text_emb.ndim == 2:
            text_emb = text_emb[None]

        # Initialize state for flow_lm_main
        state = self._init_state(self.flow_lm_main)

        empty_seq = np.zeros((1, 0, 32), dtype=np.float32)
        empty_text = np.zeros((1, 0, 1024), dtype=np.float32)

        # Voice conditioning pass
        res_voice = self.flow_lm_main.run(None, {
            "sequence": empty_seq,
            "text_embeddings": voice_embeddings,
            **state
        })
        self._update_state_from_outputs(state, res_voice, self.flow_lm_main)
        # Note: Step counters are already updated in the model's output states

        # Text conditioning pass
        res_text = self.flow_lm_main.run(None, {
            "sequence": empty_seq,
            "text_embeddings": text_emb,
            **state
        })
        self._update_state_from_outputs(state, res_text, self.flow_lm_main)
        # Note: Step counters are already updated in the model's output states

        # Autoregressive generation
        curr = np.full((1, 1, 32), np.nan, dtype=np.float32)
        dt = 1.0 / self.lsd_steps
        
        eos_step = None

        for step in range(max_frames):
            # Run main model to get conditioning and EOS
            res_step = self.flow_lm_main.run(None, {
                "sequence": curr,
                "text_embeddings": empty_text,
                **state
            })

            conditioning = res_step[0]  # [1, 1, dim]
            eos_logit = res_step[1]     # [1, 1]

            # Update state (step counters are already updated in model outputs)
            self._update_state_from_outputs(state, res_step, self.flow_lm_main)

            # Check EOS - record when EOS is first detected
            if eos_logit[0][0] > -4.0 and eos_step is None:
                eos_step = step
            
            # Stop only after frames_after_eos additional frames
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            # Flow matching with external loop (enables temperature control)
            # Initialize with noise scaled by temperature
            std = np.sqrt(self.temperature) if self.temperature > 0 else 0.0
            x = np.random.normal(0, std, (1, 32)).astype(np.float32) if std > 0 else np.zeros((1, 32), dtype=np.float32)

            # Euler integration over flow network
            for j in range(self.lsd_steps):
                s_arr, t_arr = self._st_buffers[j]
                flow_out = self.flow_lm_flow.run(None, {
                    "c": conditioning,
                    "s": s_arr,
                    "t": t_arr,
                    "x": x
                })
                x = x + flow_out[0] * dt

            latent = x.reshape(1, 1, 32)
            yield latent
            curr = latent

    def _decode_latents(self, latents: np.ndarray, chunk_size: int = 15) -> np.ndarray:
        """
        Decode latents to audio using chunked decoding with state updates.
        
        The MIMI decoder requires proper state updates between chunks to avoid
        garbled audio. Batch decoding without state updates causes artifacts.
        
        Args:
            latents: Latent array of shape [1, num_frames, 32]
            chunk_size: Number of frames to decode at once (default 15 for speed)
        
        Returns:
            Audio samples as numpy array
        """
        state = self._init_state(self.mimi_decoder)
        audio_chunks = []
        num_frames = latents.shape[1]
        
        for i in range(0, num_frames, chunk_size):
            chunk = latents[:, i:i+chunk_size, :]
            
            result = self.mimi_decoder.run(None, {"latent": chunk, **state})
            audio_chunks.append(result[0].squeeze())
            
            # Update state from decoder outputs (critical for avoiding garbling)
            for k in range(1, len(self.mimi_decoder.get_outputs())):
                out_name = self.mimi_decoder.get_outputs()[k].name
                if out_name.startswith("out_state_"):
                    idx = int(out_name.replace("out_state_", ""))
                    state[f"state_{idx}"] = result[k]
        
        return np.concatenate(audio_chunks)

    def generate(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: int = 500,
    ) -> np.ndarray:
        """
        Generate audio from text (offline/batch mode).

        Args:
            text: Text to synthesize
            voice: Audio file path for voice cloning, or pre-computed embeddings
            max_frames: Maximum latent frames to generate

        Returns:
            Audio samples as numpy array (float32, 24kHz)
        """
        voice_emb = self._get_voice_embeddings(voice)
        text_ids = self._tokenize(text)

        # Generate all latents
        latents = list(self._run_flow_lm(voice_emb, text_ids, max_frames))
        latents = np.concatenate(latents, axis=1)

        # Decode all at once
        audio = self._decode_latents(latents)
        return audio

    def stream(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: int = 500,
        first_chunk_frames: int = 2,
        target_buffer_sec: float = 0.2,
        max_chunk_frames: int = 15,
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream audio generation with adaptive chunking.

        Yields audio chunks as they become available, optimizing for:
        - Low TTFB (time to first audio)
        - Smooth real-time playback
        - High overall throughput

        Args:
            text: Text to synthesize
            voice: Audio file path for voice cloning, or pre-computed embeddings
            max_frames: Maximum latent frames to generate
            first_chunk_frames: Frames in first chunk (controls TTFB)
            target_buffer_sec: Target buffer ahead of playback
            max_chunk_frames: Maximum frames per chunk

        Yields:
            Audio chunks as numpy arrays (float32, 24kHz)
        """
        voice_emb = self._get_voice_embeddings(voice)
        text_ids = self._tokenize(text)

        # State tracking
        mimi_state = self._init_state(self.mimi_decoder)
        generated_latents = []
        decoded_frames = 0
        playback_start_time = None
        start_time = time.time()

        for latent in self._run_flow_lm(voice_emb, text_ids, max_frames):
            generated_latents.append(latent)
            pending = len(generated_latents) - decoded_frames

            # Decide chunk size
            chunk_size = 0

            if playback_start_time is None:
                # First chunk - minimize TTFB
                if pending >= first_chunk_frames:
                    chunk_size = first_chunk_frames
            else:
                # Calculate buffer status
                elapsed = time.time() - start_time
                audio_decoded_sec = decoded_frames * self.FRAME_DURATION
                playback_elapsed = elapsed - playback_start_time
                buffer_sec = audio_decoded_sec - playback_elapsed

                if buffer_sec < target_buffer_sec and pending >= 1:
                    # Buffer low - decode small chunk quickly
                    chunk_size = min(pending, 3)
                elif pending >= max_chunk_frames:
                    # Enough accumulated - decode larger chunk
                    chunk_size = max_chunk_frames

            # Decode chunk if needed
            if chunk_size > 0:
                latents_chunk = np.concatenate(
                    generated_latents[decoded_frames:decoded_frames + chunk_size],
                    axis=1
                )

                res = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
                audio_chunk = res[0].squeeze()

                for k, val in enumerate(res[1:]):
                    mimi_state[f"state_{k}"] = val

                decoded_frames += chunk_size

                if playback_start_time is None:
                    playback_start_time = time.time() - start_time

                yield audio_chunk

        # Decode remaining latents
        if decoded_frames < len(generated_latents):
            remaining_latents = np.concatenate(
                generated_latents[decoded_frames:],
                axis=1
            )
            res = self.mimi_decoder.run(None, {"latent": remaining_latents, **mimi_state})
            yield res[0].squeeze()

    def save_audio(self, audio: np.ndarray, path: Union[str, Path]):
        """Save audio to file."""
        sf.write(str(path), audio, self.SAMPLE_RATE)

    @property
    def device(self) -> str:
        """Return the device being used."""
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"PocketTTSOnnx("
            f"device={self.device!r}, "
            f"precision={self.precision!r}, "
            f"temperature={self.temperature}, "
            f"lsd_steps={self.lsd_steps}, "
            f"sample_rate={self.SAMPLE_RATE})"
        )
