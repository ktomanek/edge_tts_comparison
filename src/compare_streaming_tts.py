from tts_lib import tts_engines
import time
import soundfile as sf
import statistics
import numpy as np
import threading
import sounddevice as sd

class StreamingAudioPlayer:
    """Plays audio chunks in real-time with continuous buffering."""

    def __init__(self, sample_rate, prebuffer_chunks=4, dynamic_rebuffer=True, rebuffer_threshold_ms=500):
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.stream = None
        self.finished = False
        self.started = False
        self.prebuffer_chunks = prebuffer_chunks
        self.chunks_received = 0
        self.playback_start_time = None
        self.underrun_count = 0
        self.dynamic_rebuffer = dynamic_rebuffer
        self.rebuffer_threshold_samples = int((rebuffer_threshold_ms / 1000.0) * sample_rate)
        self.is_rebuffering = False
        self.rebuffer_count = 0

    def audio_callback(self, outdata, frames, time_info, status):
        """Callback function for sounddevice OutputStream."""
        if status:
            print(f"\n[Audio callback status: {status}]", flush=True)

        with self.buffer_lock:
            buffer_size = len(self.buffer)

            # Check if we should start rebuffering
            if self.dynamic_rebuffer and not self.finished and not self.is_rebuffering:
                if buffer_size < self.rebuffer_threshold_samples:
                    self.is_rebuffering = True
                    self.rebuffer_count += 1

            # If rebuffering, output silence until buffer fills up
            # BUT: If finished, stop rebuffering immediately to play out remaining buffer
            if self.is_rebuffering:
                if self.finished or buffer_size >= self.rebuffer_threshold_samples * 2:
                    self.is_rebuffering = False
                else:
                    outdata[:] = 0
                    return

            if buffer_size >= frames:
                # We have enough data
                outdata[:] = self.buffer[:frames].reshape(-1, 1)
                self.buffer = self.buffer[frames:]
            elif buffer_size > 0:
                # Partial data available - buffer underrun
                outdata[:buffer_size] = self.buffer.reshape(-1, 1)
                outdata[buffer_size:] = 0
                self.buffer = np.array([], dtype=np.float32)
                self.underrun_count += 1
            else:
                # No data available - buffer underrun
                outdata[:] = 0
                if not self.finished:
                    self.underrun_count += 1
                if self.finished:
                    raise sd.CallbackStop

    def add_chunk(self, chunk):
        """Add an audio chunk to the playback buffer."""
        should_start = False

        with self.buffer_lock:
            self.buffer = np.concatenate([self.buffer, chunk])
            self.chunks_received += 1

            # Check if we should start playback (but don't call while holding lock)
            if not self.started and self.chunks_received >= self.prebuffer_chunks:
                should_start = True

        # Start playback AFTER releasing the lock to avoid deadlock
        if should_start:
            self._start_playback()

    def _start_playback(self):
        """Internal method to start audio stream."""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=1024
        )
        self.stream.start()
        # Wait for stream to initialize before marking as started
        time.sleep(0.05)
        self.playback_start_time = time.time()
        self.started = True

    def get_playback_start_time(self):
        """Get the time when playback actually started (or None if not started yet)."""
        return self.playback_start_time

    def get_underrun_count(self):
        """Get the number of buffer underruns that occurred during playback."""
        return self.underrun_count

    def get_rebuffer_count(self):
        """Get the number of times dynamic rebuffering occurred."""
        return self.rebuffer_count

    def stop(self):
        """Stop the audio stream after playing remaining buffered audio."""
        # If we never started (not enough chunks), start now
        if not self.started and self.chunks_received > 0:
            self._start_playback()

        if self.stream:
            # Wait for buffer to drain before stopping
            max_wait_time = 10.0  # Maximum 10 seconds to drain
            check_interval = 0.05
            elapsed = 0.0

            while elapsed < max_wait_time:
                with self.buffer_lock:
                    remaining = len(self.buffer)

                if remaining == 0:
                    break

                time.sleep(check_interval)
                elapsed += check_interval

            # Mark as finished so callback knows to stop when buffer is empty
            self.finished = True

            # Wait for the audio callback to naturally raise CallbackStop and for stream to finish
            # The callback will stop when it sees finished=True and buffer is empty
            # We need to wait for: internal sounddevice buffer to drain
            blocksize = 1024
            blocks_to_wait = 5  # Increased from 3 to 5 for safety
            drain_time = (blocksize * blocks_to_wait) / self.sample_rate
            time.sleep(drain_time)

            # Now close the stream (it should have already stopped via CallbackStop)
            try:
                if self.stream.active:
                    self.stream.stop()
            except:
                pass  # Stream might already be stopped
            self.stream.close()

def benchmark_streaming(tts_engine, text, target_sr=16000, num_runs=10, play_audio=False, prebuffer_chunks=4, dynamic_rebuffer=True, rebuffer_threshold_ms=500):
    """Run streaming synthesis multiple times and report statistics.

    Args:
        tts_engine: TTS engine instance with synthesize_stream method
        text: Text to synthesize
        target_sr: Target sample rate
        num_runs: Number of synthesis runs (default: 10)
        play_audio: If True, play audio in real-time during first run (default: False)
        prebuffer_chunks: Number of chunks to buffer before starting playback (default: 4)
        dynamic_rebuffer: Enable dynamic rebuffering when buffer runs low (default: True)
        rebuffer_threshold_ms: Buffer threshold in ms for triggering rebuffering (default: 500)

    Returns:
        Tuple of (audio, sample_rate) from the last run
    """
    ttfb_times = []  # Time to first byte (first chunk received)
    ttfa_times = []  # Time to first audio (when playback actually starts)
    total_times = []  # Total generation time
    audio_result = None
    sr_result = None
    chunk_sizes = []  # Will store chunk sizes from last run

    if play_audio:
        print(f">> Running {num_runs} streaming synthesis iterations (playing audio on run 1)...")
    else:
        print(f">> Running {num_runs} streaming synthesis iterations...")

    for i in range(num_runs):
        chunks = []
        current_chunk_sizes = []
        t_start = time.time()
        ttfb = None
        ttfa = None

        # Play audio only on first run
        should_play = play_audio and i == 0
        player = None

        if should_play:
            print(f"   Run {i+1}: [Playing audio as it generates...]", end='', flush=True)
            player = StreamingAudioPlayer(
                target_sr,
                prebuffer_chunks=prebuffer_chunks,
                dynamic_rebuffer=dynamic_rebuffer,
                rebuffer_threshold_ms=rebuffer_threshold_ms
            )

        for chunk, sr in tts_engine.synthesize_stream(text, target_sr=target_sr):
            if ttfb is None:
                ttfb = time.time() - t_start
            chunks.append(chunk)
            current_chunk_sizes.append(len(chunk))
            sr_result = sr

            # Add chunk to continuous audio stream on first run
            if should_play and player:
                player.add_chunk(chunk)
                # Check if playback has started
                if ttfa is None:
                    playback_start = player.get_playback_start_time()
                    if playback_start is not None:
                        ttfa = playback_start - t_start

        t_total = time.time() - t_start

        # Keep the chunk sizes from last run
        chunk_sizes = current_chunk_sizes

        # Stop audio playback and wait for buffer to drain
        if should_play and player:
            underruns = player.get_underrun_count()
            rebuffers = player.get_rebuffer_count()
            player.stop()
            # Ensure we got TTFA (in case it never started during generation)
            if ttfa is None:
                playback_start = player.get_playback_start_time()
                if playback_start is not None:
                    ttfa = playback_start - t_start
            if underruns > 0 or rebuffers > 0:
                stats = []
                if rebuffers > 0:
                    stats.append(f"{rebuffers} rebuffer(s)")
                if underruns > 0:
                    stats.append(f"{underruns} underrun(s)")
                print(f" [{', '.join(stats)}]", end='')

        ttfb_times.append(ttfb)
        ttfa_times.append(ttfa if ttfa is not None else ttfb)  # Fallback to TTFB if no playback
        total_times.append(t_total)

        if should_play and ttfa is not None:
            print(f" TTFB={ttfb:.3f}s, TTFA={ttfa:.3f}s, Total={t_total:.2f}s")
        elif should_play:
            print(f" TTFB={ttfb:.3f}s, Total={t_total:.2f}s")
        else:
            print(f"   Run {i+1}: TTFB={ttfb:.3f}s, Total={t_total:.2f}s")

        # Keep the last result for saving to file
        audio_result = np.concatenate(chunks)

    mean_ttfb = statistics.mean(ttfb_times)
    stdev_ttfb = statistics.stdev(ttfb_times) if len(ttfb_times) > 1 else 0.0

    mean_ttfa = statistics.mean(ttfa_times)
    stdev_ttfa = statistics.stdev(ttfa_times) if len(ttfa_times) > 1 else 0.0

    mean_total = statistics.mean(total_times)
    stdev_total = statistics.stdev(total_times) if len(total_times) > 1 else 0.0

    # Calculate audio duration
    audio_duration = len(audio_result) / sr_result
    real_time_factor = mean_total / audio_duration

    print(f">> Time to First Byte (TTFB): {mean_ttfb:.3f}s ± {stdev_ttfb:.3f}s (mean ± stdev)")
    if play_audio:
        print(f">> Time to First Audio (TTFA): {mean_ttfa:.3f}s ± {stdev_ttfa:.3f}s (when user hears sound)")
    print(f">> Total synthesis time: {mean_total:.2f}s ± {stdev_total:.2f}s (mean ± stdev)")
    print(f">> Audio duration: {audio_duration:.2f}s")
    print(f">> Real-time factor: {real_time_factor:.2f}x (lower is faster)")

    # Chunk size statistics
    if chunk_sizes:
        avg_chunk_size = statistics.mean(chunk_sizes)
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        total_chunks = len(chunk_sizes)
        avg_chunk_duration_ms = (avg_chunk_size / sr_result) * 1000
        print(f">> Chunks: {total_chunks} total, avg size: {avg_chunk_size:.0f} samples ({avg_chunk_duration_ms:.0f}ms)")
        print(f"   Min/Max chunk size: {min_chunk_size}/{max_chunk_size} samples")

    return audio_result, sr_result

run_warmup = True
play_audio_on_first_run = True  # Set to True to hear streaming audio in real-time

# Streaming playback configuration
prebuffer_chunks = 4  # Number of chunks to buffer before starting playback (increase if stuttering)
dynamic_rebuffer = False  # Enable dynamic rebuffering when buffer runs low
rebuffer_threshold_ms = 200  # Buffer threshold in ms for triggering rebuffering

text = 'In a dramatic overnight operation, India said it launched missile and air strikes on nine sites across Pakistan.'
# text = "The quick brown fox jumps over the lazy dog. Dr Smith asked whether it's 4:30 PM today."
target_sr = 16000

# Reference text for voice cloning
ref_text = text
num_inf_runs = 1

print("="*70)
print("STREAMING TTS COMPARISON")
print("="*70)
print(f"Text: {text}")
print(f"Target sample rate: {target_sr} Hz")
print(f"Warmup: {run_warmup}")
print(f"Benchmark runs: {num_inf_runs}")
print(f"Play audio on first run: {play_audio_on_first_run}")
if play_audio_on_first_run:
    print(f"Prebuffer chunks: {prebuffer_chunks}")
    print(f"Dynamic rebuffering: {dynamic_rebuffer}")
    if dynamic_rebuffer:
        print(f"Rebuffer threshold: {rebuffer_threshold_ms}ms")
print("="*70)
print()

# =============================================================================
# PocketTTS ONNX (Streaming)
# =============================================================================
print(f">>> Running PocketTTS ONNX (Streaming)...")
t1 = time.time()
ref_audio = 'kokoro_tts.wav'
pocket_tts_onnx = tts_engines.TTS_PocketTTSOnnx(voice=ref_audio, warmup=run_warmup)
print(f'>> pockettts onnx model load and warmup time: {time.time()-t1:.2f}s')

audio, sampling_rate = benchmark_streaming(
    pocket_tts_onnx,
    text,
    target_sr,
    num_runs=num_inf_runs,
    play_audio=play_audio_on_first_run,
    prebuffer_chunks=prebuffer_chunks,
    dynamic_rebuffer=dynamic_rebuffer,
    rebuffer_threshold_ms=rebuffer_threshold_ms
)
sf.write('pocket_tts_onnx_streaming.wav', audio, target_sr)
print()


print("="*70)
print("STREAMING BENCHMARK COMPLETE")
print("="*70)
print("Output files:")
print("  - pocket_tts_onnx_streaming.wav")
print("  - piper_tts_streaming.wav")
print("="*70)
