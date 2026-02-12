#!/usr/bin/env python3
"""Simple script to synthesize text using different TTS models.

Usage:
    python synthesize_text.py "Hello world" output.wav
    python synthesize_text.py "Hello world" output.wav --model kokoro
    python synthesize_text.py "Hello world" output.wav --model pockettts
    python synthesize_text.py "Hello world" output.wav --model piper
"""

import argparse
import soundfile as sf
from tts_lib import tts_engines


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize text to speech using various TTS models"
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text to synthesize"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output audio file path (e.g., output.wav)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["piper", "kokoro", "pockettts"],
        default="piper",
        help="TTS model to use (default: piper)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)"
    )

    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Output: {args.output_file}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print()

    # Load the appropriate TTS model
    if args.model == "piper":
        print("Loading Piper TTS...")
        tts = tts_engines.TTS_Piper(warmup=False)
    elif args.model == "kokoro":
        print("Loading Kokoro TTS...")
        tts = tts_engines.TTS_Kokoro(warmup=False)
    elif args.model == "pockettts":
        print("Loading PocketTTS ONNX...")
        tts = tts_engines.TTS_PocketTTSOnnx(warmup=False)

    print(f"Model loaded (sample rate: {tts.get_sample_rate()} Hz)")
    print()

    # Synthesize
    print("Synthesizing...")
    audio, sample_rate = tts.synthesize(
        args.text,
        target_sr=args.sample_rate,
        return_as_int16=False
    )

    # Save to file
    sf.write(args.output_file, audio, sample_rate)

    audio_duration = len(audio) / sample_rate
    print(f"✓ Saved to: {args.output_file}")
    print(f"✓ Audio duration: {audio_duration:.2f} seconds")
    print(f"✓ Sample rate: {sample_rate} Hz")


if __name__ == "__main__":
    main()
