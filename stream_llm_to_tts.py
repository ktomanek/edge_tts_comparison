# prompt LLM and pipe generated output directly into Piper
#
# * uses ollama for LLM
# * synthesizes only full sentences for better audio quality (collected streamed output from LLM until full sentence is found)
#
# Example run
#
# python stream_llm_to_tts.py --piper-model en_US-lessac-low.onnx --prompt "Explain to me what a cat does all day. Use exactly 3 sentences."  --speaking-rate=5.0

import ollama
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
import nltk
from nltk.tokenize import sent_tokenize
import time
import threading
import argparse
import queue
import os
import signal
import sys

# Download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MAX_TEXT_BUFFER = 200
MIN_TEXT_BUFFER = 100

class OllamaToPiperStreamer:
    def __init__(self, piper_model_path, ollama_model_name="gemma3:1b", 
                 speaking_rate=1.0, # higher numbers means faster
                 sample_rate=None):
        """Initialize the streamer with Piper and Ollama models."""
        print(f"Initializing with Piper model: {piper_model_path}")
        self.ollama_model_name = ollama_model_name
        
        # Initialize Piper TTS
        self.piper_voice = PiperVoice.load(piper_model_path)
        
        # Get sample rate from Piper config if not specified
        self.sample_rate = sample_rate if sample_rate else self.piper_voice.config.sample_rate
        print(f"Using sample rate: {self.sample_rate} Hz")
        
        self.speaking_rate = speaking_rate

        # Initialize audio stream
        self.audio_stream = None
        
        # Text processing
        self.text_buffer = ""
        self.sentence_queue = queue.Queue()
        self.is_processing = False
        self.is_speaking = False
        self.lock = threading.Lock()
        
        # Create stop event for clean termination
        self.stop_event = threading.Event()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully."""
        print("\nReceived termination signal. Shutting down...")
        self.stop_event.set()
        self._close()
        sys.exit(0)
    
    def _start_audio_stream(self):
        """Initialize and start the audio output stream."""
        if self.audio_stream is None:
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            self.audio_stream.start()
            print("Audio stream started")
    
    def _start_sentence_processor(self):
        """Start a background thread to process sentences."""
        if self.is_processing:
            return
            
        self.is_processing = True
        threading.Thread(target=self._process_sentences, daemon=True).start()
    
    def _close(self):
        """Close all resources."""
        if self.audio_stream:
            print("Closing audio stream...")
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
    
    def _process_text_chunk(self, text_chunk):
        """Process a chunk of text from Ollama, detecting complete sentences."""
        if self.stop_event.is_set():
            return
            
        if not text_chunk:
            return
            
        with self.lock:
            self.text_buffer += text_chunk
            
            # find complete sentences
            try:
                sentences = sent_tokenize(self.text_buffer)
                if len(sentences) > 1:
                    complete_sentences = sentences[:-1]
                    
                    # Keep the last (potentially incomplete) sentence in buffer
                    self.text_buffer = sentences[-1]
                    
                    # Add complete sentences to the queue
                    for sentence in complete_sentences:
                        if sentence.strip():
                            self.sentence_queue.put(sentence)
                            print(f"Queued: {sentence}")
                
                # If buffer is getting long but no sentence breaks, force process
                elif len(self.text_buffer) > MAX_TEXT_BUFFER:
                    # Look for natural break points
                    break_points = [
                        self.text_buffer.rfind(', '),
                        self.text_buffer.rfind(' - '),
                        self.text_buffer.rfind(': '),
                        self.text_buffer.rfind(' ')
                    ]
                    
                    # Find the best break point
                    break_point = max(break_points)
                    
                    if break_point > MIN_TEXT_BUFFER:  # Ensure we're not breaking too early
                        fragment = self.text_buffer[:break_point+1]
                        self.text_buffer = self.text_buffer[break_point+1:]
                        self.sentence_queue.put(fragment)
                        print(f"Forced break: {fragment}")
            
            except Exception as e:
                print(f"Error in sentence detection: {e}")
        
        # Ensure the sentence processor is running
        if not self.is_processing:
            self._start_sentence_processor()
    
    def _process_sentences(self):
        """Process sentences from the queue and speak them."""
        self._start_audio_stream()
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get a sentence from the queue with timeout
                    sentence = self.sentence_queue.get(timeout=0.5)
                    
                    # Wait until not speaking to avoid overlap
                    while self.is_speaking and not self.stop_event.is_set():
                        time.sleep(0.05)
                    
                    if self.stop_event.is_set():
                        break
                    
                    # Speak the sentence
                    self._speak_sentence(sentence, speed=self.speaking_rate)
                    self.sentence_queue.task_done()
                
                except queue.Empty:
                    # No sentences to process
                    if self.sentence_queue.empty() and not self.text_buffer and not self.is_speaking:
                        # Exit if we've processed everything
                        break
        
        finally:
            self.is_processing = False
            
            # If there are still sentences and we're not stopped, restart processor
            if not self.sentence_queue.empty() and not self.stop_event.is_set():
                self._start_sentence_processor()
    
    def _speak_sentence(self, text, speed=1.0, noise_scale=0.667, noise_w=0.8):
        """Synthesize and play a sentence with Piper."""
        if not text.strip():
            return
            
        self.is_speaking = True
        print(f"Speaking: {text}")
        
        try:
            # Process through Piper
            synthesis_args = {
                "length_scale": 1.0 / speed,  # Higher values = slower speech
                "noise_scale": noise_scale,   # Controls variability in voice
                "noise_w": noise_w            # Controls phoneme width variation
            }
            for audio_bytes in self.piper_voice.synthesize_stream_raw(text, **synthesis_args):
                if self.stop_event.is_set():
                    break
                    
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                self.audio_stream.write(audio_data)
        
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
        
        finally:
            self.is_speaking = False
    
    def _finish_processing(self):
        """Process any remaining text in the buffer."""
        with self.lock:
            if self.text_buffer.strip():
                self.sentence_queue.put(self.text_buffer)
                self.text_buffer = ""
        
        # Wait for sentence queue to empty
        if self.sentence_queue.qsize() > 0:
            print(f"Waiting for {self.sentence_queue.qsize()} pending sentences...")
            self.sentence_queue.join()
        
        # Give a moment for audio to finish playing
        time.sleep(0.5)
    
    def process_prompt(self, prompt, chat_mode=True):
        """Process a prompt through Ollama and stream to Piper."""
        print(f"Sending prompt to {self.ollama_model_name}: {prompt}")
        
        try:
            if chat_mode:
                # Use the chat API
                response = ollama.chat(
                    model=self.ollama_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                
                for chunk in response:
                    if self.stop_event.is_set():
                        break
                        
                    if chunk and 'message' in chunk and 'content' in chunk['message']:
                        text_chunk = chunk['message']['content']
                        self._process_text_chunk(text_chunk)
            else:
                # Use the generate API
                response = ollama.generate(
                    model=self.ollama_model_name,
                    prompt=prompt,
                    stream=True
                )
                
                for chunk in response:
                    if self.stop_event.is_set():
                        break
                        
                    if chunk and 'response' in chunk:
                        text_chunk = chunk['response']
                        self._process_text_chunk(text_chunk)
            
            # Process any remaining text
            self._finish_processing()
            
        except Exception as e:
            print(f"Error from Ollama: {e}")
        
        finally:
            print("Finished processing prompt")


def main():
    """Main function to run the Ollama to Piper streamer."""
    parser = argparse.ArgumentParser(description="Stream text from Ollama to Piper TTS")
    parser.add_argument("--piper-model", required=True, help="Path to the Piper model (.onnx file)")
    parser.add_argument("--ollama-model-name", default="gemma3:1b", help="Ollama model to use (default: llama3)")
    parser.add_argument("--speaking-rate", type=float, default=1.0, help="how fast should generated speech be, 1.0 is default, higher numbers mean faster speech")
    parser.add_argument("--prompt", help="Text prompt to send to Ollama")
    parser.add_argument("--generate", action="store_true", help="Use generate API instead of chat API")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the streamer
    streamer = OllamaToPiperStreamer(
        piper_model_path=args.piper_model,
        ollama_model_name=args.ollama_model_name,
        speaking_rate=args.speaking_rate
    )
    
    try:
        if args.interactive:
            print(f"Interactive mode with {args.ollama_model_name} and Piper. Type 'exit' to quit.")
            
            while True:
                prompt = input("\nYour prompt: ").strip()
                
                if prompt.lower() in ('exit', 'quit'):
                    break
                
                if prompt:
                    streamer.process_prompt(prompt, chat_mode=not args.generate)
        
        elif args.prompt:
            streamer.process_prompt(args.prompt, chat_mode=not args.generate)
        
        else:
            parser.print_help()
    
    finally:
        streamer._close()


if __name__ == "__main__":
    main()