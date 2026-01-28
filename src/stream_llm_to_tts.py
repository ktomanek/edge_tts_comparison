# Generate text output via an LLM using Ollama and synthesize speech in streaming-fashing.
# Using Piper, this works in realtime; Kokoro seems too slow for that even on a Mac M2.

# For Kokoro, when increasing the speaking speed, first word is often cut off.
# ## Installation

# * install ollama locally: https://ollama.com/download
# * then pull the model you want ot use, eg: 
# ```ollama pull gemma3:1b```
# * then install [ollama python library](https://github.com/ollama/ollama-python) 
# ```pip install ollama```
# * other dependencies
# * ```pip install sounddevice nltk```
# * download sentence splitter: ```python -c "import nltk; nltk.download('punkt_tab')```
# ## Run
# ```
# python stream_llm_to_tts.py \
#     --ollama-model-name=gemma3:1b \
#     --tts_engine piper  \
#     --speaking-rate=2.0 \
#     --prompt "Explain to me what a cat does all day. Use exactly 3 sentences."
# ```
# ```
# python stream_llm_to_tts.py \
#     --ollama-model-name=gemma3:1b \
#     --tts_engine kokoro  \
#     --speaking-rate=1.0 \
#     --prompt "Explain to me what a cat does all day. Use exactly 3 sentences."
# ```



import json
import ollama
import sounddevice as sd
import nltk
from nltk.tokenize import sent_tokenize
import time
import threading
import argparse
import queue
import signal
import sys
import tts_lib
import time 

# Download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MAX_TEXT_BUFFER = 125
MIN_TEXT_BUFFER = 75

import re
import emoji

def clean_llm_output(text):
    """
    Remove formatting symbols.
    """
    text = text.replace('*', ' ')
    text = emoji.replace_emoji(text, replace='')
    return text

DEFAULT_SYSTEM_PROMPT = """You are an assistant that runs on an edge device. A person is interacting with you via voice. 
For that reason you should limit your answers a bit in length unless explicitly asked to give detailed responses. 
If you are asked for advise, list all relevant points but limit yourself to the top 3 items.
But most importantly, do not output any sort of formatting information.
Do not start your sentences with 'okay' always. Be friendly and helpful."""

class OllamaToPiperStreamer:
    def __init__(self, 
                 ollama_model_name="gemma3:1b", 
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 tts_engine='piper',
                 speaking_rate=1.0, # higher numbers means faster
                 tts_model_path=None,
                 ):
        """Initialize the streamer with Piper and Ollama models."""
        self.ollama_model_name = ollama_model_name
        print(f"Using Ollama model: {self.ollama_model_name}")

        self.ollama_options={
            "temperature": 0.3,  # lower temperature for speed
            "num_predict": -1,  # unlimited
            #"num_ctx": 1024
        }

        # warm up model
        t1 = time.time()
        ollama.chat(
            model=self.ollama_model_name,
            messages=[{"role": "user", "content": "hi"}],
            stream=False
        )        
        print(f"Ollama model warmed up in {time.time()-t1} secs.")

        # initialize conversation context
        self.messages = [
            {'role': 'system', 'content': system_prompt},
        ]

        # load TTS
        if tts_engine == 'piper':
            print('Initializing Piper TTS')
            if tts_model_path:
                print(f"Initializing with tts model: {tts_model_path}")                
                self.tts = tts_lib.TTS_Piper(tts_model_path)
            else:
                self.tts = tts_lib.TTS_Piper()
        elif tts_engine == 'kokoro':
            print('Initializing Kokoro TTS')
            if tts_model_path:
                print(f"Initializing with tts model: {tts_model_path}")                
                self.tts = tts_lib.TTS_Kokoro(tts_model_path)
            else:
                self.tts = tts_lib.TTS_Kokoro()
        else:
            raise ValueError('Unknown tts engine.')
        
        self.sample_rate = self.tts.get_sample_rate()
        print(f"Using sample rate: {self.sample_rate} Hz")
        
        self.speaking_rate = speaking_rate
        print(f"Using speaking rate: {self.speaking_rate}")        

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

        # increase buffer size if needed, esp on slower devices like raspberry pi
        buffer_size = 1024

        if self.audio_stream is None:
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=buffer_size,
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
            audio_data, sample_rate = self.tts.synthesize(
                text, target_sr = self.sample_rate, 
                speaking_rate=speed, return_as_int16=True)
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
    
    def process_prompt(self, user_prompt, verbose=False):
        """Process a prompt through Ollama and stream to Piper."""
        
        self.messages.append({'role': 'user', 'content': user_prompt})

        if verbose:
            print('>> context length: turns:', len(self.messages) / 2)
            print('>> context length: characters:', len(json.dumps(self.messages)))

            pretty_json = json.dumps(self.messages, indent=2)
            print(f">> Sending prompt to {self.ollama_model_name}: {pretty_json}")

        response = ollama.chat(
            model=self.ollama_model_name,
            messages=self.messages,
            stream=True,
            options=self.ollama_options
        )
        
        text_chunks = []
        for chunk in response:
            if self.stop_event.is_set():
                break
                
            if chunk and 'message' in chunk and 'content' in chunk['message']:
                text_chunk = chunk['message']['content']


                # remove asterisks and other formatting info from the text
                text_chunk = clean_llm_output(text_chunk)

                self._process_text_chunk(text_chunk)            
                text_chunks.append(text_chunk)

        assistant_response = ''.join(text_chunks)
        self.messages.append({'role': 'assistant', 'content': assistant_response})


        # Process any remaining text
        self._finish_processing()
            


def main():
    """Main function to run the Ollama to Piper streamer."""
    parser = argparse.ArgumentParser(description="Stream text from Ollama to Piper TTS")
    parser.add_argument("--ollama-model-name", default="gemma3:1b", help="Ollama model to use (default: llama3)")
    parser.add_argument("--tts_engine", choices=['piper', 'kokoro'], default="piper", help="which tts engine to use; piper is much faster than kokoro.")
    parser.add_argument("--tts_model_path", required=False, help="Path to the tts model (.onnx file)")
    parser.add_argument("--speaking-rate", type=float, default=1.0, help="how fast should generated speech be, 1.0 is default, higher numbers mean faster speech")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="Instructions for the model.")
    parser.add_argument("--user_prompt", help="Text prompt to send to Ollama")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose status info")
    
    args = parser.parse_args()
    
    # Initialize the streamer
    streamer = OllamaToPiperStreamer(
        ollama_model_name=args.ollama_model_name,
        system_prompt=args.system_prompt,
        tts_engine=args.tts_engine,
        speaking_rate=args.speaking_rate,
        tts_model_path=args.tts_model_path
    )
    
    try:
        system_prompt = args.system_prompt
        if args.interactive:
            print(f"Interactive mode with {args.ollama_model_name} and Piper. Type 'exit' to quit.")
            
            while True:
                prompt = input("\nYour prompt: ").strip()
                
                if prompt.lower() in ('exit', 'quit'):
                    break
                
                if prompt:
                    streamer.process_prompt(prompt, args.verbose)
        
        elif args.user_prompt:
            streamer.process_prompt(args.user_prompt, args.verbose)
        
        else:
            parser.print_help()
    
    finally:
        streamer._close()


if __name__ == "__main__":
    main()