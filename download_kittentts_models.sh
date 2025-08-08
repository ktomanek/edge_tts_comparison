mkdir -p models/kitten_tts

# ONNX models are here
wget https://huggingface.co/KittenML/kitten-tts-nano-0.1/resolve/main/kitten_tts_nano_v0_1.onnx -O models/kitten_tts/kitten_tts_nano_v0_1.onnx

# voices
wget https://huggingface.co/KittenML/kitten-tts-nano-0.1/resolve/main/voices.npz -O models/kitten_tts/voices.npz
