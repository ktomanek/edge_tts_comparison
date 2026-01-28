#!/bin/bash

# Download PocketTTS ONNX models from Hugging Face
# Repository: https://huggingface.co/KevinAHM/pocket-tts-onnx

set -e  # Exit on error

# Configuration
HF_REPO="KevinAHM/pocket-tts-onnx"
ONNX_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main/onnx"
ROOT_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"
TARGET_DIR="models/pockettts_onnx"

# Create target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# ONNX model files to download (from onnx subdirectory)
declare -a onnx_models=(
    "flow_lm_flow.onnx"
    "flow_lm_flow_int8.onnx"
    "flow_lm_main.onnx"
    "flow_lm_main_int8.onnx"
    "mimi_decoder.onnx"
    "mimi_decoder_int8.onnx"
    "mimi_encoder.onnx"
    "text_conditioner.onnx"
    "LICENSE"
)

# Additional files to download (from repo root)
declare -a root_files=(
    "tokenizer.model"
)

echo "=========================================="
echo "Downloading PocketTTS ONNX Models"
echo "=========================================="
echo "Target directory: ${TARGET_DIR}"
echo "Total files: $((${#onnx_models[@]} + ${#root_files[@]}))"
echo ""

# Download ONNX models
echo "Downloading ONNX models..."
for model in "${onnx_models[@]}"; do
    url="${ONNX_BASE_URL}/${model}"
    output="${TARGET_DIR}/${model}"

    echo "Downloading: ${model}"

    # Check if file already exists
    if [ -f "${output}" ]; then
        echo "  ⚠ File already exists, skipping: ${model}"
        continue
    fi

    # Download with wget (with fallback to curl)
    if command -v wget &> /dev/null; then
        wget -q --show-progress "${url}" -O "${output}"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "${url}" -o "${output}"
    else
        echo "  ✗ Error: Neither wget nor curl is available"
        exit 1
    fi

    # Verify download
    if [ -f "${output}" ]; then
        size=$(du -h "${output}" | cut -f1)
        echo "  ✓ Downloaded: ${model} (${size})"
    else
        echo "  ✗ Failed to download: ${model}"
        exit 1
    fi

    echo ""
done

# Download root files (tokenizer, etc.)
echo "Downloading additional files..."
for file in "${root_files[@]}"; do
    url="${ROOT_BASE_URL}/${file}"
    output="${TARGET_DIR}/${file}"

    echo "Downloading: ${file}"

    # Check if file already exists
    if [ -f "${output}" ]; then
        echo "  ⚠ File already exists, skipping: ${file}"
        continue
    fi

    # Download with wget (with fallback to curl)
    if command -v wget &> /dev/null; then
        wget -q --show-progress "${url}" -O "${output}"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "${url}" -o "${output}"
    else
        echo "  ✗ Error: Neither wget nor curl is available"
        exit 1
    fi

    # Verify download
    if [ -f "${output}" ]; then
        size=$(du -h "${output}" | cut -f1)
        echo "  ✓ Downloaded: ${file} (${size})"
    else
        echo "  ✗ Failed to download: ${file}"
        exit 1
    fi

    echo ""
done

echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Downloaded files in ${TARGET_DIR}:"
ls -lh "${TARGET_DIR}"
