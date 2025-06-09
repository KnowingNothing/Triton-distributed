################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

#!/bin/bash

export http_proxy=http://sys-proxy-rd-relay.byted.org:3128  https_proxy=http://sys-proxy-rd-relay.byted.org:3128  no_proxy=code.byted.org

# --- Dynamically detect CUDA version ---
cuda_version_full=""
parsed_cuda_slug="" # Slug for the URL, e.g., cu121

if command -v nvcc &> /dev/null; then
    cuda_version_output=$(nvcc --version)
    if [[ $cuda_version_output =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
        cuda_major="${BASH_REMATCH[1]}"
        cuda_minor="${BASH_REMATCH[2]}"
        cuda_version_full="${cuda_major}.${cuda_minor}"
        parsed_cuda_slug="cu${cuda_major}${cuda_minor}"
        echo "Detected CUDA version: $cuda_version_full (Using slug: $parsed_cuda_slug)"
    else
        echo "Error: Could not parse CUDA version from 'nvcc --version' output."
        echo "Output was: $cuda_version_output"
        exit 1
    fi
else
    echo "Error: 'nvcc' command not found. Cannot automatically detect CUDA version."
    echo "If CUDA is installed, please ensure 'nvcc' is in your PATH."
    exit 1
fi

# --- Dynamically detect PyTorch version ---
pytorch_version_full=""
parsed_pytorch_slug="" # Slug for the URL, e.g., torch2.1

PY_EXECUTABLE=""
if command -v python3 &> /dev/null; then
    PY_EXECUTABLE="python3"
elif command -v python &> /dev/null; then
    PY_EXECUTABLE="python"
else
    echo "Error: Neither 'python3' nor 'python' command found. Cannot detect PyTorch version."
    exit 1
fi

pytorch_version_output=$($PY_EXECUTABLE -c "import sys; sys.path.append('.'); import torch; print(torch.__version__)" 2>/dev/null)
if [[ -n "$pytorch_version_output" ]]; then
    # Remove +cudaX.X or .dev suffixes etc.
    pytorch_cleaned_version=$(echo "$pytorch_version_output" | sed -E 's/\+.*//' | sed -E 's/\.dev.*//')
    if [[ $pytorch_cleaned_version =~ ^([0-9]+)\.([0-9]+) ]]; then # Match start of string for major.minor
        torch_major="${BASH_REMATCH[1]}"
        torch_minor="${BASH_REMATCH[2]}"
        # We care about major.minor for the slug
        parsed_pytorch_slug="torch${torch_major}.${torch_minor}"
        echo "Detected PyTorch version: $pytorch_version_output (Using slug: $parsed_pytorch_slug)"
    else
        echo "Error: Could not parse major.minor PyTorch version from '$pytorch_version_output'."
        exit 1
    fi
else
    echo "Error: Could not import PyTorch or get its version."
    echo "Please ensure PyTorch is installed in the current Python environment ($PY_EXECUTABLE)."
    exit 1
fi

# --- Build URL and install ---
if [[ -z "$parsed_cuda_slug" || -z "$parsed_pytorch_slug" ]]; then
    echo "Error: CUDA slug or PyTorch slug could not be determined. Exiting."
    exit 1
fi

flashinfer_whl_url="https://flashinfer.ai/whl/${parsed_cuda_slug}/${parsed_pytorch_slug}/"
pip install flashinfer-python -i "$flashinfer_whl_url"
pip install flash-attn --no-build-isolation
pip install transformers==4.51.3 numpy==1.26

MODEL_NAME="Qwen/Qwen3-32B"
while true; do
  echo "Download: $MODEL_NAME (timeout: 60s)..."
  timeout 120s huggingface-cli download "$MODEL_NAME"

  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "model '$MODEL_NAME' download successfully!"
    break
  elif [ $EXIT_CODE -eq 124 ]; then
    echo "Timeout!"
  else

    echo "Download failed with exit code $EXIT_CODE. Retrying in 5 seconds..."
  fi

  sleep 5
done

if [[ $? -eq 0 ]]; then
    echo "e2e env installation successful."
else
    echo "Error: e2e env installation failed."
    exit 1
fi
