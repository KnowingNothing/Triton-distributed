#!/bin/bash

set -e

function prepare_env() {
  export CUDA_HOME=/usr/local/cuda
  export PATH=$PATH:$CUDA_HOME/bin
  nvcc --version >/dev/null
  CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  git submodule update --init --recursive
  pip3 install cuda-python==${CUDA_VER} setuptools==69.0.0 wheel pybind11
  echo "numpy<2" >/tmp/pip_install_constraint.txt
}

function build() {
  export USE_TRITON_DISTRIBUTED_AOT=0
  pip3 install -c /tmp/pip_install_constraint.txt -e python[build,tests,tutorials] --verbose --no-build-isolation --use-pep517
}

function build_wheel() {
  pushd python
  python3 setup.py bdist_wheel
  popd
}

prepare_env
echo "Building..."
build
echo "Building wheel..."
build_wheel
echo "Build completed successfully."
