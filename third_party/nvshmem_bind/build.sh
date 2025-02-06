#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})
ARCH=""

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
  --arch)
    # Process the arch argument
    ARCH="$2"
    shift # Skip the argument value
    shift # Skip the argument key
    ;;
  *)
    # Unknown argument
    echo "Unknown argument: $1"
    shift # Skip the argument
    ;;
  esac
done

if [[ -n $ARCH ]]; then
  build_args=" --arch ${ARCH}"
fi

function build_pynvshmem() {
  pushd ${PROJECT_ROOT}/pynvshmem
  mkdir -p build
  pushd build
  cmake .. \
    -DNVSHMEM_DIR=${NVSHMEM_DIR}/lib/cmake/nvshmem \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  make -j VERBOSE=1
  popd
  popd
}

function set_arch() {
  if [[ -z $ARCH ]]; then
    export ARCH=$(python3 -c 'import torch; print("".join([str(x) for x in torch.cuda.get_device_capability()]))')
    echo "using CUDA arch: ${ARCH}"
  fi
}

function set_nvcc_gencode() {
  NVCC_GENCODE="" # default none
  arch_list=()
  IFS=";" read -ra arch_list <<<"$ARCH"
  for _arch in "${arch_list[@]}"; do
    NVCC_GENCODE="-gencode=arch=compute_${_arch},code=sm_${_arch} ${NVCC_GENCODE}"
  done
}

function build_nvshmem_cubin() {
  pushd ${PROJECT_ROOT}/runtime
  nvcc -rdc=true -ccbin g++ $NVCC_GENCODE -I$NVSHMEM_DIR/include nvshmem_wrapper.cu -ptx -c -o nvshmem_wrapper.ptx
  IFS=";" read -ra arch_list <<<"$ARCH"
  for _arch in "${arch_list[@]}"; do
    ptxas -c nvshmem_wrapper.ptx --gpu-name=sm_${_arch} -o nvshmem_wrapper.sm${_arch}.cubin
  done
  popd
}

set_arch
set_nvcc_gencode

export NVSHMEM_DIR=${PROJECT_ROOT}/../nvshmem/build/install
bash -x build_nvshmem.sh ${build_args}
build_pynvshmem

build_nvshmem_cubin
