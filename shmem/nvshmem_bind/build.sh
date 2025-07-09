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

function move_libnvshmem_device_bc() {
  local dst_path=${PROJECT_ROOT}/../../3rdparty/triton/third_party/nvidia/backend/lib
  nvshmem_dir=${NVSHMEM_SRC:-${PROJECT_ROOT}/../../3rdparty/nvshmem}
  lib_file=${nvshmem_dir}/build/install/lib/libnvshmem_device.bc
  if ! mv -f $lib_file $dst_path; then
    echo "File move failed" >&2
    rm -rf "$tmp_dir"
    return 1
  fi

  lib_file=${nvshmem_dir}/build/install/lib/libnvshmemi_device.bc
  if ! mv -f $lib_file $dst_path; then
    echo "File move failed" >&2
    rm -rf "$tmp_dir"
    return 1
  fi

  echo "Move done."
}

set_arch
set_nvcc_gencode

bash -x ${PROJECT_ROOT}/build_nvshmem.sh ${build_args}

move_libnvshmem_device_bc

echo "done"
