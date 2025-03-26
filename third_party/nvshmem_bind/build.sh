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
  NVSHMEM_HOME=${NVSHMEM_DIR} pip3 install .
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

function download_libnvshmem_device_bc() {
  local dst_path=${PROJECT_ROOT}/../nvidia/backend/lib

  local tmp_dir
  tmp_dir=$(mktemp -d)
  if [[ $? -ne 0 ]]; then
    echo "Unable to create temporary directory" >&2
    return 1
  fi

  local target_file="$dst_path/libnvshmem_device.bc"
  if [[ -f "$target_file" ]]; then
    echo "$target_file already exists, skip downloading"
    return 0
  fi

  echo "download..."
  local url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz"
  if ! wget -q --show-progress -P $tmp_dir $url; then
    echo "download failed" >&2
    rm -rf $tmp_dir
    return 1
  fi

  echo "extract..."
  local tar_file="$tmp_dir/libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz"
  if ! tar -xf "$tar_file" -C "$tmp_dir"; then
    echo "decompress file failed。" >&2
    rm -rf $tmp_dir
    return 1
  fi

  local extract_dir=$tmp_dir/libnvshmem-linux-x86_64-3.2.5_cuda12-archive
  if [[ ! -d $extract_dir ]]; then
    echo "No such director: $extract_dir" >&2
    rm -rf $tmp_dir
    return 1
  fi

  local lib_file=$extract_dir/lib/libnvshmem_device.bc
  if [[ ! -f $lib_file ]]; then
    echo "No such file: $lib_file" >&2
    rm -rf $tmp_dir
    return 1
  fi

  if ! mv -f $lib_file $dst_path; then
    echo "File move failed" >&2
    rm -rf "$tmp_dir"
    return 1
  fi

  rm -rf $tmp_dir
}

function download_libnvshmem_device_bc_byted() {
  local dst_path=${PROJECT_ROOT}/../nvidia/backend/lib
  lib_file=/tmp/libnvshmem_device.bc
  wget -q --show-progress https://tosv.byted.org/obj/flux/dsit-triton/nvshmem/3.2.5-1/bc/libnvshmem_device.bc -O ${lib_file}
  if ! mv -f $lib_file $dst_path; then
    echo "File move failed" >&2
    rm -rf "$tmp_dir"
    return 1
  fi
}

set_arch
set_nvcc_gencode

export NVSHMEM_DIR=${PROJECT_ROOT}/../nvshmem/build/install
bash -x ${PROJECT_ROOT}/build_nvshmem.sh ${build_args}
build_pynvshmem

download_libnvshmem_device_bc_byted

echo "done"
