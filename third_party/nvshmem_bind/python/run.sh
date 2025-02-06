#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
TRITON_NVSHMEM_DIR=${SCRIPT_DIR}/
PYNVSHMEM_DIR=${SCRIPT_DIR}/../pynvshmem
NVSHMEM_ROOT=${SCRIPT_DIR}/../../nvshmem/build/install

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_ROOT}/lib
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID

export PYTHONPATH=$PYTHONPATH:${PYNVSHMEM_DIR}/build:$TRITON_NVSHMEM_DIR
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

export PYTHONPATH=$PYTHONPATH:${TRITON_NVSHMEM_DIR}:${PYNVSHMEM_DIR}/build

export TRITON_CACHE_DIR=triton_cache
export NVSHMEM_HOME=${NVSHMEM_ROOT}
mkdir -p triton_cache

# run_ag_gemm
function run_nvshmem_sample() {
  pushd ${TRITON_NVSHMEM_DIR}
  torchrun --nproc_per_node=8 --nnodes=1 example/sample.py
}
run_nvshmem_sample
