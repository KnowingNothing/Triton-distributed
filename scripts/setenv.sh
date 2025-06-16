export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

SCRIPT_DIR="$(pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
NVSHMEM_ROOT=${SCRIPT_DIR}/3rdparty/nvshmem/build/install
OMPI_BUILD=${SCRIPT_DIR}/shmem/rocshmem_bind/ompi_build/install/ompi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_ROOT}/lib:${OMPI_BUILD}/lib
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID

export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

export TRITON_CACHE_DIR=${SCRIPT_DIR}/triton_cache
export NVSHMEM_HOME=${NVSHMEM_ROOT}

export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/python
mkdir -p ${SCRIPT_DIR}/triton_cache

