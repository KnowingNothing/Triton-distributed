# Build Triton-distributed

## The best practice to use Triton-distributed with the Nvidia backend:
- Python 3.11 (suggest using virtual environment)
- CUDA 12.4
- Torch 2.4.1
- Clang 19

#### if for AMD GPU:
- ROCM 6.3.0
- Torch 2.4.1 with ROCM support



Dependencies with other versions may also work well, but this is not guaranteed. If you find any problem in installing, please tell us in Issues.

### Steps:
1. Clone Triton-distributed to your own path (e.g., `/home/Triton-distributed`)
2. Update submodules
    ```sh
    git submodule update --init --recursive
    ```
3. Install dependencies
    ```sh
    pip3 install torch==2.4.1
    pip3 install black "clang-format==19.1.2" pre-commit ruff yapf==0.43
    pip3 install ninja cmake wheel pybind11 cuda-python==12.4 numpy chardet pytest
    ```
    for AMD GPU, use torch with rocm support and hip-python
    ```sh
    python3 -m pip install -i https://test.pypi.org/simple hip-python>=6.3.0
    ```
4. Apply NVSHMEM fix
(Disclaimer: This step is because of NVSHMEM license requirements, it is illegal to release any modified codes or patch.)

    1. Download NVSHMEM 3.2.5 Source Code [NVSHMEM Open Source Packages](https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz)
    2. Extract to designated location
        ```sh
        mkdir -p /home/Triton-distributed/3rdparty/nvshmem
        tar -xvf nvshmem_src_3.2.5-1.txz -C /home/Triton-distributed/3rdparty/nvshmem/ --strip-components=1
        ```
    3. Bitcode Bug Fix: [BUG with nvshmem 3.2.5 for bitcode compiling](https://forums.developer.nvidia.com/t/bug-with-nvshmem-3-2-5-for-bitcode-compiling/327847)

       File: ```src/include/non_abi/device/common/nvshmemi_common_device.cuh``` (Line 287)
       ```cpp
        - dst = (void *)(dst_p + nelems);
        - src = (void *)(src_p + nelems);

        +#ifdef __clang_llvm_bitcode_lib__
        +    dst = (void *)(dst_p + nelems * 4);
        +    src = (void *)(src_p + nelems * 4);
        +#else
        +    dst = (void *)(dst_p + nelems);
        +    src = (void *)(src_p + nelems);
        +#endif
        ```
    4. Clang Compilation Error Fix

       File: ```src/include/device_host/nvshmem_common.cuh``` (Line 41)
       ```cpp
        - __device__ int __nvvm_reflect(const char *s);
        + __device__ int __nvvm_reflect(const void *s);
       ```

5. Build or install Clang-19 for building NVSHMEM bitcode library

    Clang-19 is required to build NVSHMEM bitcode library. To install Clang-19, we recommend pre-built binary:
    ```sh
    sudo apt install clang-19 llvm-19 libclang-19-dev
    ```
    Also, you may install Clang-19 from source by building LLVM (see [how to build LLVM](https://llvm.org/docs/CMake.html)).
    ```sh
    git clone git@github.com:llvm/llvm-project.git
    cd llvm-project
    git checkout llvmorg-19.1.0
    mkdir build
    cd build
    cmake -G Ninja ../llvm    -DLLVM_ENABLE_PROJECTS="clang;lldb;lld"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU"    -DCMAKE_BUILD_TYPE=Release    -DLLVM_ENABLE_ASSERTIONS=ON    -DMLIR_ENABLE_BINDINGS_PYTHON=ON  -DCMAKE_BUILD_TYPE=Release
    cmake --build .
    ```
    Remember to put the built binary and library path to `PATH` and `LD_LIBRARY_PATH`.
    ```sh
    export PATH=$PATH:/home/llvm-project/build/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/llvm-project/build/lib
    ```

    For ROCMSHMEM on AMD GPU, no explicit build required as the building process is integrated with Triton-distributed.

6. Build Triton-distributed
    Then you can build Triton-distributed.
    ```sh
    cd /home/Triton-distributed/python
    export USE_TRITON_DISTRIBUTED_AOT=0
    python3 setup.py build_ext
    ```

    We also provide AOT version of Triton-distributed. If you want to use AOT, then
    ```sh
    cd /home/Triton-distributed/
    source scripts/setenv.sh
    bash scripts/gen_aot_code.sh
    export USE_TRITON_DISTRIBUTED_AOT=1
    cd python
    python3 setup.py build_ext
    ```
    (Note: You have to first build non-AOT version before building AOT version, once you build AOT version, you will always build for AOT in future. To unset this, you have to remove your build directory: `python/build`)
6. Setup environment variables (Do this step at the beginning every time you use Triton-distributed)
    ```sh
    cd /home/Triton-distributed
    source scripts/setenv.sh
    ```

### Test your installation
#### AllGather GEMM example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma
```

#### GEMM ReduceScatter example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_gemm_rs.py 8192 8192 29568
```

#### NVSHMEM example in Triton-distributed
```sh
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_nvshmem_api.py
```

### Run All The Test Files
```sh
# basic
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma_multi_barrier
# ag gemm
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_autotune
# gemm rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_gemm_rs.py 8192 8192 29568
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_gemm_rs.py 8192 8192 29568 --check
# allgather
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ag_small_msg.py
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_all_gather.py
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_fast_allgather.py   --iters 10   --warmup_iters 20   --mode push_2d_ll   --minbytes 4096   --maxbytes 8192
# all-to-all
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_all_to_all.py
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ep_moe_inference.py
# nvshmem related
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_nvshmem_api.py
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ring_put.py
# flash decoding
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_persistent
USE_TRITON_DISTRIBUTED_AOT=1 bash ./scripts/launch.sh  ./python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_persistent_aot
USE_TRITON_DISTRIBUTED_AOT=1 bash ./scripts/launch.sh  ./python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_aot
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_sp_decode_attn.py --case perf
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
USE_TRITON_DISTRIBUTED_AOT=1 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
# ag moe
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ag_moe.py --M 2048
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --autotune
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_ag_moe_inter_node.py --M 2048
# moe rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2 --check
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2 --check --autotune
# ep a2a
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash ./scripts/launch.sh  ./python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash ./scripts/launch.sh  ./python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8
```

## To use Triton-distributed with the AMD backend:
Starting from the rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4 Docker container
#### Steps:
1. Clone the repo
```sh
git clone https://github.com/ByteDance-Seed/Triton-distributed.git
```
2. Update submodules
```sh
cd Triton-distributed/
git submodule update --init --recursive
```
3. Install dependencies
```sh
sudo apt-get update -y
sudo apt install -y libopenmpi-dev
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --no-deps
./third_party/rocshmem_bind/build.sh
python3 -m pip install -i https://test.pypi.org/simple hip-python~=6.3.2 (or whatever Rocm version you have)
pip3 install pybind11
```
4. Build Triton-distributed
```sh
pip3 install -e python --verbose --no-build-isolation
```
### Test your installation
#### GEMM ReduceScatter example on single node
```sh
bash ./third_party/distributed/launch_amd.sh ./third_party/distributed/distributed/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
 ```
and see the following (reduced) output
```sh
torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 ./third_party/distributed/distributed/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
✅ Triton and Torch match
```
