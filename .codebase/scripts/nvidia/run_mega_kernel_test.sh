
#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

DENSE_MODEL=".codebase/hf_configs/Qwen/Qwen3-0.6B"

function run_mega_kernel_testcases() {
  python3 python/triton_dist/mega_triton_kernel/test/ops/test_mlp_layer.py
  python3 python/triton_dist/mega_triton_kernel/test/ops/test_rms_norm.py
  python3 python/triton_dist/mega_triton_kernel/test/ops/test_add.py
  python3 python/triton_dist/mega_triton_kernel/test/ops/test_page_attn.py
  NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/ops/test_allreduce.py
  NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/test_qwen3.py --model .codebase/hf_configs/Qwen/Qwen3-8B --backend mega_kernel
  NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/bench_qwen3.py --model .codebase/hf_configs/Qwen/Qwen3-8B --seq_len 128 --allreduce_method one_shot_multimem
  # too large for CI
  # NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/bench_qwen3.py --model Qwen/Qwen3-32B --seq_len 128 --allreduce_method one_shot_multimem
}

export RANDOM_PARAMS=1
run_mega_kernel_testcases
