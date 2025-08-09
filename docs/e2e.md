# End-to-End Demo for Triton-Distributed
## Environment Set Up

First, you need to set up the environment for running the end-to-end demo. This includes installing necessary dependencies and configuring the environment variables. You can do this by running the following commands:
```bash
bash ./scripts/build_e2e_env.sh
source ./scripts/setenv.sh
```

## Layer Level End-to-end Demo

We provide TP_MLP, TP_Attn, EP_MoE, SP_Attn for end-to-end demo. You can run the end-to-end demo for these layers by executing the following commands:
```bash
# tp mlp
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model Qwen/Qwen3-32B --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 2048 --model Qwen/Qwen3-32B --mode gemm_ar

# tp attn prefill
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode gemm_ar

# tp attn decode
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode gemm_ar
```

## Model Level End-to-end Demo

We provide a model level end-to-end demo. You can run the end-to-end demo executing the following command:
```bash
# tp e2e check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model Qwen/Qwen3-32B --check --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --check --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --check --mode gemm_ar

# tp e2e prefill
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode gemm_ar

# tp e2e decode
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode gemm_ar

# e2e inference
bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150
bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --triton_dist
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --triton_dist_AR
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --triton_dist_gemm_ar
```