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
import os
import torch
import torch.distributed

import argparse

from triton_dist.kernels.nvidia import (
    create_ag_group_gemm_context,
    ag_group_gemm,
)

from triton_dist.utils import (group_profile, initialize_distributed, TP_GROUP, perf_func, dist_print,
                               get_device_max_shared_memory_size)


def torch_moe_scatter_group_gemm(in_features, expert_weights, topk_ids):
    M, K = in_features.shape
    in_features = (in_features.view(M, -1, K).repeat(1, topk_ids.shape[1], 1).reshape(-1, K))
    out = torch.zeros(
        M * topk_ids.shape[1],
        expert_weights.shape[2],
        dtype=in_features.dtype,
        device=in_features.device,
    )

    topk_ids = topk_ids.view(-1)

    for i in range(expert_weights.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = in_features[mask] @ expert_weights[i]
    return out


def torch_ag_group_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    full_topk_ids: torch.Tensor,
):
    M_per_rank, K = local_input.shape
    M = M_per_rank * pg.size()
    a_tensor_golden_part_k = torch.zeros(M, K, dtype=local_input.dtype).cuda()
    torch.distributed.all_gather_into_tensor(a_tensor_golden_part_k, local_input, group=pg)
    a_tensor_golden = (a_tensor_golden_part_k.reshape(pg.size(), 1, M_per_rank, K).transpose(1, 2).reshape(M, K))
    tensor_golden = torch_moe_scatter_group_gemm(a_tensor_golden, local_weight, full_topk_ids)
    return a_tensor_golden, tensor_golden


def estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype: torch.dtype, b_dtype: torch.dtype, stages: int):
    return (BM * BK * a_dtype.itemsize + BN * BK * b_dtype.itemsize) * (stages - 1)


def estimate_gemm_max_stages(BM, BN, BK, a_dtype, b_dtype, shared_memory_limit: int):
    return shared_memory_limit // estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype, b_dtype, 2) + 1


def perf_test(name, input_len, dtype, config):
    M = input_len
    N = config["N"]
    K = config["K"]
    E = config["E"]
    topk = config["TOPK"]
    tp_group = TP_GROUP()

    assert M % tp_group.size() == 0
    assert N % tp_group.size() == 0
    M_per_rank = M // tp_group.size()
    N_per_rank = N // tp_group.size()

    if tp_group.rank() == 0:
        print(f"shape: M={M}, N={N}, K={K}; num experts={E}, topk={topk}")

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([E, K, N_per_rank], dtype=dtype, device="cuda")

    score = ((-2 * torch.randn((M_per_rank, E), device="cuda", dtype=dtype) + 1) / 100 * (tp_group.rank() + 1))
    score = torch.softmax(score, dim=-1)
    _, local_topk_ids = torch.topk(score, topk)
    full_topk_ids = torch.zeros(M_per_rank * tp_group.size(), topk, dtype=local_topk_ids.dtype).cuda()
    torch.distributed.all_gather_into_tensor(full_topk_ids, local_topk_ids, group=tp_group)

    BM, BN, BK, stage = config["BM"], config["BN"], config["BK"], config["num_stages"]
    shared_memory_limit = get_device_max_shared_memory_size(torch.cuda.current_device())
    max_stages = estimate_gemm_max_stages(BM, BN, BK, A.dtype, B.dtype, shared_memory_limit)
    if stage > max_stages:
        print(f"stage {stage} exceeds max stages {max_stages}, force set to {max_stages}...")
        config["num_stages"] = max_stages

    ctx = create_ag_group_gemm_context(A, B, tp_group.rank(), tp_group.size(), full_topk_ids, max_M=M,
                                       BLOCK_M=config["BM"], BLOCK_N=config["BN"], BLOCK_K=config["BK"],
                                       GROUP_SIZE_M=config["GROUP_SIZE_M"], stages=config["num_stages"],
                                       warps=config["num_warps"])

    result = ag_group_gemm(A, B, ctx)

    _, C_golden = torch_ag_group_gemm(tp_group, A, B, full_topk_ids)

    assert torch.allclose(C_golden, result, atol=1e-3, rtol=1e-3)

    def sort_func():
        return ctx.sort_topk_ids_align_block_size(full_topk_ids, E, tp_group.size(), M_per_rank, config["BM"])[0]

    def triton_func():
        return ag_group_gemm(A, B, ctx)

    def torch_func():
        return torch_ag_group_gemm(tp_group, A, B, full_topk_ids)

    name = name.lower().replace(" ", "_").replace("-", "_")
    with group_profile(f"ag_moe_{name}_{os.environ['TORCHELASTIC_RUN_ID']}", do_prof=args.profile, group=tp_group):
        _, permute_perf = perf_func(sort_func, iters=args.iters, warmup_iters=args.warmup_iters)
        _, triton_perf = perf_func(triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
        _, torch_perf = perf_func(torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

    dist_print(
        f"RANK {tp_group.rank()} perf: compute permute {permute_perf:0.3f} ms, dist-triton={triton_perf:0.3f} ms, torch={torch_perf:0.3f} ms; speedup={torch_perf/triton_perf:0.2f}",
        need_sync=True,
        allowed_ranks=list(range(tp_group.size())),
    )


layer_configs = {
    "Dummy Model": {
        "N": 8192, "K": 8192, "E": 32, "TOPK": 3, "BM": 128, "BN": 128, "BK": 32, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "Qwen1.5-MoE-A2.7B": {
        "N": 1408, "K": 2048, "E": 60, "TOPK": 4, "BM": 128, "BN": 128, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "Mixtral-8x7B": {
        "N": 4096, "K": 14336, "E": 8, "TOPK": 2, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "Mixtral-8x22B": {
        "N": 6144, "K": 16384, "E": 8, "TOPK": 2, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "DeepSeek-MoE": {
        "N": 2048, "K": 1408, "E": 64, "TOPK": 6, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--autotune", default=False, action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    if args.autotune:
        import triton
        from triton_dist.autotuner import contextual_autotune
        from triton_dist.kernels.nvidia import allgather_group_gemm

        configs = [
            triton.Config({"BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK}, num_stages=s, num_warps=w)
            for BN in [128, 256]
            for BK in [32, 64]
            for s in [3, 4]
            for w in [4, 8]
        ]
        allgather_group_gemm.kernel_consumer_m_parallel_scatter_group_gemm = triton.autotune(
            configs=configs, key=["M", "N", "K"])(allgather_group_gemm.kernel_consumer_m_parallel_scatter_group_gemm)
        ag_group_gemm = contextual_autotune(is_dist=True)(ag_group_gemm)

    initialize_distributed()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    for name, config in layer_configs.items():
        perf_test(name, args.M, dtype, config)

    torch.distributed.destroy_process_group()
