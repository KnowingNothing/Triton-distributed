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
from triton_dist import pynvshmem
import torch
import torch.distributed

import argparse

import triton
import triton.language as tl

from triton_dist.kernels.nvidia import (ag_group_gemm_inter_node, create_ag_group_gemm_inter_node_context,
                                        sort_topk_ids_align_block_size)

from triton_dist.utils import (initialize_distributed, TP_GROUP, perf_func, dist_print, group_profile)

dtype = torch.float16


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


def perf_test_ag(input_len, config):
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

    full_A = torch.empty([M_per_rank * tp_group.size(), K], dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(full_A, A, group=tp_group)

    ctx = create_ag_group_gemm_inter_node_context(
        tp_group,
        A,
        B,
        full_topk_ids,
        E,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        GROUP_SIZE_M=config["GROUP_M"],
        stages=config["stage"],
        warps=config["warp"],
        ag_stream=torch.cuda.Stream(),
        group_gemm_stream=torch.cuda.Stream(),
    )
    local_rank = ctx.rank % ctx.local_world_size

    def triton_ag_func():
        ctx.local_copy_and_barrier_all(A, is_internode=True)
        inter_node_allgather(A, ctx.workspace_tensors, ctx.barrier_tensors, 1, ctx.rank, ctx.local_world_size,
                             ctx.num_ranks, torch.cuda.current_stream(), ctx.internode_ag_stream)
        pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)

    triton_ag_func()
    assert torch.allclose(full_A, ctx.workspace_tensors[local_rank], atol=1e-3, rtol=1e-3)

    ag_size = M * K * A.element_size() / 1024 / 1024 / 1024  #GB

    with group_profile(f"trace_ag_scatter_inter_node/m_{M}_n_{N}_k_{K}_e_{E}", do_prof=args.profile, group=tp_group):
        _, triton_perf = perf_func(triton_ag_func, iters=10, warmup_iters=20)

        dist_print(
            f"RANK {tp_group.rank()} allgather perf: dist-triton={triton_perf} ms, band={ag_size/triton_perf*1000*(tp_group.size()-1)/tp_group.size()} GBps",
            need_sync=True,
            allowed_ranks=list(range(tp_group.size())),
        )


def perf_test_group_gemm(input_len, config):
    M = input_len
    N = config["N"]
    K = config["K"]
    E = config["E"]
    topk = config["TOPK"]
    tp_group = TP_GROUP()

    assert M % tp_group.size() == 0
    assert N % tp_group.size() == 0
    M_per_rank = M // tp_group.size()
    K_per_rank = K // tp_group.size()

    if tp_group.rank() == 0:
        print(f"shape: M={M}, N={N}, K={K}; num experts={E}, topk={topk}")

    A = torch.randn([M_per_rank, K_per_rank], dtype=dtype, device="cuda")
    B = torch.randn([E, K_per_rank, N], dtype=dtype, device="cuda")
    score = ((-2 * torch.randn((M_per_rank, E), device="cuda", dtype=dtype) + 1) / 100 * (tp_group.rank() + 1))
    score = torch.softmax(score, dim=-1)
    _, local_topk_ids = torch.topk(score, topk)

    full_topk_ids = torch.zeros(M_per_rank * tp_group.size(), topk, dtype=local_topk_ids.dtype).cuda()
    torch.distributed.all_gather_into_tensor(full_topk_ids, local_topk_ids, group=tp_group)

    full_A = torch.empty([M_per_rank * tp_group.size(), K], dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(full_A, A, group=tp_group)

    ctx = create_ag_group_gemm_inter_node_context(
        tp_group,
        A,
        B,
        full_topk_ids,
        E,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        GROUP_SIZE_M=config["GROUP_M"],
        stages=config["stage"],
        warps=config["warp"],
        ag_stream=torch.cuda.Stream(),
        group_gemm_stream=torch.cuda.Stream(),
    )
    local_rank = ctx.rank % ctx.local_world_size

    ctx.barrier_tensors[local_rank].fill_(1)

    C = torch.empty(
        [ctx.topk * M_per_rank * ctx.num_ranks, N],
        dtype=A.dtype,
        device=A.device,
    )

    def _group_gemm():
        EM = ctx.sorted_topk_ids.shape[0]

        grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        kernel_consumer_m_parallel_scatter_group_gemm[grid](
            full_A,
            B,
            C,
            ctx.barrier_tensors[local_rank],
            ctx.sorted_topk_ids,
            ctx.aligned_expert_ids,
            ctx.aligned_barrier_ids,
            ctx.aligned_num_tokens,
            ctx.full_num_tokens,
            EM,
            N,
            K,
            full_A.stride(0),
            full_A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            ctx.BLOCK_M,
            ctx.BLOCK_N,
            ctx.BLOCK_K,
            ctx.GROUP_SIZE_M,
            ctx.topk,
            tl.float16,
            ctx.rank,
            ctx.num_ranks,
            num_stages=ctx.stages,
            num_warps=ctx.warps,
        )
        torch.cuda.synchronize()
        pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)

    with group_profile(f"trace_group_gemm_inter_node/m_{M}_n_{N}_k_{K}_e_{E}", do_prof=args.profile, group=tp_group):
        _, triton_perf = perf_func(_group_gemm, iters=10, warmup_iters=20)

        dist_print(f"RANK {tp_group.rank()} group gemm perf: dist-triton={triton_perf} ms", need_sync=True,
                   allowed_ranks=list(range(tp_group.size())))


def perf_test(input_len, config):
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

    ctx = create_ag_group_gemm_inter_node_context(
        tp_group,
        A,
        B,
        full_topk_ids,
        E,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        GROUP_SIZE_M=config["GROUP_M"],
        stages=config["stage"],
        warps=config["warp"],
        ag_stream=torch.cuda.Stream(),
        group_gemm_stream=torch.cuda.Stream(),
    )

    def sort_func():
        return sort_topk_ids_align_block_size(full_topk_ids, E, tp_group.size(), M_per_rank, config["BM"])[0]

    def triton_func():
        return ag_group_gemm_inter_node(A, B, ctx)

    def torch_func():
        return torch_ag_group_gemm(tp_group, A, B, full_topk_ids)

    result = triton_func()
    _, C_golden = torch_func()
    assert torch.allclose(C_golden, result, atol=1e-3, rtol=1e-3)

    with group_profile(f"trace_ag_scatter_group_gemm_inter_node/m_{M}_n_{N}_k_{K}_e_{E}", do_prof=args.profile,
                       group=tp_group):
        perf_func(sort_func, iters=10, warmup_iters=10)
        perf_func(triton_func, iters=10, warmup_iters=10)
        perf_func(torch_func, iters=10, warmup_iters=10)

    _, permute_perf = perf_func(sort_func, iters=100, warmup_iters=500)
    _, triton_perf = perf_func(triton_func, iters=100, warmup_iters=500)
    _, torch_perf = perf_func(torch_func, iters=100, warmup_iters=500)

    dist_print(
        f"RANK {tp_group.rank()} perf: compute permute {permute_perf} ms, dist-triton={triton_perf} ms, torch={torch_perf} ms; speedup={torch_perf/triton_perf}",
        need_sync=True,
        allowed_ranks=list(range(tp_group.size())),
    )


layer_configs = {
    "Dummy Model":
    {"N": 8192, "K": 8192, "E": 32, "TOPK": 3, "BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8, "stage": 4, "warp": 8},
    "Qwen1.5-MoE-A2.7B":
    {"N": 1408, "K": 2048, "E": 60, "TOPK": 4, "BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8, "stage": 4, "warp": 8},
    "Mixtral-8x7B":
    {"N": 4096, "K": 14336, "E": 8, "TOPK": 2, "BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8, "stage": 4, "warp": 8},
    "Mixtral-8x22B":
    {"N": 6144, "K": 16384, "E": 8, "TOPK": 2, "BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8, "stage": 4, "warp": 8},
    "DeepSeek-MoE":
    {"N": 2048, "K": 1408, "E": 64, "TOPK": 6, "BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8, "stage": 4, "warp": 8},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--ag_only", action="store_true", default=False)
    parser.add_argument("--gemm_only", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()

    initialize_distributed()

    if args.gemm_only:
        from third_party.distributed.distributed.kernels.nvidia.allgather_group_gemm import kernel_consumer_m_parallel_scatter_group_gemm
        for _, config in layer_configs.items():
            perf_test_group_gemm(args.M, config)
    elif args.ag_only:
        from third_party.distributed.distributed.kernels.nvidia.allgather_group_gemm import inter_node_allgather
        for _, config in layer_configs.items():
            perf_test_ag(args.M, config)
    else:
        for _, config in layer_configs.items():
            perf_test(args.M, config)

    torch.distributed.destroy_process_group()
