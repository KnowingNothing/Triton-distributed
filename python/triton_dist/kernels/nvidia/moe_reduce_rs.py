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
import torch
import triton
import triton.language as tl
import triton_dist.language as dl

from triton_dist import pynvshmem

from typing import Optional, List

from dataclasses import dataclass

from triton_dist.kernels.nvidia.common_ops import wait_eq, set_signal, barrier_all_on_stream, BarrierAllContext
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import (atomic_add, __syncthreads, tid, ntid)


################### helper functions ###################
def ceil_div(a, b):
    return (a + b - 1) // b


def is_power_of_two(value):
    return ((value - 1) & value) == 0


def torch_dtype_to_triton_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.int32:
        return tl.int32
    elif dtype == torch.int8:
        return tl.int8
    else:
        raise RuntimeError(f"unsupported dtype: {dtype}")


@triton.jit
def get_flat_tid():
    tid_x, tid_y, tid_z = tid(0), tid(1), tid(2)
    ntid_x, ntid_y = ntid(0), ntid(1)
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


################### compute ctx ###################
@dataclass
class MoEAgScatterGroupGemmPrecomputeContext:
    full_topk_weight = None
    full_topk_ids = None
    full_sorted_token_ids = None
    full_token_expert_ids = None
    block_wait_barriers = None
    rank_block_num = None
    full_num_tokens_post_padded_list = None
    EM: int = 0
    full_numel: int = 0
    TOP_K: int = 0
    BLOCK_M: int = 0
    num_tokens_per_rank: int = 0


def full_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    num_ranks: int,
    num_tokens_per_rank: int,
):
    sorted_ids = torch.empty(
        ((num_tokens_per_rank * topk_ids.shape[1] + num_experts * (block_size - 1)) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        ((num_tokens_per_rank * topk_ids.shape[1] + num_experts) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    block_barrier_ids = torch.empty(
        ((num_tokens_per_rank * topk_ids.shape[1] + num_experts) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    rank_block_num = torch.empty(
        num_ranks,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    # The part below adapted from the cuda kernel in Saber

    num_iterations = num_ranks
    num_tokens_per_iteration = num_tokens_per_rank * topk_ids.shape[1]
    numel = num_tokens_per_iteration
    tokens_per_thread = ceil_div(numel, num_experts)

    topk_ids_flatten = topk_ids.flatten()

    last_pad_tokens = 0
    num_tokens_post_pad[0] = 0
    sorted_ids_idx = 0
    expert_ids_idx = 0
    block_barrier_ids_idx = 0
    topk_ids_idx = 0
    for iter in range(num_iterations):
        sorted_ids_idx += last_pad_tokens
        expert_ids_idx += last_pad_tokens // block_size
        block_barrier_ids_idx += last_pad_tokens // block_size

        token_cnts = torch.zeros((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
        cumsum = torch.zeros((num_experts + 1), dtype=torch.int32, device=topk_ids.device)

        for j in range(num_experts):
            start_idx = j * tokens_per_thread
            for i in range(start_idx, min(numel, start_idx + tokens_per_thread)):
                token_cnts[j + 1, topk_ids_flatten[topk_ids_idx + i]] += 1

        for j in range(num_experts):
            for i in range(1, num_experts + 1):
                token_cnts[i, j] += token_cnts[i - 1, j]

        for i in range(1, num_experts + 1):
            cumsum[i] = cumsum[i - 1] + ceil_div(token_cnts[num_experts, i - 1], block_size) * block_size
        num_tokens_post_pad[0] += cumsum[num_experts]
        rank_block_num[iter] = cumsum[num_experts] // block_size

        last_pad_tokens = cumsum[num_experts]

        for j in range(num_experts):
            for i in range(cumsum[j], cumsum[j + 1], block_size):
                expert_ids[expert_ids_idx + i // block_size] = j
                block_barrier_ids[block_barrier_ids_idx + i // block_size] = iter

        for j in range(num_experts):
            start_idx = j * tokens_per_thread
            for i in range(start_idx, min(numel, start_idx + tokens_per_thread)):
                expert_id = topk_ids_flatten[topk_ids_idx + i]
                rank_post_pad = token_cnts[j, expert_id] + cumsum[expert_id]
                sorted_ids[sorted_ids_idx + rank_post_pad] = i + iter * num_tokens_per_iteration
                token_cnts[j, expert_id] += 1

        topk_ids_idx += num_tokens_per_iteration

    return (
        sorted_ids,
        expert_ids,
        block_barrier_ids,
        rank_block_num,
        num_tokens_post_pad,
    )


def select_experts(pg, num_ranks, topk, input_dtype, device, router_logits):
    num_tokens_per_rank = router_logits.shape[0]
    num_tokens = num_tokens_per_rank * num_ranks
    full_topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    full_topk_weight = torch.zeros(num_tokens, topk, dtype=input_dtype, device=device)
    score = torch.softmax(router_logits, dim=-1)
    local_topk_weight, local_topk_ids = torch.topk(score, topk)
    torch.distributed.all_gather_into_tensor(
        full_topk_weight,
        local_topk_weight,
        group=pg,
    )
    torch.distributed.all_gather_into_tensor(
        full_topk_ids,
        local_topk_ids.to(torch.int32),
        group=pg,
    )
    return full_topk_ids, full_topk_weight


def precompute_context_helper(
    pg,
    num_ranks: int,
    topk: int,
    num_tokens_per_rank: int,
    num_experts: int,
    input_dtype,
    device,
    router_logits,
    BLOCK_M: int = 128,
):
    ctx = MoEAgScatterGroupGemmPrecomputeContext()

    (ctx.full_topk_ids, ctx.full_topk_weight) = select_experts(pg, num_ranks, topk, input_dtype, device, router_logits)

    E = num_experts
    ctx.TOP_K = topk
    ctx.BLOCK_M = BLOCK_M

    (
        full_sorted_token_ids,
        full_token_expert_ids,
        block_wait_barriers,
        rank_block_num,
        full_num_tokens_post_padded_list,
    ) = full_moe_align_block_size(ctx.full_topk_ids, BLOCK_M, E, num_ranks, num_tokens_per_rank)
    EM = full_num_tokens_post_padded_list.cpu().tolist()[0]  # full_sorted_token_ids.shape[0]
    full_numel = ctx.full_topk_ids.numel()

    ctx.full_sorted_token_ids = full_sorted_token_ids
    ctx.full_token_expert_ids = full_token_expert_ids
    ctx.full_num_tokens_post_padded_list = full_num_tokens_post_padded_list
    ctx.block_wait_barriers = block_wait_barriers
    ctx.rank_block_num = rank_block_num
    ctx.EM = EM
    ctx.full_numel = full_numel
    ctx.num_tokens_per_rank = num_tokens_per_rank
    return ctx


@dataclass
class DataflowConfig:
    GEMM_BLOCK_M: int
    GEMM_BLOCK_N: int
    GEMM_BLOCK_K: int
    GROUP_SIZE: int
    num_stages: int
    num_warps: int
    RS_BLOCK_M: int
    RS_BLOCK_N: int


@dataclass
class MoEReduceRSContext:
    precompute_ctx: MoEAgScatterGroupGemmPrecomputeContext

    rs_buffers: List[torch.Tensor]
    rs_buffer_ptrs: torch.Tensor
    rs_per_node_buffer: torch.Tensor
    p2p_buffer: torch.Tensor
    final_output_buffer: torch.Tensor
    barrier: BarrierAllContext

    rs_stream: torch.cuda.Stream
    reduction_stream: torch.cuda.Stream
    p2p_stream: torch.cuda.Stream

    dataflow_config: DataflowConfig

    barriers_gemm_scatter_counter: List[torch.Tensor]
    barriers_gemm_scatter_counter_ptrs: torch.Tensor
    barriers_gemm_scatter_ready: List[torch.Tensor]
    barriers_gemm_scatter_ready_ptrs: torch.Tensor
    barrier_gemm_scatter_counter: torch.Tensor
    barrier_gemm_scatter_ready: torch.Tensor
    rs_per_node_signal_buffer: torch.Tensor


def create_moe_rs_context(pg, local_rank, world_size, local_world_size, max_token_num, hidden_dim, num_experts, topk,
                          input_dtype, output_dtype, device, moe_block_size, router_logits):
    num_tokens_per_rank = router_logits.shape[0]
    precompute_ctx = precompute_context_helper(
        pg,
        world_size,
        topk,
        num_tokens_per_rank,
        num_experts,
        input_dtype,
        device,
        router_logits,
        BLOCK_M=moe_block_size,
    )

    rs_buffers: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node([max_token_num, hidden_dim],
                                                                                     input_dtype)
    rs_buffer_ptrs: torch.Tensor = torch.tensor([t.data_ptr() for t in rs_buffers], device=device)

    rs_per_node_buffer = pynvshmem.nvshmem_create_tensor(
        [max_token_num // local_world_size, hidden_dim],
        input_dtype,
    )
    p2p_buffer = pynvshmem.nvshmem_create_tensor(
        [max_token_num // local_world_size, hidden_dim],
        input_dtype,
    )
    final_output_buffer = torch.zeros(
        (max_token_num * topk, hidden_dim),
        dtype=output_dtype,
        device=device,
    )

    barrier = BarrierAllContext(True)

    # stream
    rs_stream = torch.cuda.Stream()
    reduction_stream = torch.cuda.Stream()
    p2p_stream = torch.cuda.Stream()

    # Setup metadata for kernel launch
    RS_BLOCK_M = max_token_num // world_size
    RS_BLOCK_N = hidden_dim
    GEMM_BLOCK_M = moe_block_size
    GEMM_BLOCK_N = 128
    GEMM_BLOCK_K = 32
    dataflow_config = DataflowConfig(GEMM_BLOCK_M, GEMM_BLOCK_N, GEMM_BLOCK_K, 8, 4, 4, RS_BLOCK_M, RS_BLOCK_N)

    # initialize barriers
    with torch.device(torch.cuda.current_device()):

        # gemm_scatter

        barriers_gemm_scatter_counter: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node(
            [world_size, 1], torch.int32)

        barriers_gemm_scatter_counter_ptrs = torch.tensor([ptr.data_ptr()
                                                           for ptr in barriers_gemm_scatter_counter]).cuda()

        barriers_gemm_scatter_ready: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node(
            [world_size, 1], torch.uint64)

        barriers_gemm_scatter_ready_ptrs = torch.tensor([ptr.data_ptr() for ptr in barriers_gemm_scatter_ready]).cuda()

        barrier_gemm_scatter_counter = barriers_gemm_scatter_counter[local_rank]
        barrier_gemm_scatter_ready = barriers_gemm_scatter_ready[local_rank]

        barrier_gemm_scatter_counter.zero_()
        barrier_gemm_scatter_ready.zero_()

        # intra_node - p2p

        rs_per_node_signal_buffer = pynvshmem.nvshmem_create_tensor([world_size], torch.uint64)
        rs_per_node_signal_buffer.zero_()

    return MoEReduceRSContext(precompute_ctx, rs_buffers, rs_buffer_ptrs, rs_per_node_buffer, p2p_buffer,
                              final_output_buffer, barrier, rs_stream, reduction_stream, p2p_stream, dataflow_config,
                              barriers_gemm_scatter_counter, barriers_gemm_scatter_counter_ptrs,
                              barrier_gemm_scatter_ready, barriers_gemm_scatter_ready_ptrs,
                              barrier_gemm_scatter_counter, barrier_gemm_scatter_ready, rs_per_node_signal_buffer)


################### triton kernel ###################
@triton.jit
def kernel_producer_group_gemm_tp_scatter_input(
    local_world_size,
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    token_expert_ids_ptr,
    topk_weight_ptr,
    num_tokens_post_padded,
    block_wait_barrier_ptr,
    rank_block_num,
    barrier_counter,
    barriers_ready_ptrs,
    num_valid_tokens: int,
    EM,
    N,
    K_per_rank,
    E,
    stride_in_m,
    stride_in_k,
    stride_weight_e,
    stride_weight_k,
    stride_weight_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TOP_K: tl.constexpr,
    compute_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(EM, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)

    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    rank = dl.rank()
    num_ranks = dl.num_ranks()
    local_rank = rank % local_world_size
    num_block_m_per_rank = num_block_m // num_ranks
    m_offset = num_block_m_per_rank * ((rank + 5) % num_ranks)
    pid_m = (pid_m + m_offset) % num_block_m

    num_tokens_post_padded_value = tl.load(num_tokens_post_padded)

    if pid_m * BLOCK_M >= num_tokens_post_padded_value:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (a_ptr + offs_token[:, None] * stride_in_m + offs_k[None, :] * stride_in_k)

    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_be = tl.load(token_expert_ids_ptr + pid_m)

    b_ptrs = (b_ptr + offs_be * stride_weight_e + offs_k[:, None] * stride_weight_k +
              offs_bn[None, :] * stride_weight_n)

    moe_weight = tl.load(topk_weight_ptr + offs_token, mask=token_mask, other=0)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_per_rank, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K_per_rank - k * BLOCK_K),
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_bn[None, :] < N and offs_k[:, None] < K_per_rank - k * BLOCK_K),
        )

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_in_k
        b_ptrs += BLOCK_K * stride_weight_k

    accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_dtype)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_out_m + offs_cn[None, :] * stride_out_n)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

    offs_counter = tl.load(block_wait_barrier_ptr + pid_m)
    offs_ready = tl.load(block_wait_barrier_ptr + pid_m)
    threshold = tl.load(rank_block_num + offs_counter) * num_block_n
    counter_ptr = barrier_counter + offs_counter
    remote_barrier_ready_ptr = tl.load(barriers_ready_ptrs + local_rank).to(tl.pointer_type(tl.uint64))
    __syncthreads()
    thread_id = tid(0)
    value = 1
    if thread_id == 0:
        if atomic_add(counter_ptr, value, "gpu", "relaxed") == threshold - 1:
            dl.notify(remote_barrier_ready_ptr + offs_ready, rank, signal=1, sig_op="add", comm_scope="gpu")


@triton.jit
def kernel_consumer_topk_reduce_scatter_intra_node(
    local_world_size,
    consumer_output_ptr,  # shape: [M * ReduceLength, N]
    remote_buffer_ptrs,  # each tensor shape: [M, N]
    barrier_gemm_scatter_ready,
    M,
    N,
    num_pid_m,
    num_pid_n,
    # constants
    REDUCE_LENGTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    perfect_tile: tl.constexpr,
    use_tl_reduce: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    local_rank = rank % local_world_size
    nnodes = world_size // local_world_size
    dtype = consumer_output_ptr.dtype.element_ty
    M_per_rank = M // world_size
    num_blocks_m_per_rank = tl.cdiv(M_per_rank, BLOCK_M)
    num_blocks_m_per_node = num_blocks_m_per_rank * local_world_size

    tl.static_assert(perfect_tile, "Consider perfect tiling now.")

    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    num_block_per_pid_m = tl.cdiv(num_block_m, num_pid_m)
    num_block_per_pid_n = tl.cdiv(num_block_n, num_pid_n)

    for m in range(num_block_per_pid_m):
        for n in range(num_block_per_pid_n):
            mid = m * num_pid_m + pid_m
            nid = n * num_pid_n + pid_n

            mid = (mid + local_rank * nnodes * num_blocks_m_per_rank) % num_block_m

            to_rank = mid // num_blocks_m_per_rank
            to_rank_local = to_rank % local_world_size
            to_node = to_rank // local_world_size

            if use_tl_reduce:
                offs_m_reduce = tl.arange(0, BLOCK_M * REDUCE_LENGTH)
                offs_in_m = mid * BLOCK_M * REDUCE_LENGTH + offs_m_reduce
                offs_in_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                src_ptrs = (consumer_output_ptr + offs_in_m[:, None] * N + offs_in_n[None, :])
                token = dl.wait(barrier_gemm_scatter_ready + to_rank, 1, "gpu", "acquire")
                src_ptrs = dl.consume_token(src_ptrs, token)
                data = tl.load(src_ptrs)
                to_reduce_data = tl.reshape(data, [BLOCK_M, REDUCE_LENGTH, BLOCK_N])
                reduce_data = tl.sum(to_reduce_data, axis=1)
            else:
                reduce_data = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)
                token = dl.wait(barrier_gemm_scatter_ready + to_rank, 1, "gpu", "acquire")
                for i in range(REDUCE_LENGTH):
                    offs_m_reduce = tl.arange(0, BLOCK_M) + i
                    offs_in_m = (mid * BLOCK_M * REDUCE_LENGTH + offs_m_reduce * REDUCE_LENGTH)
                    offs_in_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                    src_ptrs = (consumer_output_ptr + offs_in_m[:, None] * N + offs_in_n[None, :])
                    src_ptrs = dl.consume_token(src_ptrs, token)
                    data = tl.load(src_ptrs)
                    reduce_data += data

            # scatter
            dst_ptr = tl.load(remote_buffer_ptrs + to_rank_local).to(tl.pointer_type(dtype))
            offs_out_m = (to_node * num_blocks_m_per_node + local_rank * num_blocks_m_per_rank +
                          mid % num_blocks_m_per_rank) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_out_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
            dst_ptrs = dst_ptr + offs_out_m[:, None] * N + offs_out_n[None, :]
            tl.store(dst_ptrs, reduce_data)


@triton.jit
def kernel_consumer_reduce(
    local_world_size,
    c_ptr,  # [M_per_node, N]
    out_ptr,  # [M_per_rank, N]
    # shape of matrix
    M,
    N,
    # strides
    stride_m,
    stride_n,
    # reduce tile shape
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    local_rank = rank % local_world_size
    m_per_rank = tl.cdiv(M, world_size)
    pid = tl.program_id(axis=0)
    reduce_n_blocks_per_rank = tl.cdiv(N, BLOCK_N)
    pid_m = pid // reduce_n_blocks_per_rank
    pid_n = pid % reduce_n_blocks_per_rank

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    out_ptrs = out_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)

    org_data = tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty)

    for rid in range(0, local_world_size):
        swizzle_rid = (rid + local_rank) % local_world_size
        full_offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M) + swizzle_rid * m_per_rank) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        ptrs = c_ptr + (full_offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
        data = tl.load(ptrs)
        org_data += data

    tl.store(out_ptrs, org_data)


@triton.jit
def kernel_inter_node_p2p_for_same_local_rank(
    local_world_size,
    M_per_rank,
    N,
    input,  # [M_per_rank * nnodes, N]
    output,  # [M_per_rank * nnodes, N]
    rs_per_node_signal,
):
    pid = tl.program_id(axis=0)
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    num_pid = tl.num_programs(axis=0)
    nelem_per_rank = M_per_rank * N
    elem_size = tl.constexpr(input.dtype.element_ty.primitive_bitwidth) // 8

    for i in range(pid, nnodes - 1, num_pid):
        remote_node_id = (i + 1 + node_id) % nnodes
        remote_rank = local_rank + remote_node_id * local_world_size
        libshmem_device.signal_wait_until(
            rs_per_node_signal + remote_node_id,
            libshmem_device.NVSHMEM_CMP_EQ,
            1,
        )
        libshmem_device.putmem_block(
            output + node_id * nelem_per_rank,
            input + remote_node_id * nelem_per_rank,
            nelem_per_rank * elem_size,
            remote_rank,
        )


@triton.jit
def kernel_ring_reduce(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_split, N]
    # shape of matrix
    M_per_rank,
    N,
    begin_idx,
    num_splits: tl.constexpr,
    # reduce tile shape
    BLOCK_SIZE_M: tl.constexpr = 256,
    BLOCK_SIZE_N: tl.constexpr = 64,
):
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M_per_rank * num_splits, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    output_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    for tile_id in range(pid, total_tiles, num_pid):
        tile_id_m = tile_id // num_tiles_n
        tile_id_n = tile_id % num_tiles_n
        # accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=out_ptr.dtype.element_ty)
        cur_rank = (begin_idx + 1) % num_splits
        accum = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
        for i in range(1, num_splits):
            cur_rank = (i + begin_idx + 1) % num_splits
            data = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
            accum += data

        output_desc.store([tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N], accum)


################### kernel calls ###################
def topk_reduce_scatter_reduce_for_each_node(
    rank,
    world_size,
    local_world_size,
    M,
    N,
    TOP_K,
    local_tensor: torch.Tensor,  # Output of GroupGEMM
    rs_buffers: List[torch.Tensor],  # [M, N] for each rank
    rs_buffer_ptrs: torch.Tensor,
    rs_per_node_buffer: torch.Tensor,  # [M // local_world_size, N]
    barrier_gemm_scatter_ready: torch.Tensor,
    rs_per_node_signal_buf: torch.Tensor,
    barrier: BarrierAllContext,
    rs_stream: torch.cuda.Stream,
    reduction_stream: torch.cuda.Stream,
):
    local_rank = rank % local_world_size
    nnodes = world_size // local_world_size
    node_id = rank // local_world_size
    M_per_node = M // nnodes
    M_per_rank = M // world_size

    with torch.cuda.stream(rs_stream):
        grid = lambda _: (128, 1, 1)

        kernel_consumer_topk_reduce_scatter_intra_node[grid](
            local_world_size,
            local_tensor,
            rs_buffer_ptrs,
            barrier_gemm_scatter_ready,
            M,
            N,
            16,  # num pid m
            8,  # num pid n
            TOP_K,  # REDUCE_LENGTH: tl.constexpr,
            128,  # BLOCK_M: tl.constexpr,
            128,  # BLOCK_N: tl.constexpr,
            True,  # perfect_tile: tl.constexpr,
            is_power_of_two(TOP_K),  # use_tl_reduce: tl.constexpr,
            num_warps=32,
        )

        barrier_all_on_stream(barrier, rs_stream)
        reduction_stream.wait_stream(rs_stream)

    with torch.cuda.stream(reduction_stream):

        for n in range(0, nnodes):
            cur_node_id = (node_id + n + 1) % nnodes
            rs_buffer_cur_node = rs_buffers[local_rank][cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]
            rs_per_node_buffer_cur_node = rs_per_node_buffer[cur_node_id * M_per_rank:(cur_node_id + 1) * M_per_rank]

            grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

            kernel_consumer_reduce[grid](
                local_world_size,
                rs_buffer_cur_node,  # c_ptr
                rs_per_node_buffer_cur_node,  # out_ptr
                M,
                N,
                N,  # stride_m
                1,  # stride_n
                128,  # BLOCK_M
                128,  # BLOCK_N
                num_warps=32,
            )

            set_signal(rs_per_node_signal_buf[cur_node_id].data_ptr(), 1, reduction_stream, require_i64=True)

    return rs_per_node_buffer[:M_per_rank * nnodes]


def p2p_inter_node(
    rank,
    world_size,
    local_world_size,
    input,
    output,
    rs_per_node_signal_buf,
    stream,
):
    nnodes = world_size // local_world_size
    node_id = rank // local_world_size
    if nnodes == 1:
        wait_eq(
            rs_per_node_signal_buf[node_id].data_ptr(),
            1,
            stream,
            require_i64=True,
        )
        return input
    M, N = input.shape
    M_per_rank = M // nnodes
    with torch.cuda.stream(stream):
        grid = lambda META: (nnodes - 1, )
        kernel_inter_node_p2p_for_same_local_rank[grid](
            local_world_size,
            M_per_rank,
            N,
            input,
            output,
            rs_per_node_signal_buf,
            num_warps=16,
        )
        wait_eq(
            rs_per_node_signal_buf[node_id].data_ptr(),
            1,
            stream,
            require_i64=True,
        )
        output[M_per_rank * node_id:M_per_rank * (node_id + 1)].copy_(input[M_per_rank * node_id:M_per_rank *
                                                                            (node_id + 1)])
    return output[:M_per_rank * nnodes]


def ring_reduce(
    input,  # [M_per_rank * nnodes, N]
    output,  # [M_per_rank, N]
    begin_idx,
    num_splits,  # nnodes
    stream,
):
    total_M, N = input.shape
    M_per_split = total_M // num_splits
    assert output.shape[0] == M_per_split and total_M % num_splits == 0

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    with torch.cuda.stream(stream):
        kernel_ring_reduce[grid](
            input,
            output,
            M_per_split,
            N,
            begin_idx,
            num_splits,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=64,
            num_warps=4,
        )

    return output


def consumer_reduce_scatter_reduce_2d(
    rank,
    world_size,
    local_world_size,
    M,
    N,
    TOP_K,
    local_tensor: torch.Tensor,
    rs_buffers: List[torch.Tensor],
    rs_buffer_ptrs: torch.Tensor,
    rs_per_node_buffer: torch.Tensor,
    p2p_buffer: torch.Tensor,
    barrier_gemm_scatter_ready: torch.Tensor,
    rs_per_node_signal_buffer: torch.Tensor,
    barrier: BarrierAllContext,
    rs_stream: torch.cuda.Stream,
    reduction_stream: torch.cuda.Stream,
    p2p_stream: torch.cuda.Stream,
):
    M_per_rank = M // world_size
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size

    reduction_stream.wait_stream(rs_stream)
    barrier_all_on_stream(None, rs_stream)
    p2p_stream.wait_stream(rs_stream)
    rs_result_intra_node = topk_reduce_scatter_reduce_for_each_node(
        rank,
        world_size,
        local_world_size,
        M,
        N,
        TOP_K,
        local_tensor,
        rs_buffers,
        rs_buffer_ptrs,
        rs_per_node_buffer,
        barrier_gemm_scatter_ready,
        rs_per_node_signal_buffer,
        barrier,
        rs_stream,
        reduction_stream,
    )
    p2p_result = p2p_inter_node(
        rank,
        world_size,
        local_world_size,
        rs_result_intra_node,
        p2p_buffer,
        rs_per_node_signal_buffer,
        p2p_stream,
    )
    rs_stream.wait_stream(p2p_stream)
    barrier_all_on_stream(None, rs_stream)
    output = torch.empty((M_per_rank, N), dtype=local_tensor.dtype, device=local_tensor.device)
    ring_reduce(
        p2p_result,
        output,
        node_id,
        nnodes,
        rs_stream,
    )
    return output


def moe_reduce_rs(
    rank,
    world_size,
    local_world_size,
    # input
    a: torch.Tensor,
    b: torch.Tensor,
    # context
    ctx: MoEReduceRSContext,
    # option
    dump_ir=False,
    debug_sync=False,
    do_initial_sync=True,
    do_final_sync=True,
):
    padded_EM, K_per_rank = a.shape
    E = b.shape[0]
    M = ctx.precompute_ctx.full_topk_ids.shape[0]
    TOP_K = ctx.precompute_ctx.full_topk_ids.shape[1]
    dtype = a.dtype
    assert (dtype == torch.float16 or dtype == torch.float8_e4m3fn), "Currently only used for float16 or float8_e4m3fn"
    assert a.dtype == b.dtype

    GEMM_BLOCK_M = ctx.dataflow_config.GEMM_BLOCK_M
    GEMM_BLOCK_N = ctx.dataflow_config.GEMM_BLOCK_N
    GEMM_BLOCK_K = ctx.dataflow_config.GEMM_BLOCK_K
    GROUP_SIZE_M = ctx.dataflow_config.GROUP_SIZE
    num_stages = ctx.dataflow_config.num_stages
    num_warps = ctx.dataflow_config.num_warps

    compiled = None

    if do_initial_sync:
        ctx.barrier_gemm_scatter_counter.zero_()
        ctx.barrier_gemm_scatter_ready.zero_()
        ctx.rs_stream.wait_stream(torch.cuda.current_stream())
        ctx.reduction_stream.wait_stream(torch.cuda.current_stream())
        ctx.p2p_stream.wait_stream(torch.cuda.current_stream())
        barrier_all_on_stream(None, torch.cuda.current_stream())

    with torch.cuda.stream(torch.cuda.current_stream()):
        full_sorted_token_ids = ctx.precompute_ctx.full_sorted_token_ids
        full_token_expert_ids = ctx.precompute_ctx.full_token_expert_ids
        block_wait_barriers = ctx.precompute_ctx.block_wait_barriers
        rank_block_num = ctx.precompute_ctx.rank_block_num
        full_num_tokens_post_padded_list = (ctx.precompute_ctx.full_num_tokens_post_padded_list)
        EM = ctx.precompute_ctx.EM
        full_numel = ctx.precompute_ctx.full_numel

        (
            _,
            _,
            N,
        ) = b.shape

        grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

        compiled = kernel_producer_group_gemm_tp_scatter_input[grid](
            local_world_size,
            a,
            b,
            ctx.final_output_buffer,
            full_sorted_token_ids,
            full_token_expert_ids,
            ctx.precompute_ctx.full_topk_weight,
            full_num_tokens_post_padded_list,
            block_wait_barriers,
            rank_block_num,
            ctx.barrier_gemm_scatter_counter,
            ctx.barriers_gemm_scatter_ready_ptrs,
            full_numel,
            EM,
            N,
            K_per_rank,
            E,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            N,
            1,
            GEMM_BLOCK_M,
            GEMM_BLOCK_N,
            GEMM_BLOCK_K,
            GROUP_SIZE_M,
            TOP_K,
            torch_dtype_to_triton_dtype(a.dtype),
            num_stages=num_stages,
            num_warps=num_warps,
        )

    # debug sync
    if debug_sync:
        ctx.rs_stream.wait_stream(torch.cuda.current_stream())
        ctx.reduction_stream.wait_stream(ctx.rs_stream)
        ctx.p2p_stream.wait_stream(ctx.reduction_stream)
        torch.cuda.current_stream().wait_stream(ctx.rs_stream)
        torch.cuda.current_stream().wait_stream(ctx.reduction_stream)
        torch.cuda.current_stream().wait_stream(ctx.p2p_stream)
        barrier_all_on_stream(None, torch.cuda.current_stream())

    output = None

    with torch.cuda.stream(ctx.rs_stream):
        output = consumer_reduce_scatter_reduce_2d(
            rank,
            world_size,
            local_world_size,
            M,
            N,
            TOP_K,
            ctx.final_output_buffer,
            ctx.rs_buffers,
            ctx.rs_buffer_ptrs,
            ctx.rs_per_node_buffer,
            ctx.p2p_buffer,
            ctx.barrier_gemm_scatter_ready,
            ctx.rs_per_node_signal_buffer,
            ctx.barrier,
            ctx.rs_stream,
            ctx.reduction_stream,
            ctx.p2p_stream,
        )

    if dump_ir:
        if rank == 0:
            if compiled is not None:
                for suffix in ["ptx", "ttir", "ttgir", "llir"]:
                    with open(f"trace_{compiled.name}.{suffix}", "w") as fout:
                        print(compiled.asm[suffix], file=fout)

    if do_final_sync:
        torch.cuda.current_stream().wait_stream(ctx.rs_stream)
        torch.cuda.current_stream().wait_stream(ctx.reduction_stream)
        torch.cuda.current_stream().wait_stream(ctx.p2p_stream)
        barrier_all_on_stream(None, torch.cuda.current_stream())

    return output
