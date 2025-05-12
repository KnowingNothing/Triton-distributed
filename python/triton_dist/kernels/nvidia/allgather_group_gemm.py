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
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from triton.language.extra import libshmem_device
import triton_dist.language as dl
from triton_dist.utils import TP_GROUP
from typing import Optional, List
from cuda import cudart
from triton._C.libtriton_distributed.distributed import moe_ag_scatter_align_block_size
from triton_dist.kernels.nvidia.common_ops import wait_eq, set_signal
from triton_dist.kernels.nvidia.allgather import cp_engine_producer_all_gather_full_mesh_push

from triton_dist import pynvshmem


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


def sort_topk_ids_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    num_tokens_per_rank: int,
    block_size: int,
    moe_stream: torch.cuda.Stream = None,
):
    """
    Sort and align the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - num_experts: The total number of experts.
    - num_ranks: The total number of ranks.
    - num_tokens_per_rank: The total number of tokens (not topked) per rank.
    - block_size: The block size used in block matrix multiplication.
    -  moe_stream (torch.cuda.streams.Stream, optional): The stream used for cuda kernel, if not provided, use current stream. Defaults to None.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - block_barrier_ids: A tensor indicating the assigned barrier index for each block.
    - rank_block_num: A tensor indicating the number of blocks for each rank of tokens.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, where each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    num_topk = topk_ids.shape[1]
    sorted_ids = torch.empty(
        ((num_tokens_per_rank * num_topk + num_experts * (block_size - 1)) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        ((num_tokens_per_rank * num_topk + num_experts) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    block_barrier_ids = torch.empty(
        ((num_tokens_per_rank * num_topk + num_experts) * num_ranks, ),
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
    moe_stream = torch.cuda.current_stream() if moe_stream is None else moe_stream
    moe_ag_scatter_align_block_size(
        topk_ids,
        num_experts,
        num_ranks,
        num_tokens_per_rank * topk_ids.shape[1],
        block_size,
        sorted_ids,
        expert_ids,
        block_barrier_ids,
        rank_block_num,
        num_tokens_post_pad,
        moe_stream.cuda_stream,
    )
    torch.cuda.current_stream().wait_stream(moe_stream)
    return (
        sorted_ids,
        expert_ids,
        block_barrier_ids,
        rank_block_num,
        num_tokens_post_pad,
    )


@triton.jit
def nvshmem_device_producer_p2p_put_block_kernel(
    ag_buffer_ptr,
    signal_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size

    for i in range(pid, n_nodes - 1, num_pid):
        peer = local_rank + (node_rank + i + 1) % n_nodes * local_world_size
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,
            ag_buffer_ptr + rank * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


def inter_node_allgather(local_tensor: torch.Tensor, ag_buffer: list[torch.Tensor], signal_buffer: list[torch.Tensor],
                         signal_target, rank, local_world_size, world_size, intranode_ag_stream=None,
                         internode_ag_stream=None):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    node_rank = rank // local_world_size
    M_per_rank, N = local_tensor.shape

    with torch.cuda.stream(internode_ag_stream):
        grid = lambda META: (int(local_world_size + n_nodes - 2), )
        nvshmem_device_producer_p2p_put_block_kernel[grid](
            ag_buffer[local_rank],
            signal_buffer[local_rank],
            M_per_rank * N,
            local_tensor.element_size(),
            signal_target,
            rank,
            local_world_size,
            world_size,
            num_warps=32,
        )

    with torch.cuda.stream(intranode_ag_stream):
        for i in range(1, local_world_size):
            segment = rank * M_per_rank * N
            local_dst_rank = (local_rank + local_world_size - i) % local_world_size
            src_ptr = ag_buffer[local_rank].data_ptr() + segment * local_tensor.element_size()
            dst_ptr = ag_buffer[local_dst_rank].data_ptr() + segment * local_tensor.element_size()
            (err, ) = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                M_per_rank * N * local_tensor.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                intranode_ag_stream.cuda_stream,
            )
            set_signal(signal_buffer[local_dst_rank][rank].data_ptr(), signal_target, intranode_ag_stream, True)

        for i in range(1, n_nodes):
            recv_rank = local_rank + (node_rank + i) % n_nodes * local_world_size
            recv_segment = recv_rank * M_per_rank * N
            wait_eq(signal_buffer[local_rank][recv_rank].data_ptr(), signal_target, intranode_ag_stream, True)
            src_ptr = ag_buffer[local_rank].data_ptr() + recv_segment * local_tensor.element_size()
            for j in range(1, local_world_size):
                local_dst_rank = (local_rank + local_world_size - j) % local_world_size
                dst_ptr = ag_buffer[local_dst_rank].data_ptr() + recv_segment * local_tensor.element_size()
                (err, ) = cudart.cudaMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    M_per_rank * N * local_tensor.element_size(),
                    cudart.cudaMemcpyKind.cudaMemcpyDefault,
                    intranode_ag_stream.cuda_stream,
                )
                set_signal(signal_buffer[local_dst_rank][recv_rank].data_ptr(), signal_target, intranode_ag_stream,
                           True)

    intranode_ag_stream.wait_stream(internode_ag_stream)


@triton.jit
def kernel_consumer_m_parallel_scatter_group_gemm(
    in_features_ptr,
    expert_weights_ptr,
    out_features_ptr,
    block_barrier_ptr,
    sorted_token_ids_ptr,
    token_expert_ids_ptr,
    num_tokens_post_padded,
    block_barrier_id_ptr,
    num_valid_tokens,
    M,
    N,
    K,
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
    rank: tl.constexpr,
    world_size: tl.constexpr,
    swizzle_offset: tl.constexpr = 3,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)

    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    # swizzle along m-dimension
    num_rank_m = world_size
    num_rank_n = 1
    m_per_rank = num_block_m // num_rank_m
    rank_m = rank // num_rank_n
    m_offset = m_per_rank * ((rank_m + swizzle_offset) % num_rank_m)
    pid_m = (pid_m + m_offset) % num_block_m

    num_tokens_post_padded_value = tl.load(num_tokens_post_padded)

    if pid_m * BLOCK_M >= num_tokens_post_padded_value:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (in_features_ptr + offs_token[:, None] // TOP_K * stride_in_m + offs_k[None, :] * stride_in_k)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_be = tl.load(token_expert_ids_ptr + pid_m)

    b_ptrs = (expert_weights_ptr + offs_be * stride_weight_e + offs_k[:, None] * stride_weight_k +
              offs_bn[None, :] * stride_weight_n)

    offs_barrier = tl.load(block_barrier_id_ptr + pid_m)
    token = dl.wait(block_barrier_ptr + offs_barrier, 1, "gpu", "acquire")
    a_ptrs = dl.consume_token(a_ptrs, token)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_in_k
        b_ptrs += BLOCK_K * stride_weight_k

    accumulator = accumulator.to(compute_dtype)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (out_features_ptr + offs_token[:, None] * stride_out_m + offs_cn[None, :] * stride_out_n)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def rowise_intranode_ag_scatter_group_gemm(
    a,  # local tensor
    a_buffer,
    a_tensors: List[torch.Tensor],
    b,  # local weight
    c,  # output
    block_barriers,
    topk,
    full_sorted_token_ids,
    full_token_expert_ids,
    block_barrier_ids,
    full_num_tokens_post_padded_list,
    full_numel,
    rank,
    world_size,
    ag_stream=None,
    group_gemm_stream=None,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    GROUP_SIZE_M=8,
    stages=3,
    warps=4,
):
    M_per_rank, K_per_rank = a.shape
    _, K = (
        M_per_rank * world_size,
        K_per_rank,
    )
    _, _, N = b.shape
    dtype = a.dtype
    assert (dtype == torch.float16 or dtype == torch.float8_e4m3fn), "Currently only used for float16 or float8_e4m3fn"

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    group_gemm_stream = (torch.cuda.current_stream() if group_gemm_stream is None else group_gemm_stream)
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    group_gemm_stream.wait_stream(current_stream)

    cp_engine_producer_all_gather_full_mesh_push(
        rank,
        world_size,
        a,
        a_tensors,
        block_barriers,
        ag_stream,
    )

    with torch.cuda.stream(group_gemm_stream):
        EM = full_sorted_token_ids.shape[0]

        grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        compiled = kernel_consumer_m_parallel_scatter_group_gemm[grid](
            a_buffer,
            b,
            c,  #
            block_barriers[rank],
            full_sorted_token_ids,
            full_token_expert_ids,
            full_num_tokens_post_padded_list,
            block_barrier_ids,
            full_numel,
            EM,
            N,
            K,  #
            a_buffer.stride(0),
            a_buffer.stride(1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_SIZE_M,
            topk,
            torch_dtype_to_triton_dtype(dtype),
            rank,
            world_size,
            num_stages=stages,
            num_warps=warps,
        )

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(group_gemm_stream)
    block_barriers[rank].zero_()
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream.cuda_stream)
    return compiled


def rowise_internode_ag_scatter_group_gemm(
    a,  # local tensor
    a_tensors: List[torch.Tensor],
    b,  # local weight
    c,  # output
    block_barriers: List[torch.Tensor],
    topk,
    full_sorted_token_ids,
    full_token_expert_ids,
    block_barrier_ids,
    full_num_tokens_post_padded_list,
    full_numel,
    rank,
    world_size,
    local_world_size,
    internode_ag_stream,
    ag_stream=None,
    group_gemm_stream=None,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    GROUP_SIZE_M=8,
    stages=3,
    warps=4,
    signal_target=1,
):
    M_per_rank, K_per_rank = a.shape
    _, K = (
        M_per_rank * world_size,
        K_per_rank,
    )
    _, _, N = b.shape
    dtype = a.dtype
    local_rank = rank % local_world_size
    assert (dtype == torch.float16 or dtype == torch.float8_e4m3fn), "Currently only used for float16 or float8_e4m3fn"

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    group_gemm_stream = (torch.cuda.current_stream() if group_gemm_stream is None else group_gemm_stream)
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    group_gemm_stream.wait_stream(current_stream)

    inter_node_allgather(a, a_tensors, block_barriers, signal_target, rank, local_world_size, world_size, ag_stream,
                         internode_ag_stream)

    with torch.cuda.stream(group_gemm_stream):
        EM = full_sorted_token_ids.shape[0]

        grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        compiled = kernel_consumer_m_parallel_scatter_group_gemm[grid](
            a_tensors[local_rank],
            b,
            c,  #
            block_barriers[local_rank],
            full_sorted_token_ids,
            full_token_expert_ids,
            full_num_tokens_post_padded_list,
            block_barrier_ids,
            full_numel,
            EM,
            N,
            K,  #
            a_tensors[local_rank].stride(0),
            a_tensors[local_rank].stride(1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_SIZE_M,
            topk,
            torch_dtype_to_triton_dtype(dtype),
            rank,
            world_size,
            num_stages=stages,
            num_warps=warps,
        )

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(group_gemm_stream)

    return compiled


@dataclass
class MoE_AllGatherGroupGEMMTensorParallelContext:
    rank: int
    num_ranks: int
    num_experts: int
    topk: int
    full_num_tokens: int
    sorted_topk_ids: torch.Tensor
    aligned_expert_ids: torch.Tensor
    aligned_barrier_ids: torch.Tensor
    aligned_num_tokens: torch.Tensor
    workspace_tensors: List[torch.Tensor]
    barrier_tensors: List[torch.Tensor]
    internode_ag_stream: Optional[torch.cuda.streams.Stream] = None
    ag_stream: Optional[torch.cuda.streams.Stream] = None
    group_gemm_stream: Optional[torch.cuda.streams.Stream] = None
    local_world_size: int = 8
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_SIZE_M: int = 8
    stages: int = 3
    warps: int = 8

    def update(self, full_topk_ids: torch.Tensor, rank, num_ranks, num_experts, topk, BLOCK_M=128, BLOCK_N=256,
               BLOCK_K=64, GROUP_SIZE_M=8, stages=3, warps=8, ag_stream=None, group_gemm_stream=None,
               local_world_size=8):
        self.rank = rank
        self.num_ranks = num_ranks
        self.local_world_size = local_world_size
        self.num_experts = num_experts
        self.topk = topk
        self.full_num_tokens = full_topk_ids.numel()
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_SIZE_M = GROUP_SIZE_M
        self.stages = stages
        self.warps = warps
        self.ag_stream = ag_stream
        self.group_gemm_stream = group_gemm_stream
        M_per_rank = full_topk_ids.shape[0] // num_ranks
        (
            self.sorted_topk_ids,
            self.aligned_expert_ids,
            self.aligned_barrier_ids,
            _,
            self.aligned_num_tokens,
        ) = sort_topk_ids_align_block_size(full_topk_ids, num_experts, num_ranks, M_per_rank, BLOCK_M)

    def local_copy_and_barrier_all(self, local_data, is_internode=False):
        M_per_rank, K = local_data.shape
        local_rank = self.rank % self.local_world_size
        self.barrier_tensors[local_rank].fill_(0)
        pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
        dst = self.workspace_tensors[local_rank][self.rank * M_per_rank:(self.rank + 1) * M_per_rank, :]
        dst.copy_(local_data)
        set_signal(self.barrier_tensors[local_rank][self.rank].data_ptr(), 1, torch.cuda.current_stream(), is_internode)


def create_ag_group_gemm_intra_node_context(
    tp_pg,
    tensor_A,
    tensor_B,
    full_topk_ids,
    num_experts,
    max_M=2**14,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    GROUP_SIZE_M=8,
    stages=3,
    warps=8,
    ag_stream=None,
    group_gemm_stream=None,
):
    """create context for allgather group gemm intra-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [E, K, N_per_rank]
        topk_id (torch.Tensor<int32_t>): local topk ids. shape: [M_per_rank, topk]
        max_M: max value of M
        BLOCK_M (int, optional): Group GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): Group GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): Group GEMM tiling factor for K dim. Defaults to 64.
        GROUP_SIZE_M (int, optional): Group size of block for M dim (not size of group GEMM). Defaults to 8.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        warps (int, optional): No.of used warps. Defaults to 8.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        group_gemm_stream (torch.cuda.streams.Stream, optional): The stream used for group gemm, if not provided, use current stream. Defaults to None.

    Returns:
        MoE_AllGatherGroupGEMMTensorParallelContext
    """
    M_per_rank, K = tensor_A.shape
    _, topk = full_topk_ids.shape
    assert (
        tensor_B.shape[1] == K), f"tensor_B should has shape (col_major) [N_per_rank, {K}], but get [{tensor_B.shape}]"
    assert tensor_A.dtype == tensor_B.dtype
    dtype = tensor_A.dtype
    num_ranks = tp_pg.size()
    rank = tp_pg.rank()

    workspaces = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, K], dtype)
    barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], torch.int32)

    # sort the full topk ids.
    (
        full_sorted_token_ids,
        full_token_expert_ids,
        block_wait_barriers,
        _,
        full_num_tokens_post_padded_list,
    ) = sort_topk_ids_align_block_size(full_topk_ids, num_experts, num_ranks, M_per_rank, BLOCK_M)

    barriers[rank].fill_(0)
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    ret = MoE_AllGatherGroupGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        num_experts=num_experts,
        topk=topk,
        full_num_tokens=full_topk_ids.numel(),
        sorted_topk_ids=full_sorted_token_ids,
        aligned_expert_ids=full_token_expert_ids,
        aligned_barrier_ids=block_wait_barriers,
        aligned_num_tokens=full_num_tokens_post_padded_list,
        workspace_tensors=workspaces,
        barrier_tensors=barriers,
        ag_stream=ag_stream,
        group_gemm_stream=group_gemm_stream,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        stages=stages,
        warps=warps,
    )

    return ret


def ag_group_gemm_intra_node(a, b, ctx=None, full_topk_ids=None, num_experts=None):
    """intra-node allgather group gemm

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [E, K, N_per_rank]
        ctx: (Optional[MoE_AllGatherGroupGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M * topk, N_per_rank]
    """
    if ctx is None:
        assert full_topk_ids is not None and num_experts is not None
        tp_pg = TP_GROUP()
        ctx = create_ag_group_gemm_intra_node_context(tp_pg, a, b, full_topk_ids, num_experts)
    M_per_rank, _ = a.shape
    _, _, N_per_rank = b.shape
    C = torch.empty(
        [ctx.topk * M_per_rank * ctx.num_ranks, N_per_rank],
        dtype=a.dtype,
        device=a.device,
    )

    rowise_intranode_ag_scatter_group_gemm(
        a,
        ctx.workspace_tensors[ctx.rank],
        ctx.workspace_tensors,
        b,
        C,
        ctx.barrier_tensors,
        ctx.topk,
        ctx.sorted_topk_ids,
        ctx.aligned_expert_ids,
        ctx.aligned_barrier_ids,
        ctx.aligned_num_tokens,
        ctx.full_num_tokens,
        ctx.rank,
        ctx.num_ranks,
        ag_stream=ctx.ag_stream,
        group_gemm_stream=ctx.group_gemm_stream,
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N=ctx.BLOCK_N,
        BLOCK_K=ctx.BLOCK_K,
        GROUP_SIZE_M=ctx.GROUP_SIZE_M,
        stages=ctx.stages,
        warps=ctx.warps,
    )

    return C


def create_ag_group_gemm_inter_node_context(tp_pg, tensor_A, tensor_B, full_topk_ids, num_experts, max_M=2**14,
                                            BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_SIZE_M=8, stages=3, warps=8,
                                            ag_stream=None, group_gemm_stream=None, local_world_size=8):
    """create context for allgather group gemm inter-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [E, K, N_per_rank]
        topk_id (torch.Tensor<int32_t>): local topk ids. shape: [M_per_rank, topk]
        max_M: max value of M
        BLOCK_M (int, optional): Group GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): Group GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): Group GEMM tiling factor for K dim. Defaults to 64.
        GROUP_SIZE_M (int, optional): Group size of block for M dim (not size of group GEMM). Defaults to 8.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        warps (int, optional): No.of used warps. Defaults to 8.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        group_gemm_stream (torch.cuda.streams.Stream, optional): The stream used for group gemm, if not provided, use current stream. Defaults to None.

    Returns:
        MoE_AllGatherGroupGEMMTensorParallelContext
    """
    M_per_rank, K = tensor_A.shape
    _, topk = full_topk_ids.shape
    assert (
        tensor_B.shape[1] == K), f"tensor_B should has shape (col_major) [N_per_rank, {K}], but get [{tensor_B.shape}]"
    assert tensor_A.dtype == tensor_B.dtype
    dtype = tensor_A.dtype
    num_ranks = tp_pg.size()
    rank = tp_pg.rank()
    local_rank = rank % local_world_size

    workspaces = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, K], dtype)
    barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], torch.uint64)

    # sort the full topk ids.
    (
        full_sorted_token_ids,
        full_token_expert_ids,
        block_wait_barriers,
        _,
        full_num_tokens_post_padded_list,
    ) = sort_topk_ids_align_block_size(full_topk_ids, num_experts, num_ranks, M_per_rank, BLOCK_M)

    barriers[local_rank].fill_(0)
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    ret = MoE_AllGatherGroupGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        local_world_size=local_world_size,
        num_experts=num_experts,
        topk=topk,
        full_num_tokens=full_topk_ids.numel(),
        sorted_topk_ids=full_sorted_token_ids,
        aligned_expert_ids=full_token_expert_ids,
        aligned_barrier_ids=block_wait_barriers,
        aligned_num_tokens=full_num_tokens_post_padded_list,
        workspace_tensors=workspaces,
        barrier_tensors=barriers,
        internode_ag_stream=torch.cuda.Stream(),
        ag_stream=ag_stream,
        group_gemm_stream=group_gemm_stream,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        stages=stages,
        warps=warps,
    )

    return ret


def ag_group_gemm_inter_node(a, b, ctx=None, full_topk_ids=None, num_experts=None):
    """inter-node allgather group gemm

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [E, K, N_per_rank]
        ctx: (Optional[MoE_AllGatherGroupGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M * topk, N_per_rank]
    """
    if ctx is None:
        assert full_topk_ids is not None and num_experts is not None
        tp_pg = TP_GROUP()
        ctx = create_ag_group_gemm_inter_node_context(tp_pg, a, b, full_topk_ids, num_experts)
    M_per_rank, _ = a.shape
    _, _, N_per_rank = b.shape
    C = torch.empty(
        [ctx.topk * M_per_rank * ctx.num_ranks, N_per_rank],
        dtype=a.dtype,
        device=a.device,
    )

    ctx.local_copy_and_barrier_all(a, is_internode=True)

    rowise_internode_ag_scatter_group_gemm(
        a,
        ctx.workspace_tensors,
        b,
        C,
        ctx.barrier_tensors,
        ctx.topk,
        ctx.sorted_topk_ids,
        ctx.aligned_expert_ids,
        ctx.aligned_barrier_ids,
        ctx.aligned_num_tokens,
        ctx.full_num_tokens,
        ctx.rank,
        ctx.num_ranks,
        ctx.local_world_size,
        internode_ag_stream=ctx.internode_ag_stream,
        ag_stream=ctx.ag_stream,
        group_gemm_stream=ctx.group_gemm_stream,
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N=ctx.BLOCK_N,
        BLOCK_K=ctx.BLOCK_K,
        GROUP_SIZE_M=ctx.GROUP_SIZE_M,
        stages=ctx.stages,
        warps=ctx.warps,
    )

    return C
