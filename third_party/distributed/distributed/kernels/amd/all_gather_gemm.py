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
import dataclasses
import triton
import triton.language as tl

from hip import hip
from triton.distributed.utils import HIP_CHECK
from typing import Optional, List
import pyrocshmem
from triton.distributed.kernels.amd.common_ops import (
    wait_eq_sys,
    barrier_all_ipc,
    set_signal,
)

SIGNAL_DTYPE = torch.int32


def create_tensor_ipc_intra_node(shape, dtype, local_rank, local_world_size, tp_group):

    return pyrocshmem.hipipc_create_tensor_list(tp_group, shape, dtype)


def cp_engine_producer_all_gather_full_mesh_push(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape
    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    for src_rank in rank_orders:
        if src_rank == rank:
            continue
        dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
        src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
        # dst.copy_(src)
        nbytes = src.numel() * src.element_size()
        cp_res = hip.hipMemcpyAsync(
            dst.data_ptr(),
            src.data_ptr(),
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            ag_stream.cuda_stream,
        )
        HIP_CHECK(cp_res)

        set_signal(barrier_buffers[rank][src_rank].data_ptr(), 1, ag_stream)


def cp_engine_producer_all_gather_full_mesh_pull_multi_stream(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    M_PER_CHUNK: int,
    ag_stream_pool: List[torch.cuda.Stream],
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape
    chunk_num_per_rank = M_per_rank // M_PER_CHUNK
    num_stream = len(ag_stream_pool)
    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    stream_offset = rank
    #stream_offset = 0
    for src_rank in rank_orders:
        if src_rank == rank:
            stream_offset += chunk_num_per_rank
            continue
        for chunk_idx_intra_rank in range(chunk_num_per_rank):
            stream_offset += 1
            chunk_pos = src_rank * chunk_num_per_rank + chunk_idx_intra_rank
            stream_pos = stream_offset % num_stream
            ag_stream = ag_stream_pool[stream_pos]
            M_start_pos = src_rank * M_per_rank + chunk_idx_intra_rank * M_PER_CHUNK
            M_end_pos = M_start_pos + M_PER_CHUNK
            dst = remote_tensor_buffers[rank][M_start_pos:M_end_pos, :]
            src = remote_tensor_buffers[src_rank][M_start_pos:M_end_pos, :]
            # dst.copy_(src)
            nbytes = src.numel() * src.element_size()
            cp_res = hip.hipMemcpyAsync(
                dst.data_ptr(),
                src.data_ptr(),
                nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                #hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                ag_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)

            set_signal(barrier_buffers[rank][chunk_pos].data_ptr(), 1, ag_stream)


def cp_engine_producer_all_gather_full_mesh_push_multi_stream(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    one: torch.Tensor,
    M_PER_CHUNK: int,
    ag_stream_pool: List[torch.cuda.Stream],
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape
    chunk_num_per_rank = M_per_rank // M_PER_CHUNK
    num_stream = len(ag_stream_pool)
    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    stream_offset = rank
    stream_offset = 0
    data_elem_size = local_tensor.element_size()
    barrier_elem_size = one.element_size()

    for idx, remote_rank in enumerate(rank_orders):
        if remote_rank == rank:
            stream_offset += chunk_num_per_rank
            continue
        for chunk_idx_intra_rank in range(chunk_num_per_rank):
            stream_offset += 1
            chunk_pos = rank * chunk_num_per_rank + chunk_idx_intra_rank
            stream_pos = idx % num_stream
            ag_stream = ag_stream_pool[stream_pos]
            M_dst_start_pos = rank * M_per_rank + chunk_idx_intra_rank * M_PER_CHUNK
            # M_dst_end_pos = M_dst_start_pos + M_PER_CHUNK
            # dst = remote_tensor_buffers[remote_rank][M_dst_start_pos:M_dst_end_pos, :]
            M_src_start_pos = chunk_idx_intra_rank * M_PER_CHUNK
            # M_src_end_pos = M_src_start_pos + M_PER_CHUNK
            # src = local_tensor[M_src_start_pos:M_src_end_pos, :]
            src_ptr = local_tensor.data_ptr() + M_src_start_pos * N * data_elem_size
            dst_ptr = remote_tensor_buffers[remote_rank].data_ptr() + M_dst_start_pos * N * data_elem_size
            # dst.copy_(src)
            nbytes = M_PER_CHUNK * N * data_elem_size
            cp_res = hip.hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                #hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                ag_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)
            cp_res = hip.hipMemcpyAsync(
                barrier_buffers[remote_rank].data_ptr() + chunk_pos * barrier_elem_size,
                one.data_ptr(),
                barrier_elem_size,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                #hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                ag_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)
            # set_signal(barrier_buffers[remote_rank][chunk_pos].data_ptr(), 1, ag_stream)


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr,
    stream,
):
    with torch.cuda.stream(stream):
        barrier_all_ipc[(1, )](rank, num_ranks, sync_bufs_ptr)


@triton.jit
def kernel_local_copy_and_barrier_all(
    rank,
    num_ranks,
    local_buf_ptr,
    global_buf_ptr,
    barrier_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = sm_id // num_pid_n
    pid_n = sm_id % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    data_ptr = local_buf_ptr + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
    dst_ptr = global_buf_ptr + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
    mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)
    mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)

    data = tl.load(data_ptr, mask=mask_data)
    tl.store(dst_ptr, data, mask=mask_dst)


def local_copy_and_barrier_all(rank, num_ranks, local_data, global_data, comm_buf_ptr, barrier_ptr, M_per_rank,
                               M_PER_CHUNK, N):
    barrier_all_ipc[(1, )](rank, num_ranks, comm_buf_ptr)
    grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel_local_copy_and_barrier_all[grid](rank, num_ranks, local_data, global_data, barrier_ptr, M_per_rank, N,
                                            local_data.stride(0), local_data.stride(1), global_data.stride(0),
                                            global_data.stride(1), 128, 256)
    barrier_ptr.fill_(0)
    # set_signal(barrier_ptr[rank].data_ptr(), 1, torch.cuda.current_stream())
    chunk_num_per_rank = M_per_rank // M_PER_CHUNK
    for chunk_idx_intra_rank in range(chunk_num_per_rank):
        chunk_pos = rank * chunk_num_per_rank + chunk_idx_intra_rank
        set_signal(barrier_ptr[chunk_pos].data_ptr(), 1, torch.cuda.current_stream())

    barrier_all_ipc[(1, )](rank, num_ranks, comm_buf_ptr)


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 1,
                'waves_per_eu': 2,
            }, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def ag_gemm_consumer_kernel(
        # Pointers to matrices
        a_ptr, local_a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn, rank, world_size, barrier_ptr,
        # Meta-parameters
        M_PER_CHUNK: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    ### threadblock swizzle
    # num_pid_m_per_rank = tl.cdiv(num_pid_m, world_size)
    # pid_m = (pid_m + num_pid_m_per_rank * rank) % num_pid_m
    # segment = pid_m // num_pid_m_per_rank
    num_chunk = tl.cdiv(M, M_PER_CHUNK)
    num_pid_m_per_rank = tl.cdiv(num_pid_m, world_size)
    num_pid_m_per_chunk = tl.cdiv(num_pid_m, num_chunk)
    pid_m = (pid_m + num_pid_m_per_rank * rank) % num_pid_m
    seg_chunk = pid_m // num_pid_m_per_chunk
    seg_rank = pid_m // num_pid_m_per_rank

    ##### ****************
    if seg_rank != rank:
        wait_eq_sys(barrier_ptr + seg_chunk, 1)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if seg_rank == rank:
        # NOTE: assume a_ptr and local_a_ptr has same strides.
        M_PER_RANK = M // world_size
        # offs_am = ((pid_m - num_pid_m_per_rank * rank) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_PER_RANK
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_PER_RANK
        a_ptrs = local_a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


## TODO: adjust
@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2,
                'kpack': 1, 'matrix_instr_nonkdim': 16
            }, num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2,
                'kpack': 1, 'matrix_instr_nonkdim': 16
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
                'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
                'kpack': 1
            }, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
    use_cuda_graph=True,
)
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit
def kernel_consumer_gemm_persistent(
    A,
    localA,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    rank,
    world_size,
    barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_PER_CHUNK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    M_per_rank = M // world_size
    pid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        ## swizzle
        num_pid_m_per_copy_chunk = M_PER_CHUNK // BLOCK_SIZE_M
        chunk_offset = pid_m // (num_pid_m_per_copy_chunk * world_size)
        rank_offset = pid_m % (num_pid_m_per_copy_chunk * world_size) // num_pid_m_per_copy_chunk
        block_offset = pid_m % num_pid_m_per_copy_chunk

        rank_offset = (rank_offset + rank) % world_size
        pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_CHUNK + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M

        offs_am = pid_m * BLOCK_SIZE_M
        offs_sig = offs_am // M_PER_CHUNK
        offs_rank = pid_m // pid_m_per_rank

        if offs_rank != rank:
            wait_eq_sys(barrier_ptr + offs_sig, 1)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        if offs_rank == rank:
            rm = rm % M_per_rank
            A_BASE = localA + rm[:, None] * stride_am + rk[None, :] * stride_ak
        else:
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak

        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def ag_gemm_intra_node_op(a, b, c, rank, num_ranks, workspace_tensors, barrier_tensors, comm_buf_ptr, ag_streams,
                          gemm_stream, serial=False, M_PER_CHUNK=1024, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3,
                          autotune=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    _, N_per_rank = b.shape

    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)
    for ag_stream in ag_streams:
        ag_stream.wait_stream(current_stream)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]), )

    def call_ag():
        cp_engine_producer_all_gather_full_mesh_push_multi_stream(rank, num_ranks, a, workspace_tensors, M_PER_CHUNK,
                                                                  ag_streams, barrier_tensors)

    if serial:
        call_ag()
        for ag_stream in ag_streams:
            current_stream.wait_stream(ag_stream)
        torch.cuda.synchronize()
        torch.distributed.barrier()
    else:
        call_ag()

    with torch.cuda.stream(gemm_stream):
        compiled = None
        full_input = workspace_tensors[rank][:M]
        assert full_input.stride() == a.stride()
        compiled = ag_gemm_consumer_kernel[grid](
            full_input,
            a,
            b,
            c,  #
            M,
            N_per_rank,
            K,  #
            full_input.stride(0),
            full_input.stride(1),  #
            b.stride(0),
            b.stride(1),  #
            c.stride(0),
            c.stride(1),  #
            rank,
            num_ranks,
            barrier_tensors[rank],
            M_PER_CHUNK=M_PER_CHUNK,
        )

    current_stream.wait_stream(gemm_stream)
    for ag_stream in ag_streams:
        current_stream.wait_stream(ag_stream)

    return compiled


def ag_gemm_consumer_persistent_op(a, b, c, rank, num_ranks, workspace_tensors, one, barrier_tensors, comm_buf_ptr,
                                   ag_streams, gemm_stream, serial=False, M_PER_CHUNK=1024, BLOCK_M=128, BLOCK_N=256,
                                   BLOCK_K=64, stages=3, autotune=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    _, N_per_rank = b.shape

    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)
    for ag_stream in ag_streams:
        ag_stream.wait_stream(current_stream)

    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    NUM_XCDS = 4

    grid = lambda META: (min(
        NUM_SMS,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
    ), )

    def call_ag():
        cp_engine_producer_all_gather_full_mesh_push_multi_stream(rank, num_ranks, a, workspace_tensors, one,
                                                                  M_PER_CHUNK, ag_streams, barrier_tensors)

    if serial:
        call_ag()
        for ag_stream in ag_streams:
            current_stream.wait_stream(ag_stream)
        torch.cuda.synchronize()
        torch.distributed.barrier()
    else:
        call_ag()

    ##persistent_gemm:kernel_consumer_gemm_persistent
    with torch.cuda.stream(gemm_stream):
        compiled = None
        full_input = workspace_tensors[rank][:M]
        assert full_input.stride() == a.stride()

        compiled = kernel_consumer_gemm_persistent[grid](
            full_input,
            a,
            b,
            c,  #
            M,
            N_per_rank,
            K,  #
            full_input.stride(0),
            full_input.stride(1),  #
            b.stride(0),
            b.stride(1),  #
            c.stride(0),
            c.stride(1),  #
            rank,
            num_ranks,
            barrier_tensors[rank],
            M_PER_CHUNK=M_PER_CHUNK,
            NUM_SMS=NUM_SMS,  #
            NUM_XCDS=NUM_XCDS,
        )

    current_stream.wait_stream(gemm_stream)
    # for ag_stream in ag_streams:
    #     current_stream.wait_stream(ag_stream)

    return compiled


@dataclasses.dataclass
class AllGatherGEMMTensorParallelContext:
    rank: int
    num_ranks: int
    workspace_tensors: List[torch.Tensor]
    barrier_tensors: List[torch.Tensor]
    comm_bufs: List[torch.Tensor]
    one: torch.Tensor
    comm_buf_ptr: torch.Tensor
    ag_streams: Optional[List[torch.cuda.streams.Stream]] = None
    gemm_stream: Optional[torch.cuda.streams.Stream] = None
    serial: bool = False
    M_PER_CHUNK: int = 1024
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    stages: int = 3
    autotune: bool = False

    def update(self, rank, num_ranks, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3, ag_streams=None, gemm_stream=None,
               serial=False, autotune=False):
        self.rank = rank
        self.num_ranks = num_ranks
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.stages = stages
        self.ag_streams = ag_streams
        self.gemm_stream = gemm_stream
        self.serial = serial
        self.autotune = autotune


def create_ag_gemm_intra_node_context(max_M, N, K, input_dtype, output_dtype, rank, num_ranks, tp_group,
                                      ag_streams=None, gemm_stream=None, M_PER_CHUNK=1024, BLOCK_M=128, BLOCK_N=256,
                                      BLOCK_K=64, stages=3, serial=False, autotune=False):
    """create context for allgather gemm intra-node

    Args:
        max_M: max number of M shape
        N(int): N
        K(int): K
        input_dtype(torch.dtype): dtype of input
        output_dtype(torch.dtype): dtype of output
        rank (int): current rank
        num_ranks (int): total number of ranks
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        ag_streams (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        gemm_stream (torch.cuda.streams.Stream, optional): The stream used for gemm, if not provided, use current stream. Defaults to None.
        serial (bool, optional): Make the execution serialized, for debug. Defaults to False.
        autotune (bool, optional): whether to enable autotune. Defaults to False.

    Returns:
        AllGatherGEMMTensorParallelContext
    """
    assert M_PER_CHUNK % BLOCK_M == 0
    assert max_M % num_ranks == 0
    M_per_rank = max_M // num_ranks  # noqa: F841
    assert M_per_rank % M_PER_CHUNK == 0
    dtype = input_dtype
    workspaces = create_tensor_ipc_intra_node([max_M, K], dtype, rank, num_ranks, tp_group)

    m_chunk_num = (max_M + M_PER_CHUNK - 1) // M_PER_CHUNK
    barriers = create_tensor_ipc_intra_node([m_chunk_num], torch.int32, rank, num_ranks, tp_group)
    barriers[rank].fill_(0)

    comm_bufs = create_tensor_ipc_intra_node([num_ranks], torch.int32, rank, num_ranks, tp_group)
    comm_bufs[rank].fill_(0)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_bufs], device=torch.cuda.current_device(),
                                requires_grad=False)

    current_stream = torch.cuda.current_stream()

    torch.cuda.synchronize()
    barrier_all_on_stream(rank, num_ranks, comm_buf_ptr, current_stream)

    _ag_streams = [torch.cuda.Stream(priority=-1) for i in range(num_ranks)] if ag_streams is None else ag_streams
    _gemm_stream = current_stream if gemm_stream is None else gemm_stream
    one = torch.ones((1024, ), dtype=torch.int32).cuda()

    ret = AllGatherGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        workspace_tensors=workspaces,
        barrier_tensors=barriers,
        comm_bufs=comm_bufs,
        one=one,
        comm_buf_ptr=comm_buf_ptr,
        ag_streams=_ag_streams,
        gemm_stream=_gemm_stream,
        serial=serial,
        M_PER_CHUNK=M_PER_CHUNK,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        stages=stages,
        autotune=autotune,
    )

    return ret


def ag_gemm_intra_node(a, b, transed_b, ctx, rank=None, num_ranks=None):
    """allgather gemm for intra-node

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [K, N_per_rank]
        transed_b (bool): indicates whether tensor b is transposed
        ctx: (Optional[AllGatherGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
    """
    if ctx is None:
        raise RuntimeError("requires ctx for ipc handle")

    M_per_rank, K = a.shape
    N_per_rank = b.shape[1]
    C = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank], dtype=a.dtype, device=a.device)

    # Use our own customized barrier kernel
    #local_copy_and_barrier_all(ctx.rank, ctx.num_ranks, a, ctx.workspace_tensors[ctx.rank], ctx.comm_buf_ptr,
    #                           ctx.barrier_tensors[ctx.rank], M_per_rank, ctx.M_PER_CHUNK, K)

    def reset_barrier_and_barrier_all():
        barrier_ptr = ctx.barrier_tensors[ctx.rank]
        barrier_all_ipc[(1, )](ctx.rank, ctx.num_ranks, ctx.comm_buf_ptr)
        barrier_ptr.fill_(0)

    ## TODO: add option, use persistent gemm as default
    use_persistent = True
    if use_persistent:
        ag_gemm_consumer_persistent_op(a, b, C, ctx.rank, ctx.num_ranks, ctx.workspace_tensors, ctx.one,
                                       ctx.barrier_tensors, ctx.comm_buf_ptr, ag_streams=ctx.ag_streams,
                                       gemm_stream=ctx.gemm_stream, serial=ctx.serial, M_PER_CHUNK=ctx.M_PER_CHUNK,
                                       autotune=ctx.autotune)
        barrier_ptr = ctx.barrier_tensors[ctx.rank]
        barrier_ptr.fill_(0)
        barrier_all_ipc[(1, )](ctx.rank, ctx.num_ranks, ctx.comm_buf_ptr)

    else:
        reset_barrier_and_barrier_all()
        ag_gemm_intra_node_op(a, b, C, ctx.rank, ctx.num_ranks, ctx.workspace_tensors, ctx.barrier_tensors,
                              ctx.comm_buf_ptr, ag_streams=ctx.ag_streams, gemm_stream=ctx.gemm_stream,
                              serial=ctx.serial, M_PER_CHUNK=ctx.M_PER_CHUNK, autotune=ctx.autotune)

    return C
