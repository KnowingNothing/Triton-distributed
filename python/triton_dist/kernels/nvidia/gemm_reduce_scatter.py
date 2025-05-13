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
import dataclasses
from typing import List, Optional

import torch

import triton
import triton_dist.language as dl
import triton.language as tl
from triton_dist import pynvshmem

from triton.language.extra.cuda.language_extra import atomic_add, __syncthreads, tid
from triton_dist.kernels.nvidia.reduce_scatter import ReduceScatter2DContext, create_reduce_scater_2d_ctx, reduce_scatter_2d_op


################### context ###################
@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    rs_ctx: ReduceScatter2DContext
    output_dtype: torch.dtype

    # gemm bufs (symm address)
    gemm_out_bufs: List[torch.Tensor]

    # stream
    rs_stream: torch.cuda.Stream

    # gemm kernel config
    num_gemm_sms: int
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_M: int = 8
    stages: int = 3

    def update(self, rank, num_ranks, rs_stream, output_dtype=None, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_M=8,
               stages=3):
        self.rank = rank
        self.num_ranks = num_ranks
        self.rs_stream = rs_stream
        self.output_dtype = output_dtype
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_M = GROUP_M
        self.stages = stages

    def get_gemm_out_buf(self, input):
        M, _ = input.shape
        local_rank = self.rs_ctx.local_rank
        return self.gemm_out_bufs[local_rank][:M]


def create_gemm_rs_context(max_M, N, rank, world_size, local_world_size, output_dtype, rs_stream, BLOCK_M=128,
                           BLOCK_N=256, BLOCK_K=64, GROUP_M=8, stages=3) -> GEMMReduceScatterTensorParallelContext:
    rs_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, output_dtype,
                                         overlap_with_gemm=True)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms
    gemm_out_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, N], output_dtype)
    ctx = GEMMReduceScatterTensorParallelContext(rs_ctx=rs_ctx, output_dtype=output_dtype, gemm_out_bufs=gemm_out_bufs,
                                                 rs_stream=rs_stream, num_gemm_sms=num_gemm_sms, BLOCK_M=BLOCK_M,
                                                 BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, stages=stages)
    return ctx


# TMA related test
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    barrier_ptr,
    counter_ptr,
    local_world_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):  #
    # Matmul using TMA and device-side descriptor creation
    rank = dl.rank()
    num_ranks = dl.num_ranks()
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = rank // local_world_size
    nnodes = num_ranks // local_world_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // num_ranks
    # M_per_rank % BLOCK_SIZE_M == 0 is guaranteed by the caller
    num_pid_m_per_rank = M_per_rank // BLOCK_SIZE_M

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            m_rank = pid_m // num_pid_m_per_rank
            pid_m_intra_rank = pid_m - m_rank * num_pid_m_per_rank
            m_node_id = m_rank // local_world_size
            m_local_rank = m_rank % local_world_size
            swizzle_m_node_id = (m_node_id + node_id + 1) % nnodes
            swizzle_m_local_rank = (m_local_rank + rank + 1) % local_world_size
            swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank

            # rank swizzle
            pid_m = swizzle_m_rank * num_pid_m_per_rank + pid_m_intra_rank

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            counter_start = offs_am // M_per_rank
            counter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
            counter_end = min(counter_end, num_ranks - 1)
            for counter_id in range(counter_start, counter_end + 1):
                m_start = M_per_rank * counter_id
                m_end = M_per_rank * (counter_id + 1) - 1
                tiled_m_start = m_start // BLOCK_SIZE_M
                tiled_m_end = m_end // BLOCK_SIZE_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                tiled_n = tl.cdiv(N, BLOCK_SIZE_N)
                val = tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")
                if val == tiled_m_size * tiled_n - 1:
                    dl.notify(barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu")
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


@triton.jit(do_not_specialize=["local_world_size"], launch_metadata=_matmul_launch_metadata)
def kernel_gemm_rs_producer_non_persistent(
    # Pointers to matrices
    a_ptr,  # [M, K]_Ti
    b_ptr,  # [K, N]_Ti
    c_ptr,  # [M, N]_To
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    barrier_ptr,
    counter_ptr,
    local_world_size,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    tl.static_assert(a_ptr.dtype.is_ptr(), "A should be a pointer")
    tl.static_assert(b_ptr.dtype.is_ptr(), "B should be a pointer")
    tl.static_assert(c_ptr.dtype.is_ptr(), "C should be a pointer")
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B should have the same dtype")
    # IS_FP8 = tl.constexpr(a_dtype == tl.float8e4nv) or tl.constexpr(a_dtype == tl.float8e5)

    rank = dl.rank()
    num_ranks = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = num_ranks // local_world_size

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    M_per_rank = M // num_ranks
    num_pid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    # TODO(houqi.1993) M_per_rank % BLOCK_SIZE_M == 0 is guaranteed by the caller
    tile_id = pid

    # perfect shape is required.
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    m_rank = pid_m // num_pid_m_per_rank
    pid_m_intra_rank = pid_m - m_rank * num_pid_m_per_rank
    m_node_id = m_rank // local_world_size
    m_local_rank = m_rank % local_world_size
    swizzle_m_node_id = (m_node_id + node_id + 1) % nnodes
    swizzle_m_local_rank = (m_local_rank + rank + 1) % local_world_size
    swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank

    # rank swizzle
    pid_m = swizzle_m_rank * num_pid_m_per_rank + pid_m_intra_rank

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if a_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator, mask=out_mask)

    # inc barrier
    segment_start = pid_m * BLOCK_SIZE_M // M_per_rank
    segment_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
    __syncthreads()
    segment = segment_start + tid(axis=0)
    if segment <= segment_end:
        m_start = M_per_rank * segment
        m_end = M_per_rank * (segment + 1) - 1
        tiled_m_start = m_start // BLOCK_SIZE_M
        tiled_m_end = m_end // BLOCK_SIZE_M
        tiled_m_size = tiled_m_end - tiled_m_start + 1
        tiled_n = tl.cdiv(N, BLOCK_SIZE_N)
        val = atomic_add(counter_ptr + segment, 1, semantic="release", scope="gpu")
        if (val == tiled_n * tiled_m_size - 1):
            atomic_add(barrier_ptr + segment, 1, semantic="relaxed", scope="gpu")


def gemm_rs_producer_persistent(a, b, c, barrier, workspace, world_size, local_world_size, num_gemm_sms, gemm_stream,
                                BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, STAGES=3):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    M_per_rank = M // world_size

    assert M_per_rank % BLOCK_SIZE_M == 0

    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    ), )

    with torch.cuda.stream(gemm_stream):
        compiled = kernel_gemm_rs_producer_persistent[grid](
            a,
            b,
            c,
            M,
            N,
            local_K,
            barrier,
            workspace,
            local_world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            False,
            NUM_SMS=num_gemm_sms,  #
            num_stages=STAGES,
            num_warps=8,
        )

    current_stream.wait_stream(gemm_stream)

    return compiled


def gemm_rs_producer_non_persistent(a, b, c, barrier, workspace, world_size, local_world_size, gemm_stream,
                                    BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, STAGES=3):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    M_per_rank = M // world_size

    assert M_per_rank % BLOCK_SIZE_M == 0

    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    with torch.cuda.stream(gemm_stream):
        compiled = kernel_gemm_rs_producer_non_persistent[grid](
            a,
            b,
            c,
            M,
            N,
            local_K,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(0),
            c.stride(0),
            c.stride(1),
            barrier,
            workspace,
            local_world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_stages=STAGES,
            num_warps=8,
        )

    current_stream.wait_stream(gemm_stream)

    return compiled


def padded_to_BLOCK_M(input, world_size, BLOCK_SIZE_M):
    M, local_K = input.shape

    M_per_rank = M // world_size
    pad_size = (M_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * BLOCK_SIZE_M
    if pad_size == M_per_rank:
        return input
    input = input.reshape(world_size, M_per_rank, local_K)
    pad_input = torch.empty((world_size, pad_size, local_K), dtype=input.dtype, device=input.device)
    pad_input[:, :M_per_rank].copy_(input)
    pad_input = pad_input.reshape(-1, local_K)
    return pad_input


def gemm_rs_op(input, weight, ctx: GEMMReduceScatterTensorParallelContext, persistent: bool = True):
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    num_gemm_sms = ctx.num_gemm_sms

    orig_M = input.shape[0]
    orig_M_per_rank = orig_M // world_size
    input = padded_to_BLOCK_M(input, world_size, ctx.BLOCK_M)
    M, local_K = input.shape
    N = weight.shape[0]
    assert N == ctx.rs_ctx.N

    assert M % world_size == 0
    assert weight.shape[1] == local_K
    local_M = M // world_size
    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(current_stream)

    output = torch.empty((local_M, N), dtype=output_dtype, device=input.device)
    workspace = torch.zeros((world_size, ), dtype=torch.int32, device=input.device)
    gemm_out = ctx.get_gemm_out_buf(input)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf

    if persistent:
        gemm_rs_producer_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size, local_world_size,
                                    num_gemm_sms, current_stream, BLOCK_SIZE_M=ctx.BLOCK_M, BLOCK_SIZE_N=ctx.BLOCK_N,
                                    BLOCK_SIZE_K=ctx.BLOCK_K, GROUP_SIZE_M=ctx.GROUP_M, STAGES=ctx.stages)
    else:
        gemm_rs_producer_non_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size,
                                        local_world_size, current_stream, BLOCK_SIZE_M=ctx.BLOCK_M,
                                        BLOCK_SIZE_N=ctx.BLOCK_N, BLOCK_SIZE_K=ctx.BLOCK_K, GROUP_SIZE_M=ctx.GROUP_M,
                                        STAGES=ctx.stages)

    with torch.cuda.stream(rs_stream):
        output = reduce_scatter_2d_op(gemm_out, ctx.rs_ctx)
    current_stream.wait_stream(rs_stream)

    return output[:orig_M_per_rank]


def gemm_rs(a, b, ctx, persistent=True):
    """GEMM Reduce-Scatter for Multi-Node

    computes local GEMM (a x b) to generate partial results, followed by `reduce_scatter` to produce c

    Args:
        a (torch.Tensor<bfloat16/float16>): local matmul A matrix. shape: [M, local_K]
        b (torch.Tensor<bfloat16/float16>): local matmul B matrix. shape: [N, local_K]
        ctx(GEMMReduceScatterTensorParallelContext): context

    Returns:
        c (torch.Tensor<bfloat16/float16>): local matmul C matrix. shape: [M // world_size, N]
    """
    c = gemm_rs_op(a, b, ctx, persistent)
    return c
