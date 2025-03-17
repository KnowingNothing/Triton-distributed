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
import triton.distributed.language as dl

from typing import Optional, List
import pynvshmem
from triton.distributed.kernels.nvidia.common_ops import barrier_all, wait_eq, set_signal


def cp_engin_scatter_and_notify_push_mode(
    rank,
    num_ranks,
    gemm_out,  # [M, N]
    scatter_bufs,
    scatter_barrier,
    reduce_barriers,
    scatter_stream,
):
    M = gemm_out.shape[0]
    M_per_rank = M // num_ranks

    with torch.cuda.stream(scatter_stream):
        for i in range(0, num_ranks):
            remote_rank = (rank + i + 1) % num_ranks
            wait_eq(scatter_barrier[remote_rank].data_ptr(), 1,  # signal
                    scatter_stream, True)
            remote_buf = scatter_bufs[remote_rank][rank * M_per_rank:(rank + 1) * M_per_rank, :]
            local_buf = gemm_out[remote_rank * M_per_rank:(remote_rank + 1) * M_per_rank, :]
            remote_buf.copy_(local_buf)
            set_signal(
                reduce_barriers[remote_rank][rank].data_ptr(),
                1,
                scatter_stream,
                require_i64=True,
            )


def ring_reduce_after_scatter(
    rank,
    num_ranks,
    scatter_out,  # [M, N]
    reduce_barrier,
    stream,
):
    M, N = scatter_out.shape
    M_per_rank = M // num_ranks
    output = torch.empty((M_per_rank, N), dtype=scatter_out.dtype, device=scatter_out.device)
    grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]), )
    with torch.cuda.stream(stream):
        kernel_consumer_reduce[grid](
            scatter_out,
            output,
            reduce_barrier,
            M_per_rank,
            N,
            BLOCK_SIZE=4096,
            num_warps=8,
        )

    return output


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_buf,
    stream,
):

    with torch.cuda.stream(stream):
        barrier_all[(1, )](rank, num_ranks, sync_buf)


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

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
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

    nxt_rank = (rank + 1) % num_ranks
    M_per_rank = M // num_ranks
    rank_swizzle_offset = M_per_rank * nxt_rank // BLOCK_SIZE_M

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

            # rank swizzle
            pid_m = (pid_m + rank_swizzle_offset) % num_pid_m

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


def gemm_rs_producer_persistent(a, b, c, barrier, workspace, gemm_stream, BLOCK_SIZE_M=128, BLOCK_SIZE_N=256,
                                BLOCK_SIZE_K=64, GROUP_SIZE_M=8, STAGES=3):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(
        NUM_SMS,
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
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            False,
            NUM_SMS=NUM_SMS,  #
            num_stages=STAGES,
            num_warps=8,
        )

    current_stream.wait_stream(gemm_stream)

    return compiled


@triton.jit
def kernel_consumer_reduce(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_rank, N]
    reduce_barrier_ptr,
    # shape of matrix
    M_per_rank,
    N,
    # reduce tile shape
    BLOCK_SIZE: tl.constexpr,
):
    rank = dl.rank()
    num_ranks = dl.num_ranks()
    pid = tl.program_id(axis=0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.where(offs < M_per_rank * N, offs, 0)

    out_ptrs = out_ptr + offs

    accum = tl.zeros((BLOCK_SIZE, ), dtype=out_ptr.dtype.element_ty)
    for i in range(0, num_ranks):
        cur_rank = (i + rank + 1) % num_ranks
        c_ptrs = c_ptr + offs + cur_rank * M_per_rank * N
        data = tl.load(c_ptrs)
        accum += data

    tl.store(out_ptrs, accum)


def gemm_rs_intra_node_persistent_op(a, b, output_dtype, rank, num_ranks, scatter_bufs, scatter_barrier,
                                     reduce_barriers, sync_buf, scatter_stream, reduce_stream, BLOCK_M=128, BLOCK_N=256,
                                     BLOCK_K=64, GROUP_M=8, stages=3):
    """gemm reduce scatter for intra-node
    
    Local matrix A and do matmul with local matrix B, produces local matrix C, then reduce scatter

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M, K_per_rank]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N, K_per_rank]
        output_dtype (torch.dtype): output data type
        rank (int): current rank
        num_ranks (int): total number of ranks
        scatter_bufs (List[torch.Tensor<float>]): A list of symm-tensors used for inter-rank allgather.
            Each tensor shape: [maxM, N]. Created by `create_gemm_rs_intra_node_context`.
        scatter_barrier (torch.Tensor<int64>): A symm-tensors used for scatter.
            Tensor shape: [num_ranks]. Created by `create_gemm_rs_intra_node_context`.
        reduce_barriers (List[torch.Tensor<int64>]): A list of symm-tensors used for reduce.
            Each tensor shape: [num_ranks]. Created by `create_gemm_rs_intra_node_context`.
        sync_buf (torch.Tensor<int32>): A symm-tensor used for global synchronization.
            Shape: [MAX_NUM_BLOCKS_ON_GPU(65536)*num_ranks]. Created by `create_gemm_rs_intra_node_context`.
        scatter_stream (torch.cuda.streams.Stream): The stream used for scatter.
        reduce_stream (torch.cuda.streams.Stream): The stream used for reduce.
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        GROUP_M (int, optional): GEMM group M size. Defaults to 8.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.

    Returns:
        result matrix C. shape [M_per_rank, N]
    """
    M, local_K = a.shape
    N, _ = b.shape

    assert b.shape[1] == local_K
    assert a.dtype == b.dtype
    local_M = M // num_ranks
    current_stream = torch.cuda.current_stream()
    barrier_all_on_stream(rank, num_ranks, sync_buf, current_stream)
    scatter_stream.wait_stream(current_stream)
    reduce_stream.wait_stream(current_stream)

    output = torch.empty((local_M, N), dtype=output_dtype, device=a.device)
    workspace = torch.zeros((num_ranks, ), dtype=torch.int32, device=a.device)
    gemm_out = torch.empty((M, N), dtype=output_dtype, device=a.device)
    gemm_rs_producer_persistent(a, b, gemm_out, scatter_barrier, workspace, current_stream, BLOCK_SIZE_M=BLOCK_M,
                                BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K, GROUP_SIZE_M=GROUP_M, STAGES=stages)

    cp_engin_scatter_and_notify_push_mode(rank, num_ranks, gemm_out, scatter_bufs, scatter_barrier, reduce_barriers,
                                          scatter_stream)
    current_stream.wait_stream(scatter_stream)
    barrier_all_on_stream(rank, num_ranks, sync_buf, current_stream)

    output = ring_reduce_after_scatter(rank, num_ranks, scatter_bufs[rank][:M], reduce_barriers[rank], current_stream)
    current_stream.wait_stream(current_stream)
    reduce_barriers[rank].zero_()
    scatter_barrier.zero_()

    return output


@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    rank: int
    num_ranks: int
    scatter_bufs: List[torch.Tensor]
    scatter_barriers: List[torch.Tensor]
    reduce_barriers: List[torch.Tensor]
    sync_buf: torch.Tensor
    scatter_stream: torch.cuda.streams.Stream
    reduce_stream: torch.cuda.streams.Stream
    output_dtype: torch.dtype
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_M: int = 8
    stages: int = 3

    def update(self, rank, num_ranks, scatter_stream, reduce_stream, output_dtype=None, BLOCK_M=128, BLOCK_N=256,
               BLOCK_K=64, GROUP_M=8, stages=3):
        self.rank = rank
        self.num_ranks = num_ranks
        self.scatter_stream = scatter_stream
        self.reduce_stream = reduce_stream
        self.output_dtype = output_dtype
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_M = GROUP_M
        self.stages = stages


def create_gemm_rs_intra_node_context(tensor_A, tensor_B, rank, num_ranks, scatter_stream, reduce_stream,
                                      output_dtype=None, max_M=2**14, max_blocks=65536, BLOCK_M=128, BLOCK_N=256,
                                      BLOCK_K=64, GROUP_M=8, stages=3):
    """create context for allgather gemm intra-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M, K_per_rank]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [N, K_per_rank]
        rank (int): current rank
        num_ranks (int): total number of ranks
        scatter_stream (torch.cuda.streams.Stream, optional): The stream used for scatter.
        reduce_stream (torch.cuda.streams.Stream, optional): The stream used for reduce.
        output_dtype (torch.dtype, optional): if not provided, the same as tensor_A.dtype
        max_M (int): max size of M
        max_blocks (int): max number of blocks
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        GROUP_M (int, optional): GEMM group M size. Defaults to 8.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.

    Returns:
        GEMMReduceScatterTensorParallelContext
    """
    N = tensor_B.shape[0]
    K_per_rank = tensor_B.shape[1]
    assert tensor_A.shape[1] == K_per_rank

    input_dtype = tensor_A.dtype
    output_dtype = output_dtype if output_dtype is not None else input_dtype
    signal_dtype = torch.uint64

    scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, N], output_dtype)
    scatter_barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], signal_dtype)
    reduce_barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], signal_dtype)

    sync_buf = pynvshmem.nvshmem_create_tensor([max_blocks * num_ranks], torch.int32)
    sync_buf.fill_(0)
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    ret = GEMMReduceScatterTensorParallelContext(rank=rank, num_ranks=num_ranks, scatter_bufs=scatter_bufs,
                                                 scatter_barriers=scatter_barriers, reduce_barriers=reduce_barriers,
                                                 sync_buf=sync_buf, scatter_stream=scatter_stream,
                                                 reduce_stream=reduce_stream, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                                 BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, stages=stages,
                                                 output_dtype=output_dtype)

    return ret


def gemm_rs_intra_node(a, b, ctx=None, reduce_stream=None, scatter_stream=None, rank=None, num_ranks=None):
    """allgather gemm for intra-node
    
    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M, K_per_rank]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N, K_per_rank]
        ctx: (Optional[AllGatherGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
    """
    if ctx is None:
        assert rank is not None and num_ranks is not None
        reduce_stream = reduce_stream if reduce_stream is not None else torch.cuda.Stream()
        scatter_stream = scatter_stream if scatter_stream is not None else torch.cuda.Stream()
        ctx = create_gemm_rs_intra_node_context(a, b, rank, num_ranks, scatter_stream=scatter_stream,
                                                reduce_stream=reduce_stream)

    C = gemm_rs_intra_node_persistent_op(a, b, ctx.output_dtype, ctx.rank, ctx.num_ranks, ctx.scatter_bufs,
                                         ctx.scatter_barriers[ctx.rank], ctx.reduce_barriers, ctx.sync_buf,
                                         ctx.scatter_stream, ctx.reduce_stream, BLOCK_M=ctx.BLOCK_M,
                                         BLOCK_N=ctx.BLOCK_N, BLOCK_K=ctx.BLOCK_K, GROUP_M=ctx.GROUP_M,
                                         stages=ctx.stages)

    return C
