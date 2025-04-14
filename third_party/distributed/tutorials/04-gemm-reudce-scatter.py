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
"""
Gemm reduce-scatter
===================
In this tutorial, you will write a Multi Node Gemm reduce-scatter operation that is significantly faster
than PyTorch's native op.

In doing so, you will learn about:

* Write an efficient reduce-scatter kernel.

* Overlap reduce-scatter with gemm operations to hide communication.

.. code-block:: bash

    CUDA_DEVICE_MAX_CONNECTIONS=8 ./third_party/distributed/launch.sh  ./third_party/distributed/tutorials/04-gemm-rs.py 8192 8192 29568

"""

import random
import torch
import dataclasses
import triton
import triton.language as tl
import triton.distributed.language as dl

from typing import Optional, List
import pynvshmem
from triton.distributed.kernels.nvidia.common_ops import wait_eq
from triton.language.extra.cuda.language_extra import __syncthreads, tid, atomic_cas
from triton.distributed.utils import CUDA_CHECK
from triton.language.extra import libshmem_device
from cuda import cudart

import argparse
import os
import datetime
import numpy as np

from functools import partial

from triton.distributed.utils import (
    generate_data,
    get_torch_prof_ctx,
    perf_func,
    dist_print,
)

SIGNAL_DTYPE = torch.uint64


# GEMM and reduce-scatter context. Stores comm info (dtype, nnodes, etc), symm buffers, and signals.
################### GEMM RS context ###################
@dataclasses.dataclass
class ReduceScatter2DContext:
    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    overlap_with_gemm: bool

    # comm buffer
    scatter_bufs: List[torch.Tensor]
    rs_per_node_bufs: List[torch.Tensor]
    p2p_bufs: List[torch.Tensor]

    # barrier bufs
    signal_bufs: List[torch.Tensor]  # need reset: signal_buf =  scatter_signal | rs_per_node_signal
    sync_buf: torch.Tensor  # no need to reset

    # stream
    reduction_stream: torch.cuda.Stream
    p2p_stream: torch.cuda.Stream

    # sms
    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int

    # preprocess to reduce cpu overhead
    # comm barriers
    scatter_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)
    rs_per_node_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)

    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    scatter_signal_buf_list_for_each_node: List[torch.Tensor] = dataclasses.field(init=False)

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.scatter_signal_buf_list_for_each_node = []
        for buf in self.signal_bufs:
            assert buf.shape[0] >= 2 * self.world_size

        self.scatter_signal_bufs = [buf[:self.world_size] for buf in self.signal_bufs]
        self.rs_per_node_signal_bufs = [buf[self.world_size:self.world_size * 2] for buf in self.signal_bufs]

        for node_id in range(self.nnodes):
            self.scatter_signal_buf_list_for_each_node.append(
                self.scatter_signal_bufs[self.local_rank][node_id * self.local_world_size:(node_id + 1) *
                                                          self.local_world_size])

    def reset_barriers(self) -> int:
        self.signal_bufs[self.local_rank].fill_(0)

    def get_scatter_bufs_and_signal_for_each_node(self, input, node_id):
        M = input.shape[0]
        M_per_rank = M // self.world_size
        M_per_node = M_per_rank * self.local_world_size
        scatter_bufs_intra_node = [
            self.scatter_bufs[i][node_id * M_per_node:(node_id + 1) * M_per_node] for i in range(self.local_world_size)
        ]
        return scatter_bufs_intra_node, self.scatter_signal_buf_list_for_each_node[node_id]

    @property
    def rs_per_node_buf(self) -> torch.Tensor:
        return self.rs_per_node_bufs[self.local_rank]

    @property
    def rs_per_node_signal_buf(self) -> torch.Tensor:
        return self.rs_per_node_signal_bufs[self.local_rank]

    @property
    def p2p_buf(self) -> torch.Tensor:
        return self.p2p_bufs[self.local_rank]

    @property
    def num_rs_sms(self) -> int:
        if self.nnodes > 1:
            return self.num_sync_sms + self.num_p2p_sms + self.num_reduction_sms
        else:
            # for intra node rs, no need sm.
            return 0

    @property
    def scatter_signal_buf(self) -> torch.Tensor:
        return self.scatter_signal_bufs[self.local_rank]


def create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, dtype, overlap_with_gemm=True,
                                num_reduction_sms=15) -> ReduceScatter2DContext:
    """
        for num_reduction_sms: tunable param, 16 are enough for H800
            For H800, we overlap local reduce and inter-node p2p with intra-node scatter.
            The reduction kernel bandwidth is not a bottleneck if it exceeds 450GB, so only a few SMs are needed.
            For machines with higher intra_node bandwidth(e.g. H100), we may need to increase the number of SMs or redesign overlapping.
    """
    assert world_size % local_world_size == 0
    assert max_M % world_size == 0

    scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, N], dtype)

    rs_per_node_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M // local_world_size, N], dtype)

    p2p_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M // local_world_size, N], dtype)

    # signal_buf: scatter_signal | rs_per_node_signal
    num_signal_bufs = 2
    signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([
        world_size * num_signal_bufs,
    ], SIGNAL_DTYPE)

    sync_buf = pynvshmem.nvshmem_create_tensor([
        local_world_size,
    ], torch.int32)
    sync_buf.fill_(0)
    barrier_all_on_stream(torch.cuda.current_stream())

    p2p_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
    reduction_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

    num_sync_sms = 0
    num_p2p_sms = 1
    ctx = ReduceScatter2DContext(max_M=max_M, N=N, rank=rank, world_size=world_size, local_world_size=local_world_size,
                                 dtype=dtype, overlap_with_gemm=overlap_with_gemm, scatter_bufs=scatter_bufs,
                                 rs_per_node_bufs=rs_per_node_bufs, p2p_bufs=p2p_bufs, signal_bufs=signal_bufs,
                                 sync_buf=sync_buf, reduction_stream=reduction_stream, p2p_stream=p2p_stream,
                                 num_sync_sms=num_sync_sms, num_p2p_sms=num_p2p_sms,
                                 num_reduction_sms=num_reduction_sms)
    return ctx


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


################### triton kernel ###################
"""
This kernel performs P2P communication, sending corresponding data
to the GPU with the same local rank on node `cur_node_id + offset + 1`.
"""


@triton.jit
def kernel_inter_node_p2p_for_same_local_rank(
    offset,
    local_world_size,
    M_per_rank,
    N,
    input,  # [M, N]
    output,  # [M, N]
    elem_size: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    nelem_per_rank = M_per_rank * N

    remote_node_id = (offset + 1 + node_id) % nnodes
    remote_rank = local_rank + remote_node_id * local_world_size
    libshmem_device.putmem_block(
        output + node_id * nelem_per_rank,
        input + remote_node_id * nelem_per_rank,
        nelem_per_rank * elem_size,
        remote_rank,
    )


"""
The difference between this kernel and a normal reduction operation lies in the accumulation order.
In a normal reduction, the accumulation starts from the 0th row.
This kernel starts the accumulation from the `offset`.
"""


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


"""
This function is used for intra-node barrier synchronization.
It is based on the Compare-And-Swap(CAS) operation to ensure that
all GPUs within the current node reach the barrier.
"""


@triton.jit
def barrier_all_intra_node(local_world_size, comm_buf_ptr):
    thread_id = tid(axis=0).to(tl.int32)
    rank = dl.rank()
    local_rank = rank % local_world_size
    node_id = rank // local_world_size
    rank_offset = node_id * local_world_size
    if thread_id < local_world_size:
        remote_ptr = dl.symm_at(comm_buf_ptr, thread_id + rank_offset)
        while atomic_cas(remote_ptr + local_rank, 0, 1, "sys", "release") != 0:
            pass
        while (atomic_cas(comm_buf_ptr + thread_id, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


def barrier_all_on_stream(
    stream,
    is_intra_node=False,
    barrier_all_buf=None,
    local_world_size=0,
):
    if not is_intra_node:
        pynvshmem.nvshmem_barrier_all_on_stream(stream.cuda_stream)
    else:
        assert barrier_all_buf is not None and local_world_size > 0
        with torch.cuda.stream(stream):
            barrier_all_intra_node[(1, )](local_world_size, barrier_all_buf)


def ring_reduce(
    input,  # [M_per_node, N]
    output,  # [M_per_rank, N]
    begin_idx,
    num_splits,
    stream,
    num_sms=-1,
):
    total_M, N = input.shape
    M_per_split = total_M // num_splits
    assert output.shape[0] == M_per_split and total_M % num_splits == 0
    if num_sms == -1:
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
    else:
        grid = lambda META: (min(
            triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), num_sms), )
        with torch.cuda.stream(stream):
            kernel_ring_reduce[grid](
                input,
                output,
                M_per_split,
                N,
                begin_idx,
                num_splits,
                BLOCK_SIZE_M=256,
                BLOCK_SIZE_N=128,
                num_warps=8,
            )

    return output


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


def intra_node_scatter(input_intra_node, scatter_bufs_intra_node: List[torch.Tensor],
                       scatter_signal_buf_intra_node: torch.Tensor, local_rank, stream, overlap_with_gemm=True):
    M, N = input_intra_node.shape
    local_world_size = len(scatter_bufs_intra_node)
    M_per_rank = M // local_world_size
    """
        use flattern pointer and driver api to reduce the overhead of slice, plus the offset is equal to tensor slice op:
            `signal_base_ptr + nbytes_per_scatter_signal * remote_local_rank`: `scatter_signal_buf_intra_node[remote_local_rank].data_ptr()`
            `scatter_bufs_intra_node[remote_local_rank].data_ptr() + remote_offset`: `scatter_bufs_intra_node[remote_local_rank][local_rank * M_per_rank:(local_rank + 1) * M_per_rank, :]`
            `local_buf_base_ptr + remote_local_rank * nbytes_per_rank`: `input_intra_node[remote_local_rank * M_per_rank:(remote_local_rank + 1) * M_per_rank, :]`
    """
    nbytes_per_rank = M_per_rank * N * input_intra_node.dtype.itemsize
    local_buf_base_ptr = input_intra_node.data_ptr()
    remote_offset = local_rank * nbytes_per_rank
    signal_base_ptr = scatter_signal_buf_intra_node.data_ptr()
    nbytes_per_scatter_signal = scatter_signal_buf_intra_node.dtype.itemsize

    # send input_intra_node[remote_rank * M_per_rank : (remote_rank + 1) * M_per_rank] on the current rank to
    # input_intra_node[rank * M_per_rank : (rank + 1)] on the remote rank.
    with torch.cuda.stream(stream):
        for i in range(0, local_world_size):
            # Rank level swizzle: Each rank perform scatter start from the next rank of the current.
            # In this way, the send/recv communication volume of each rank is balanced.
            remote_local_rank = (local_rank + i + 1) % local_world_size
            if overlap_with_gemm:
                # wait for the corresponding barrier to be set to 1, indicate the corresponding GEMM tile computation is complete.
                wait_eq(signal_base_ptr + nbytes_per_scatter_signal * remote_local_rank, 1,  # signal
                        stream, True)
            remote_buf_ptr = scatter_bufs_intra_node[remote_local_rank].data_ptr() + remote_offset
            local_buf_ptr = local_buf_base_ptr + remote_local_rank * nbytes_per_rank
            # use copy engine to perform scatter
            (err, ) = cudart.cudaMemcpyAsync(
                remote_buf_ptr,
                local_buf_ptr,
                nbytes_per_rank,
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                stream.cuda_stream,
            )
            CUDA_CHECK(err)


def reducer_scatter_for_each_node(input, stream, ctx: ReduceScatter2DContext):
    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    M, N = input.shape
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    rs_per_node_buf = ctx.rs_per_node_buf
    p2p_buf = ctx.p2p_buf
    # For `target_node_id` in the range [0, nnodes - 1], perform an intra-node reduce-scatter on the
    # input[target_node_id * M_per_node: (target_node_id + 1) * M_per_node]. Then send the result via
    # P2P to the same local rank on the node 'target_node_id'.
    with torch.cuda.stream(stream):
        for n in range(0, nnodes):
            """
            Node-level swizzle: Each node perform intra-node reduce-scatter start from the next node of the current.
            In this way, the P2P communication volume of each node is balanced, and the last intra-node reduce-scatter operation
            is performed on the current node without the need for inter-node communication.

            For time start from 0 to nnode - 1, the communication order between nodes:
                time 0: 0->1, 1->2, 2->3, 3->0
                time 1: 0->2, 1->3, 2->0, 3->1
                time 2: 0->3, 1->0, 2->1, 3->2
                time 3: 0->0, 1->1, 2->2, 3->3
            """
            cur_node_id = (node_id + n + 1) % nnodes
            input_intra_node = input[cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]
            scatter_bufs_intra_node, scatter_signal_buf_intra_node = ctx.get_scatter_bufs_and_signal_for_each_node(
                input, cur_node_id)
            # step1: intra node reduce-scatter
            # step1-1: intra node scatter, the corresponding data has been computed by GEMM.
            intra_node_scatter(input_intra_node, scatter_bufs_intra_node, scatter_signal_buf_intra_node, local_rank,
                               stream)

            rs_buf_cur_node = rs_per_node_buf[M_per_rank * cur_node_id:(cur_node_id + 1) * M_per_rank]
            # step1-2: perform barrier_all, wait for all peers within the node to complete the scatter operation
            barrier_all_on_stream(stream, is_intra_node=True, barrier_all_buf=ctx.sync_buf,
                                  local_world_size=local_world_size)
            reduction_stream.wait_stream(stream)
            # step1-3: perform reduction operation to get the result of the intra-node reduce-scatter.
            ring_reduce(scatter_bufs_intra_node[local_rank], rs_buf_cur_node, local_rank, local_world_size,
                        reduction_stream, num_sms=-1 if n == nnodes - 1 else num_reduction_sms)

            # step2: inter node p2p, send result to the same local rank on the node `(n + 1 + node_id) % nnodes`.
            if nnodes > 1:
                with torch.cuda.stream(reduction_stream):
                    if n == nnodes - 1:
                        p2p_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)].copy_(
                            rs_per_node_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)])
                    else:
                        grid = lambda META: (ctx.num_p2p_sms, )
                        kernel_inter_node_p2p_for_same_local_rank[grid](
                            n,
                            local_world_size,
                            M_per_rank,
                            N,
                            rs_per_node_buf,
                            p2p_buf,
                            input.dtype.itemsize,
                            num_warps=16,
                        )

    stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return rs_per_node_buf[:M_per_rank * nnodes]
    return p2p_buf[:M_per_rank * nnodes]


def reduce_scatter_multi_node(input, stream, ctx: ReduceScatter2DContext):
    """
    A hierarchical reduce-scatter implementation that overlaps the intra-node scatter
    with the local reduce and the inter-node p2p(after reduce). It also provides a rank-wise
    signal and supports overlap with gemm.
    """
    M, N = input.shape
    M_per_rank = M // ctx.world_size
    ctx.p2p_stream.wait_stream(stream)
    """
    step1:
        Leveraging the characteristics of reduce-scatter, we first partition the input data according to the target nodes for communication.
        For the data send to each node, we perform an intra-node reduce-scatter operation within the current node.
        Finally, we use P2P communication to send the data to the same local rank on the target node.
        This can reduce the inter-node communication volume by a factor of local_world_size.
    """
    rs_resutl_per_node = reducer_scatter_for_each_node(input, stream, ctx)
    barrier_all_on_stream(stream)
    output = torch.empty((M_per_rank, N), dtype=input.dtype, device=input.device)

    # Step 2: After receiving data sent via P2P from all nodes, perform a reduction to get the final result.
    ring_reduce(rs_resutl_per_node, output, ctx.node_id, ctx.nnodes, stream)
    return output


def reduce_scatter_2d_op(input, ctx: ReduceScatter2DContext):
    reduction_stream = ctx.reduction_stream
    M, N = input.shape
    assert input.dtype == ctx.dtype
    assert ctx.max_M >= M and ctx.N == N
    assert M % ctx.world_size == 0

    current_stream = torch.cuda.current_stream()
    reduction_stream.wait_stream(current_stream)
    # Wait for the completion of the previous iteration.
    barrier_all_on_stream(current_stream)

    # perform reduce-scatter
    output = reduce_scatter_multi_node(input, current_stream, ctx)

    # Reset the barriers for the next iteration.
    ctx.reset_barriers()
    return output


"""
The 'kernel_gemm_rs_producer_persistent' kernel is almost identical to a regular Triton GEMM kernel, with only two minor differences:
1. The computation order of tiles is swizzled according to the rank.
2. There is an additional operation to set the barrier in the epilogue.
"""


@triton.jit
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
            """
            Difference 1: Based on the m dimension, calculate the target rank where the output data will be scattered to.
            Then, perform a swizzle operation according to the local rank and the node_id of the current GPU.
            This ensures that during communication, the data sent and received by each rank is balanced, maximizing the utilization of all communication bandwidth.
            """
            # original rank and node_id
            m_node_id = m_rank // local_world_size
            m_local_rank = m_rank % local_world_size
            swizzle_m_node_id = (m_node_id + node_id + 1) % nnodes
            swizzle_m_local_rank = (m_local_rank + rank + 1) % local_world_size
            swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank

            # perform swizzle
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
            """
            Difference 2: # Compute the rank that the current tile will be sent
            If the current tile is the last one to complete for that rank, set its barrier to 1 (indicating the ready state).
            the reduce-scatter on another stream waits for the barrier to be ready and then performs the scatter operation.
            """
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
                # `tiled_m_size * tiled_n` represents the total number of tiles within the rank.
                val = tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")
                # If `val` is equal to `tiled_m_size * tiled_n - 1`, it means the current tile
                # is the last one to be completed for this rank.
                if val == tiled_m_size * tiled_n - 1:
                    dl.notify(barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu")
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


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

    # Launch the Triton GEMM kernel. Once the kernel has completed the computation of the output tiles
    # that send to a specific rank, will set the corresponding barrier to 1.
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


def gemm_rs_multi_node_persistent_op(input, weight, ctx: GEMMReduceScatterTensorParallelContext):
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    num_gemm_sms = ctx.num_gemm_sms

    orig_M = input.shape[0]
    orig_M_per_rank = orig_M // world_size
    """
        Pad the `input` tensor so that all data within a output tile is associated with a single rank.
        It enables the scatter operation to wait and send data to only one rank at a time,
        which significantly enhances communication efficiency and simplifies control logic.
    """
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
    """
    Perform the GEMM operation. The output tiles sent to different ranks each correspond to a barrier.
    If the computation of the corresponding tiles is completed, set the barrier to 1.
    """
    gemm_rs_producer_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size, local_world_size,
                                num_gemm_sms, current_stream, BLOCK_SIZE_M=ctx.BLOCK_M, BLOCK_SIZE_N=ctx.BLOCK_N,
                                BLOCK_SIZE_K=ctx.BLOCK_K, GROUP_SIZE_M=ctx.GROUP_M, STAGES=ctx.stages)
    """
    Perform reduce-scatter on the rs_stream, overlapping with the gemm operation.
    This implementation is based on tile level barriers, enabling the overlap of
    communication and computation. Once the data corresponding to each barrier is
    computed(barrier[wait_rank] = 1), the corresponding reduce-scatterwill be perform.
    """
    with torch.cuda.stream(rs_stream):
        output = reduce_scatter_2d_op(gemm_out, ctx.rs_ctx)
    current_stream.wait_stream(rs_stream)

    return output[:orig_M_per_rank]


def gemm_rs_multi_node(a, b, ctx):
    """GEMM Reduce-Scatter for Multi-Node

    computes local GEMM (a x b) to generate partial results, followed by `reduce_scatter` to produce c

    Args:
        a (torch.Tensor<bfloat16/float16>): local matmul A matrix. shape: [M, local_K]
        b (torch.Tensor<bfloat16/float16>): local matmul B matrix. shape: [N, local_K]
        ctx(GEMMReduceScatterTensorParallelContext): context

    Returns:
        c (torch.Tensor<bfloat16/float16>): local matmul C matrix. shape: [M // world_size, N]
    """
    c = gemm_rs_multi_node_persistent_op(a, b, ctx)
    return c


def torch_gemm_rs(
    input: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [N, local_K]
    TP_GROUP,
):
    M, local_K = input.shape
    N = weight.shape[0]
    output = torch.matmul(input, weight.T)
    rs_output = torch.empty((M // WORLD_SIZE, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=TP_GROUP)
    return rs_output


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 6e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")

    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)
    random.seed(42)

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_K = args.K // TP_GROUP.size()

    scale = TP_GROUP.rank() + 1

    def _make_data(M):
        data_config = [((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
                       ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
                       ]
        generator = generate_data(data_config)
        input, weight = next(generator)
        return input, weight

    rs_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
    dist_gemm_rs_ctx = create_gemm_rs_context(args.M, args.N, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, output_dtype,
                                              rs_stream)
    ctx = get_torch_prof_ctx(args.profile)
    input, weight = _make_data(args.M)
    with ctx:
        torch_output, torch_perf = perf_func(partial(torch_gemm_rs, input, weight, TP_GROUP), iters=100,
                                             warmup_iters=20)

        pynvshmem.nvshmem_barrier_all()
        torch.cuda.synchronize()

        dist_triton_output, dist_triton_perf = perf_func(partial(gemm_rs_multi_node, input, weight, dist_gemm_rs_ctx),
                                                         iters=100, warmup_iters=20)

    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
    torch.testing.assert_close(torch_output, dist_triton_output, atol=atol, rtol=rtol)
    torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", dist_triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    torch.distributed.destroy_process_group()
