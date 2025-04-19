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
import triton
import torch
import triton.language as tl
import triton.distributed.language as dl
from triton.language.extra import libshmem_device
from triton.distributed.utils import (
    CUDA_CHECK, )
from cuda import cuda
from triton.language.extra.cuda.language_extra import (
    tid,
    __syncthreads,
)

import pynvshmem


@tl.core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope: tl.constexpr,
    semantic: tl.constexpr,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"atom.{semantic.value}.{scope.value}.global.cas.b32 $0, [$1], $2, $3;",
        constraints=("=r,l,r,r"),
        args=[
            ptr,
            value,
            target_value,
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def barrier_all(rank, num_ranks, comm_buf_ptr):
    thread_idx = tid(axis=0)
    sm_id = tl.program_id(axis=0)
    if thread_idx < num_ranks:
        remote_ptr = libshmem_device.remote_ptr(comm_buf_ptr + sm_id * num_ranks + rank,
                                                thread_idx.to(tl.int32)).to(tl.pointer_type(tl.int32))
        while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
            pass
        while (atomic_cas(comm_buf_ptr + sm_id * num_ranks + thread_idx, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


@triton.jit
def barrier_all_intra_node(local_world_size, comm_buf_ptr):
    """
    This function is used for intra-node barrier synchronization.
    It is based on the Compare-And-Swap(CAS) operation to ensure that
    all GPUs within the current node reach the barrier.
    """
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


def wait_eq(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
    else:
        (err, ) = cuda.cuStreamWaitValue64(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
    CUDA_CHECK(err)


def set_signal(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    else:
        (err, ) = cuda.cuStreamWriteValue64(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    CUDA_CHECK(err)
