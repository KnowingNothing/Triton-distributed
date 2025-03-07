################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
import triton
import triton.language as tl
from triton.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid, __syncthreads
import torch
import torch.distributed
import pynvshmem
import os
import datetime

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def test_nvshmemx_getmem_with_scope():

    @triton.jit
    def _nvshmemx_getmem(ptr, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.getmem_nbi_block(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.getmem_nbi_warp(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < bytes_per_rank:
                        libshmem_device.getmem_nbi(
                            ptr + pid * bytes_per_rank + thread_idx,
                            ptr + pid * bytes_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.getmem_block(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.getmem_warp(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < bytes_per_rank:
                        libshmem_device.getmem(
                            ptr + pid * bytes_per_rank + thread_idx,
                            ptr + pid * bytes_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = pynvshmem.nvshmem_create_tensor((1024, ), torch.int8)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {("block", False): "nvshmemx_getmem_block", ("warp", False): "nvshmemx_getmem_warp",
                   ("thread", False): "nvshmem_getmem", ("block", True): "nvshmemx_getmem_nbi_block", ("warp", True):
                   "nvshmemx_getmem_nbi_warp", ("thread", True): "nvshmem_getmem_nbi"}[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemx_getmem[(WORLD_SIZE, )](
                t,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=1 if scope == "warp" else 4,
            )
            pynvshmem.nvshmem_barrier_all()
            print(t.reshape(8, -1))


def test_nvshmemx_putmem_with_scope():

    @triton.jit
    def _nvshmemx_putmem(ptr, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_nbi_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_nbi_warp(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < bytes_per_rank:
                        libshmem_device.putmem_nbi(
                            ptr + mype * bytes_per_rank + thread_idx,
                            ptr + mype * bytes_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_warp(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < bytes_per_rank:
                        libshmem_device.putmem(
                            ptr + mype * bytes_per_rank + thread_idx,
                            ptr + mype * bytes_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = pynvshmem.nvshmem_create_tensor((1024, ), torch.int8)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {("block", False): "nvshmemx_putmem_block", ("warp", False): "nvshmemx_putmem_warp",
                   ("thread", False): "nvshmem_putmem", ("block", True): "nvshmemx_putmem_nbi_block", ("warp", True):
                   "nvshmemx_putmem_nbi_warp", ("thread", True): "nvshmem_putmem_nbi"}[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemx_putmem[(WORLD_SIZE, )](
                t,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=1 if scope == "warp" else 4,
            )
            pynvshmem.nvshmem_barrier_all()
            print(t.reshape(8, -1))


def test_nvshmem_signal():

    @triton.jit
    def _pingpong(t, iters):
        # pingpong for rank 0-1, 2-3, ...
        mype = libshmem_device.my_pe()
        thread_idx = tid(axis=0)
        if thread_idx == 0:
            for n in range(iters):
                if mype == 0:
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        1,
                    )
                elif mype == 1:
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        0,
                    )
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
        __syncthreads()

    print("test nvshmemx_signal with pingpong...")
    t = pynvshmem.nvshmem_create_tensor((1, ), torch.uint64)
    _pingpong[(1, )](t, 100, num_warps=1)
    pynvshmem.nvshmem_barrier_all()
    print(t)
    torch.cuda.synchronize()


def test_nvshmemx_putmem_signal_with_scope():

    @triton.jit
    def _nvshmemx_putmem_signal(ptr, signal, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_signal_nbi_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        signal + mype,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        pid,
                    )
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_nbi_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                elif scope == "thread":
                    if thread_idx == 0:
                        libshmem_device.putmem_signal_nbi(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_signal_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        signal + mype,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        pid,
                    )
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                elif scope == "thread":
                    if thread_idx == 0:
                        libshmem_device.putmem_signal(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = pynvshmem.nvshmem_create_tensor((1024, ), torch.int8)
    signal = pynvshmem.nvshmem_create_tensor((WORLD_SIZE, ), torch.uint64)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {("block", False): "nvshmemx_putmem_signal_block", ("warp", False): "nvshmemx_putmem_signal_warp",
                   ("thread", False): "nvshmem_putmem_signal", ("block", True): "nvshmemx_putmem_signal_nbi_block",
                   ("warp", True): "nvshmemx_putmem_signal_nbi_warp", ("thread", True):
                   "nvshmem_putmem_signal_nbi"}[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            signal.fill_(0)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemx_putmem_signal[(WORLD_SIZE, )](
                t,
                signal,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=4,
            )
            pynvshmem.nvshmem_barrier_all()
            print(t.reshape(8, -1))
            print(signal)


def test_nvshmem_barrier_sync_quiet_fence():
    """ only test runs, no result checked
    """

    @triton.jit
    def _nvshmem_barrier_sync_quiet_fence():
        libshmem_device.barrier_all()
        libshmem_device.sync_all()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid == 0:
            libshmem_device.barrier_all_block()
            libshmem_device.sync_all_block()

            if thread_idx / 32 == 0:
                libshmem_device.barrier_all_warp()
                libshmem_device.sync_all_warp()

        libshmem_device.quiet()
        libshmem_device.fence()

    print("test nvshmem_barrier/nvshmem_sync/nvshmem_quiet/nvshmem_fence all in one...")
    _nvshmem_barrier_sync_quiet_fence[(1, )](num_warps=4)


if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    test_nvshmemx_getmem_with_scope()
    test_nvshmemx_putmem_with_scope()
    test_nvshmemx_putmem_signal_with_scope()
    test_nvshmem_signal()
    test_nvshmem_barrier_sync_quiet_fence()
