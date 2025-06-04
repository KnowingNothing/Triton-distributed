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
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid, ntid, __syncthreads, multimem_st_b64, load_v2_b64, st
import torch
import torch.distributed
from triton_dist import pynvshmem
import os
import datetime

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def test_nvshmem_basic():

    @triton.jit
    def _nvshmem_basic(output):
        thread_idx = tid(axis=0)
        if thread_idx == 0:
            st(output, libshmem_device.my_pe())
            output += 1
            st(output, libshmem_device.team_my_pe(libshmem_device.NVSHMEM_TEAM_WORLD))
            output += 1
            st(output, libshmem_device.team_my_pe(libshmem_device.NVSHMEMX_TEAM_NODE))
            output += 1

            st(output, libshmem_device.n_pes())
            output += 1
            st(output, libshmem_device.team_n_pes(libshmem_device.NVSHMEM_TEAM_WORLD))
            output += 1
            st(output, libshmem_device.team_n_pes(libshmem_device.NVSHMEMX_TEAM_NODE))

    print("nvshmem basic start...")
    output = pynvshmem.nvshmem_create_tensor((6, ), torch.int32)
    _nvshmem_basic[(1, )](output)
    pynvshmem.nvshmem_barrier_all()
    try:
        torch.testing.assert_close(
            output,
            torch.tensor([RANK, RANK, LOCAL_RANK, WORLD_SIZE, WORLD_SIZE, LOCAL_WORLD_SIZE], dtype=torch.int32,
                         device="cuda")), output
    except Exception as e:
        print(" ❌ nvshmem basic failed")
        raise (e)
    else:
        print("✅ nvshmem basic pass")


def test_nvshmemx_getmem_with_scope(N, dtype: torch.dtype = torch.int8):

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

    t = pynvshmem.nvshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemx_getmem_block",
                ("warp", False): "nvshmemx_getmem_warp",
                ("thread", False): "nvshmem_getmem",
                ("block", True): "nvshmemx_getmem_nbi_block",
                ("warp", True): "nvshmemx_getmem_nbi_warp",
                ("thread", True): "nvshmem_getmem_nbi",
            }[(scope, nbi)]
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
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                raise (e)
            else:
                print(f"✅ {api} pass")


def test_nvshmemx_putmem_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemx_putmem(
        ptr,
        elems_per_rank,
        scope: tl.constexpr,
        nbi: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
    ):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_nbi_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_nbi_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem_nbi(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = pynvshmem.nvshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemx_putmem_block",
                ("warp", False): "nvshmemx_putmem_warp",
                ("thread", False): "nvshmem_putmem",
                ("block", True): "nvshmemx_putmem_nbi_block",
                ("warp", True): "nvshmemx_putmem_nbi_warp",
                ("thread", True): "nvshmem_putmem_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemx_putmem[(WORLD_SIZE, )](
                t,
                N // WORLD_SIZE,
                scope,
                nbi,
                ELEM_SIZE=dtype.itemsize,
                num_warps=1 if scope == "warp" else 4,
            )
            pynvshmem.nvshmem_barrier_all()
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                raise (e)
            else:
                print(f"✅ {api} pass")


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
    t.fill_(0)
    pynvshmem.nvshmem_barrier_all()
    _pingpong[(1, )](t, 100, num_warps=1)
    pynvshmem.nvshmem_barrier_all()
    if pynvshmem.nvshmem_my_pe() == 0:
        try:
            torch.testing.assert_close(t.to(torch.int32), torch.ones([1], dtype=torch.int32, device="cuda") * 100)
        except Exception as e:
            print("❌ nvshmemx_signal with pingpong failed")
            raise e
        else:
            print("✅ nvshmemx_signal with pingpong pass")


def test_nvshmemx_putmem_signal_with_scope(N, dtype: torch.dtype = torch.int8):

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

    t = pynvshmem.nvshmem_create_tensor((N, ), dtype)
    signal = pynvshmem.nvshmem_create_tensor((WORLD_SIZE, ), torch.uint64)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemx_putmem_signal_block",
                ("warp", False): "nvshmemx_putmem_signal_warp",
                ("thread", False): "nvshmem_putmem_signal",
                ("block", True): "nvshmemx_putmem_signal_nbi_block",
                ("warp", True): "nvshmemx_putmem_signal_nbi_warp",
                ("thread", True): "nvshmem_putmem_signal_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            signal.fill_(0)
            signal[RANK].fill_(1)
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
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
                torch.testing.assert_close(signal, torch.ones((WORLD_SIZE, ), dtype=torch.uint64, device="cuda"))
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                print(signal)
                raise (e)
            else:
                print(f"✅ {api} pass")


def test_nvshmem_barrier_sync_quiet_fence():
    """only test runs, no result checked"""

    @triton.jit
    def _nvshmem_barrier_sync_quiet_fence():
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid == 0:
            libshmem_device.barrier_all_block()
            libshmem_device.sync_all_block()

            if thread_idx / 32 == 0:
                libshmem_device.barrier_all_warp()
                libshmem_device.sync_all_warp()

            if thread_idx == 0:
                libshmem_device.barrier_all()
                libshmem_device.sync_all()

        libshmem_device.quiet()
        libshmem_device.fence()

    @triton.jit
    def _nvshmem_barrier_sync_quiet_fence_with_team(team):
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid == 0:
            libshmem_device.barrier_block(team)
            libshmem_device.team_sync_block(team)

            if thread_idx / 32 == 0:
                libshmem_device.barrier_warp(team)
                libshmem_device.team_sync_warp(team)

            if thread_idx == 0:
                libshmem_device.barrier(team)

    print("test nvshmem_barrier/nvshmem_sync/nvshmem_quiet/nvshmem_fence all in one...")
    _nvshmem_barrier_sync_quiet_fence[(1, )](num_warps=4)
    torch.cuda.synchronize()
    print("✅ nvshmem_barrier_all/nvshmem_sync/nvshmem_quiet/nvshmem_fence pased...")
    _nvshmem_barrier_sync_quiet_fence_with_team[(1, )](pynvshmem.NVSHMEMX_TEAM_NODE, num_warps=4)
    torch.cuda.synchronize()
    print("✅ nvshmem_barrier/nvshmemx_team_sync pased...")


def test_nvshmem_broadcast(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmem_broadcast(dst, src, nbytes, scope: tl.constexpr):
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if scope == "block":
            libshmem_device.broadcast_block(libshmem_device.NVSHMEM_TEAM_WORLD, dst, src, nbytes, 0)
        if scope == "warp":
            if wid == 0:
                libshmem_device.broadcast_warp(libshmem_device.NVSHMEM_TEAM_WORLD, dst, src, nbytes, 0)
                __syncthreads()
        if scope == "thread":
            if thread_idx == 0:
                libshmem_device.broadcast(libshmem_device.NVSHMEM_TEAM_WORLD, dst, src, nbytes, 0)
            __syncthreads()

    src = pynvshmem.nvshmem_create_tensor((N, ), dtype)
    dst = pynvshmem.nvshmem_create_tensor((N, ), dtype)
    for scope in ["block", "warp", "thread"]:
        api = {
            "block": "nvshmemx_broadcast_block",
            "warp": "nvshmemx_broadcast_warp",
            "thread": "nvshmem_broadcast",
        }[scope]
        print(f"running {api}...")
        src.fill_(RANK + 1)
        dst.fill_(-1)
        pynvshmem.nvshmem_barrier_all()
        _nvshmem_broadcast[(1, )](
            dst,
            src,
            src.nbytes,
            scope,
            num_warps=4,
        )
        pynvshmem.nvshmem_barrier_all()
        t_expected = torch.ones_like(dst)
        try:
            torch.testing.assert_close(dst, t_expected)
        except Exception as e:
            print(f" ❌ {api} failed")
            print(dst)
            raise (e)
        else:
            print(f"✅ {api} pass")


def test_nvshmem_fcollect(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmem_fcollect(dst, src, nbytes, scope: tl.constexpr):
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if scope == "block":
            libshmem_device.fcollect_block(
                libshmem_device.NVSHMEM_TEAM_WORLD,
                dst,
                src,
                nbytes,
            )
        if scope == "warp":
            if wid == 0:
                libshmem_device.fcollect_warp(
                    libshmem_device.NVSHMEM_TEAM_WORLD,
                    dst,
                    src,
                    nbytes,
                )
            __syncthreads()
        if scope == "thread":
            if thread_idx == 0:
                libshmem_device.fcollect(
                    libshmem_device.NVSHMEM_TEAM_WORLD,
                    dst,
                    src,
                    nbytes,
                )
            __syncthreads()

    src = pynvshmem.nvshmem_create_tensor((N, ), dtype)
    dst = pynvshmem.nvshmem_create_tensor((N * WORLD_SIZE, ), dtype)
    for scope in ["block", "warp", "thread"]:
        api = {
            "block": "nvshmemx_fcollect_block",
            "warp": "nvshmemx_fcollect_warp",
            "thread": "nvshmem_fcollect",
        }[scope]
        print(f"running {api}...")
        src.fill_(RANK + 1)
        dst.fill_(-1)
        pynvshmem.nvshmem_barrier_all()
        _nvshmem_fcollect[(1, )](
            dst,
            src,
            src.nbytes // src.itemsize,
            scope,
            num_warps=4,
        )
        pynvshmem.nvshmem_barrier_all()
        torch.cuda.synchronize()
        t_expected = (torch.ones_like(dst).reshape(
            (WORLD_SIZE, -1)) * torch.arange(1, 1 + WORLD_SIZE, device="cuda").to(dtype)[:, None]).flatten()
        try:
            torch.testing.assert_close(dst, t_expected)
        except Exception as e:
            print(f" ❌ {api} failed")
            print(dst)
            raise (e)
        else:
            print(f"✅ {api} pass")


def _if_nvls_supported():
    """  NOTE: Hopper + NVSHMEM_DISABLE_CUDA_VMM=0 does not guarantee that NVLS is supported. for test only """
    major, _ = torch.cuda.get_device_capability()
    return major >= 9 and os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0"


def test_nvshmem_mc_ptr(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmem_multimem_st(ptr, nbytes):
        thread_idx = tid(axis=0)
        block_dim = ntid(axis=0)
        pid = tl.program_id(0)
        npid = tl.num_programs(0)
        ptr = tl.cast(ptr, tl.pointer_type(tl.int8))
        mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, ptr)
        for n in range(thread_idx + block_dim * pid, nbytes // 16, block_dim * npid):
            val0, val1 = load_v2_b64(ptr + n * 16)
            multimem_st_b64(tl.cast(mc_ptr, tl.pointer_type(tl.int8)) + n * 16, val0)
            multimem_st_b64(mc_ptr + n * 16 + 8, val1)

    t: torch.Tensor = pynvshmem.nvshmem_create_tensor((N, ), dtype)
    t.fill_(1 + RANK)
    pynvshmem.nvshmem_barrier_all()
    if not _if_nvls_supported():
        print("not support MultiCast memory. only works on NVLS hardware and NVSHMEM_DISABLE_CUDA_VMM=0")
        return

    if RANK == 0:
        _nvshmem_multimem_st[(4, )](t, t.nbytes, num_warps=4)
    pynvshmem.nvshmem_barrier_all()
    try:
        torch.testing.assert_close(t, torch.ones_like(t))
    except Exception as e:
        print(f"t: {t}")
        raise e
    else:
        print("_nvshmem_multimem_st done")


def test_nvshmemi_putmem_rma(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemi_putmem_rma_kernel(
        ptr,
        elems_per_rank,
        scope: tl.constexpr,
        nbi: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
    ):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_rma_nbi_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_rma_nbi_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem_rma_nbi(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_rma_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_rma_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem_rma(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = pynvshmem.nvshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemi_transfer_rma_put_block",
                ("warp", False): "nvshmemi_transfer_rma_put_warp",
                ("thread", False): "nvshmemi_transfer_rma",
                ("block", True): "nvshmemi_transfer_rma_put_nbi_block",
                ("warp", True): "nvshmemi_transfer_rma_put_nbi_warp",
                ("thread", True): "nvshmemi_transfer_rma_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemi_putmem_rma_kernel[(WORLD_SIZE, )](
                t,
                N // WORLD_SIZE,
                scope,
                nbi,
                ELEM_SIZE=dtype.itemsize,
                num_warps=1 if scope == "warp" else 4,
            )
            pynvshmem.nvshmem_barrier_all()
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                raise (e)
            else:
                print(f"✅ {api} pass")


def test_nvshmemi_putmem_rma_signal_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemi_putmem_rma_signal_kernel(ptr, signal, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_signal_rma_nbi_block(
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
                        libshmem_device.putmem_signal_rma_nbi_warp(
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
                        libshmem_device.putmem_signal_rma_nbi(
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
                    libshmem_device.putmem_signal_rma_block(
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
                        libshmem_device.putmem_signal_rma_warp(
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
                        libshmem_device.putmem_signal_rma(
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

    t = pynvshmem.nvshmem_create_tensor((N, ), dtype)
    signal = pynvshmem.nvshmem_create_tensor((WORLD_SIZE, ), torch.uint64)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemi_transfer_put_signal_block",
                ("warp", False): "nvshmemi_transfer_put_signal_warp",
                ("thread", False): "nvshmemi_transfer_put_signal",
                ("block", True): "nvshmemi_transfer_put_signal_nbi_block",
                ("warp", True): "nvshmemi_transfer_put_signal_nbi_warp",
                ("thread", True): "nvshmemi_transfer_put_signal_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            signal.fill_(0)
            signal[RANK].fill_(1)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemi_putmem_rma_signal_kernel[(WORLD_SIZE, )](
                t,
                signal,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=4,
            )
            pynvshmem.nvshmem_barrier_all()
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
                torch.testing.assert_close(signal, torch.ones((WORLD_SIZE, ), dtype=torch.uint64, device="cuda"))
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                print(signal)
                raise (e)
            else:
                print(f"✅ {api} pass")


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

    test_nvshmem_basic()
    test_nvshmemx_getmem_with_scope(31 * WORLD_SIZE, torch.int8)
    test_nvshmemx_putmem_with_scope(16 * WORLD_SIZE, torch.int8)
    test_nvshmemx_putmem_signal_with_scope(20 * WORLD_SIZE, torch.int8)
    test_nvshmem_signal()
    test_nvshmem_barrier_sync_quiet_fence()
    test_nvshmem_broadcast(32 * WORLD_SIZE, torch.int8)
    # some ranks hangs. don't know why
    # test_nvshmem_fcollect(1024, torch.int8)
    test_nvshmem_mc_ptr(1024, torch.int16)

    test_nvshmemi_putmem_rma(16 * WORLD_SIZE, torch.int8)
    test_nvshmemi_putmem_rma_signal_with_scope(16 * WORLD_SIZE, torch.int8)

    torch.distributed.barrier(TP_GROUP)
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()
