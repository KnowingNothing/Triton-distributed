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
import pynvshmem
import torch
import torch.distributed

import triton
import triton.language as tl
from triton.language.extra import libshmem_device

from triton.language.extra.cuda.language_extra import (
    __syncthreads,
    tid,
    ntid,
    load_v4_u32,
    store_v2_u32,
    atomic_add,
    ld_u32_acquire,
)


@triton.jit
def _forward_pull_kernel(symm_ptr, bytes_per_rank, symm_flag, world_size, rank, signal_target):
    pid = tl.program_id(0)
    thread_idx = tid(0)
    if pid == rank:
        if thread_idx != rank and thread_idx < world_size:
            libshmem_device.signal_op(
                symm_flag + rank,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                thread_idx,
            )
        __syncthreads()
    else:
        peer = pid
        if thread_idx == 0:
            libshmem_device.signal_wait_until(symm_flag + peer, libshmem_device.NVSHMEM_CMP_GE, signal_target)
        __syncthreads()
        libshmem_device.getmem_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + peer * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + peer * bytes_per_rank,
            bytes_per_rank,
            peer,
        )


@triton.jit
def _forward_push_2d_kernel(symm_ptr, bytes_per_rank, symm_flag, nnodes, world_size, rank, signal_target):
    local_world_size = world_size // nnodes
    local_rank = rank % local_world_size
    # tl.static_assert(world_size % nnodes == 0)
    nid = rank // local_world_size
    rank_base = nid * local_world_size

    pid = tl.program_id(0)
    thread_idx = tid(0)
    if pid == local_rank:  # remote push
        for n in range(nnodes - 1):
            sid = (n + 1 + nid) % nnodes
            peer = sid * local_world_size + local_rank
            segment = rank
            libshmem_device.putmem_signal_nbi_block(
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )  # write and tell peer remote that remote copy is done
        if thread_idx < world_size and thread_idx != rank:
            libshmem_device.signal_wait_until(
                symm_flag + thread_idx,
                libshmem_device.NVSHMEM_CMP_GE,
                signal_target,
            )
        __syncthreads()
    else:  # local push
        peer = rank_base + pid
        for n in range(nnodes):
            sid = (n + nid) % nnodes
            segment = sid * local_world_size + local_rank
            if n != 0:  # wait for data from other nodes
                if thread_idx == 0:
                    libshmem_device.signal_wait_until(
                        symm_flag + segment,
                        libshmem_device.NVSHMEM_CMP_GE,
                        signal_target,
                    )
                __syncthreads()
            libshmem_device.putmem_signal_block(
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )  # write and tell peer remote that remote copy is done


@triton.jit
def _recv_ll_block(dest_ptr, src_ptr, num_ints, ll_flag):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    num_ints: of the pre-LL-packed num_ints.
    """
    thread_idx = tid(0)
    block_size = ntid(0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    # manual load per vec
    for n in range(thread_idx, num_ints // 2, block_size):
        data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        while flag1 != ll_flag or flag2 != ll_flag:
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        store_v2_u32(dest_ptr + n * 2, data1, data2)


@triton.jit
def _pack_ll_block(dest_ptr, src_ptr, num_ints, ll_flag, BLOCK_SIZE: tl.constexpr):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    nbytes: of the pre-LL-packed bytes.
    BLOCK_SIZE: count by ints, not bytes.
    """
    iters = tl.cdiv(num_ints, BLOCK_SIZE)
    src_ptr = tl.cast(src_ptr, dtype=tl.pi32_t)
    dest_ptr = tl.cast(dest_ptr, dtype=tl.pi32_t)
    for n in range(iters):
        src_offsets = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        src_mask = src_offsets < num_ints
        src = tl.load(src_ptr + src_offsets, mask=src_mask)
        flags = tl.full((BLOCK_SIZE, ), ll_flag, tl.int32)
        dst = tl.interleave(src, flags)
        dest_offset = n * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
        dest_mask = dest_offset < num_ints * 2
        tl.store(dest_ptr + dest_offset, dst, mask=dest_mask)


@triton.jit
def _is_cta_master():
    thread_idx_x = tid(0)
    thread_idx_y = tid(1)
    thread_idx_z = tid(2)
    return (thread_idx_x + thread_idx_y + thread_idx_z) == 0


@triton.jit
def _is_gpu_master():
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)
    return (pid_x + pid_y + pid_z) == 0


@triton.jit
def barrier_on_this_grid(ptr):
    __syncthreads()
    pid_size_x = tl.num_programs(axis=0)
    pid_size_y = tl.num_programs(axis=1)
    pid_size_z = tl.num_programs(axis=2)
    expected = pid_size_x * pid_size_y * pid_size_z
    if _is_cta_master():
        nb = tl.where(
            _is_gpu_master(),
            tl.cast(0x80000000, tl.uint32, bitcast=True) - (expected - 1),
            1,
        )
        old_arrive = atomic_add(ptr, nb, scope="gpu", semantic="release")
    else:
        old_arrive = tl.cast(0, tl.uint32)

    if _is_cta_master():
        current_arrive = ld_u32_acquire(ptr)
        while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
            current_arrive = ld_u32_acquire(ptr, scope=tl.constexpr("gpu"))

    __syncthreads()


@triton.jit
def _forward_push_2d_ll_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer,
    nnodes,
    world_size,
    rank,
    signal_target,
    grid_barrier,
):
    local_world_size = world_size // nnodes
    local_rank = rank % local_world_size
    nid = rank // local_world_size
    rank_base = nid * local_world_size

    pid = tl.program_id(0)
    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4

    ll_buffer_int8 = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if pid == local_rank:  # remote push
        _pack_ll_block(
            ll_buffer_int8 + rank * bytes_per_rank * 2,
            symm_ptr + rank * bytes_per_rank,
            num_ints,
            signal_target,
            2048,
        )  # magic number here
        for n in range(1, nnodes):
            peer_nid_from = (n + nid) % nnodes
            peer_nid_to = (nid - n + nnodes) % nnodes
            peer_from = peer_nid_from * local_world_size + local_rank
            peer_to = peer_nid_to * local_world_size + local_rank
            # tl.device_print("peer_from/to", peer_from, peer_to)
            libshmem_device.putmem_nbi_block(
                ll_buffer_int8 + rank * bytes_per_rank * 2,
                ll_buffer_int8 + rank * bytes_per_rank * 2,
                bytes_per_rank * 2,
                peer_to,
            )  # write and tell peer remote that remote copy is done
            segment = peer_from
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
            barrier_on_this_grid(grid_barrier)

        if thread_idx < world_size and thread_idx % local_world_size != local_rank:
            libshmem_device.signal_wait_until(
                symm_flag + thread_idx,
                libshmem_device.NVSHMEM_CMP_GE,
                signal_target,
            )
        __syncthreads()

    else:  # local push
        peer = rank_base + pid
        for n in range(nnodes):
            peer_nid = (n + nid) % nnodes
            segment = peer_nid * local_world_size + local_rank
            if n != 0:  # wait for remote done
                barrier_on_this_grid(grid_barrier)
            libshmem_device.putmem_signal_block(
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )  # write and tell peer remote that remote copy is done


@triton.jit
def _forward_push_2d_ll_perf_only_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer0,
    symm_ll_buffer1,
    nnodes,
    world_size,
    rank,
    signal_target,
    grid_barrier,
    iters,
):
    for n in range(iters - 1):
        _forward_push_2d_ll_kernel(
            symm_ptr,
            bytes_per_rank,
            symm_flag,
            tl.where(n % 2 == 0, symm_ll_buffer0, symm_ll_buffer1),
            nnodes,
            world_size,
            rank,
            signal_target,
            grid_barrier,
        )
        barrier_on_this_grid(grid_barrier)
        signal_target += 1


class AllGatherOp:

    def __init__(self, nnodes, world_size, rank, max_buffer_size: int = 2 * 32 * 1024 * 1024):
        self.rank = rank
        self.size = world_size
        self.signal = pynvshmem.nvshmem_create_tensor((self.size, ), torch.uint64)
        self.max_buffer_size = max_buffer_size
        self.ll_buffers = [pynvshmem.nvshmem_create_tensor((self.max_buffer_size, ), torch.int8) for _ in range(2)]
        self.signal.zero_()
        self.signal_target = 15  # avoid 1 to constexpr
        self.nnodes = nnodes
        self.grid_barrier = torch.zeros((1, ), dtype=torch.uint32, device="cuda")

    def forward_pull(self, symm_buffer: torch.Tensor):
        self.signal_target += 1
        # print(f"_forward_pull_kernel: cache_key {_forward_pull_kernel.cache_key} hash: {_forward_pull_kernel.hash}")
        return _forward_pull_kernel[(self.size, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            self.signal,
            self.size,
            self.rank,
            self.signal_target,
            num_warps=32,
        )

    def forward_push_2d(self, symm_buffer: torch.Tensor):
        self.signal_target += 1
        _forward_push_2d_kernel[(self.size // self.nnodes, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            self.signal,
            self.nnodes,
            self.size,
            self.rank,
            self.signal_target,
            num_warps=32,
        )
        return symm_buffer

    def forward_push_2d_ll(self, symm_buffer: torch.Tensor):
        assert symm_buffer.nbytes * 2 < self.max_buffer_size
        self.signal_target += 1
        ll_buffer = self.ll_buffers[self.signal_target % 2]
        _forward_push_2d_ll_kernel[(self.size // self.nnodes, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            self.signal,
            ll_buffer,
            self.nnodes,
            self.size,
            self.rank,
            self.signal_target,
            self.grid_barrier,
            num_warps=32,
        )

        return symm_buffer

    def _forward_push_2d_ll_perf_only(self, symm_buffer: torch.Tensor, iters):
        assert symm_buffer.nbytes * 2 < self.max_buffer_size
        _forward_push_2d_ll_perf_only_kernel[(self.size // self.nnodes, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            self.signal,
            self.ll_buffers[0],
            self.ll_buffers[1],
            self.nnodes,
            self.size,
            self.rank,
            self.signal_target,
            self.grid_barrier,
            iters,
            num_warps=32,
        )
        self.signal_target += iters

        return symm_buffer
