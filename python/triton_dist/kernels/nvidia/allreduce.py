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
from enum import IntEnum
import math
import os
import warnings
import torch
import torch.distributed
import triton
from torch import Tensor
import triton.language as tl
from typing import List, Union
from triton_dist import pynvshmem
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid, __syncthreads, atomic_cas, ntid, multimem_st_b64, load_v2_b64, multimem_ld_reduce_p_v4, multimem_st_v4_p_b32
from triton_dist.kernels.nvidia.common_ops import (barrier_all_on_stream, load_128, add_v8_bf16, barrier_on_this_grid,
                                                   get_flat_tid)
from triton_dist.kernels.nvidia.reduce_scatter import kernel_ring_reduce_tma
import functools
import dataclasses

SIGNAL_TARGET = 1
SIGNAL_DTYPE = torch.uint64
MAX_DOUBLE_TREE_BLOCKS = 1024  # for double tree op


class AllReduceMethod(IntEnum):
    OneShot_TMA = 1
    OneShot = 2
    DoubleTree = 3
    TwoShot_Multicast = 4
    OneShot_Ld_Reduce = 5
    TwoShot = 6


STR_TO_METHOD = {
    "one_shot_tma": AllReduceMethod.OneShot_TMA,
    "one_shot_non_tma": AllReduceMethod.OneShot,
    "double_tree": AllReduceMethod.DoubleTree,
    "two_shot_multicast": AllReduceMethod.TwoShot_Multicast,
    "one_shot_ld_reduce": AllReduceMethod.OneShot_Ld_Reduce,
    "two_shot_ld_reduce": AllReduceMethod.TwoShot,
}


def str_to_method(method_str: str) -> AllReduceMethod:
    if method_str not in STR_TO_METHOD:
        raise ValueError(f"Invalid method name {method_str}. Supported methods: {list(STR_TO_METHOD.keys())}")
    return STR_TO_METHOD[method_str]


@dataclasses.dataclass
class AllReduceContext:
    M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype

    # comm buffer
    scatter_bufs: Tensor
    buf_M: int
    recv_buffer: Tensor

    # barrier bufs
    signal_bufs: Tensor
    signal_stages: int
    signal_len: int

    # use this to sync all grids
    grid_barrier: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.zeros(1, dtype=torch.int32, device="cuda"))

    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)
    current_stage: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        assert self.world_size % self.local_world_size == 0
        self.nnodes = self.world_size // self.local_world_size
        self.current_stage = 0

    def reset_barriers(self):
        self.signal_bufs.fill_(0)

    def split_signals_by_length(self, len_list: List[int], signal_tensor=None) -> List[Tensor]:
        res = []
        signal_pad = signal_tensor if signal_tensor is not None else self.signal_bufs
        assert signal_pad.numel() >= sum(len_list), "No enough signals"
        start = 0
        for l in len_list:
            res.append(signal_pad[start:start + l])
            start += l
        return res

    def get_stage_signals(self) -> Tensor:
        assert self.signal_stages > 1
        signal_pad = self.signal_bufs[self.current_stage]
        self.current_stage = (self.current_stage + 1) % self.signal_stages
        return signal_pad

    def get_signal_list_stage(self, l: List[Tensor]) -> List[Tensor]:
        assert self.signal_stages > 1
        ret = []
        for t in l:
            ret.append(t[self.current_stage])
        self.current_stage = (self.current_stage + 1) % self.signal_stages
        return ret

    def get_symm_list(self):
        rank_offset = self.rank - self.local_rank
        bufs = [pynvshmem.symm_tensor(self.scatter_bufs, i + rank_offset) for i in range(self.local_world_size)]
        signals = [pynvshmem.symm_tensor(self.signal_bufs, i + rank_offset) for i in range(self.local_world_size)]
        return bufs, signals


def create_allreduce_ctx(
    input: Tensor,
    method: Union[str, AllReduceMethod],
    signal_stages: int = 1,
) -> AllReduceContext:
    if isinstance(method, str):  # valid method check
        method = str_to_method(method)
    input = input.view(-1, input.shape[-1])
    M, N = input.shape[0], input.shape[1]
    dtype = input.dtype

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    if method in [AllReduceMethod.TwoShot, AllReduceMethod.OneShot_Ld_Reduce]:
        signal_len = 1  # dummy
        recv_shape = None
    elif method in [AllReduceMethod.OneShot_TMA, AllReduceMethod.OneShot]:
        signal_len = world_size
        recv_shape = (M * world_size, N)
    elif method == AllReduceMethod.TwoShot_Multicast:
        recv_shape = (M, N)
        signal_len = world_size
    elif method == AllReduceMethod.DoubleTree:
        signal_len = MAX_DOUBLE_TREE_BLOCKS * world_size
        recv_shape = None
    else:
        raise ValueError("Invalid method!")

    signal_shape = [
        signal_stages,
        signal_len,
    ] if signal_stages > 1 else [
        signal_len,
    ]

    scatter_bufs = pynvshmem.nvshmem_create_tensor([M, N], dtype)
    recv_buffer = pynvshmem.nvshmem_create_tensor(
        recv_shape,
        dtype,
    ) if recv_shape is not None else None
    signal_bufs = pynvshmem.nvshmem_create_tensor(signal_shape, SIGNAL_DTYPE)
    signal_bufs.fill_(0)
    barrier_all_on_stream(None, torch.cuda.current_stream())

    ctx = AllReduceContext(M=M, N=N, rank=rank, world_size=world_size, local_world_size=local_world_size, dtype=dtype,
                           scatter_bufs=scatter_bufs, signal_len=signal_len, buf_M=M, signal_bufs=signal_bufs,
                           recv_buffer=recv_buffer, signal_stages=signal_stages)
    return ctx


def _run_straggler(ctx, straggler_option):
    if straggler_option:
        rank, cyles = straggler_option
        if rank == ctx.rank:
            torch.cuda._sleep(cyles)


@functools.lru_cache()
def get_tree_parent_and_children(N, rank):
    """
    Calculate parent and children nodes for a given rank in two complementary binary trees
    Integrated from https://github.com/cchan/tccl/blob/main/tree.py

    Args:
        N (int): Total number of nodes, must be a power of 2
        rank (int): Node identifier

    Returns:
        tuple: (tree_a_parent, tree_a_left, tree_a_right, tree_b_parent, tree_b_left, tree_b_right)
               -1 indicates no parent or child node exists
    """

    # Check if N is a power of 2
    assert (N & (N - 1)) == 0, "len(ranks) is not a power of 2!"

    # Build first tree
    height = int(math.log2(N))
    tree_a = []
    prev_row = None
    for h in range(height):
        if prev_row is None:
            row = [i for i in range(N) if i % 2 != 0]
        else:
            row = []
            for i in range(0, len(prev_row), 2):
                row.append((prev_row[i] + prev_row[i + 1]) // 2)
        tree_a = row + tree_a
        prev_row = row
    tree_a = [0] + tree_a

    # Build second complementary tree
    tree_b = list(map(lambda e: e - 1, tree_a))
    tree_b[0] = N - 1

    # Calculate parent and children in tree_a
    rank_index_a = tree_a.index(rank)
    if rank_index_a == 0:
        parent_a = -1
        left_child_a = tree_a[rank_index_a + 1]
        right_child_a = -1
    else:
        parent_idx_a = rank_index_a // 2
        left_child_idx_a = rank_index_a * 2
        right_child_idx_a = left_child_idx_a + 1

        parent_a = tree_a[parent_idx_a]
        left_child_a = tree_a[left_child_idx_a] if left_child_idx_a < len(tree_a) else -1
        right_child_a = tree_a[right_child_idx_a] if right_child_idx_a < len(tree_a) else -1

    # Calculate parent and children in tree_b
    rank_index_b = tree_b.index(rank)
    if rank_index_b == 0:
        parent_b = -1
        left_child_b = tree_b[rank_index_b + 1]
        right_child_b = -1
    else:
        parent_idx_b = rank_index_b // 2
        left_child_idx_b = rank_index_b * 2
        right_child_idx_b = left_child_idx_b + 1

        parent_b = tree_b[parent_idx_b]
        left_child_b = tree_b[left_child_idx_b] if left_child_idx_b < len(tree_b) else -1
        right_child_b = tree_b[right_child_idx_b] if right_child_idx_b < len(tree_b) else -1

    return parent_a, left_child_a, right_child_a, parent_b, left_child_b, right_child_b


@triton.jit
def st_p_v4_b32(ptr, packed_v4_b32, mask):
    f1, f2, f3, f4 = packed_v4_b32
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @%p0 st.global.v4.b32 [$1], {$2, $3, $4, $5};
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,r,r,r,r,r"),
        args=[ptr, f1, f2, f3, f4, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["rank", "world_size"])
def double_tree_all_reduce_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    tree0_parent,
    tree0_child0,
    tree0_child1,
    tree1_parent,
    tree1_child0,
    tree1_child1,
    numel,
    rank,
    world_size,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
):
    tl.static_assert(output_ptr.dtype.element_ty == tl.bfloat16, "Only supports dtype=Bf16")
    block_id = (tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0) + tl.program_id(1) * tl.num_programs(0) +
                tl.program_id(0))
    pid = tl.program_id(axis=0)
    thread_idx = get_flat_tid()
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.int64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.int64))
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    block_start = pid * BLOCK_SIZE

    if pid < tl.num_programs(axis=0) // 2:
        tree_child0 = tree0_child0
        tree_child1 = tree0_child1
        tree_parent = tree0_parent
    else:
        tree_child0 = tree1_child0
        tree_child1 = tree1_child1
        tree_parent = tree1_parent

    if tree_child0 != -1:
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint64))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_child0
        if thread_idx == 0:  # wait
            while (atomic_cas(wait_addrs, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()
    if tree_child1 != -1:
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint64))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_child1
        if thread_idx == 0:  # wait
            while (atomic_cas(wait_addrs, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()

    while block_start < (numel // NUMEL_PER_THREAD):

        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < numel // NUMEL_PER_THREAD

        acc_hi = tl.zeros((BLOCK_SIZE, ), tl.int64)
        acc_lo = tl.zeros((BLOCK_SIZE, ), tl.int64)
        if tree_child0 != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_child0).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        # All the computation in else Region will be eliminated by DSE pass, but it's necessary in triton's frontend parser now
        else:
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            hi = tl.zeros((BLOCK_SIZE, ), tl.int64)
            lo = tl.zeros((BLOCK_SIZE, ), tl.int64)
            (acc_hi, acc_lo) = (acc_hi, acc_lo)

        if tree_child1 != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_child1).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        else:
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            hi = tl.zeros((BLOCK_SIZE, ), tl.int64)
            lo = tl.zeros((BLOCK_SIZE, ), tl.int64)
            (acc_hi, acc_lo) = (acc_hi, acc_lo)

        buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
        (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
        (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        tl.store(buffer_ptr + offsets + 0, acc_hi, mask=mask)
        tl.store(buffer_ptr + offsets + 1, acc_lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    if tree_parent != -1:
        remote_signal_pad_addr = tl.load(signal_pad_ptrs + tree_parent).to(tl.pointer_type(tl.uint64))
        send_addr = remote_signal_pad_addr + block_id * world_size + rank
        __syncthreads()
        if thread_idx == 0:  # send
            while atomic_cas(send_addr, 0, 1, "sys", "release") != 0:
                pass

        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint64))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_parent
        if thread_idx == 0:  # wait
            while (atomic_cas(wait_addrs, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()

    block_start = pid * BLOCK_SIZE
    while block_start < (numel // NUMEL_PER_THREAD):
        # Each thread processes 128 bits. Since Triton doesn't yet natively
        # support 128-bit dtypes, we achieve this by having each thread process
        # two 64-bit elements.

        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < numel // NUMEL_PER_THREAD

        if tree_parent != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_parent).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            tl.store(buffer_ptr + offsets + 0, hi, mask=mask)
            tl.store(buffer_ptr + offsets + 1, lo, mask=mask)
            tl.store(output_ptr + offsets + 0, hi, mask=mask)
            tl.store(output_ptr + offsets + 1, lo, mask=mask)
        else:
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets + 0, hi, mask=mask)
            tl.store(output_ptr + offsets + 1, lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    if tree_child0 != -1:
        remote_signal_pad_addr = tl.load(signal_pad_ptrs + tree_child0).to(tl.pointer_type(tl.uint64))
        send_addr = remote_signal_pad_addr + block_id * world_size + rank
        __syncthreads()
        if thread_idx == 0:  # send
            while atomic_cas(send_addr, 0, 1, "sys", "release") != 0:
                pass
    if tree_child1 != -1:
        remote_signal_pad_addr = tl.load(signal_pad_ptrs + tree_child1).to(tl.pointer_type(tl.uint64))
        send_addr = remote_signal_pad_addr + block_id * world_size + rank
        __syncthreads()
        if thread_idx == 0:  # send
            while atomic_cas(send_addr, 0, 1, "sys", "release") != 0:
                pass


@triton.jit(do_not_specialize=[
    "rank",
    "signal_target",
])
def one_shot_push_1d_kernel(
    input_ptr,
    output,
    signal_pad,
    recv_buffer,
    rank,
    world_size: tl.constexpr,
    n_elements,
    signal_target,
    BLOCK_SIZE: tl.constexpr,
):
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    num_total_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    start_tile_id = rank * num_total_tiles // world_size + min(rank, num_total_tiles % world_size)

    for peer in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_block(
            recv_buffer + n_elements * rank,
            input_ptr,
            n_elements * elem_size,
            signal_pad + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size and thread_idx != rank:
        libshmem_device.signal_wait_until(
            signal_pad + thread_idx,
            libshmem_device.NVSHMEM_CMP_EQ,
            signal_target,
        )
    __syncthreads()
    for i in range(pid, num_total_tiles, num_pid):
        tile_id = (start_tile_id + i) % num_total_tiles
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        acc = tl.load(input_ptr + offsets, mask=mask)
        for k in range(1, world_size):
            peer_id = (rank + k) % world_size
            buffer_offset = peer_id * n_elements + tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            val = tl.load(recv_buffer + buffer_offset)
            acc += val
        tl.store(output + offsets, acc, mask=mask)


@triton.jit(do_not_specialize=["rank", "signal_target"])
def one_shot_push_tma_kernel(
    M,
    N,
    input_ptr,
    output_ptr,
    symm_signal_ptr,
    symm_recv_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    signal_target,
    BLOCK_SIZE_SCATTER: tl.constexpr,
    BLOCK_SIZE_REDUCE_M: tl.constexpr,
    BLOCK_SIZE_REDUCE_N: tl.constexpr,
):
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8

    for peer in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_block(
            symm_recv_ptr + n_elements * rank,
            input_ptr,
            n_elements * elem_size,
            symm_signal_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    # TODO(houqi.1993) using more CTAs for copy can archive higher bandwidth. but with more flags
    if False:
        npid = tl.num_programs(axis=0)
        npid_per_rank = npid // world_size
        num_blocks = tl.cdiv(n_elements, BLOCK_SIZE_SCATTER)
        if pid < world_size * npid_per_rank:
            peer_id = pid % world_size
            segment = pid // world_size
            if peer_id != rank:
                peer_ptr = libshmem_device.remote_ptr(symm_recv_ptr, peer_id)
                # otherwise tl.store will not use v4 optimization
                peer_ptr = tl.multiple_of(peer_ptr, 16)
                for i in range(segment, num_blocks, npid_per_rank):
                    offsets = i * BLOCK_SIZE_SCATTER + tl.arange(0, BLOCK_SIZE_SCATTER)
                    mask = offsets < n_elements
                    value = tl.load(input_ptr + offsets, mask=mask)
                    tl.store(peer_ptr + offsets, value, mask=mask)

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.NVSHMEM_CMP_EQ,
            signal_target,
        )
    __syncthreads()

    kernel_ring_reduce_tma(
        symm_recv_ptr,
        output_ptr,
        M,
        N,
        rank,
        world_size,
        BLOCK_SIZE_M=BLOCK_SIZE_REDUCE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_REDUCE_N,
    )


@triton.jit(do_not_specialize=["signal_target", "rank"])
def two_shot_push_multicast_1d_kernel(
    input_ptr,
    sym_ptr,
    recv_signal,
    recv_buffer,
    rank,
    world_size,
    n_elements,
    signal_target,
    BLOCK_SIZE: tl.constexpr,
):
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    num_total_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    tiles_per_rank = num_total_tiles // world_size
    elem_per_rank = tl.cdiv(n_elements, world_size)

    if pid < world_size - 1:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_block(
            recv_buffer + rank * elem_per_rank,
            input_ptr + peer * elem_per_rank,
            elem_per_rank * elem_size,
            recv_signal + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size and thread_idx != rank:
        libshmem_device.signal_wait_until(
            recv_signal + thread_idx,
            libshmem_device.NVSHMEM_CMP_EQ,
            signal_target,
        )
    __syncthreads()
    for i in range(pid, tiles_per_rank, num_pid):
        tile_id = tiles_per_rank * rank + i
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        acc = tl.load(input_ptr + offsets, mask=mask)
        for stage in range(1, world_size):
            from_rank = (rank - stage + world_size) % world_size
            buffer_offsets = (tiles_per_rank * from_rank + i) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            buffer_mask = buffer_offsets < n_elements
            val = tl.load(recv_buffer + buffer_offsets, mask=buffer_mask)
            acc += val
        tl.store(sym_ptr + offsets, acc, mask=mask)
        __syncthreads()
        block_N = ntid(axis=0)
        ptr = tl.cast(sym_ptr + tile_id * BLOCK_SIZE, tl.pointer_type(tl.int8))
        mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, ptr)
        for n in range(thread_idx, BLOCK_SIZE * elem_size // 16, block_N):
            val0, val1 = load_v2_b64(ptr + n * 16)
            multimem_st_b64(tl.cast(mc_ptr, tl.pointer_type(tl.int8)) + n * 16, val0)
            multimem_st_b64(mc_ptr + n * 16 + 8, val1)


@triton.jit
def intra_node_one_shot_ld_reduce_kernel(data_ptr, out_ptr, elems, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(data_ptr.dtype.element_ty == out_ptr.dtype.element_ty)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    num_blocks = tl.cdiv(elems, BLOCK_SIZE)

    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, data_ptr)
    VEC_SIZE = 128 // tl.constexpr(data_ptr.dtype.element_ty.primitive_bitwidth)
    for block_id in range(pid, num_blocks, num_pid):
        offs = block_id * BLOCK_SIZE + tl.arange(
            0,
            BLOCK_SIZE * tl.constexpr(data_ptr.dtype.element_ty.primitive_bitwidth) // 128) * VEC_SIZE
        mask = offs < elems
        packed_v4_b32 = multimem_ld_reduce_p_v4(data_mc_ptr + offs, mask)
        st_p_v4_b32(out_ptr + offs, packed_v4_b32, mask)


@triton.jit
def intra_node_two_shot_multimem_kernel(symm_ptr, grid_barrier_ptr, elems, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    num_blocks = tl.cdiv(elems, BLOCK_SIZE)
    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ptr)
    VEC_SIZE = 128 // tl.constexpr(symm_ptr.dtype.element_ty.primitive_bitwidth)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)

    for block_id in range(pid, num_blocks, num_pid):
        # NOTE: use BLOCK_SIZE // VEC_SIZE as tl.arange won't compile. sad
        offs = block_id * BLOCK_SIZE + tl.arange(
            0,
            BLOCK_SIZE * tl.constexpr(symm_ptr.dtype.element_ty.primitive_bitwidth) // 128) * VEC_SIZE
        packed_v4_b32 = multimem_ld_reduce_p_v4(data_mc_ptr + offs, offs < elems)
        # TODO(houqi.1993) ptxas BUG that multimem.st does not support %p syntax
        multimem_st_v4_p_b32(data_mc_ptr + offs, packed_v4_b32, offs < elems)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)


def intra_node_one_shot_push_1d_op(
    ctx: AllReduceContext,
    local_input: Tensor,
    output: Tensor,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 32,
):
    """ Performs a reduction operation using a simple, native implementation.
    This op uses the most basic reduction algorithm, which is only efficient
    for small data size. However, it causes symmetric memory capacity issues
    when processing large inputs.

    """
    block_size = num_warps * 32 * 16 // local_input.itemsize
    num_elem = local_input.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    pad = ctx.get_stage_signals() if ctx.signal_stages > 1 else None
    signal = ctx.split_signals_by_length([ctx.world_size], signal_tensor=pad)[0]
    if ctx.signal_stages == 1:
        ctx.reset_barriers()
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream)

    _run_straggler(ctx, straggler_option)
    grid = (min(num_tiles, max_sm), ) if max_sm > 0 else (num_tiles, )
    one_shot_push_1d_kernel[grid](
        local_input,
        output,
        signal,
        ctx.recv_buffer,
        ctx.rank,
        ctx.world_size,
        num_elem,
        SIGNAL_TARGET,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def intra_node_one_shot_nvshmem_push_tma_op(ctx: AllReduceContext, local_input: Tensor, output: Tensor,
                                            straggler_option=None, max_sm: int = -1, num_warps: int = 32,  # dummy arg
                                            ):
    """ P2P sends full-length copy of local tensor to other ranks, then uses TMA reduce.

    Args:
        straggler_option (_type_, optional): Simulates the straggler. Defaults to None.
    """
    M, N = local_input.shape
    num_elem = local_input.numel()

    # use as many resource as we can in a CTA

    # heuristic choose M, N, and block_size_m, block_size_n with a reshape (M_per_rank, N) => (M_per_rank * N // 64, N) with N % 64 == 0
    if N % 64 == 0:
        M = M * N // 64
        N = 64
    block_size_n = 64
    block_size_m = 128

    pad = ctx.get_stage_signals() if ctx.signal_stages > 1 else None
    signal = ctx.split_signals_by_length([ctx.world_size], signal_tensor=pad)[0]

    if ctx.signal_stages == 1:
        ctx.reset_barriers()
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream)
    _run_straggler(ctx, straggler_option)

    num_sms = 32 if max_sm < 0 else max_sm
    one_shot_push_tma_kernel[(num_sms, )](
        M,
        N,
        local_input,
        output,
        signal,
        ctx.recv_buffer,
        ctx.rank,
        ctx.world_size,
        num_elem,
        SIGNAL_TARGET,
        BLOCK_SIZE_SCATTER=num_warps * 32 * 16 // local_input.itemsize,
        BLOCK_SIZE_REDUCE_M=block_size_m,
        BLOCK_SIZE_REDUCE_N=block_size_n,
        num_warps=num_warps,
    )
    return output


def intra_node_one_shot_multicast_all_reduce_op(
    ctx: AllReduceContext,
    local_input,
    output,
    straggler_option=None,
    num_warps=32,
    max_sm=-1,
):
    """ One-shot reduces symm buffer using the multimem.ld_reduce instruction.

    Raises:
        NotImplementedError: Currently supports bf16 and float32.
    """
    assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0", "Set NVSHMEM_DISABLE_CUDA_VMM=0 for multicast."
    ctx.scatter_bufs.copy_(local_input)
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream)
    _run_straggler(ctx, straggler_option)

    assert local_input.dtype in [torch.float, torch.bfloat16, torch.float16]

    block_size = num_warps * 32 * 16 // local_input.itemsize
    num_blocks = 4 if max_sm < 0 else max_sm
    intra_node_one_shot_ld_reduce_kernel[(num_blocks, )](
        ctx.scatter_bufs,
        output,
        local_input.numel(),
        block_size,
        num_warps=num_warps,
    )
    return output


def intra_node_double_tree_all_reduce_op(
    ctx: AllReduceContext,
    local_input: Tensor,
    output: Tensor,
    max_sm: int = -1,
    num_warps: int = 32,
    straggler_option=None,
):
    """
    Args:
        max_sm (int, optional): Max grid size. Defaults to -1.
        num_warps (int, optional): The This operator does not specify a block size directly;
                                instead, it calculates the block size by multiplying num_warps by 32. Defaults to 8.
        straggler_option (_type_, optional): _description_. Defaults to None.

    """
    world_size, local_world_size = ctx.world_size, ctx.local_world_size
    rank = ctx.rank

    block_size = num_warps * 32
    NUMEL_PER_THREAD = 8

    assert local_input.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert (local_input.numel() % NUMEL_PER_THREAD == 0), "The number of elements must be 128-bit aligned."

    if max_sm > 0:
        num_blocks = min(
            triton.cdiv(triton.cdiv(local_input.numel(), NUMEL_PER_THREAD), block_size),
            max_sm,
        )
    else:
        num_blocks = triton.cdiv(triton.cdiv(local_input.numel(), NUMEL_PER_THREAD), block_size)
    assert world_size == local_world_size, "Inter-node is not currently supported"
    #assert num_blocks % 2 == 0, "Better strike a balance between two trees"
    assert ctx.signal_len >= world_size * num_blocks, "Alloc at least one signal for each block"

    tree0_parent, tree0_child0, tree0_child1, tree1_parent, tree1_child0, tree1_child1 = get_tree_parent_and_children(
        world_size, rank)

    def list2tensor_ptr(l: List[Tensor]):
        ptrs = []
        for t in l:
            ptrs.append(t.data_ptr())
        ptrs = torch.tensor(ptrs, ).cuda()
        return ptrs

    bufs, signals = ctx.get_symm_list()
    if ctx.signal_stages > 1:
        signals = ctx.get_signal_list_stage(signals)
    buf_ptrs = list2tensor_ptr(bufs)  # construct a C-style pointer array
    sig_ptrs = list2tensor_ptr(signals)

    bufs[rank].copy_(local_input)
    if ctx.signal_stages == 1:
        ctx.reset_barriers()

    _run_straggler(ctx, straggler_option)

    pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    double_tree_all_reduce_kernel[(num_blocks, 1, 1)](
        buf_ptrs,
        sig_ptrs,
        output,
        tree0_parent,
        tree0_child0,
        tree0_child1,
        tree1_parent,
        tree1_child0,
        tree1_child1,
        numel=local_input.numel(),
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        num_warps=num_warps,
    )
    return output


# TODO(houqi.1993) support very small shape. now only support 16 Byte aligned
def intra_node_two_shot_multicast_all_reduce_op(
    ctx: AllReduceContext,
    local_input: torch.Tensor,
    output: torch.Tensor,
    straggler_option=None,
    num_warps=8,
    max_sm=-1,
):
    """ Reduces corresponding shard using ld_reduce PTX instruction,
        then use multimem.st to gather all shards.

        Notes:
            - This is an out-of-place operator. The `output` parameter is a dummy argument,
            only to ensure consistency of the interface with other operators.
            The result is not copied into `output`; instead, it returns the symmetric buffer directly.
    """
    assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0", "Set NVSHMEM_DISABLE_CUDA_VMM=0 for multicast."
    elems = local_input.numel()
    elems_per_rank = triton.cdiv(elems, ctx.world_size)
    assert elems_per_rank * local_input.itemsize % 16 == 0, "use vectorize of 128bit mulitmem.ld_reduce/st"

    ctx.scatter_bufs.copy_(local_input)
    _run_straggler(ctx, straggler_option)

    assert local_input.dtype in [torch.float, torch.bfloat16, torch.float16]
    #  each thread got 4 float or 8 bfloat16
    block_size = num_warps * 32 * 16 // local_input.itemsize
    ngrids = 4 if max_sm < 0 else max_sm  # TODO(houqi.1993) maybe H20 need more than 4
    ngrids = min(triton.cdiv(elems_per_rank, block_size), ngrids)

    intra_node_two_shot_multimem_kernel[(ngrids, )](
        ctx.scatter_bufs.flatten()[ctx.rank * elems_per_rank:],
        ctx.grid_barrier,
        min(elems_per_rank, elems - elems_per_rank * ctx.rank),
        block_size,
        num_warps=num_warps,
    )
    return ctx.scatter_bufs


def intra_node_two_shot_push_multicast_1d_op(
    ctx: AllReduceContext,
    local_input: Tensor,
    block_size: int = 256,
    straggler_option=None,
    num_warps: int = 32,
    max_sm: int = -1,
):
    """ This function is DEPRECATED. Please use `intra_node_two_shot_multicast_all_reduce_op` instead.

    Notes:
        - Out-of-place op, results are stored in symm buffer (see `return ctx.scatter_bufs`)
        - Uses multimem asm instrution
        - NVLink communication cannot be fused with reduce, leading to lower bandwidth utilization
        - Relies on nvshmem for per-block data transfer, which limits scalability when using multiple ranks per peer.
          Especially problematic for high-bandwidth hardware like H20.

    Args:
        block_size (int, optional): Num of rows (M dimension) for one block. Defaults to 256.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Use 'intra_node_two_shot_multicast_all_reduce_op' instead for better performance.", DeprecationWarning,
        stacklevel=2)
    assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0", "Set NVSHMEM_DISABLE_CUDA_VMM=0 for multicast."
    rank, world_size = ctx.rank, ctx.world_size
    num_elem = local_input.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    tiles_per_rank = triton.cdiv(num_tiles, world_size)
    pad = ctx.get_stage_signals() if ctx.signal_stages > 1 else None
    local_counter_pad = ctx.split_signals_by_length([world_size], signal_tensor=pad)[0]

    if ctx.signal_stages == 1:
        ctx.reset_barriers()
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream)
    _run_straggler(ctx, straggler_option)

    if max_sm > 0:
        grid_size = max(min(tiles_per_rank, max_sm), world_size)  # ensure enough blocks for p2p push
    else:
        grid_size = max(tiles_per_rank, world_size)

    two_shot_push_multicast_1d_kernel[(grid_size, )](local_input, ctx.scatter_bufs, local_counter_pad, ctx.recv_buffer,
                                                     rank, world_size, num_elem, SIGNAL_TARGET, num_warps=num_warps,
                                                     BLOCK_SIZE=block_size)
    return ctx.scatter_bufs


def all_reduce(input: Tensor, output: Tensor, method: Union[str, AllReduceMethod] = None, ctx: AllReduceContext = None,
               return_ctx: bool = False, max_sm: int = -1, straggler_option=None,  # for straggler simulation
               ):
    if isinstance(method, str):
        method = str_to_method(method)
    if ctx is None:
        ctx = create_allreduce_ctx(
            input,
            method,
        )
    if method is None:  # methods using multimem reduce are recommended
        if input.element_size() * input.numel() <= 64 * 1024:  #64KB
            method = AllReduceMethod.OneShot_Ld_Reduce
        else:
            method = AllReduceMethod.TwoShot

    METHOD_TO_OP = {
        AllReduceMethod.OneShot_TMA: intra_node_one_shot_nvshmem_push_tma_op,
        AllReduceMethod.OneShot: intra_node_one_shot_push_1d_op,
        AllReduceMethod.DoubleTree: intra_node_double_tree_all_reduce_op,
        AllReduceMethod.OneShot_Ld_Reduce: intra_node_one_shot_multicast_all_reduce_op,
        AllReduceMethod.TwoShot: intra_node_two_shot_multicast_all_reduce_op,
    }  # TwoShot_Multicast is not listed here since it will be deprecated
    op_handle = METHOD_TO_OP[method]

    warps_iters = triton.cdiv(input.element_size() * input.numel(),
                              32 * 16 // input.itemsize)  # each thread can process 16B
    num_warps = max(4, min(warps_iters, 32))
    output = op_handle(
        ctx=ctx,
        local_input=input,
        output=output,
        max_sm=max_sm,
        num_warps=num_warps,
        straggler_option=straggler_option,
    )

    if return_ctx:
        return output, ctx
    return output
