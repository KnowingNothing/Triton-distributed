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
Low Latency All-to-All Communication
====================================
In this tutorial, we demonstrate how to implement the All-to-All communication
paradigm in Expert Parallelism (EP) for MoE models using Triton-distributed.

First, let's quickly review the EP workflow:
In MoE, the `E` experts are distributed across `N` devices (EP ranks).
For simplicity, we assume that `N` divides `E` evenly, so experts are distributed
uniformly. For example, when `E = 128` and `N = 32`, each device will handle 4 experts.

During inference with EP, each device is assigned a subset of tokens, as determined
by the MoE router module. The router on each device generates a tensor of
shape `[num_tokens, topk]`, containing the indices of the top `k` experts selected
for each token. The experts chosen for a token may reside on other devices,
necessitating communication to send the tokens to the appropriate devices.
Similarly, if other devices have tokens that select experts located on the current device,
those tokens need to be sent to the current device as well. This process is called Dispatch.

After the tokens are processed by their corresponding experts, they need to be
returned to their original devices. This operation mirrors Dispatch and is referred
to as Combine. From a communication perspective, both Dispatch and Combine are
essentially All-to-All collective communication operations.

Next, we demonstrate how to implement an efficient All-to-All operation in Triton-distributed with minimal code.

Triton-distributed provides a programming model that allows fine-grained control
over data movement between devices, optimizing hardware utilization.
At the core of our implementation are low-level primitives that manage the communication logic.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/sentenv.sh
    bash ./third_party/distributed/launch.sh ./third_party/distributed/tutorials/04-deepseek-infer-all2all.py

"""
import torch
import torch.distributed
import triton
import triton.language as tl
from triton import pynvshmem

import os
import random
import argparse
import datetime
import numpy as np

from enum import Enum
from tabulate import tabulate
from typing import Optional
from triton.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid


@triton.jit
def ceil_div(a, b):
    return (a + b - 1) // b


FP8_MAX = tl.constexpr(torch.finfo(torch.float8_e4m3fn).max)
FP8_MAX_INV = tl.constexpr(1 / 448.)


# use static config for fast compile
@triton.autotune(configs=[triton.Config(kwargs={'BM': BM}, num_warps=w) for BM in [16] for w in [16]], key=[])
@triton.jit
def all_to_all_kernel(
    send_tensor,
    send_scale,
    data_src,
    data_dst,
    splits_src,
    splits_dst,
    signal,
    send_splits_cumsum,
    recv_offset,
    scale_src,
    scale_dst,
    rank: int,
    call_count: int,
    act_pos: int,
    MODE: tl.constexpr,
    ONLINE_QUANT_FP8: tl.constexpr,
    FP8_GSIZE: tl.constexpr,
    WITH_SCALE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_TOT_EXPERTS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BN_SCALE: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr = 2,
    SCALE_ELEMENT_SIZE: tl.constexpr = 4,
):
    """
    All-to-All kernel for the Dispatch and Combine phases.

    - MODE: Determines whether the operation is Dispatch (0) or Combine (1).
    - ONLINE_QUANT_FP8: A flag indicating whether FP8 quantization is used.
    - FP8_GSIZE: The group size for FP8 quantization.
    - WITH_SCALE: A flag indicating whether to send scale (already computed).
    - WORLD_SIZE: number of EP ranks.
    - HIDDEN: The hidden size for each token.
    - MAX_M: The maximum number of tokens that can be processed per rank.
    - EXPERTS_PER_RANK: The number of experts handled by each rank.
    - NUM_TOT_EXPERTS: The total number of experts.
    - BM, BN, BN_SCALE: Block size used to copy data to send buffer
    - ELEMENT_SIZE: The size of each element in bytes.
    - SCALE_ELEMENT_SIZE: The size of each scale in bytes.
    """
    pid = tl.program_id(0)
    # Triton-distributed exposes `tid` that can be used to identify the thread index within a CTA
    threadidx = tid(axis=0)
    NUM_GROUPS: tl.constexpr = HIDDEN // FP8_GSIZE

    # 1. Calculate the token range for the current program (rank), get the corresponding pointer
    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK
    m_st = tl.load(send_splits_cumsum + exp_st)
    m_ed = tl.load(send_splits_cumsum + exp_ed)
    num_rows_cur_block = m_ed - m_st

    # Signal pointer to communicate when data is ready
    signal_ptr = signal + act_pos * WORLD_SIZE + rank
    if MODE == 0:  # dispatch mode
        # Calculate source and destination offsets based on the expert-level token number cumsum
        split_src_ptr = splits_src + (exp_st + pid)
        split_dst_ptr = splits_dst + act_pos * (NUM_TOT_EXPERTS + WORLD_SIZE) + rank * (EXPERTS_PER_RANK + 1)

        off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
        off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
        cumsum_sts = tl.load(send_splits_cumsum + off0)
        cumsum_eds = tl.load(send_splits_cumsum + off1)
        tl.store(split_src_ptr + tl.arange(0, EXPERTS_PER_RANK), cumsum_eds - cumsum_sts)
        tl.store(split_src_ptr + EXPERTS_PER_RANK, m_st)

        # Calculate the source and destination data offsets for the dispatch operation
        src_off = m_st
        dst_off = rank * MAX_M
        data_src_ptr = data_src + src_off * HIDDEN
        data_dst_ptr = data_dst + act_pos * WORLD_SIZE * MAX_M * HIDDEN + dst_off * HIDDEN
        scale_src_ptr = scale_src + src_off * NUM_GROUPS
        scale_dst_ptr = scale_dst + act_pos * WORLD_SIZE * MAX_M * NUM_GROUPS + dst_off * NUM_GROUPS
    else:  # combine mode
        # For the combine phase, source and destination offsets are updated accordingly
        src_off = pid * MAX_M
        dst_off = tl.load(recv_offset + pid)
        data_src_ptr = data_src + act_pos * WORLD_SIZE * MAX_M * HIDDEN + src_off * HIDDEN
        data_dst_ptr = data_dst + dst_off * HIDDEN
        scale_src_ptr = scale_src + act_pos * WORLD_SIZE * MAX_M * NUM_GROUPS + src_off * NUM_GROUPS
        scale_dst_ptr = scale_dst + dst_off * NUM_GROUPS

    # 2. Copy the data (may be online quantized to FP8) to send buffer
    off_m = tl.arange(0, BM)
    if ONLINE_QUANT_FP8 and MODE == 0:
        # TODO: adaptive UNROLL_FACTOR
        UNROLL_FACTOR: tl.constexpr = 4
        group_offs = off_m[:, None] * HIDDEN + tl.arange(0, FP8_GSIZE * UNROLL_FACTOR)[None, :]
        send_tensor_ptrs = send_tensor + m_st * HIDDEN + group_offs
        data_src_ptrs = tl.cast(data_src_ptr, tl.pointer_type(tl.float8e4nv)) + group_offs
        scale_src_ptrs = scale_src_ptr + off_m[:, None] * NUM_GROUPS + tl.arange(0, UNROLL_FACTOR)[None, :]
        for i in tl.range(ceil_div(num_rows_cur_block, BM)):
            group_mask = off_m[:, None] < num_rows_cur_block - i * BM
            for _ in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
                group = tl.reshape(tl.load(send_tensor_ptrs, group_mask), (BM * UNROLL_FACTOR, FP8_GSIZE))
                scale = tl.max(tl.abs(group), 1, keep_dims=True).to(tl.float32) * FP8_MAX_INV
                quant = tl.reshape((group.to(tl.float32) / scale).to(tl.float8e4nv), (BM, UNROLL_FACTOR * FP8_GSIZE))
                tl.store(data_src_ptrs, quant, group_mask)
                tl.store(scale_src_ptrs, tl.reshape(scale, (BM, UNROLL_FACTOR)), group_mask)

                send_tensor_ptrs += UNROLL_FACTOR * FP8_GSIZE
                data_src_ptrs += UNROLL_FACTOR * FP8_GSIZE
                scale_src_ptrs += UNROLL_FACTOR
            send_tensor_ptrs += (BM - 1) * HIDDEN
            data_src_ptrs += (BM - 1) * HIDDEN
            scale_src_ptrs += (BM - 1) * NUM_GROUPS
    else:
        off_n = tl.arange(0, BN)
        send_tensor_ptrs = send_tensor + m_st * HIDDEN + off_m[:, None] * HIDDEN + off_n[None, :]
        data_src_ptrs = data_src_ptr + off_m[:, None] * HIDDEN + off_n[None, :]
        if WITH_SCALE:
            off_g = tl.arange(0, BN_SCALE)
            send_scale_ptrs = send_scale + m_st * NUM_GROUPS + off_m[:, None] * NUM_GROUPS + off_g[None, :]
            scale_src_ptrs = scale_src_ptr + off_m[:, None] * NUM_GROUPS + off_g[None, :]
        for i in tl.range(ceil_div(num_rows_cur_block, BM)):
            data_mask = (off_m[:, None] < num_rows_cur_block - i * BM) & (off_n[None, :] < HIDDEN)
            tl.store(data_src_ptrs, tl.load(send_tensor_ptrs, data_mask), data_mask)
            send_tensor_ptrs += BM * HIDDEN
            data_src_ptrs += BM * HIDDEN
            if WITH_SCALE:
                scale_mask = (off_m[:, None] < num_rows_cur_block - i * BM) & (off_g[None, :] < NUM_GROUPS)
                tl.store(scale_src_ptrs, tl.load(send_scale_ptrs, scale_mask), scale_mask)
                send_scale_ptrs += BM * NUM_GROUPS
                scale_src_ptrs += BM * NUM_GROUPS

    # 3. Perform the memory copy operation using shared memory for inter-rank communication.
    # the last argument is the peer id (id of target rank)
    libshmem_device.putmem_nbi_block(
        data_dst_ptr,
        data_src_ptr,
        num_rows_cur_block * HIDDEN * (1 if (ONLINE_QUANT_FP8 and MODE == 0) else ELEMENT_SIZE),
        pid,
    )
    if MODE == 0:
        # Dispatch mode: send split information to the target rank
        libshmem_device.putmem_nbi_block(
            split_dst_ptr,
            split_src_ptr,
            (EXPERTS_PER_RANK + 1) * 4,  # now we use `int32` for splits
            pid,
        )
    # If online quantization is enbaled or scale is calculated ahead of time,
    # signal the target rank with the scale data
    if WITH_SCALE or ONLINE_QUANT_FP8:
        libshmem_device.putmem_signal_nbi_block(
            scale_dst_ptr,
            scale_src_ptr,
            num_rows_cur_block * NUM_GROUPS * SCALE_ELEMENT_SIZE,
            signal_ptr,
            call_count,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            pid,
        )

    # 4. fence data transfer. Then wait for signal
    libshmem_device.fence()
    if threadidx == 0:
        # notify the target rank (here is the `pid`-th rank) that the data is ready by setting the signal
        if not (WITH_SCALE or ONLINE_QUANT_FP8):
            libshmem_device.signal_op(
                signal_ptr,
                call_count,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                pid,
            )
        # wait for the signal from the source rank (here is the `pid`-th rank)
        libshmem_device.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            libshmem_device.NVSHMEM_CMP_EQ,
            call_count,
        )


def dtype_size_in_bytes(dtype: torch.dtype) -> int:
    return {
        torch.float32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float8_e4m3fn: 1,
    }[dtype]


class AllToAllContext:

    def __init__(
        self,
        max_m: int,
        hidden: int,
        online_quant_fp8: bool,
        rank: int,
        num_tot_experts: int,
        WORLD_SIZE: int,
        FP8_GSIZE: int = 128,
        dtype=torch.bfloat16,
        scale_dtype=torch.float,
    ):
        """
        params:
            - max_m: max number of tokens per rank


        - In this context, we pre-define the max number of tokens that can be sent from
            one device `max_m`, which is typically 128 or 256, and reserve corresponding send/receive buffer size.

        - We also need to allocate split_buffer and send splits information to record
            the number of tokens received by each expert for subsequent calculations and communication.

        - The signal buffer is used to notify the target rank that the data is already ready.
            `pynvshmem.nvshmem_create_tensor` is the low-level API to create shared memory
            between different devices (see [nvshmem](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html#memory-model)).

        - We record `call_count` of the kernel as the unique signal to notify target rank.
        """
        self.send_buf = pynvshmem.nvshmem_create_tensor([max_m, hidden], dtype)
        self.recv_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * max_m * 2, hidden], dtype)
        self.scale_send_buf = pynvshmem.nvshmem_create_tensor([max_m, hidden // FP8_GSIZE], scale_dtype)
        self.scale_recv_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * max_m * 2, hidden // FP8_GSIZE],
                                                              scale_dtype)
        # `+WORLD_SIZE` because we need to send/receive the start offset in `send_buf` of the tokens dispatched at dispatch phase
        self.split_send_buf = pynvshmem.nvshmem_create_tensor([num_tot_experts + WORLD_SIZE], torch.int32)
        self.split_recv_buf = pynvshmem.nvshmem_create_tensor([(num_tot_experts + WORLD_SIZE) * 2], torch.int32)
        self.signal_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * 2], torch.uint64)

        self.max_m = max_m
        self.hidden = hidden
        self.online_quant_fp8 = online_quant_fp8
        self.FP8_GSIZE = FP8_GSIZE
        self.dtype = dtype
        self.scale_dtype = scale_dtype
        self.ele_size = dtype_size_in_bytes(self.dtype)
        self.scale_ele_size = dtype_size_in_bytes(self.scale_dtype)

        self.num_tot_experts = num_tot_experts
        self.experts_per_rank = num_tot_experts // WORLD_SIZE

        self.WORLD_SIZE = WORLD_SIZE
        self.rank = rank

        # start from 1, becase the initial values of signal buffer is 0
        self.call_count = 1
        # switch double buffer
        self.act_pos = 0
        self.MOD_VALUE = 1000000


def next_power_of_2(x: int) -> int:
    if x == 0:
        return 1
    return 1 << (x - 1).bit_length()


class AllToAllMode(Enum):
    DISPATCH = 0
    COMBINE = 1


def fast_all_to_all(
    ctx: AllToAllContext,
    mode: AllToAllMode,
    send_tensor: torch.Tensor,
    send_split_cumsum: torch.Tensor,
    recv_offset: Optional[torch.Tensor],
    send_scale: Optional[torch.Tensor],
):
    """
    low-latency all-to-all communication

    `mode`: dispatch / combine
    `send_tensor`: [num_tokens, HIDDEN] input tensor
    `send_split_cumsum`: [num_experts + 1] cumulative sum of the number of tokens
    `recv_offset`: [WORLD_SIZE] used in combine mode, base offset of the received tokens
    `send_scale`: [num_tokens] scale tensor. used for quantization
    """
    with_scale = send_scale is not None
    online_quant = ctx.online_quant_fp8
    assert not (online_quant == with_scale and with_scale), "`online_quant_fp8` and `with_scale` cannot be both True"
    if online_quant or with_scale:
        assert send_tensor.shape[
            1] % ctx.FP8_GSIZE == 0, "the last dimension of `send_tensor` must be divisible by `ctx.FP8_GSIZE`"

    num_tokens = send_tensor.shape[0]
    if mode == AllToAllMode.DISPATCH:
        assert num_tokens <= ctx.max_m
        send_buf = ctx.send_buf
        recv_buf = ctx.recv_buf
        scale_send_buf = ctx.scale_send_buf
        scale_recv_buf = ctx.scale_recv_buf
        split_send_buf = ctx.split_send_buf
        split_recv_buf = ctx.split_recv_buf
        MODE = 0
    else:
        assert num_tokens <= ctx.WORLD_SIZE * ctx.max_m
        send_buf = ctx.recv_buf
        recv_buf = ctx.send_buf
        scale_send_buf = ctx.scale_recv_buf
        scale_recv_buf = ctx.scale_send_buf
        split_send_buf = ctx.split_recv_buf
        split_recv_buf = ctx.split_send_buf
        MODE = 1

    grid = (ctx.WORLD_SIZE, )
    # TODO: adaptive block size
    BN = next_power_of_2(send_tensor.shape[1])
    BN_SCALE = next_power_of_2(send_tensor.shape[1] // ctx.FP8_GSIZE) if online_quant else BN
    all_to_all_kernel[grid](
        send_tensor,
        send_scale,
        data_src=send_buf,
        data_dst=recv_buf,
        splits_src=split_send_buf,
        splits_dst=split_recv_buf,
        signal=ctx.signal_buf,
        send_splits_cumsum=send_split_cumsum,
        recv_offset=recv_offset,
        scale_src=scale_send_buf,
        scale_dst=scale_recv_buf,
        rank=ctx.rank,
        call_count=ctx.call_count,
        act_pos=ctx.act_pos,
        MODE=MODE,
        ONLINE_QUANT_FP8=online_quant,
        FP8_GSIZE=ctx.FP8_GSIZE,
        WITH_SCALE=with_scale,
        WORLD_SIZE=ctx.WORLD_SIZE,
        HIDDEN=ctx.hidden,
        MAX_M=ctx.max_m,
        EXPERTS_PER_RANK=ctx.experts_per_rank,
        NUM_TOT_EXPERTS=ctx.num_tot_experts,
        BN=BN,
        BN_SCALE=BN_SCALE,
        ELEMENT_SIZE=ctx.ele_size,
        SCALE_ELEMENT_SIZE=ctx.scale_ele_size,
    )

    # for double buffer
    split_buf_st = ctx.act_pos * (ctx.num_tot_experts + ctx.WORLD_SIZE)
    split_buf_ed = split_buf_st + (ctx.num_tot_experts + ctx.WORLD_SIZE)
    data_buf_st = ctx.act_pos * ctx.WORLD_SIZE * ctx.max_m
    data_buf_ed = data_buf_st + ctx.WORLD_SIZE * ctx.max_m
    scale_buf_st = ctx.act_pos * ctx.WORLD_SIZE * ctx.max_m
    scale_buf_ed = scale_buf_st + ctx.WORLD_SIZE * ctx.max_m

    out_lis: list[torch.Tensor] = []
    if mode == AllToAllMode.DISPATCH:
        out_lis.append(split_recv_buf[split_buf_st:split_buf_ed])
        out_lis.append(recv_buf[data_buf_st:data_buf_ed, :])
        out_lis.append(scale_recv_buf[scale_buf_st:scale_buf_ed, :] if (with_scale or online_quant) else None)
    else:
        out_lis.append(None)
        out_lis.append(recv_buf)
        out_lis.append(scale_recv_buf if with_scale else None)
        ctx.act_pos ^= 1
    ctx.call_count = (ctx.call_count + 1) % ctx.MOD_VALUE

    return out_lis


def all_to_all_post_process(
    ctx: AllToAllContext,
    input_splits: torch.Tensor,
    recv_buffer: torch.Tensor,
    scale_buffer: Optional[torch.Tensor] = None,
):
    with_scale = scale_buffer is not None
    world_size = ctx.WORLD_SIZE
    combine_offset = input_splits[torch.arange(1, world_size + 1) * (ctx.experts_per_rank + 1) - 1]
    combine_send_splits = input_splits.reshape(world_size, -1)[:, :ctx.experts_per_rank].flatten()
    num_tokens_from_each_rank = combine_send_splits.reshape(world_size, -1).sum(dim=1)

    data_vec, scale_vec = [], []
    for i in range(world_size):
        n_token_from_tgt_rank = num_tokens_from_each_rank[i]
        _start = i * ctx.max_m
        if ctx.online_quant_fp8:
            data_vec.append(recv_buffer.reshape(-1, ctx.hidden // 2)[_start * 2:_start * 2 + n_token_from_tgt_rank])
        else:
            data_vec.append(recv_buffer[_start:_start + n_token_from_tgt_rank])
        if with_scale or ctx.online_quant_fp8:
            scale_vec.append(scale_buffer[_start:_start + n_token_from_tgt_rank])
    output = torch.concat(data_vec)
    output_scale = torch.concat(scale_vec) if (with_scale or ctx.online_quant_fp8) else None

    return combine_offset, combine_send_splits, output, output_scale


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.empty(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    out[0] = 0
    _ = torch.cumsum(splits, 0, out=out[1:])
    return out


def calc_scatter_index_stable(choosed_experts: torch.Tensor):
    return (choosed_experts.flatten().argsort(stable=True).argsort().int().view(choosed_experts.shape))


def calc_gather_index(
    scatter_index: torch.Tensor,
    row_start: int,
    row_end: int,
    BLOCK_SIZE: int = 1024,
):

    @triton.jit
    def _kernel(
        scatter_index: torch.Tensor,
        gather_index: torch.Tensor,
        topk_index: torch.Tensor,
        ntokens: int,
        topk: int,
        row_start: int,
        row_end: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < ntokens * topk
        scatter_idx = tl.load(scatter_index + offset, mask=mask, other=-1)
        token_idx = offset // topk
        topk_idx = offset % topk
        token_idx_mask = (scatter_idx >= row_start) & (scatter_idx < row_end)
        tl.store(gather_index + scatter_idx - row_start, token_idx, mask=token_idx_mask)
        tl.store(topk_index + scatter_idx - row_start, topk_idx, mask=token_idx_mask)

    ntokens, topk = scatter_index.shape
    gather_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    topk_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    grid = lambda META: (triton.cdiv(ntokens * topk, META["BLOCK_SIZE"]), )
    _kernel[grid](
        scatter_index,
        gather_index,
        topk_index,
        ntokens,
        topk,
        row_start,
        row_end,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )
    return gather_index, topk_index


@triton.autotune(configs=[triton.Config(kwargs={'BM': BM}, num_warps=w) for BM in [16] for w in [16]], key=[])
@triton.jit
def _quant_kernel(out, out_scale, t, m, N: tl.constexpr, FP8_GSIZE: tl.constexpr = 128, BM: tl.constexpr = 32):
    pid = tl.program_id(0)
    FP8_MAX_INV = tl.constexpr(1 / 448.)
    NUM_GROUPS: tl.constexpr = N // FP8_GSIZE
    UNROLL_FACTOR: tl.constexpr = 4
    off_m = pid * BM + tl.arange(0, BM)
    off_n = tl.arange(0, UNROLL_FACTOR * FP8_GSIZE)
    input_ptrs = t + off_m[:, None] * N + off_n[None, :]
    out_ptrs = tl.cast(out, tl.pointer_type(tl.float8e4nv)) + off_m[:, None] * N + off_n[None, :]
    out_scale_ptrs = out_scale + off_m[:, None] * NUM_GROUPS + tl.arange(0, UNROLL_FACTOR)[None, :]
    for _ in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
        group_mask = off_m[:, None] < m
        group = tl.reshape(tl.load(input_ptrs, group_mask), (BM * UNROLL_FACTOR, FP8_GSIZE))
        scale = tl.max(tl.abs(group), 1, keep_dims=True).to(tl.float32) * FP8_MAX_INV
        quant = (group.to(tl.float32) / scale).to(tl.float8e4nv)
        tl.store(out_ptrs, tl.reshape(quant, (BM, UNROLL_FACTOR * FP8_GSIZE)), mask=group_mask)
        tl.store(out_scale_ptrs, tl.reshape(scale, (BM, UNROLL_FACTOR)), mask=group_mask)
        input_ptrs += UNROLL_FACTOR * FP8_GSIZE
        out_ptrs += UNROLL_FACTOR * FP8_GSIZE
        out_scale_ptrs += UNROLL_FACTOR


@triton.autotune(configs=[triton.Config(kwargs={'BM': BM}, num_warps=w) for BM in [16] for w in [16]], key=[])
@triton.jit
def _dequant_kernel(out, input, scales, m, N: tl.constexpr, FP8_GSIZE: tl.constexpr = 128, BM: tl.constexpr = 32):
    pid = tl.program_id(0)
    NUM_GROUPS: tl.constexpr = N // FP8_GSIZE
    UNROLL_FACTOR: tl.constexpr = 4
    off_m = pid * BM + tl.arange(0, BM)
    off_n = tl.arange(0, UNROLL_FACTOR * FP8_GSIZE)
    input_ptrs = tl.cast(input, tl.pointer_type(tl.float8e4nv)) + off_m[:, None] * N + off_n[None, :]
    input_scale_ptrs = scales + off_m[:, None] * NUM_GROUPS + tl.arange(0, UNROLL_FACTOR)[None, :]
    out_ptrs = out + off_m[:, None] * N + off_n[None, :]
    for _ in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
        group_mask = off_m[:, None] < m
        group = tl.reshape(tl.load(input_ptrs, group_mask), (BM * UNROLL_FACTOR, FP8_GSIZE))
        scale = tl.reshape(tl.load(input_scale_ptrs, group_mask), (BM * UNROLL_FACTOR, 1))
        deq = (group.to(tl.float32) * scale).to(tl.bfloat16)
        tl.store(out_ptrs, tl.reshape(deq, (BM, UNROLL_FACTOR * FP8_GSIZE)), mask=group_mask)
        input_ptrs += UNROLL_FACTOR * FP8_GSIZE
        input_scale_ptrs += UNROLL_FACTOR
        out_ptrs += UNROLL_FACTOR * FP8_GSIZE


def quant_bf16_fp8(tensor: torch.Tensor, gsize: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    m, N = tensor.shape
    grid = lambda meta: (triton.cdiv(m, meta["BM"]), )
    out = torch.empty((m, N // 2), dtype=torch.bfloat16, device="cuda")
    out_scale = torch.empty(m, N // gsize, dtype=torch.float32, device="cuda")
    _quant_kernel[grid](out, out_scale, tensor, m, N)
    return out, out_scale


def dequant_fp8_bf16(q_tensor: torch.Tensor, scales: torch.Tensor):
    m, N = q_tensor.shape
    grid = lambda meta: (triton.cdiv(m, meta["BM"]), )
    out = torch.empty([m, N * 2], dtype=torch.bfloat16, device=q_tensor.device)
    _dequant_kernel[grid](out, q_tensor, scales, m, N * 2)
    return out


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)


EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def initialize_distributed():
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP has already been initialized"
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    EP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    init_seed(seed=RANK)
    pynvshmem.init_nvshmem_by_uniqueid(EP_GROUP)
    return EP_GROUP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-N", type=int, default=7168)
    parser.add_argument("-G", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--bench_iters", default=1000, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--with_scale", action="store_true")
    parser.add_argument("--quant_gsize", type=int, default=128, help="quantization group size")
    parser.add_argument("--online_quant_fp8", action="store_true")
    return parser.parse_args()


def generate_random_exp_indices(token_num, total_num_experts, topk):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for t in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def bench_func(f, bench_iters: int):
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(1000000000)
    for _ in range(20):
        f()
    st.record()
    for _ in range(bench_iters):
        _ = f()
    ed.record()
    torch.cuda.synchronize()
    return st.elapsed_time(ed) / args.bench_iters


def perf_torch(input: torch.Tensor, scale_tensor: torch.Tensor, exp_indices: torch.Tensor):
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    # calculate the scatter and gather idx
    scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
    gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, num_tokens * args.topk)
    num_groups = input.shape[1] // args.quant_gsize
    scattered_input = torch.empty((input.size(0) * args.topk, input.size(1)), dtype=input.dtype,
                                  device=input.device).copy_(torch.index_select(input, dim=0,
                                                                                index=gather_idx_cur_rank))
    scattered_scale = torch.empty(
        (scale_tensor.size(0) * args.topk, num_groups),
        dtype=scale_tensor.dtype,
        device=scale_tensor.device,
    ).copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank)) if args.with_scale else None

    send_tensor, send_scale = scattered_input, scattered_scale
    if args.online_quant_fp8:
        send_tensor, send_scale = quant_bf16_fp8(scattered_input)

    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    ep_size = EP_GROUP.size()
    a2a_dispatch_output = torch.empty(
        [a2a_splits_cpu.sum(), input.size(1) // (2 if args.online_quant_fp8 else 1)], dtype=send_tensor.dtype,
        device=input.device)
    a2a_dispatch_scale = torch.empty([a2a_splits_cpu.sum(), num_groups], dtype=torch.float32,
                                     device=scale_tensor.device)
    torch.cuda.synchronize()

    dispatch_time, combine_time, quant_time = 0., 0., 0.

    # 1. Dispatch
    def _quant_input():
        return quant_bf16_fp8(scattered_input)

    if args.online_quant_fp8:
        quant_time = bench_func(_quant_input, args.bench_iters)

    def fwd():
        torch.distributed.all_to_all_single(
            output=a2a_dispatch_output,
            input=send_tensor,
            output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
            input_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
            group=EP_GROUP,
        )
        if args.with_scale or args.online_quant_fp8:
            torch.distributed.all_to_all_single(
                output=a2a_dispatch_scale,
                input=send_scale,
                output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
                input_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
                group=EP_GROUP,
            )

    dispatch_time = bench_func(fwd, args.bench_iters)

    # 2. Combine
    a2a_combine_output = torch.empty_like(scattered_input)
    combine_input = a2a_dispatch_output
    if args.online_quant_fp8:
        combine_input = dequant_fp8_bf16(a2a_dispatch_output, a2a_dispatch_scale)

    def cmb():
        torch.distributed.all_to_all_single(
            output=a2a_combine_output,
            input=combine_input,
            output_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
            input_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
            group=EP_GROUP,
        )

    combine_time = bench_func(cmb, args.bench_iters)

    comb_ref = (dequant_fp8_bf16(send_tensor, send_scale) if args.online_quant_fp8 else send_tensor)
    torch.testing.assert_close(a2a_combine_output.float(), comb_ref.float(), rtol=1e-5, atol=1e-5)

    return a2a_dispatch_output, a2a_dispatch_scale, quant_time, dispatch_time, combine_time


def perf_triton(input: torch.Tensor, scale_tensor: torch.Tensor, exp_indices: torch.Tensor):
    # 0. pre-process: duplicate the input tensor `topk` times then scatter

    # splits_gpu_cur_rank: [num_experts]; indicates the number of tokens for each expert
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    # split_cumsum: [num_experts + 1]; cumulative sum of the number of tokens for each expert
    split_cumsum = splits_to_cumsum(splits_gpu_cur_rank)

    scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
    gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, num_tokens * args.topk)
    scattered_input = torch.empty(input.size(0) * args.topk, input.size(1), dtype=input.dtype,
                                  device=input.device).copy_(torch.index_select(input, dim=0,
                                                                                index=gather_idx_cur_rank))
    scattered_scale = torch.empty(
        (scale_tensor.size(0) * args.topk, input.shape[1] // args.quant_gsize),
        dtype=scale_tensor.dtype,
        device=scale_tensor.device,
    ).copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank)) if args.with_scale else None

    # 1. Dispatch
    def fwd():
        out = fast_all_to_all(all_to_all_ctx, AllToAllMode.DISPATCH, scattered_input, split_cumsum, None,
                              scattered_scale)
        # flip for test
        all_to_all_ctx.act_pos ^= 1
        return out

    avg_time_dispatch = bench_func(fwd, args.bench_iters)

    dispatch_splits, dis_token, dis_scale = fwd()
    comb_offset, comb_send_splits, dis_token, dis_scale = all_to_all_post_process(all_to_all_ctx, dispatch_splits,
                                                                                  dis_token, dis_scale)

    # 2. Combine
    combine_input = (dequant_fp8_bf16(dis_token, dis_scale) if args.online_quant_fp8 else dis_token)
    combine_split_cumsum = splits_to_cumsum(comb_send_splits)

    def comb():
        return fast_all_to_all(all_to_all_ctx, AllToAllMode.COMBINE, combine_input, combine_split_cumsum, comb_offset,
                               None)

    _, combined_tokens, _ = comb()

    # check the correctness of combine
    combine_ref = (dequant_fp8_bf16(*quant_bf16_fp8(scattered_input)) if args.online_quant_fp8 else scattered_input)
    torch.testing.assert_close(combined_tokens[:scattered_input.shape[0]].float(), combine_ref.float(), rtol=1e-5,
                               atol=1e-5)

    avg_time_combine = bench_func(comb, args.bench_iters)

    return dis_token, dis_scale, avg_time_dispatch, avg_time_combine


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()

    assert (args.G % WORLD_SIZE == 0), f"args.G:{args.G} should be divisible by WORLD_SIZE:{WORLD_SIZE}"
    experts_per_rank = args.G // WORLD_SIZE
    num_tokens = args.M
    print(f"Rank-{RANK}: Received {num_tokens} tokens")

    all_to_all_ctx = AllToAllContext(
        args.M * args.topk,
        args.N,
        args.online_quant_fp8,
        RANK,
        args.G,
        WORLD_SIZE,
        args.quant_gsize,
        DTYPE_MAP[args.dtype],
        torch.float,
    )

    # exp_indices: [num_tokens, topk]
    exp_indices = generate_random_exp_indices(num_tokens, args.G, args.topk).to("cuda")
    input = (torch.rand(num_tokens, args.N, dtype=torch.float32).to(DTYPE_MAP[args.dtype]).to("cuda"))
    scale = torch.rand((num_tokens, args.N // args.quant_gsize), dtype=torch.float32).to("cuda")

    ref_out, ref_scale, torch_quant, torch_dis, torch_comb = perf_torch(input, scale, exp_indices)
    torch.cuda.synchronize()
    triton_out, triton_scale, triton_dis, triton_comb = perf_triton(input, scale, exp_indices)
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # collect the results then print
    def gather_benchmark(time_value):
        tensor = torch.tensor(time_value, device="cuda")
        gather_list = ([torch.zeros_like(tensor) for _ in range(WORLD_SIZE)] if RANK == 0 else None)
        torch.distributed.gather(tensor, gather_list, dst=0)
        return [t.item() for t in gather_list] if RANK == 0 else None

    torch_quant_ts = gather_benchmark(torch_quant)
    torch_dis_ts = gather_benchmark(torch_dis)
    torch_comb_ts = gather_benchmark(torch_comb)
    triton_dis_ts = gather_benchmark(triton_dis)
    triton_comb_ts = gather_benchmark(triton_comb)
    if RANK == 0:
        print("\n=== Results ===")
        headers = [
            "Rank", "Torch Quant (ms)", "Torch Dispatch (ms)", "Torch Combine (ms)", "Triton Dispatch (ms)",
            "Triton Combine (ms)"
        ]
        rows = [[
            r, f"{torch_quant_ts[r]:.3f}", f"{torch_dis_ts[r]:.3f}", f"{torch_comb_ts[r]:.3f}",
            f"{triton_dis_ts[r]:.3f}", f"{triton_comb_ts[r]:.3f}"
        ] for r in range(WORLD_SIZE)] + [[
            "Avg", f"{sum(torch_quant_ts)/WORLD_SIZE:.3f}", f"{sum(torch_dis_ts)/WORLD_SIZE:.3f}",
            f"{sum(torch_comb_ts)/WORLD_SIZE:.3f}", f"{sum(triton_dis_ts)/WORLD_SIZE:.3f}",
            f"{sum(triton_comb_ts)/WORLD_SIZE:.3f}"
        ]]
        print(tabulate(rows, headers=headers, floatfmt=".3f", tablefmt="grid"))
    torch.distributed.barrier()

    # check the correctness
    def check(out: torch.Tensor, ref: torch.Tensor, msg: str = "Triton"):
        try:
            torch.testing.assert_close(out.float(), ref.float(), rtol=1e-5, atol=1e-5)
            print(f"✅ RANK[{RANK}] check {msg} passed")
        except Exception as e:
            print(f"❌ RANK[{RANK}] check {msg} failed")
            raise e

    check(triton_out, ref_out, "Triton out")
    if args.with_scale or args.online_quant_fp8:
        check(triton_scale, ref_scale, "Triton scale")
    torch.distributed.destroy_process_group(EP_GROUP)
