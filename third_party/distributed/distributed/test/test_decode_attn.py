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
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_decode_attention.py
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flashinfer.py
# which was originally adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py
#
# The original copyright is:
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
################################################################################

import torch
import triton
import triton.language as tl
import pytest
from typing import List, Optional, Tuple

import argparse
import os
import sys
from cuda import cuda, cudart
import datetime
import numpy as np
import pynvshmem

from triton.distributed.kernels import (gqa_fwd_batch_decode_persistent, gqa_fwd_batch_decode_persistent_aot,
                                        gqa_fwd_batch_decode, gqa_fwd_batch_decode_aot)


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        output = func()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


def dist_print(*args, **kwargs):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    prefix = False
    if "allowed_ranks" in kwargs:
        allowed_ranks = kwargs["allowed_ranks"]
        if isinstance(allowed_ranks, str) and allowed_ranks == "all":
            allowed_ranks = list(range(world_size))

        del kwargs["allowed_ranks"]
    else:
        allowed_ranks = [0]
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]

        del kwargs["prefix"]

    need_sync = False
    if "need_sync" in kwargs:
        need_sync = kwargs["need_sync"]

        del kwargs["need_sync"]

    for allowed in allowed_ranks:
        if need_sync:
            torch.distributed.barrier()
        if rank == allowed:
            if prefix:
                print(f"[rank:{rank}]", end="")
            print(*args, **kwargs)


def broadcast_cpu(tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


def init_nvshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = pynvshmem.nvshmemx_get_uniqueid()
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.numpy().tobytes()
    pynvshmem.nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)


ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="")

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@tl.core.extern
def thread_id(axis: tl.constexpr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %tid.{axis.value};",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def __syncthreads(_builder=None):
    return tl.core.inline_asm_elementwise(
        asm="""
        bar.sync 0;
        mov.u32 $0, 0;
        """,
        constraints="=r",  # force have a return value, even not used.
        args=[],
        dtype=tl.uint32,
        is_pure=False,  # no optimize this!
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def __atomic_add(
    ptr,
    value,
    scope: tl.constexpr = "gpu",
    semantic: tl.constexpr = "relaxed",
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"atom.{semantic.value}.{scope.value}.global.add.s32 $0, [$1], $2;",
        constraints=("=r,l,r"),
        args=[
            ptr,
            value,
        ],
        is_pure=False,
        pack=1,
        dtype=tl.int32,
        _builder=_builder,
    )


@triton.jit
def atomic_add(barrier_ptr, value, scope: tl.constexpr, semantic: tl.constexpr):
    """custom atomic_add implementation using extern_elementwise

    :param scope: one of "gpu", "sys". default to "gpu"
    :param semantic: one of "release", "acquire", "relaxed", "acq_rel". default to "relaxed"
    :returns: the result of atomic_add
    :rtype: int
    """
    return __atomic_add(barrier_ptr, value, scope, semantic)


NUM_BLOCKS = 32000  # Large enough to test overflow in index calculation.


@pytest.mark.parametrize("kv_lens", [[1320, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", [(16, 16), (32, 8), (64, 8), (6, 1)])
@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("block_size", [1, 16])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("soft_cap", [0, 30, 50])
def test_triton_decode_with_paged_kv(
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(NUM_BLOCKS + 1, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].contiguous()
    value_cache = key_value_cache[:, 1, :, :, :].contiguous()
    workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    output = gqa_fwd_batch_decode(query, key_cache, value_cache, workspace, [1] * num_seqs,
                                  torch.tensor(kv_lens, dtype=torch.int32, device=query.device), block_tables, scale,
                                  soft_cap)

    ref_output = ref_paged_attn(query=query, key_cache=key_cache, value_cache=value_cache, query_lens=[1] * num_seqs,
                                kv_lens=kv_lens, block_tables=block_tables, scale=scale, soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("kv_lens", [[32], [1320], [18], [463], [1], [54], [293], [70]])
@pytest.mark.parametrize("num_heads", [(96, 12)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("soft_cap", [0])
def test_triton_decode_with_paged_kv_aot(
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].contiguous()
    value_cache = key_value_cache[:, 1, :, :, :].contiguous()
    workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    stream = torch.cuda.current_stream()
    for i in range(10):
        output = gqa_fwd_batch_decode_aot(stream, query, key_cache, value_cache, workspace, [1] * num_seqs,
                                          torch.tensor(kv_lens, dtype=torch.int32, device=query.device), block_tables,
                                          scale, soft_cap)

    ref_output = ref_paged_attn(query=query, key_cache=key_cache, value_cache=value_cache, query_lens=[1] * num_seqs,
                                kv_lens=kv_lens, block_tables=block_tables, scale=scale, soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("kv_lens", [[1320], [18], [463], [1], [54], [293], [70]])
@pytest.mark.parametrize("num_heads", [(64, 8)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("soft_cap", [0])
def test_triton_decode_with_paged_kv_persistent(
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].contiguous()
    value_cache = key_value_cache[:, 1, :, :, :].contiguous()
    workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    output = gqa_fwd_batch_decode_persistent(query, key_cache, value_cache, workspace, [1] * num_seqs,
                                             torch.tensor(kv_lens, dtype=torch.int32, device=query.device),
                                             block_tables, scale, soft_cap)

    ref_output = ref_paged_attn(query=query, key_cache=key_cache, value_cache=value_cache, query_lens=[1] * num_seqs,
                                kv_lens=kv_lens, block_tables=block_tables, scale=scale, soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("kv_lens", [[32], [1320], [18], [463], [1], [54], [293], [70]])
@pytest.mark.parametrize("num_heads", [(96, 12)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("soft_cap", [0])
def test_triton_decode_with_paged_kv_persistent_aot(
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].contiguous()
    value_cache = key_value_cache[:, 1, :, :, :].contiguous()
    workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    stream = torch.cuda.current_stream()
    output = gqa_fwd_batch_decode_persistent_aot(stream, query, key_cache, value_cache, workspace, [1] * num_seqs,
                                                 torch.tensor(kv_lens, dtype=torch.int32, device=query.device),
                                                 block_tables, scale, soft_cap)

    ref_output = ref_paged_attn(query=query, key_cache=key_cache, value_cache=value_cache, query_lens=[1] * num_seqs,
                                kv_lens=kv_lens, block_tables=block_tables, scale=scale, soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@register_test("perf_8k")
def perf_8k_decode(args):
    for kv_len in [2**i for i in range(15, 16)]:
        kv_lens = [kv_len]
        torch.set_default_device("cuda")
        num_seqs = len(kv_lens)
        num_query_heads = 96
        num_kv_heads = 12
        head_size = 128
        assert num_query_heads % num_kv_heads == 0
        max_kv_len = max(kv_lens)
        scale = head_size**-0.5
        dtype = torch.float16
        soft_cap = 0.0

        block_size = 1
        NUM_BLOCKS = 2**16 + 100

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

        key_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

        kv_split = 32
        output_split = torch.empty([num_seqs, num_query_heads, kv_split, head_size + 1], dtype=torch.float32)
        output_combine = torch.empty([num_seqs, num_query_heads, head_size], dtype=query.dtype)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
        # block_tables = torch.arange(0, (num_seqs * max_num_blocks_per_seq)).view(num_seqs, -1).to(torch.int32)
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=query.device)

        def func():
            gqa_fwd_batch_decode(query, key_cache, value_cache, workspace, [1] * num_seqs, kv_lens, block_tables, scale,
                                 soft_cap, output_split=output_split, output_combine=output_combine, kv_split=kv_split)

        _, perf = perf_func(func, iters=1000, warmup_iters=200)

        torch.distributed.barrier(args.default_group)
        dist_print(f"rank: {args.rank} KV len={kv_len} Performance is {perf} ms", allowed_ranks="all", need_sync=True)

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=True,
                profile_memory=True,
        ) as profiler:
            for i in range(20):
                func()

        prof_dir = f"prof/trace_flash_decode_kvlen_{kv_len}"
        os.makedirs(prof_dir, exist_ok=True)
        profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")


@register_test("perf_8k_aot")
def perf_8k_decode_aot(args):
    for kv_len in [2**i for i in range(0, 20)]:
        kv_lens = [kv_len]
        torch.set_default_device("cuda")
        num_seqs = len(kv_lens)
        num_query_heads = 96
        num_kv_heads = 12
        head_size = 128
        assert num_query_heads % num_kv_heads == 0
        max_kv_len = max(kv_lens)
        scale = head_size**-0.5
        dtype = torch.float16
        soft_cap = 0

        block_size = 1
        NUM_BLOCKS = 2**20 + 100

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

        key_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

        kv_split = 32
        output_split = torch.empty([num_seqs, num_query_heads, kv_split, head_size + 1], dtype=torch.float16)
        output_combine = torch.empty([num_seqs, num_query_heads, head_size], dtype=query.dtype)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        # block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
        block_tables = torch.arange(0, (num_seqs * max_num_blocks_per_seq)).view(num_seqs, -1).to(torch.int32)
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=query.device)

        stream = torch.cuda.current_stream()

        def func():
            gqa_fwd_batch_decode_aot(stream, query, key_cache, value_cache, workspace, [1] * num_seqs, kv_lens,
                                     block_tables, scale, soft_cap, output_split=output_split,
                                     output_combine=output_combine, kv_split=kv_split)

        _, perf = perf_func(func, iters=1000, warmup_iters=200)

        torch.distributed.barrier(args.default_group)
        dist_print(f"rank: {args.rank} KV len={kv_len} Performance is {perf} ms", allowed_ranks="all", need_sync=True)

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=True,
                profile_memory=True,
        ) as profiler:
            for i in range(20):
                func()

        prof_dir = f"prof/trace_flash_decode_kvlen_{kv_len}_aot"
        os.makedirs(prof_dir, exist_ok=True)
        profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")


@register_test("perf_8k_persistent")
def perf_8k_decode_persistent(args):
    for kv_len in [2**i for i in range(16)]:
        kv_lens = [kv_len]
        torch.set_default_device("cuda")
        num_seqs = len(kv_lens)
        num_query_heads = 96
        num_kv_heads = 12
        head_size = 128
        assert num_query_heads % num_kv_heads == 0
        max_kv_len = max(kv_lens)
        scale = head_size**-0.5
        dtype = torch.float16
        soft_cap = 0

        block_size = 1
        NUM_BLOCKS = 2**13 + 100

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

        key_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

        kv_split = 32
        output_split = torch.empty([num_seqs, num_query_heads, kv_split, head_size + 1], dtype=query.dtype)
        output_combine = torch.empty([num_seqs, num_query_heads, head_size], dtype=query.dtype)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=query.device)

        def func():
            gqa_fwd_batch_decode_persistent(query, key_cache, value_cache, workspace, [1] * num_seqs, kv_lens,
                                            block_tables, scale, soft_cap, output_split=output_split,
                                            output_combine=output_combine, kv_split=kv_split)

        _, perf = perf_func(func, iters=1000, warmup_iters=200)

        torch.distributed.barrier(args.default_group)
        dist_print(f"rank: {args.rank} KV len={kv_len} Performance is {perf} ms", allowed_ranks="all", need_sync=True)

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=True,
                profile_memory=True,
        ) as profiler:
            for i in range(20):
                func()

        prof_dir = f"prof/trace_flash_decode_kvlen_{kv_len}_persistent"
        os.makedirs(prof_dir, exist_ok=True)
        profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")


@register_test("perf_8k_persistent_aot")
def perf_8k_decode_persistent_aot(args):
    for kv_len in [2**i for i in range(14)]:
        kv_lens = [kv_len]
        torch.set_default_device("cuda")
        num_seqs = len(kv_lens)
        num_query_heads = 96
        num_kv_heads = 12
        head_size = 128
        assert num_query_heads % num_kv_heads == 0
        max_kv_len = max(kv_lens)
        scale = head_size**-0.5
        dtype = torch.float16
        soft_cap = 0

        block_size = 1
        NUM_BLOCKS = 2**13 + 100

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

        key_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache = torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype)
        workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

        kv_split = 32
        output_split = torch.empty([num_seqs, num_query_heads, kv_split, head_size + 1], dtype=query.dtype)
        output_combine = torch.empty([num_seqs, num_query_heads, head_size], dtype=query.dtype)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=query.device)
        stream = torch.cuda.current_stream()

        def func():
            gqa_fwd_batch_decode_persistent_aot(stream, query, key_cache, value_cache, workspace, [1] * num_seqs,
                                                kv_lens, block_tables, scale, soft_cap, output_split=output_split,
                                                output_combine=output_combine, kv_split=kv_split)

        _, perf = perf_func(func, iters=1000, warmup_iters=200)

        torch.distributed.barrier(args.default_group)
        dist_print(f"rank: {args.rank} KV len={kv_len} Performance is {perf} ms", allowed_ranks="all", need_sync=True)

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=True,
                profile_memory=True,
        ) as profiler:
            for i in range(20):
                func()

        prof_dir = f"prof/trace_flash_decode_kvlen_{kv_len}_persistent_aot"
        os.makedirs(prof_dir, exist_ok=True)
        profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
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

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    init_nvshmem_by_uniqueid(TP_GROUP)
    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    args = get_args()
    args.default_group = TP_GROUP
    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    torch.distributed.destroy_process_group()
