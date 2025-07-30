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

import argparse
import datetime
import os
import sys
from typing import List, Optional, Tuple

import math
import numpy as np
import pytest
import torch
import nvshmem.core

from triton_dist.kernels.nvidia import (mla_decode, mla_inter_rank_combine)
from triton_dist.utils import dist_print, perf_func, init_nvshmem_by_torch_process_group

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


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(
            s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q,
                  h_kv, d, dv, causal, dtype):
    # q: [b, s_q, h_q, d]
    # block_table: [b, max_seqlen_pad // block_size]
    # blocked_k: [b * max_seqlen_pad // block_size, block_size, h_kv, d]
    # cache_seqlens: [b]
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q,
                h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out.to(dtype), lse.to(dtype)

    out_torch, _ = ref_mla()
    return out_torch


NUM_BLOCKS = 32000  # Large enough to test overflow in index calculation.



@register_test("correctness_single")
def test_tilelang_mla_decode_with_paged_kv(args):
    b = 1
    h_q = 128
    h_kv = 1
    cache_seqlen = 8192
    d = 576
    dv = 512
    device = "cuda"
    dtype = torch.float16

    s_q = 1  # for decode, s_q = 1
    block_size = 64
    cache_seqlens = torch.tensor([cache_seqlen + 2 * i for i in range(b)],
                                 dtype=torch.int32,
                                 device=device)
    causal = True

    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256

    q = torch.randn(b, s_q, h_q, d, dtype=dtype, device=device)
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32,
        device=device).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=dtype, device=device)
    out_internal = mla_decode(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, dtype)
    out_flash = out_internal[..., :-1]
    
    out_ref = run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q,
                            cache_seqlens, h_q, h_kv, d, dv, causal, dtype)
    
    torch.testing.assert_close(out_flash, out_ref, rtol=0.01, atol=0.01)
    


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

        _, perf = perf_func(func, iters=100, warmup_iters=20)

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
    init_nvshmem_by_torch_process_group(TP_GROUP)

    args = get_args()
    args.default_group = TP_GROUP
    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
