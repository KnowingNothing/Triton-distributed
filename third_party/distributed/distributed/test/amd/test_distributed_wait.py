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
import triton
import triton.language as tl
import triton.distributed.language as dl

import argparse
import os
import sys

DEVICE = triton.runtime.driver.active.get_active_torch_device()

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

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


@triton.jit
def consumer_task_kernel(
    # Pointers to inputs Matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Distributed parameters
    rank,
    num_ranks,
    ready_ptr,
    # Input Matrix dimensions
    M,
    N,
    stride_am,
    stride_an,  #
    stride_bm,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_n

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    num_barriers_to_wait = 1
    token = dl.wait(ready_ptr, num_barriers_to_wait, "gpu", "acquire")
    a_ptrs = dl.consume_token(a_ptrs, token)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    c = a + b
    tl.store(c_ptrs, c, mask=mask)


def consumer_task(A, B, C, rank, num_ranks, barrier, needs_wait=True):
    M, N = A.shape
    assert A.shape == B.shape and A.shape == C.shape
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    compiled = consumer_task_kernel[grid](A, B, C, rank, num_ranks, barrier, M, N, *A.stride(), *B.stride(),
                                          *C.stride(), BLOCK_SIZE_M, BLOCK_SIZE_N)

    return compiled


@register_test("lower")
def test_lower_wait(args):
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "1"

    device = DEVICE
    dtype = torch.float16

    rank = 0
    num_ranks = 8
    barrier_tensor = torch.ones([num_ranks], dtype=torch.int32, device=device)
    M = 1024
    N = 1024

    assert M % num_ranks == 0
    M_per_rank = M // num_ranks  # noqa: F841
    N_per_rank = N // num_ranks

    A = torch.randn([M, N_per_rank], dtype=dtype, device=device)
    B = torch.randn([M, N_per_rank], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)

    compiled = consumer_task(A, B, C, rank, num_ranks, barrier_tensor)
    print(compiled.asm["ptx"])

    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["MLIR_ENABLE_DUMP"] = "0"


if __name__ == "__main__":
    args = get_args()
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)
