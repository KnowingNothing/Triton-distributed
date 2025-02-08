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
import torch
import triton
import triton.language as tl
import triton.distributed.language as dl

import os
import time


@triton.jit
def kernel_consumer_gemm(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Distributed parameters
    rank,
    num_ranks,
    ready_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    is_fp8: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    dtype = tl.float16 if not is_fp8 else tl.float8e4nv
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    num_k_blocks_per_rank = num_k_blocks // num_ranks
    # a_ptrs = dl.wait(a_ptrs, ready_ptr, "gpu", "acquire")
    for k in range(0, num_k_blocks):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a_ptrs = dl.wait(a_ptrs, ready_ptr + k // (num_k_blocks_per_rank), "gpu", "acquire")
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(dtype)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    
def consumer_gemm(A, B, C, rank, num_ranks, barrier):
    M, K = A.shape
    _, N = B.shape
    grid = lambda META: (
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )
    compiled = kernel_consumer_gemm[grid](
        A, B, C,
        rank, num_ranks, barrier,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        128, 128, 32,
        8,
        False,
        num_stages=4
    )
    return compiled

def test_lower_wait():
    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["MLIR_ENABLE_DUMP"] = "0"
    
    device="cuda"
    dtype=torch.float16
    
    rank = 0
    num_ranks = 8
    barrier_tensor = torch.ones([num_ranks], dtype=torch.int32, device=device)
    M = 1024
    N = 1024
    K = 1024
    
    assert M % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    
    ag_A = torch.randn([M, K], dtype=dtype, device=device)
    B = torch.randn([K, N_per_rank], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
    
    compiled = consumer_gemm(ag_A, B, C, rank, num_ranks, barrier_tensor)
    print(compiled.asm["ptx"])
    
    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["MLIR_ENABLE_DUMP"] = "0"

    
def test_1024_gemm_single_device():
    device="cuda"
    dtype=torch.float16
    rank = 0
    num_ranks = 8
    # TODO(zhengsize): why needs + 1?
    barrier_tensor = torch.zeros([num_ranks + 1], dtype=torch.int32, device=device)
    M = 1024
    N = 1024
    K = 1024
    
    assert M % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    
    ag_A = torch.randn([M, K], dtype=dtype, device=device)
    B = torch.randn([K, N_per_rank], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
    
    C_golden = torch.matmul(ag_A, B)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        consumer_gemm(ag_A, B, C, rank, num_ranks, barrier_tensor)
    
    print("Consumer GEMM launched!")
    print("signals are:")
    print(barrier_tensor)
    print("sleeping...")
    time.sleep(6)
    print("wake up!")
    barrier_tensor.fill_(1)
    print("signals are:")
    print(barrier_tensor)
    
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    print("Pass!")
    
if __name__ == "__main__":
    # test_lower_wait()
    test_1024_gemm_single_device()
    
    