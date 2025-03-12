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
from triton.language.extra import libshmem_device

import time
import argparse
import os
import sys
from typing import Optional, List
from cuda import cuda, cudart
import datetime
import numpy as np

import pynvshmem


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


def cp_engine_producer_all_gather_full_mesh_push(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
    for_correctness=False,
):
    M_per_rank, N = local_tensor.shape

    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.cuda.stream(ag_stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            time.sleep(3)
        for src_rank in rank_orders:
            if src_rank == rank:
                continue
            dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            dst.copy_(src)

            (err, ) = cuda.cuStreamWriteValue32(
                ag_stream.cuda_stream,
                barrier_buffers[rank][src_rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


@triton.jit
def __syncthreads():
    tl.inline_asm_elementwise(
        asm="""
        bar.sync 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


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


@triton.jit
def kernel_local_copy_and_barrier_all(
    rank,
    num_ranks,
    local_buf_ptr,
    global_buf_ptr,
    barrier_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = sm_id // num_pid_n
    pid_n = sm_id % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    data_ptr = local_buf_ptr + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
    dst_ptr = global_buf_ptr + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
    mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)
    mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)

    data = tl.load(data_ptr, mask=mask_data)
    tl.store(dst_ptr, data, mask=mask_dst)


@triton.jit
def barrier_all(rank, num_ranks, comm_buf_ptr):
    tid = thread_id(axis="x")
    sm_id = tl.program_id(axis=0)
    if tid < num_ranks:
        remote_ptr = libshmem_device.remote_ptr(comm_buf_ptr + sm_id * num_ranks + rank,
                                                tid.to(tl.int32)).to(tl.pointer_type(tl.int32))
        while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
            pass
        while (atomic_cas(comm_buf_ptr + sm_id * num_ranks + tid, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


def local_copy_and_barrier_all(rank, num_ranks, local_data, global_data, comm_buf, barrier_ptr, M_per_rank, N):
    barrier_all[(1, )](rank, num_ranks, comm_buf)
    grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel_local_copy_and_barrier_all[grid](rank, num_ranks, local_data, global_data, barrier_ptr, M_per_rank, N,
                                            local_data.stride(0), local_data.stride(1), global_data.stride(0),
                                            global_data.stride(1), 128, 256)
    barrier_ptr.fill_(0)
    # global_data[rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(local_data)
    (err, ) = cuda.cuStreamWriteValue32(
        torch.cuda.current_stream().cuda_stream,
        barrier_ptr[rank].data_ptr(),
        1,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    CUDA_CHECK(err)
    barrier_all[(1, )](rank, num_ranks, comm_buf)


# TMA related test
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr,
    comm_buf_ptr,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # swizzle m
            alpha = 0
            beta = 0
            pid_m = (pid_m + ((((rank ^ alpha) + beta) % num_ranks) * pid_ms_per_rank)) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            rank_beg = offs_am // M_per_rank
            rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            token = dl.wait(ready_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire")
            a_desc = dl.consume_token(a_desc, token)

        # You can also put the barrier here with a minor performance drop
        # if needs_wait:
        #     num_barriers_to_wait = num_barriers_wait_per_block
        #     token = dl.wait(ready_ptr + (ki * BLOCK_SIZE_K) // (K // num_ranks), num_barriers_to_wait, "gpu", "acquire")
        #     a_desc = dl.consume_token(a_desc, token)

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def ag_gemm_persistent(a, b, c, rank, num_ranks, workspace_tensors, barrier_tensors, comm_buf, for_correctness=False,
                       ag_stream=None, gemm_stream=None, serial=False, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, K = b.shape

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    gemm_stream = torch.cuda.current_stream() if gemm_stream is None else gemm_stream
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    gemm_stream.wait_stream(current_stream)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(
        NUM_SMS,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
    ), )

    def call_ag():
        cp_engine_producer_all_gather_full_mesh_push(
            rank,
            num_ranks,
            a,
            workspace_tensors,
            ag_stream,
            barrier_tensors,
            for_correctness=for_correctness,
        )

    if serial:
        call_ag()
        current_stream.wait_stream(ag_stream)
        torch.cuda.synchronize()
        # print(barrier_tensors[rank])
    else:
        call_ag()
    with torch.cuda.stream(gemm_stream):
        compiled = kernel_consumer_gemm_persistent[grid](
            workspace_tensors[rank],
            b,
            c,  #
            M,
            N_per_rank,
            K,  #
            rank,
            num_ranks,
            barrier_tensors[rank],
            comm_buf,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            8,
            False,
            NUM_SMS=NUM_SMS,  #
            num_stages=stages,
            num_warps=8,
            # cluster_dims=(2,2,1)
        )

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(gemm_stream)

    return compiled


@register_test("correctness_tma")
def test_ag_gemm_tma_intra_node(args):
    device = "cuda"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    M = 999 * num_ranks
    N = 1024
    K = 1024

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    workspaces = pynvshmem.nvshmem_create_tensor_list_intra_node([M, K], dtype)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], torch.int32)

    # at most 65536 blocks, each block world_size barriers
    max_blocks = 65536
    comm_buf = pynvshmem.nvshmem_create_tensor([max_blocks * num_ranks], torch.int32)
    comm_buf.fill_(0)
    barriers[rank].fill_(0)
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    debug = False

    ag_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.Stream()

    def func():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        # The following version doesn't rely on our customized barrier kernel
        # Just reuse nvshmem barrier, they should produce similar performance

        # barriers[rank].fill_(0)
        # pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
        # workspaces[rank][rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)
        # (err, ) = cuda.cuStreamWriteValue32(
        #     torch.cuda.current_stream().cuda_stream,
        #     barriers[rank][rank].data_ptr(),
        #     1,
        #     cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        # )
        # CUDA_CHECK(err)
        # pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)

        # Use our own customized barrier kernel
        local_copy_and_barrier_all(rank, num_ranks, A, workspaces[rank], comm_buf, barriers[rank], M_per_rank, K)
        compiled = ag_gemm_persistent(A, B, C, rank, num_ranks, workspaces, barriers, comm_buf, for_correctness=True,
                                      ag_stream=ag_stream, gemm_stream=gemm_stream, serial=False)
        if rank == 0 and debug:
            print(compiled.asm["ptx"])
        return C

    if rank == 0 and debug:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        func()
        os.environ["TRITON_ALWAYS_COMPILE"] = "0"
        os.environ["MLIR_ENABLE_DUMP"] = "0"

    for i in range(5):
        # every time, use a new input data to check correctness
        A.copy_(torch.randn([M_per_rank, K], dtype=dtype, device=device))
        B.copy_(torch.randn([N_per_rank, K], dtype=dtype, device=device))
        workspaces[rank].copy_(torch.randn([M, K], dtype=dtype, device=device))
        pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
        torch.cuda.synchronize()
        C = func()

    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=args.default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    for i in range(num_ranks):
        torch.distributed.barrier(args.default_group)
        if rank == i:
            print(f"Rank {rank}")
            if not torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3):
                print("Golden")
                print(C_golden)
                print("Output")
                print(C)
                print("Max diff", torch.max(torch.abs(C_golden - C)))
                print("Avg diff", torch.mean(torch.abs(C_golden - C)))
                print("Wrong Answer!")
            else:
                print("Pass!")


configs = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}


@register_test("perf_tma")
def test_perf_ag_gemm_tma_intra_node(args):
    device = "cuda"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    if args.shape_id:
        shape_config = configs[args.shape_id]
        M = shape_config["M"]
        N = shape_config["N"]
        K = shape_config["K"]
        BLOCK_M = shape_config["BM"]
        BLOCK_N = shape_config["BN"]
        BLOCK_K = shape_config["BK"]
        stages = shape_config["Stage"]
    else:
        M = 1024 * num_ranks
        N = 28672
        K = 8192
        BLOCK_M = 128
        BLOCK_N = 256
        BLOCK_K = 64
        stages = 3

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    workspaces = pynvshmem.nvshmem_create_tensor_list_intra_node([M, K], dtype)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], torch.int32)

    # at most 65536 blocks, each block world_size barriers
    max_blocks = 65536
    comm_buf = pynvshmem.nvshmem_create_tensor([max_blocks * num_ranks], torch.int32)
    comm_buf.fill_(0)
    barriers[rank].fill_(0)
    ag_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.Stream()
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    def func():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        # The following version doesn't rely on our customized barrier kernel
        # Just reuse nvshmem barrier, they should produce similar performance

        # barriers[rank].fill_(0)
        # pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
        # workspaces[rank][rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)
        # (err, ) = cuda.cuStreamWriteValue32(
        #     torch.cuda.current_stream().cuda_stream,
        #     barriers[rank][rank].data_ptr(),
        #     1,
        #     cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        # )
        # CUDA_CHECK(err)
        # pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)

        # Use our own customized barrier kernel
        local_copy_and_barrier_all(rank, num_ranks, A, workspaces[rank], comm_buf, barriers[rank], M_per_rank, K)
        ag_gemm_persistent(
            A,
            B,
            C,
            rank,
            num_ranks,
            workspaces,
            barriers,
            comm_buf,
            for_correctness=False,
            ag_stream=ag_stream,
            gemm_stream=gemm_stream,
            serial=False,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            stages=stages,
        )
        return C

    C, perf = perf_func(func, iters=1000, warmup_iters=200)
    dist_print(f"rank{RANK}", perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

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

    prof_dir = "prof/trace_ag_gemm_intra_node"
    os.makedirs(prof_dir, exist_ok=True)
    profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")
    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=args.default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    return perf


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
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
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
