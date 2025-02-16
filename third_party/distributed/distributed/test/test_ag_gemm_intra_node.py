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

import time
import argparse
import os
import sys
from typing import Optional, List
from cuda import cuda, cudart
import datetime

import pynvshmem
from triton_nvshmem.nvshmem import (
    nvshmem_ptr, )


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

    rank_orders = [
        rank,
        rank ^ 1,
        (rank + 4) % num_ranks,
        ((rank + 4) ^ 1) % num_ranks,
        (rank + 2) % num_ranks,
        ((rank + 2) ^ 1) % num_ranks,
        (rank + 6) % num_ranks,
        ((rank + 6) ^ 1) % num_ranks,
    ]

    with torch.cuda.stream(ag_stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            time.sleep(3)
        for target_rank in rank_orders:
            # swizzle m
            dst = remote_tensor_buffers[target_rank][rank * M_per_rank:(rank + 1) * M_per_rank, :]
            src = local_tensor
            dst.copy_(src)

            (err, ) = cuda.cuStreamWriteValue32(
                ag_stream.cuda_stream,
                barrier_buffers[target_rank][rank].data_ptr(),
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
    need_tail_reset: tl.constexpr,
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
            alpha = 1
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

    # TODO(zhengsize):
    # Triton will incorrectly emit a predicate for store here, so tail reset doesn't
    # work unless we add a new op
    if need_tail_reset:
        # reset buffers
        tid = thread_id(axis="x")
        sm_id = tl.program_id(axis=0)

        if tid < num_ranks:
            tl.store(ready_ptr + tid + 1, 0)
            remote_ptr = nvshmem_ptr(comm_buf_ptr + sm_id * num_ranks + rank, tid)
            while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
                pass
            while (atomic_cas(comm_buf_ptr + sm_id * num_ranks + tid, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()


def ag_gemm_persistent(
    a,
    b,
    c,
    rank,
    num_ranks,
    workspace_tensors,
    barrier_tensors,
    comm_buf,
    for_correctness=False,
    need_tail_reset=False,
):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, K = b.shape

    ag_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    cp_engine_producer_all_gather_full_mesh_push(
        rank,
        num_ranks,
        a,
        workspace_tensors,
        ag_stream,
        barrier_tensors,
        for_correctness=for_correctness,
    )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(
        NUM_SMS,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
    ), )

    # current_stream.wait_stream(ag_stream)
    # torch.cuda.synchronize()
    # print(barrier_tensors[rank])

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
        128,
        128,
        32,
        8,
        True,
        NUM_SMS=NUM_SMS,  #
        need_tail_reset=need_tail_reset,
        num_stages=3,
        num_warps=8,
    )

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

    # at most NUM_SMS blocks, each block world_size barriers
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    comm_buf = pynvshmem.nvshmem_create_tensor([NUM_SMS * num_ranks], torch.int32)
    comm_buf.fill_(0)
    barriers[rank].fill_(0)
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    need_tail_reset = False
    debug = False

    def func():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        compiled = ag_gemm_persistent(
            A,
            B,
            C,
            rank,
            num_ranks,
            workspaces,
            barriers,
            comm_buf,
            for_correctness=True,
            need_tail_reset=need_tail_reset,
        )
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
        workspaces[rank].fill_(-1)
        if not need_tail_reset:
            # reset at beginning
            barriers[rank].fill_(0)
        else:
            print(barriers[rank])
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
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    print("Pass!")


@register_test("perf_tma")
def test_perf_ag_gemm_tma_intra_node(args):
    device = "cuda"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    M = 1024 * num_ranks
    N = 11008
    K = 4096

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    workspaces = pynvshmem.nvshmem_create_tensor_list_intra_node([M, K], dtype)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], torch.int32)

    # at most NUM_SMS blocks, each block world_size barriers
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    comm_buf = pynvshmem.nvshmem_create_tensor([NUM_SMS * num_ranks], torch.int32)
    comm_buf.fill_(0)
    barriers[rank].fill_(0)
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    def func():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        barriers[rank].fill_(0)
        pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
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
            need_tail_reset=False,
        )
        return C

    C, perf = perf_func(func, iters=100, warmup_iters=10)
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
