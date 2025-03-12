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
import dataclasses
import triton
import triton.language as tl
import triton.distributed.language as dl
import random

import argparse
import os
from typing import Optional
from cuda import cuda
import datetime
import numpy as np

from functools import partial

import pynvshmem

from utils import (
    generate_data,
    get_torch_prof_ctx,
    perf_func,
    dist_print,
    CUDA_CHECK,
)

SIGNAL_DTYPE = torch.uint64


@dataclasses.dataclass
class GemmConfig:
    BLOCK_SIZE_M: int
    BLCOK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    STAGES: int


def get_gemm_config(M, N, K):
    return GemmConfig(128, 256, 64, 8, 3)


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
def barrier_all(rank, num_ranks, comm_buf_ptr):
    tid = thread_id(axis="x").to(tl.int32)
    sm_id = tl.program_id(axis=0)
    if tid < num_ranks:
        remote_ptr = dl.symm_at(comm_buf_ptr + sm_id * num_ranks + rank, tid)
        while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
            pass
        while (atomic_cas(comm_buf_ptr + sm_id * num_ranks + tid, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


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
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    barrier_ptr,
    counter_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):  #
    # Matmul using TMA and device-side descriptor creation
    rank = dl.rank()
    num_ranks = dl.num_ranks()
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

    nxt_rank = (rank + 1) % num_ranks
    M_per_rank = M // num_ranks
    rank_swizzle_offset = M_per_rank * nxt_rank // BLOCK_SIZE_M

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

            # rank swizzle
            pid_m = (pid_m + rank_swizzle_offset) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

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

            counter_start = offs_am // M_per_rank
            counter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
            counter_end = min(counter_end, num_ranks - 1)
            for counter_id in range(counter_start, counter_end + 1):
                m_start = M_per_rank * counter_id
                m_end = M_per_rank * (counter_id + 1) - 1
                tiled_m_start = m_start // BLOCK_SIZE_M
                tiled_m_end = m_end // BLOCK_SIZE_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                tiled_n = tl.cdiv(N, BLOCK_SIZE_N)
                val = tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")
                if val == tiled_m_size * tiled_n - 1:
                    dl.notify(barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu")
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def gemm_rs_producer_persistent(a, b, c, barrier, workspace, gemm_stream):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, STAGES = dataclasses.astuple(get_gemm_config(M, N, local_K))
    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(
        NUM_SMS,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    ), )

    with torch.cuda.stream(gemm_stream):
        compiled = kernel_gemm_rs_producer_persistent[grid](
            a,
            b,
            c,
            M,
            N,
            local_K,
            barrier,
            workspace,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            False,
            NUM_SMS=NUM_SMS,  #
            num_stages=STAGES,
            num_warps=8,
        )

    current_stream.wait_stream(gemm_stream)

    return compiled


@triton.jit
def kernel_consumer_reduce(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_rank, N]
    reduce_barrier_ptr,
    # shape of matrix
    M_per_rank,
    N,
    # reduce tile shape
    BLOCK_SIZE: tl.constexpr,
):
    rank = dl.rank()
    num_ranks = dl.num_ranks()
    pid = tl.program_id(axis=0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.where(offs < M_per_rank * N, offs, 0)

    out_ptrs = out_ptr + offs

    accum = tl.zeros((BLOCK_SIZE, ), dtype=out_ptr.dtype.element_ty)
    for i in range(0, num_ranks):
        cur_rank = (i + rank + 1) % num_ranks
        c_ptrs = c_ptr + offs + cur_rank * M_per_rank * N
        data = tl.load(c_ptrs)
        accum += data

    tl.store(out_ptrs, accum)


def torch_gemm_rs(
    input: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [N, local_K]
    bias: Optional[torch.Tensor],
    TP_GROUP,
):
    M, local_K = input.shape
    N = weight.shape[0]
    output = torch.matmul(input, weight.T)
    if bias:
        output = output + bias
    rs_output = torch.empty((M // WORLD_SIZE, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=TP_GROUP)
    return rs_output


class GemmRSIntraNode(torch.nn.Module):

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_M: int,
        N: int,
        K: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ):
        self.tp_group = tp_group
        self.rank: int = tp_group.rank()
        self.world_size = tp_group.size()
        self.max_M: int = max_M
        self.N = N
        self.K = K
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        self.scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([self.max_M, self.N], output_dtype)
        self.scatter_barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([self.world_size], SIGNAL_DTYPE)
        self.reduce_barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([self.world_size], SIGNAL_DTYPE)

        self.scatter_buf = self.scatter_bufs[self.rank]
        self.scatter_barrier = self.scatter_barriers[self.rank]
        self.reduce_barrier = self.reduce_barriers[self.rank]

        self.scatter_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
        self.reduce_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

        self.max_blocks = 65536
        self.sync_buf = pynvshmem.nvshmem_create_tensor([self.max_blocks * self.world_size], torch.int32)
        self.sync_buf.fill_(0)

    def wait_eq(self, ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
        if not require_i64:
            (err, ) = cuda.cuStreamWaitValue32(
                stream.cuda_stream,
                ptr,
                signal,
                cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
            )
        else:
            (err, ) = cuda.cuStreamWaitValue64(
                stream.cuda_stream,
                ptr,
                signal,
                cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
            )
        CUDA_CHECK(err)

    def set_signal(self, ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
        if not require_i64:
            (err, ) = cuda.cuStreamWriteValue32(
                stream.cuda_stream,
                ptr,
                signal,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
        else:
            (err, ) = cuda.cuStreamWriteValue64(
                stream.cuda_stream,
                ptr,
                signal,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
        CUDA_CHECK(err)

    def cp_engin_scatter_and_notify_push_mode(self, gemm_out  # [M, N]
                                              ):
        M = gemm_out.shape[0]
        M_per_rank = M // self.world_size

        with torch.cuda.stream(self.scatter_stream):
            for i in range(0, self.world_size):
                remote_rank = (self.rank + i + 1) % self.world_size
                self.wait_eq(self.scatter_barrier[remote_rank].data_ptr(), 1,  # signal
                             self.scatter_stream, True)
                remote_buf = self.scatter_bufs[remote_rank][self.rank * M_per_rank:(self.rank + 1) * M_per_rank, :]
                local_buf = gemm_out[remote_rank * M_per_rank:(remote_rank + 1) * M_per_rank, :]
                remote_buf.copy_(local_buf)
                self.set_signal(
                    self.reduce_barriers[remote_rank][self.rank].data_ptr(),
                    1,
                    self.scatter_stream,
                    require_i64=True,
                )

    def ring_reduce_after_scatter(
        self,
        scatter_out,  # [M, N]
        stream,
    ):
        M, N = scatter_out.shape
        M_per_rank = M // self.world_size
        output = torch.empty((M_per_rank, N), dtype=self.output_dtype, device=input.device)
        grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]), )
        with torch.cuda.stream(stream):
            kernel_consumer_reduce[grid](
                scatter_out,
                output,
                self.reduce_barrier,
                M_per_rank,
                N,
                BLOCK_SIZE=4096,
                num_warps=8,
            )

        return output

    def barrier_all_on_stream(
        self,
        stream,
    ):

        with torch.cuda.stream(stream):
            barrier_all[(1, )](self.rank, self.world_size, self.sync_buf)
        # pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)

    def forward(
        self,
        input: torch.Tensor,  # [M, local_K]
        weight: torch.Tensor,  # [N, local_K]
        bias: Optional[torch.Tensor],
    ):
        M, local_K = input.shape
        N = weight.shape[0]
        assert N == self.N

        assert M % self.world_size == 0
        assert weight.shape[1] == local_K
        local_M = M // self.world_size
        current_stream = torch.cuda.current_stream()
        self.barrier_all_on_stream(current_stream)
        self.scatter_stream.wait_stream(current_stream)
        self.reduce_stream.wait_stream(current_stream)

        # self.scatter_barrier.fill_(1)
        output = torch.empty((local_M, N), dtype=self.output_dtype, device=input.device)
        workspace = torch.zeros((self.world_size, ), dtype=torch.int32, device=input.device)
        gemm_out = torch.empty((M, N), dtype=self.output_dtype, device=input.device)
        gemm_rs_producer_persistent(input, weight, gemm_out, self.scatter_barrier, workspace, current_stream)

        # torch.distributed.reduce_scatter_tensor(output, gemm_out, group=TP_GROUP)
        self.cp_engin_scatter_and_notify_push_mode(gemm_out)
        current_stream.wait_stream(self.scatter_stream)
        self.barrier_all_on_stream(current_stream)

        output = self.ring_reduce_after_scatter(self.scatter_buf[:M], current_stream)
        current_stream.wait_stream(current_stream)
        self.reduce_barrier.zero_()
        self.scatter_barrier.zero_()

        return output


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--verify-iters", default=10, type=int)

    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=True,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()

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
    random.seed(args.seed)

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_K = args.K // TP_GROUP.size()

    scale = TP_GROUP.rank() + 1

    def _make_data(M):
        data_config = [
            ((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
            ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((M, args.N), input_dtype, (1, 0))),
        ]
        generator = generate_data(data_config)
        input, weight, bias = next(generator)
        return input, weight, bias

    dist_gemm_rs_op = GemmRSIntraNode(TP_GROUP, args.M, args.N, args.K, input_dtype, output_dtype)

    if args.check:
        for n in range(args.iters):
            torch.cuda.empty_cache()
            input_list = [
                _make_data(random.randint(1, args.M // WORLD_SIZE) * WORLD_SIZE) for _ in range(args.verify_iters)
            ]
            dist_out_list, torch_out_list = [], []

            # torch impl
            for input, weight, bias in input_list:
                torch_out = torch_gemm_rs(
                    input,
                    weight,
                    bias,
                    TP_GROUP,
                )
                torch_out_list.append(torch_out)

            # dist triton impl
            for input, weight, bias in input_list:
                dist_out = dist_gemm_rs_op.forward(input, weight, bias)
                dist_out_list.append(dist_out)
            # torch.cuda.synchronize()
            # verify
            for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
                # if RANK == 0:
                #     print(f"shape = {torch_out.shape}, {torch_out[0]} {dist_out[0]}")
                try:
                    torch.testing.assert_close(torch_out, dist_out, atol=atol, rtol=rtol)
                except Exception as e:
                    raise e
        print(f"RANK[{RANK}]: pass.")
        exit(0)

    ctx = get_torch_prof_ctx(args.profile)
    input, weight, bias = _make_data(args.M)
    with ctx:
        torch_output, torch_perf = perf_func(partial(torch_gemm_rs, input, weight, bias, TP_GROUP), iters=100,
                                             warmup_iters=20)

        pynvshmem.nvshmem_barrier_all()
        torch.cuda.synchronize()

        dist_triton_output, dist_triton_perf = perf_func(partial(dist_gemm_rs_op.forward, input, weight, bias),
                                                         iters=100, warmup_iters=20)

    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
    torch.testing.assert_close(torch_output, dist_triton_output, atol=atol, rtol=rtol)
    torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", dist_triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    torch.distributed.destroy_process_group()
