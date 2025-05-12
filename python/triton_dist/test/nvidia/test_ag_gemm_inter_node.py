import argparse
import datetime
import os
import sys

import torch
import torch.distributed

from triton_dist import pynvshmem
from triton_dist.autotuner import contextual_autotune
from triton_dist.kernels.nvidia import (
    ag_gemm_inter_node,
    create_ag_gemm_inter_node_context,
    gemm_non_persistent,
    gemm_persistent,
    get_auto_all_gather_method,
)
from triton_dist.utils import dist_print, group_profile, perf_func, assert_allclose

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = int(os.environ.get("MASTER_PORT", 10000))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

torch.cuda.set_device(LOCAL_RANK)

torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.getenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

torch.distributed.init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=RANK,
    timeout=datetime.timedelta(seconds=1800),
)

TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

device = "cuda"
dtype = torch.bfloat16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--ag_only", default=False, action="store_true")
    parser.add_argument("--gemm_only", default=False, action="store_true")
    parser.add_argument("--autotune", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.get_device_capability() >= (9, 0),
    )
    return parser.parse_args()


def perf_test_gemm_only(name, M, config, persistent: bool = True):
    print(f">>> GEMM-ONLY of {name}")
    N = config["N"]
    K = config["K"]

    assert M % WORLD_SIZE == 0
    assert N % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE
    N_per_rank = N // WORLD_SIZE

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda")
    ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(ag_buffer, A, TP_GROUP)

    ctx = create_ag_gemm_inter_node_context(
        A,
        B,
        RANK,
        WORLD_SIZE,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        stages=config["stage"],
        ag_stream=torch.cuda.Stream(),
        gemm_stream=torch.cuda.Stream(),
        autotune=args.autotune,
    )

    def triton_func():
        if persistent:
            return gemm_persistent(ag_buffer, B, ctx=ctx)
        else:
            return gemm_non_persistent(ag_buffer, B, ctx=ctx)

    if args.autotune:
        _func = triton_func
        ctx.autotune = True
        triton_func = contextual_autotune(is_dist=True)(lambda: _func())

    if RANK == 0 and args.debug:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        triton_func()
        os.environ["TRITON_ALWAYS_COMPILE"] = "0"
        os.environ["MLIR_ENABLE_DUMP"] = "0"

    C = triton_func()

    def torch_func():
        return torch.matmul(ag_buffer, B.T)

    C_golden = torch_func()

    for i in range(WORLD_SIZE):
        torch.distributed.barrier(TP_GROUP)
        if RANK == i:
            assert_allclose(C_golden, C, atol=1e-3, rtol=1e-3)

    with group_profile("trace_gemm_only_inter_node", do_prof=args.profile, group=TP_GROUP):
        _, triton_perf = perf_func(triton_func, iters=10, warmup_iters=10)
        _, torch_perf = perf_func(torch_func, iters=10, warmup_iters=10)

        dist_print(
            f"Rank {RANK} gemm perf: triton={triton_perf:.2f} ms, torch={torch_perf:.2f} ms, speedup {torch_perf/triton_perf:.2f}",
            need_sync=True,
            allowed_ranks=list(range(WORLD_SIZE)),
        )


def perf_test_ag_only(name, M, config):
    N = config["N"]
    K = config["K"]

    assert M % WORLD_SIZE == 0
    assert N % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE
    N_per_rank = N // WORLD_SIZE

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda")
    ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    internode_ag_stream = torch.cuda.Stream(priority=-1)

    ctx = create_ag_gemm_inter_node_context(
        A,
        B,
        RANK,
        WORLD_SIZE,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        stages=config["stage"],
        ag_stream=torch.cuda.Stream(),
        gemm_stream=torch.cuda.Stream(),
        autotune=args.autotune,
    )

    all_gather_method = get_auto_all_gather_method(LOCAL_WORLD_SIZE, WORLD_SIZE)
    print(f"use all_gather_method: {all_gather_method}", flush=True)

    def triton_func():
        local_copy_and_barrier_all(
            ctx.rank,
            ctx.num_ranks,
            A,
            ctx.workspace_tensors[ctx.local_rank],
            ctx.comm_buf,
            ctx.barrier_tensors[ctx.local_rank],
            M_per_rank,
            K,
            phase=ctx.phase,
            is_internode=True,
        )
        ctx.phase += 2
        inter_node_allgather(
            A,
            ctx.workspace_tensors,
            ctx.barrier_tensors,
            1,
            RANK,
            LOCAL_WORLD_SIZE,
            WORLD_SIZE,
            torch.cuda.current_stream(),
            internode_ag_stream,
            True,  # cpengine_dispatch
            all_gather_method=all_gather_method,
        )
        pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
        return ctx.workspace_tensors[LOCAL_RANK]

    C = triton_func()

    def torch_func():
        torch.distributed.all_gather_into_tensor(ag_buffer, A, TP_GROUP)
        return ag_buffer

    C_golden = torch_func()

    for i in range(WORLD_SIZE):
        torch.distributed.barrier(TP_GROUP)
        if RANK == i:
            assert_allclose(C_golden, C, atol=1e-3, rtol=1e-3)

    ag_size = M * K * A.element_size() / 1024 / 1024 / 1024  # GB
    with group_profile("trace_ag_only_inter_node", do_prof=args.profile, group=TP_GROUP):
        _, triton_perf = perf_func(triton_func, iters=10, warmup_iters=10)
        _, torch_perf = perf_func(torch_func, iters=10, warmup_iters=10)

        dist_print(
            f"Rank {RANK} ag perf: triton={triton_perf:.2f} ms, {ag_size/triton_perf*1000*(WORLD_SIZE-1)/WORLD_SIZE:.2f} GBps, torch={torch_perf:.2f} ms {ag_size/torch_perf*1000*(WORLD_SIZE-1)/WORLD_SIZE:.2f} GBps, speedup {torch_perf/triton_perf:.2f}",
            need_sync=True,
            allowed_ranks=list(range(WORLD_SIZE)),
        )


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    ag_out: torch.Tensor,
):
    torch.distributed.all_gather_into_tensor(ag_out, local_input, pg)
    ag_gemm_output = torch.matmul(ag_out, local_weight)
    return ag_gemm_output


def perf_test(name, M, config, persistent: bool = True):
    N = config["N"]
    K = config["K"]
    if RANK == 0:
        print(f"test shape: M {M}, N {N}, K {K}")
    assert M % WORLD_SIZE == 0
    assert N % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE
    N_per_rank = N // WORLD_SIZE

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda")

    torch_ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")

    ctx = create_ag_gemm_inter_node_context(
        A,
        B,
        RANK,
        WORLD_SIZE,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        stages=config["stage"],
        ag_stream=torch.cuda.Stream(),
        gemm_stream=torch.cuda.Stream(),
        autotune=args.autotune,
    )

    def triton_func():
        return ag_gemm_inter_node(A, B, ctx=ctx, persistent=persistent)

    if args.autotune:
        _func = triton_func
        ctx.autotune = True
        triton_func = contextual_autotune(is_dist=True)(lambda: _func())

    if RANK == 0 and args.debug:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        triton_func()
        os.environ["TRITON_ALWAYS_COMPILE"] = "0"
        os.environ["MLIR_ENABLE_DUMP"] = "0"

    for i in range(5):
        A.random_()
        B.random_()
        C = triton_func()

    def torch_func():
        return torch_ag_gemm(TP_GROUP, A, B.T, torch_ag_buffer)

    C_golden = torch_func()

    for i in range(WORLD_SIZE):
        torch.distributed.barrier(TP_GROUP)
        if RANK == i:
            assert_allclose(C_golden, C, atol=1e-3, rtol=1e-3)

    with group_profile(
            f"trace_ag_gemm_inter_node/m_{M}_n_{N}_k_{K}",
            do_prof=args.profile,
            group=TP_GROUP,
    ):
        _, triton_duration_ms = perf_func(triton_func, iters=10, warmup_iters=5)
        _, torch_duration_ms = perf_func(torch_func, iters=10, warmup_iters=5)

        dist_print(
            f"{name} Rank {RANK} perf: triton={triton_duration_ms:.2f} ms, torch={torch_duration_ms:.2f} ms, speedup {torch_duration_ms/triton_duration_ms:.2f}",
            need_sync=True,
            allowed_ranks=list(range(WORLD_SIZE)),
        )


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.get_device_capability() < (9, 0):
        if args.persistent:
            print("Persistent is not supported on device with capability < (9, 0). exit...")
            sys.exit()

    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    configs = {
        "Model-1": {"N": 49152, "K": 12288, "BM": 128, "BN": 256, "BK": 64, "stage": 3},
        "LLaMA-7B": {"N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "stage": 3},
        "LLaMA-3.1-8B": {
            "N": 14336,
            "K": 4096,
            "BM": 128,
            "BN": 128,
            "BK": 64,
            "stage": 3,
        },
        "LLaMA-3.1-70B": {
            "N": 28672,
            "K": 8192,
            "BM": 128,
            "BN": 256,
            "BK": 64,
            "stage": 3,
        },
        "LLaMA-3.1-405B": {
            "N": 53248,
            "K": 16384,
            "BM": 128,
            "BN": 256,
            "BK": 64,
            "stage": 3,
        },
        "Qwen2-72B": {
            "N": 29568,
            "K": 8192,
            "BM": 128,
            "BN": 256,
            "BK": 64,
            "stage": 3,
        },
    }

    if args.gemm_only:
        for testcase, value in configs.items():
            perf_test_gemm_only(testcase, args.M, value, args.persistent)
    elif args.ag_only:
        from third_party.distributed.distributed.kernels.nvidia.allgather_gemm import (
            inter_node_allgather,
            local_copy_and_barrier_all,
        )

        for testcast, value in configs.items():
            perf_test_ag_only(testcast, args.M, value)
    else:
        for testcast, value in configs.items():
            perf_test(testcast, args.M, value, args.persistent)

    torch.distributed.destroy_process_group()
