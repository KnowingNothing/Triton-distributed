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
import os
import random
from typing import Optional
import torch
import triton
import torch.distributed as dist
from triton_dist.kernels.nvidia.allreduce import (
    create_allreduce_ctx,
    all_reduce,
)
from triton_dist.utils import (
    group_profile,
    initialize_distributed,
    perf_func,
)

DATA_SIZES = [
    128,  # 128B
    1024,  # 1KB
    16 * 1024,  # 16K
    32 * 1024,  # 32K
    64 * 1024,  # 64K
    128 * 1024,  # 128K
    256 * 1024,  # 256K
    512 * 1024,  # 512K
    1024 * 1024,  # 1M
    2 * 1024 * 1024,  # 2M
    4 * 1024 * 1024,  # 4M
    8 * 1024 * 1024,  # 8M
    16 * 1024 * 1024,  # 16M
    32 * 1024 * 1024,  # 32M
    64 * 1024 * 1024,  # 64M
    128 * 1024 * 1024  # 128M
]


def _pretty_format(nbytes):
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024}KB"
    if nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / 1024 / 1024}MB"
    return f"{nbytes / 1024 / 1024 / 1024}GB"


def _generate_shape(num_elem: int):
    HIDDEN_DIMS = [768, 1024, 2048, 3072, 4096, 5120, 6144]  # common hidden dims
    for n in sorted(HIDDEN_DIMS, reverse=True):
        if num_elem % n == 0:
            return (num_elem // n, n)

    for m in range(int(num_elem**0.5), 0, -1):
        if num_elem % m == 0:
            return (m, num_elem // m)


def _create_data(
    M,
    N,
    method,
    dtype=torch.float32,
    signal_stages=1,
):
    input = torch.rand([M, N], dtype=dtype, device="cuda")
    if args.debug:
        input = input.fill_((RANK + 1) / 100)
    ctx = create_allreduce_ctx(input, method, signal_stages)
    return input, ctx


def torch_all_reduce(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
):
    output = local_input.clone()
    dist.all_reduce(output, group=pg)
    return output


def stress_test(dtype, args, test_method="two_shot_multicast"):
    random.seed(args.seed)  # set all ranks to the same seed

    def _randint_with_align(alignment: int):
        test_m = random.randint(1, args.max_M)
        return triton.cdiv(test_m, alignment) * alignment

    for n in range(args.iters):
        # generate data for verify
        tensor_inputs = []
        for _ in range(args.verify_shapes):
            test_M = _randint_with_align(WORLD_SIZE * args.alignment)
            input, ctx = _create_data(
                test_M,
                args.N,
                method=test_method,
                dtype=dtype,
            )
            tensor_inputs.append((input, ctx))

        triton_out_list, torch_out_list = [], []

        for input, ctx in tensor_inputs:
            output = torch.empty_like(input)
            res = all_reduce(input, output, method=test_method, ctx=ctx)
            triton_out_list.append(res)

        for input, _ in tensor_inputs:
            res = torch_all_reduce(
                TP_GROUP,
                input,
            )
            torch_out_list.append(res)

        # verify
        for idx, (triton_res, torch_res) in enumerate(zip(triton_out_list, torch_out_list)):
            check_failed = False
            for i in range(TP_GROUP.size()):
                torch.distributed.barrier(TP_GROUP)
                if TP_GROUP.rank() == i:
                    if not torch.allclose(triton_res, torch_res, atol=6e-2, rtol=6e-2):
                        check_failed = True
                        print("❌ check failed")
                        print(f"Rank {TP_GROUP.rank()}")
                        print("shape, numel:", triton_res.shape, triton_res.shape.numel())
                        print("Diff")
                        print(torch_res - triton_res)
                        diff_loc = ~torch.isclose(triton_res, torch_res, atol=6e-2, rtol=6e-2)
                        diff_indices = diff_loc.nonzero(as_tuple=False)
                        #print(f"diff locations:\n{diff_indices}")
                        print(f"diff rows:\n{torch.unique(diff_indices[:, 0])}")
                        num_diff = torch.sum(diff_loc)
                        diff_rate = num_diff / triton_res.shape.numel()
                        print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(triton_res.shape)}")
                        print("triton_res@diff:")
                        print(triton_res[diff_loc])
                        print("input@diff:")
                        print(tensor_inputs[idx][0][diff_loc])
                        print("golden@diff:")
                        print(torch_res[diff_loc])
                        print("Max diff", torch.max(torch.abs(torch_res - triton_res)))
                        print("Avg diff", torch.mean(torch.abs(torch_res - triton_res)))
                        print("---------------------Wrong Answer!---------------------")
            if check_failed:
                exit(1)

        # just runs, check if hangs
        straggler_option = None if not args.simulate_straggler else (random.randint(
            0,
            TP_GROUP.size() - 1), random.randint(1e9, 1e9 + 1e8))  # straggler id, straggler_latency (ns)
        if straggler_option and RANK == straggler_option[0]:
            print(f"straggler id {straggler_option[0]}, latency {straggler_option[1] / 1000 / 1000 / 1000} s")

        output = torch.empty_like(input)
        for j in range(args.verify_hang):
            res = all_reduce(
                input,
                output,
                method=test_method,
                ctx=ctx,
                straggler_option=straggler_option,
            )

        if (n + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"runs {n + 1} iterations done")

    if TP_GROUP.rank() == 0:
        print(f"✅ {test_method} Pass!")


def _is_one_shot(method):
    return "one_shot" in method


def run_perf(dtype, test_method, warmup=5, iters=10):
    bytes_per_elem = torch.finfo(dtype).bits // 8
    if test_method in ["double_tree", "one_shot_non_tma", "one_shot_tma"]:
        available_ds = DATA_SIZES[:13]
    else:
        available_ds = DATA_SIZES

    for nbytes in available_ds:
        num_elem = nbytes // bytes_per_elem
        M, N = _generate_shape(num_elem)

        local_input, ctx = _create_data(M, N, method=test_method, signal_stages=warmup + iters, dtype=dtype)
        output = torch.empty_like(local_input)

        def allreduce_op():
            all_reduce(
                local_input,
                output,
                method=test_method,
                ctx=ctx,
            )

        torch.cuda._sleep(1000000000)  # in case CPU bound
        _, duration_ms = perf_func(
            allreduce_op,
            warmup_iters=warmup,
            iters=iters,
        )

        algo_bw = nbytes * 1e-9 / (duration_ms * 1e-3) * 2
        hw_bw = algo_bw * (WORLD_SIZE - 1) / WORLD_SIZE
        if _is_one_shot(test_method):
            hw_bw = algo_bw * WORLD_SIZE // 2

        if RANK == 0:
            print(
                f"RANK = {RANK}, " + _pretty_format(nbytes) +
                f" Latency = {duration_ms * 1000:0.2f} us, HW Bandwith = {hw_bw:0.2f} GB/s, Algo Bandwith = {algo_bw:0.2f} GB/s   "
            )


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--verify_shapes", type=int, default=25)
    parser.add_argument("--verify_hang", type=int, default=5)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--method", type=str, default="double_tree")
    parser.add_argument("--simulate_straggler", default=False, action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--stress", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true")
    args = parser.parse_args()

    DTYPE = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    # for TMA reduce
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=DTYPE)

    triton.set_allocator(alloc_fn)

    if "multicast" in args.method or "ld_reduce" in args.method:
        assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0"  # for multicast

    if args.stress:
        stress_test(DTYPE, args, test_method=args.method)
    else:
        with group_profile(f"all_reduce_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=TP_GROUP):
            run_perf(DTYPE, args.method, warmup=args.warmup_iters, iters=args.iters)

    torch.distributed.destroy_process_group()
