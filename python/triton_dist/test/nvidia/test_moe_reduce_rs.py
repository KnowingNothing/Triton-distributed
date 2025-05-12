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
import random
from functools import partial

import numpy as np
import torch

from triton_dist import pynvshmem
from triton_dist.kernels.nvidia import (create_moe_rs_context, moe_reduce_rs, select_experts)
from triton_dist.utils import dist_print, get_torch_prof_ctx, perf_func


def create_ones_tensor(rank, shape, dtype=torch.float16, device="cuda"):
    return torch.ones(shape, dtype=dtype, device=device)


def create_rand_tensor(rank, shape, dtype=torch.float16, device="cuda"):
    return (-2 * torch.rand(shape, dtype=dtype, device=device) + 1) / 100 * (rank + 1)


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    # torch.bfloat16: 1e-2,
    # torch.float8_e4m3fn: 1e-2,
    # torch.float8_e5m2: 1e-2,
}


class TorchGroupGemmReduceRS(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        hidden_dim: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        router_logits: torch.Tensor,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
    ):
        super(TorchGroupGemmReduceRS, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()
        self.max_token_num = max_token_num
        assert (
            max_token_num %
            self.world_size == 0), f"max_token_num({max_token_num}) should be multiple of world_size({self.world_size})"
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size % self.world_size == 0
                ), f"intermediate_size({intermediate_size}) should be multiple of world_size({self.world_size})"
        self.intermediate_size_per_rank = intermediate_size // self.world_size

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device

        self.router_logits = router_logits

        self.full_topk_ids, self.full_topk_weight = select_experts(self.pg, self.world_size, self.topk,
                                                                   self.input_dtype, self.device, self.router_logits)

        self.rs_buffer: torch.Tensor = torch.zeros(
            [self.max_token_num // self.world_size, self.hidden_dim],
            dtype=self.output_dtype,
            device=self.device,
        )

    def forward(self, intermediate_states, w):
        final_output_buffer = torch.zeros(
            self.max_token_num * self.topk,
            self.hidden_dim,
            dtype=self.output_dtype,
            device=self.device,
        )
        num_tokens_topk, intermediate_size_per_rank = intermediate_states.shape
        topk_ids = self.full_topk_ids[:num_tokens_topk // self.topk].view(-1)
        topk_weight = self.full_topk_weight[:num_tokens_topk // self.topk].view(-1)
        out = final_output_buffer[:num_tokens_topk, :]
        for i in range(self.num_experts):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = (intermediate_states[mask] @ w[i]) * topk_weight[mask, None]
        output = torch.sum(
            out.reshape(num_tokens_topk // self.topk, self.topk, -1),
            dim=1,
            keepdim=False,
        )
        torch.distributed.reduce_scatter_tensor(
            self.rs_buffer[:num_tokens_topk // self.topk // self.world_size, :],
            output,
            group=self.pg,
        )

        return self.rs_buffer[:num_tokens_topk // self.topk // self.world_size, :]


class MoEReduceRSTensorParallel(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        local_world_size: int,
        hidden_dim: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        router_logits: torch.Tensor,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        moe_block_size=128,
        debug_sync=False,
    ):
        super(MoEReduceRSTensorParallel, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()
        self.local_world_size = local_world_size
        self.local_rank = self.rank % self.local_world_size
        self.max_token_num = max_token_num
        assert (
            max_token_num %
            self.world_size == 0), f"max_token_num({max_token_num}) should be multiple of world_size({self.world_size})"
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size % self.world_size == 0
                ), f"intermediate_size({intermediate_size}) should be multiple of world_size({self.world_size})"
        self.intermediate_size_per_rank = intermediate_size // self.world_size

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device

        self.moe_block_size = moe_block_size
        self.debug_sync = debug_sync

        self.router_logits = router_logits

        self.ctx = create_moe_rs_context(
            self.pg,
            self.local_rank,
            self.world_size,
            self.local_world_size,
            self.max_token_num,
            self.hidden_dim,
            self.num_experts,
            self.topk,
            self.input_dtype,
            self.output_dtype,
            self.device,
            self.moe_block_size,
            self.router_logits,
        )

    def forward(self, intermediate_states, w):
        assert hasattr(self, "ctx") and self.ctx is not None
        num_tokens_per_rank = self.ctx.precompute_ctx.num_tokens_per_rank
        num_tokens = num_tokens_per_rank * self.world_size

        self.ctx.dataflow_config.RS_BLOCK_M = num_tokens // self.world_size

        output = moe_reduce_rs(
            self.rank,
            self.world_size,
            self.local_world_size,
            intermediate_states,
            w,
            self.ctx,
            dump_ir=False,
            debug_sync=self.debug_sync,
        )

        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)  # num_tokens
    parser.add_argument("N", type=int)  # hidden_size
    parser.add_argument("K", type=int)  # intermediate_size
    parser.add_argument("E", type=int)  # num_experts
    parser.add_argument("TOPK", type=int)
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--debug_sync", default=False, action="store_true", help="sync between compute and comm")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--autotune", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.autotune:
        import importlib

        import triton
        from triton_dist.autotuner import contextual_autotune
        moe_reduce_rs_module = importlib.import_module('triton_dist.kernels.nvidia.moe_reduce_rs')

        configs = [
            triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=s, num_warps=w)
            for BN in [128, 256]
            for BK in [32, 64]
            for s in [3, 4]
            for w in [4, 8]
        ]
        moe_reduce_rs_module.kernel_producer_group_gemm_tp_scatter_input = triton.autotune(
            configs=configs, key=["EM", "N",
                                  "K_per_rank"])(moe_reduce_rs_module.kernel_producer_group_gemm_tp_scatter_input)
        moe_reduce_rs = contextual_autotune(is_dist=True)(moe_reduce_rs)

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

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

    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)  # True or False
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

    num_tokens_per_rank = args.M // WORLD_SIZE
    hidden_size = args.N
    intermediate_size = args.K
    num_experts = args.E
    topk = args.TOPK

    max_token_num = 16 * 1024
    input_dtype = torch.float16
    output_dtype = torch.float16
    device = "cuda"

    iters = args.iters
    warmup_iters = args.warmup

    rank = TP_GROUP.rank()
    world_size = TP_GROUP.size()

    debug_sync = args.debug_sync
    check = args.check

    with torch.no_grad():
        router_logits = create_rand_tensor(
            rank,
            (num_tokens_per_rank, num_experts),
            device="cuda",
            dtype=input_dtype,
        )

        moe_block_size = 128

        module = MoEReduceRSTensorParallel(
            TP_GROUP,
            LOCAL_WORLD_SIZE,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            router_logits,
            max_token_num,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            device=device,
            debug_sync=debug_sync,
        )

        torch_module = TorchGroupGemmReduceRS(
            TP_GROUP,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            router_logits,
            max_token_num,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            device=device,
        )

        if not check:
            intermediate_states = create_rand_tensor(
                rank,
                (num_tokens_per_rank * world_size * topk, intermediate_size // world_size),
                device="cuda",
                dtype=input_dtype,
            )
            down_weight = create_rand_tensor(
                rank,
                (num_experts, intermediate_size // world_size, hidden_size),
                device="cuda",
                dtype=input_dtype,
            )
        else:
            intermediate_states = create_ones_tensor(
                rank,
                (num_tokens_per_rank * world_size * topk, intermediate_size // world_size),
                device="cuda",
                dtype=input_dtype,
            )
            down_weight = create_ones_tensor(
                rank,
                (num_experts, intermediate_size // world_size, hidden_size),
                device="cuda",
                dtype=input_dtype,
            )

        prof_ctx = get_torch_prof_ctx(args.profile)
        with prof_ctx:
            torch_output, torch_perf = perf_func(partial(torch_module.forward, intermediate_states, down_weight),
                                                 iters=iters, warmup_iters=warmup_iters)

            pynvshmem.nvshmem_barrier_all()
            torch.cuda.synchronize()

            output, perf = perf_func(partial(module.forward, intermediate_states, down_weight), iters=iters,
                                     warmup_iters=warmup_iters)

        pynvshmem.nvshmem_barrier_all()
        torch.cuda.synchronize()

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        prof_ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    if check:
        atol = THRESHOLD_MAP[output_dtype]
        rtol = THRESHOLD_MAP[output_dtype]
        torch.testing.assert_close(output, torch_output, atol=atol, rtol=rtol)
        torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    torch.distributed.destroy_process_group()
