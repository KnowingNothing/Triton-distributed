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

import os

import torch
from typing import Callable, List, Tuple, Union, Sequence
from contextlib import contextmanager, nullcontext
from cuda import cuda, cudart


# Some code from python/flux/util.py in flux project
@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point


def _make_tensor(
    shape: List[Union[int, Callable[[], int]]],
    dtype: torch.dtype,
    init_args: Union[Tuple[float, float], Tuple[int, int]],
    device: str = "cuda",
):
    """
    rand() * scale + bias
    randint(-scale, scale) + bias
    """
    if isinstance(shape, Sequence):
        shape = tuple([x() if isinstance(x, Callable) else x for x in shape])
    elif isinstance(shape, int):
        shape = (shape, )
    elif isinstance(shape, Callable):
        shape = shape()
    else:
        raise ValueError(f"unsupported shape {shape}")

    scale, bias = init_args
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        out = (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * scale + bias
    elif dtype == torch.int8:
        out = torch.randint(-scale, scale, shape, dtype=torch.int8, device=device)
        out = out + bias
    elif is_fp8_dtype(dtype):
        out = (torch.rand(shape, dtype=torch.float16, device=device) * 2 - 1) * scale + bias
        with with_torch_deterministic(False):
            out = out.to(dtype)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    return out


def generate_data(configs):
    while True:
        yield (_make_tensor(*args) if args else None for args in configs)


def get_torch_prof_ctx(do_prof: bool):
    ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) if do_prof else nullcontext())
    return ctx


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


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")
