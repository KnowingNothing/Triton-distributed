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
from ..builtin_base import builtin_class, builtin, Builtin
from little_kernel.core.type_system import *
from functools import partial


def codegen_thread_idx(thread_idx):
    return Builtin(body="", includes=[], return_val=f"threadIdx.{thread_idx}")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "x"))
def thread_x():
    """Thread index in the x dimension."""
    raise RuntimeError("thread_x should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "y"))
def thread_y():
    """Thread index in the y dimension."""
    raise RuntimeError("thread_y should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "z"))
def thread_z():
    """Thread index in the z dimension."""
    raise RuntimeError("thread_z should never be called in compilation")


def codegen_block_idx(block_idx):
    return Builtin(body="", includes=[], return_val=f"blockIdx.{block_idx}")

@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_idx, "x"))
def block_x():
    """Block index in the x dimension."""
    raise RuntimeError("block_x should never be called in compilation")

@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_idx, "y"))
def block_y():
    """Block index in the y dimension."""
    raise RuntimeError("block_y should never be called in compilation")

@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_idx, "z"))
def block_z():
    """Block index in the z dimension."""
    raise RuntimeError("block_z should never be called in compilation")



def codegen_get_lane_idx():
    return Builtin(
        body="""
__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}
""",
        includes=[],
        return_val="get_lane_idx()",
    )


@builtin(eval_return_type=int32, codegen_func=codegen_get_lane_idx)
def get_lane_idx():
    """Get the lane index of the current thread."""
    raise RuntimeError("get_lane_idx should never be called in compilation")


def codegen_syncwarp():
    return Builtin(body="", includes=[], return_val="__syncwarp()")


@builtin(eval_return_type=void, codegen_func=codegen_syncwarp)
def __syncwarp():
    """Sync all threads in the warp."""
    raise RuntimeError("__syncwarp should never be called in compilation")
