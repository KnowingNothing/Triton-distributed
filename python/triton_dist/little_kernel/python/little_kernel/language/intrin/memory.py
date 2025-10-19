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

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import LLType, void, Tensor, uint8
from enum import Enum


def codegen_alloc_dynamic_shared_memory(size_bytes, align_bytes, name):
    stmt = f"extern __shared__ __align__({align_bytes}) uint8_t {name}[]"

    return Builtin(body="", includes=[], return_val=stmt)


def alloc_dynamic_shared_memory_eval_arg_type(ctx, size_bytes, align_bytes,
                                              name):
    ctx[name] = uint8


@builtin(eval_return_type=void,
         eval_arg_type=alloc_dynamic_shared_memory_eval_arg_type,
         codegen_func=codegen_alloc_dynamic_shared_memory)
def alloc_dynamic_shared_memory(size_bytes: int, align_bytes: int, name: str):
    """Allocate dynamic shared memory with the given size in bytes."""
    raise RuntimeError(
        "alloc_dynamic_shared_memory should never be called in compilation")


def codegen_alloc_local_memory(var_name, dtype, elems):
    stmt = f"{dtype} {var_name}[{elems}]"

    return Builtin(body="", includes=[], return_val=stmt)


def alloc_local_memory_eval_arg_type(ctx, var_name, dtype, elems):
    ctx[var_name] = dtype


@builtin(eval_return_type=lambda var_name, dtype, elems: dtype,
         eval_arg_type=alloc_local_memory_eval_arg_type,
         codegen_func=codegen_alloc_local_memory)
def alloc_local_memory(var_name: str, dtype: LLType, elems: int):
    """Allocate local memory with the given number of elements."""
    raise RuntimeError(
        "alloc_local_memory should never be called in compilation")


def codegen_slice_dynamic_shared_memory(var_name, dtype, start, size,
                                        shmem_buf_name):
    # TODO(zhengsize): find a better way to handle such hacky case
    if '[' in var_name and ']' in var_name:
        stmt = f"{var_name} = reinterpret_cast<{dtype}>({shmem_buf_name} + {start}); /*size = {size} bytes*/"
    else:
        stmt = f"{dtype} {var_name} = reinterpret_cast<{dtype}>({shmem_buf_name} + {start}); /*size = {size} bytes*/"

    return Builtin(body="", includes=[], return_val=stmt)


def slice_dynamic_shared_memory_eval_arg_type(ctx, var_name, dtype, start,
                                              size, shmem_buf_name):
    ctx[var_name] = dtype


@builtin(eval_return_type=lambda var_name, dtype, start, size, shmem_buf_name:
         dtype,
         eval_arg_type=slice_dynamic_shared_memory_eval_arg_type,
         codegen_func=codegen_slice_dynamic_shared_memory)
def slice_dynamic_shared_memory(var_name: str, dtype: LLType, start: int,
                                size: int, shmem_buf_name: str):
    """Slice dynamic shared memory with the given start and size."""
    raise RuntimeError(
        "slice_dynamic_shared_memory should never be called in compilation")


def codegen_align_memory(align_bytes, scope):
    return Builtin(body="",
                   includes=[],
                   return_val=f"/* align {scope} to {align_bytes} */")


@builtin(eval_return_type=void, codegen_func=codegen_align_memory)
def align_memory(align_bytes: int, scope: str):
    """Align memory to the given byte alignment."""
    raise RuntimeError("align_memory should never be called in compilation")


def codegen_empty(shape, dtype, scope):
    raise RuntimeError("empty should never be called in code generation")


@builtin(eval_return_type=lambda shape, dtype, scope: Tensor[dtype],
         codegen_func=codegen_empty)
def empty(shape: list, dtype: LLType, scope: str):
    """Create an empty tensor with the given shape, dtype, and scope."""
    raise RuntimeError("empty should never be called in compilation")
