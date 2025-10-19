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
from ..builtin_base import inline_func, const_func, const_type_func, builtin, Builtin
from little_kernel.core.type_system import LLType, int32


def codegen_sizeof(dtype):
    return Builtin(body="", includes=[], return_val=f"sizeof({dtype})")


@const_func
@builtin(eval_return_type=int32, codegen_func=codegen_sizeof)
def sizeof(dtype):
    assert isinstance(dtype, LLType)
    assert dtype.is_scalar()
    assert dtype.bits >= 8
    return dtype.bits // 8


def codegen_typeof(val):
    return Builtin(body="", includes=[], return_val=f"decltype({val})")


@const_type_func
@builtin(eval_return_type=lambda val_type: val_type,
         codegen_func=codegen_typeof)
def typeof(val):
    """Get the type of value. This is interpreted by const_fold pass"""
    assert hasattr(val, "dtype"), "value should has dtype"
    return val.dtype


def codegen_val_cast(val, dtype):
    return Builtin(body="", includes=[], stmt=f"({dtype})({val})")


@builtin(eval_return_type=lambda val_type, dtype: dtype,
         codegen_func=codegen_val_cast)
def val_cast(val, dtype):
    raise RuntimeError("val_cast should not be called in compilation")


def codegen_ptr_cast(val, dtype):
    return Builtin(body="",
                   includes=[],
                   stmt=f"reinterpret_cast<{dtype}>({val})")


@builtin(eval_return_type=lambda val_type, dtype: dtype,
         codegen_func=codegen_ptr_cast)
def ptr_cast(val, dtype):
    raise RuntimeError("ptr_cast should not be called in compilation")
