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
from dataclasses import dataclass
import little_kernel.core.type_system as ll_type


@dataclass
class MMA:
    M: int
    N: int
    K: int
    A_dtype: ll_type.LLType
    B_dtype: ll_type.LLType
    ACC_dtype: ll_type.LLType


@dataclass
class WgMMA(MMA):
    A_in_SMEM: bool
    B_in_SMEM: bool


def WgMMA_64_X_16_F32BF16BF16_SS(N: int):
    assert N in [8 * i for i in range(1, 33)]
    return WgMMA(M=64,
                 N=N,
                 K=16,
                 A_dtype=ll_type.bfloat16,
                 B_dtype=ll_type.bfloat16,
                 ACC_dtype=ll_type.float32,
                 A_in_SMEM=True,
                 B_in_SMEM=True)
