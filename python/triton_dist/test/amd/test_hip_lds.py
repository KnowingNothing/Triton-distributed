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
import numpy as np

import triton
import torch
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.hip import libdevice  # noqa: F811

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# kernel


@triton.jit
def load_acquire_system_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # noqa: F841
    x_ptr_plus_offsets = x_ptr + offsets

    x = libdevice.load_acquire_system(x_ptr_plus_offsets)

    libdevice.syncthreads()

    y_ptr_plus_offsets = y_ptr + offsets
    tl.store(y_ptr_plus_offsets, x)


@triton.jit
def store_release_system_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # noqa: F841
    x_ptr_plus_offsets = x_ptr + offsets

    x = tl.load(x_ptr_plus_offsets)

    libdevice.syncthreads()

    y_ptr_plus_offsets = y_ptr + offsets
    libdevice.store_release_system(y_ptr_plus_offsets, x)


# test


def test_hip_device_load_store(op):
    torch.manual_seed(0)
    size = 98432
    dtype = torch.int32
    x = torch.randint(low=0, high=65536, size=(size, ), dtype=dtype, device=DEVICE)
    output_triton = torch.zeros(size, device=DEVICE, dtype=dtype)
    output_torch = x
    assert x.is_cuda and output_triton.is_cuda
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    if op == "hip_store_release_system":
        store_release_system_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
    elif op == "hip_load_acquire_system":
        load_acquire_system_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
    else:
        raise RuntimeError(f"unsupport test case: {op}")

    print(f"output_torch:\n{output_torch}")
    print(f"output_triton:\n{output_triton}")
    print(f'The maximum difference between torch and triton is '
          f'{np.max(np.abs(output_torch.cpu().numpy() - output_triton.cpu().numpy()))}')
    torch.testing.assert_close(output_torch, output_triton, rtol=None, atol=None)
    print("✅ Triton and Torch match on HIP")


test_hip_device_load_store("hip_load_acquire_system")
test_hip_device_load_store("hip_store_release_system")
