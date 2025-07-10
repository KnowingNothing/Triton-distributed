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
import triton
import triton.language as tl
from triton_dist.utils import (
    initialize_distributed,
    dist_print,
)
from triton_dist import pynvshmem
from triton_dist.kernels.nvidia.common_ops import barrier_all_on_stream
from triton_dist.language.extra import libshmem_device


### Use branch to work around the issue with multimem instruction
@triton.jit
def multimem_st(ptr, mask):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            .reg .f32 %f0;
            mov.b32 %f0, 0x12345678;
            setp.eq.s32 %p0, $2, 1;
            @!%p0 bra mask_skip;       //<----- branch around multimem store
            multimem.st.global.b32 [$1], %f0;
            mask_skip:
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,r"),
        args=[ptr.to(tl.pointer_type(tl.uint64)), mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def st(ptr, mask):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            .reg .f32 %f0;
            mov.b32 %f0, 0x12345678;
            setp.eq.s32 %p0, $2, 1;
            @%p0 st.global.b32 [$1], %f0;
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,r"),
        args=[ptr.to(tl.pointer_type(tl.uint64)), mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def test_multimem_st_predicate_kernel(
    data_ptr,
    rank,
):
    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, data_ptr)
    mask = tl.arange(0, 32) > 15

    if rank == 0:
        multimem_st(data_mc_ptr + tl.arange(0, 32), mask)


@triton.jit
def test_st_predicate_kernel(
    data_ptr,
    rank,
):
    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, data_ptr)
    mask = tl.arange(0, 32) > 15

    if rank == 0:
        st(data_mc_ptr + tl.arange(0, 32), mask)


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))

    symm_buf0 = pynvshmem.nvshmem_create_tensor([32], torch.float32)
    barrier_all_on_stream(None, torch.cuda.current_stream())
    torch.cuda.synchronize()

    test_multimem_st_predicate_kernel[(1, )](symm_buf0, RANK, num_warps=1)
    barrier_all_on_stream(None, torch.cuda.current_stream())

    symm_buf1 = pynvshmem.nvshmem_create_tensor([32], torch.float32)
    barrier_all_on_stream(None, torch.cuda.current_stream())
    torch.cuda.synchronize()

    test_st_predicate_kernel[(1, )](symm_buf1, RANK, num_warps=1)
    barrier_all_on_stream(None, torch.cuda.current_stream())

    try:
        torch.testing.assert_close(symm_buf0, symm_buf1, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ RANK[{RANK}] check failed")
        dist_print(f"Multimem StoreRank{RANK} res: ", symm_buf0, need_sync=True,
                   allowed_ranks=list(range(TP_GROUP.size())))
        dist_print(f"Store Rank{RANK} res: ", symm_buf1, need_sync=True, allowed_ranks=list(range(TP_GROUP.size())))
        raise e
    else:
        print(f"✅ RANK[{RANK}] check passed")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
