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
from little_kernel.core.type_system import void
from little_kernel.core.compile import ll_kernel


def codegen_cluster_arrive():
    body = """
__forceinline__ __device__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.aligned;\\n" : : );
}
"""
    stmt = "cluster_arrive()"
    return Builtin(body=body, includes=[], return_val=stmt)


@builtin(eval_return_type=void, codegen_func=codegen_cluster_arrive)
def cluster_arrive():
    raise RuntimeError(f"should not call cluster_arrive in compilation")


def codegen_cluster_wait():
    body = """
__forceinline__ __device__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.aligned;\\n" : : );
}
"""
    stmt = "cluster_wait()"
    return Builtin(body=body, includes=[], return_val=stmt)


@builtin(eval_return_type=void, codegen_func=codegen_cluster_wait)
def cluster_wait():
    raise RuntimeError(f"should not call cluster_wait in compilation")


@ll_kernel(backend="cuda", is_entry=False)
def cluster_sync() -> void:
    cluster_arrive()
    cluster_wait()


def codegen_block_sync():
    body = ""
    stmt = "__syncthreads()"
    return Builtin(body=body, includes=[], return_val=stmt)


@builtin(eval_return_type=void, codegen_func=lambda: codegen_block_sync)
def block_sync():
    raise RuntimeError(f"should not call block_sync in compilation")


def codegen_init_smem_barrier(smem_bar_ptr, arrive_cnt):
    body = """
__forceinline__ __device__ void init_smem_barrier(uint64_t const* smem_bar_ptr, uint32_t arrive_cnt) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_bar_ptr);
    asm volatile(
        "{\\n\\t"
        "mbarrier.init.shared::cta.b64 [%1], %0; \\n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
}
"""
    stmt = f"init_smem_barrier({smem_bar_ptr}, {arrive_cnt})"
    return Builtin(body=body,
                   includes=["\"cute/arch/util.hpp\""],
                   return_val=stmt)


@builtin(eval_return_type=void, codegen_func=codegen_init_smem_barrier)
def init_smem_barrier(smem_bar_ptr, arrive_cnt: int):
    """
    Initialize the shared memory barrier.
    """
    raise RuntimeError(f"should not call init_smem_barrier in compilation")


def codegen_fence_smem_barrier_init():
    body = """
__forceinline__ __device__ void fence_smem_barrier_init() {
    asm volatile(
        "{\\n\\t"
        "fence.mbarrier_init.release.cluster; \\n"
        "}"
        :
        :);
}
"""
    stmt = "fence_smem_barrier_init()"
    return Builtin(body=body, includes=[], return_val=stmt)


@builtin(eval_return_type=void, codegen_func=codegen_fence_smem_barrier_init)
def fence_smem_barrier_init():
    """
    Fence the shared memory barrier initialization.
    """
    raise RuntimeError(
        f"should not call fence_smem_barrier_init in compilation")
