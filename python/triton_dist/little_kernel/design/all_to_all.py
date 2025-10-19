import little_kernel as lk
import little_kernel.language as ll
from enum import Enum
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

dtype = ll.bfloat16
num_local_ranks = 8
M = lk.Var("M", ll.int32)
N = 8192
backend = "cuda"

@ll_kernel(backend=backend, is_entry=True)
def all_to_all(
    src: ll.Tensor[dtype],
    comm_buf: ll.Tensor[dtype],
    dst: ll.Tensor[dtype],
    M: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
    num_local_ranks: ll.int32,
) -> ll.void:
    M_per_rank = ll.cdiv(M, num_ranks)
    tx = ll.thread_x()
    bx = ll.block_x()
    


passes = PASSES[backend]
code = all_to_all.compile(passes, codegen_cuda)
print(code)