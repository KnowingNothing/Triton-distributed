"""
Allgather Kernel
===============

In this tutorial, you will write a distributed Allgather kernel using  Dist-Triton.

In doing so, you will learn about:

* Writing the AllGather kernel with symmetric pointers directly.

* Writing the AllGather kernel with NVSHMEM device functions.

"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid, __syncthreads
import pynvshmem

from typing import List
from cuda import cuda
from triton.distributed.utils import CUDA_CHECK

from triton.distributed.utils import initialize_distributed

# %%
# In the tensor parallelism, allgather is used to collect the partitioned input tensors among all workers.
# Before Allgather: worker 0 [0, -, -, -], worker 1 [-,1,-,-], worker 2 [-.-, 2, -], worker 3 [-, -, -, 3]
# After Allgather: worker 0 [0, 1, 2, 3], worker 1 [0, 1, 2, 3], worker 2 [0, 1, 2, 3], worker 3 [0, 1, 2, 3],
# --------------

# %%
# For inranode communication, we can directly use pointers returned by NVSHMEM to copy data.


def cp_engine_producer_all_gather_full_mesh_pull(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape

    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.cuda.stream(ag_stream):
        for src_rank in rank_orders:
            if src_rank == rank:
                continue
            # peer: src_rank, offset src_rank[src_rank] -> rank[src_rank]
            dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            dst.copy_(src)
            (err, ) = cuda.cuStreamWriteValue32(
                ag_stream.cuda_stream,
                barrier_buffers[rank][src_rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


# %%
# We can also use NVSHMEM device function (libshmem_device) to get/put data.


@triton.jit
def nvshmem_device_producer_all_gather_2d_put_block_kernel(
    ag_buffer_ptr,
    signal_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
    DISPATCH_BLOCK_NUM: tl.constexpr,
    SEND_BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    thread_idx = tid(axis=0)

    n_nodes = world_size // local_world_size
    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size

    if pid < DISPATCH_BLOCK_NUM:  # intra dispatch block
        peer = (local_rank + pid + 1) % local_world_size + node_rank * local_world_size
        for i in range(n_nodes):
            segment = local_rank + (
                (node_rank + i) % n_nodes) * local_world_size  # calculate the transfer offset of this sm
            if thread_idx == 0:
                libshmem_device.signal_wait_until(  # wait for the segment ready
                    signal_buffer_ptr + segment,
                    libshmem_device.NVSHMEM_CMP_GE,
                    signal_target,
                )
            __syncthreads()
            libshmem_device.putmem_signal_block(  # send the segment to the peer and notify the segment is ready
                ag_buffer_ptr + segment * elem_per_rank,
                ag_buffer_ptr + segment * elem_per_rank,
                elem_per_rank * size_per_elem,
                signal_buffer_ptr + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )
    else:  # inter send block
        if thread_idx == 0:
            libshmem_device.signal_wait_until(
                signal_buffer_ptr + rank,
                libshmem_device.NVSHMEM_CMP_GE,
                signal_target,
            )
        __syncthreads()
        global_send_pid = pid % SEND_BLOCK_NUM + 1
        peer = local_rank + (node_rank + global_send_pid) % n_nodes * local_world_size
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,
            ag_buffer_ptr + rank * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    num_ranks = TP_GROUP.size()
    LOCAL_WORLD_SIZE = 8  # we assume each machine contains 8 GPUs
    n_nodes = num_ranks // LOCAL_WORLD_SIZE
    local_rank = rank % LOCAL_WORLD_SIZE

    M = 8192
    N = 12288
    M_per_rank = M // TP_GROUP.size()
    dtype = torch.float16
    signal_dtype = torch.uint64  # internode barrier must be torch.uint64

    local_data = torch.randn([M_per_rank, N], dtype=dtype, device="cuda")
    ag_buffer_ptrs = pynvshmem.nvshmem_create_tensor_list_intra_node([M, N], dtype)  # buffer for dist-triton allgather
    signal = pynvshmem.nvshmem_create_tensor_list_intra_node(([num_ranks]),
                                                             signal_dtype)  # each rank corresponds to one barrier
    ag_buffer_ptrs[local_rank][
        rank * M_per_rank:(rank + 1) * M_per_rank,
    ].copy_(local_data)  # copy local data to symmetric memory for communication
    signal[local_rank].fill_(0)  # The initial value of signal should be 0s
    pynvshmem.nvshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)

    golden = torch.empty([M, N], dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(golden, local_data, group=TP_GROUP)

    if num_ranks == LOCAL_WORLD_SIZE:  # only intra comm
        cp_engine_producer_all_gather_full_mesh_pull(
            rank, num_ranks, local_data, ag_buffer_ptrs, torch.cuda.current_stream(),
            signal)  # Here we use current stream for allgather, we can pass any other stream for comm-comp fusion.
    else:
        grid = lambda META: (int(LOCAL_WORLD_SIZE + n_nodes - 1), )
        nvshmem_device_producer_all_gather_2d_put_block_kernel[grid](
            ag_buffer_ptrs[local_rank], signal[local_rank], M_per_rank * N,  # No. of elems of local data
            local_data.element_size(),  # element size
            1,  # signal target, can be any other value in practice
            rank, LOCAL_WORLD_SIZE, num_ranks, LOCAL_WORLD_SIZE, n_nodes - 1)

    pynvshmem.nvshmem_barrier_all()

    print(ag_buffer_ptrs[local_rank])
    assert torch.allclose(golden, ag_buffer_ptrs[local_rank], atol=1e-3, rtol=1e-3)
    print("Pass!")

    torch.distributed.destroy_process_group()

# To run this tutorial
# source ./scripts/sentenv.sh
# bash ./third_party/distributed/launch.sh ./third_party/distributed/tutorials/02-allgather.py
# for internode test, need to modify the value of master_node, node_rank, nnodes in launch.sh
