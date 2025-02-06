import triton
import triton.language as tl
import torch
import pynvshmem
import os
import datetime
from triton_nvshmem.nvshmem import (
    nvshmem_ptr,
    nvshmem_my_pe_wrapper,
    nvshmem_n_pes_wrapper,
    nvshmem_int_p_wrapper,
)


def broadcast_cpu(
    tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup
):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


def init_nvshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = pynvshmem.nvshmemx_get_uniqueid()
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.numpy().tobytes()
    pynvshmem.nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)


@triton.jit
def ring_put(ptr):
    # ptr_out = nvshmem_ptr(ptr, 1)
    mype = nvshmem_my_pe_wrapper()
    npes = nvshmem_n_pes_wrapper()
    peer = (mype + 1) % npes
    nvshmem_int_p_wrapper(ptr, mype, peer)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    init_nvshmem_by_uniqueid(TP_GROUP)

    t = pynvshmem.nvshmem_create_tensor((32,), torch.int32)
    print(t.data_ptr())
    ring_put[(1,)](t)

    pynvshmem.nvshmem_barrier_all()
    print(f"RANK {RANK}: {t}")
