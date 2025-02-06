# torchrun --nproc_per_node=8 --nnodes=1 run_ring_put.py
import datetime
import os

import pynvshmem
import torch
import torch.distributed


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


def ring_put():
    t = pynvshmem.nvshmem_create_tensor([1024], torch.int)
    print("create torch tensor with nvshmem")
    torch.cuda.synchronize()
    print(t)
    pynvshmem.nvshmem_int_p(t.data_ptr(), TP_GROUP.rank(), (RANK + 1) % WORLD_SIZE)
    print("after put_rank_to_next")
    print(t.to(torch.int32))


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
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

torch.cuda.synchronize()
init_nvshmem_by_uniqueid(TP_GROUP)
ring_put()
