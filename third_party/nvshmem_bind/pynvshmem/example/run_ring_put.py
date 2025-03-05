# torchrun --nproc_per_node=8 --nnodes=1 run_ring_put.py
import datetime
import os

import pynvshmem
import torch
import torch.distributed


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
pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
ring_put()
