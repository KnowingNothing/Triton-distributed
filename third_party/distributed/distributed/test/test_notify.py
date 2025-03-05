################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
import torch
import triton
import pynvshmem
import triton.distributed.language as dl

import os
import datetime


@triton.jit
def test_notify_set(ptr):
    mype = dl.rank()
    npes = dl.num_ranks()
    peer = (mype + 1) % npes
    dl.notify(ptr, peer, signal=mype, sig_op="set", comm_scope="inter_node")


@triton.jit
def test_notify_add(ptr):
    dl.notify(ptr, 0, signal=1, sig_op="add", comm_scope="intra_node")


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
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    t = pynvshmem.nvshmem_create_tensor((8, ), torch.uint64)
    t.fill_(0)
    pynvshmem.nvshmem_barrier_all()
    test_notify_set[(1, )](t)
    pynvshmem.nvshmem_barrier_all()

    assert t[0].item() == (RANK + WORLD_SIZE - 1) % WORLD_SIZE

    t.fill_(0)
    pynvshmem.nvshmem_barrier_all()
    test_notify_add[(1, )](t)
    pynvshmem.nvshmem_barrier_all()
    ref = WORLD_SIZE if RANK == 0 else 0
    assert t[0].item() == ref

    print(f"RANK {RANK}: pass.")
