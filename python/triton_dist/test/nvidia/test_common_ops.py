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
import datetime
import os
import random

import torch
import torch.distributed

from triton_dist import pynvshmem
from triton_dist.kernels.nvidia.common_ops import (barrier_all_intra_node_non_atomic,
                                                   barrier_all_intra_node_non_atomic_block, barrier_on_this_grid,
                                                   barrier_all_intra_node_atomic_cas_block)
from triton_dist.utils import check_p2p_native_atomic_supported

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def _random_sleep():
    if random.random() > 0.9:
        torch.cuda._sleep(int(random.random() * 1000000))
    elif random.random() > 0.5:
        torch.cuda._sleep(int(random.random() * 30000))


def test_barrier_on_this_grid():
    print(">> barrier_on_this_grid start...")
    flag = torch.zeros((1, ), dtype=torch.int32, device="cuda")
    for _ in range(1000):
        barrier_on_this_grid[(random.randint(1, 1024), )](flag)
    print("✅ barrier_on_this_grid passed")


def test_barrier_all_intra_node_non_atomic():
    print(">> barrier_all_intra_node_non_atomic start...")
    symm_flags = pynvshmem.nvshmem_create_tensor_list_intra_node((LOCAL_WORLD_SIZE * 3, ), torch.int32)
    symm_flags[LOCAL_RANK].fill_(0)
    pynvshmem.nvshmem_barrier_all()

    for n in range(1000):
        _random_sleep()
        # print(f"iter {n}", flush=True)
        barrier_all_intra_node_non_atomic_block[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, symm_flags[LOCAL_RANK], n + 1)

    print("✅ barrier_all_intra_node_non_atomic_block passed")

    for n in range(1000):
        _random_sleep()
        # print(f"iter {n}", flush=True)
        barrier_all_intra_node_non_atomic[(random.randint(1, 1024), )](LOCAL_RANK, RANK, LOCAL_WORLD_SIZE,
                                                                       symm_flags[LOCAL_RANK], n + 1)

    print("✅ barrier_all_intra_node_non_atomic passed")


def test_barrier_all_intra_node():
    if not check_p2p_native_atomic_supported():
        print("P2P native atomic access is not supported. skip this test...")
        return

    print(">> barrier_all_intra_node_atomic_cas_block start...")
    symm_flag = pynvshmem.nvshmem_create_tensor((LOCAL_WORLD_SIZE, ), torch.int32)
    symm_flag.fill_(0)
    pynvshmem.nvshmem_barrier_all()

    for n in range(1000):
        _random_sleep()
        barrier_all_intra_node_atomic_cas_block[(1, )](LOCAL_RANK, RANK, LOCAL_WORLD_SIZE, symm_flag)

    print("✅ barrier_all_intra_node_atomic_cas_block passed")


if __name__ == "__main__":
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

    test_barrier_on_this_grid()
    test_barrier_all_intra_node_non_atomic()
    test_barrier_all_intra_node()
