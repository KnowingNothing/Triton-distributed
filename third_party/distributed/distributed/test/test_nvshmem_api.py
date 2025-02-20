import triton
import triton.language as tl
from triton.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid, __syncthreads
import torch
import torch.distributed
import pynvshmem
import os
import datetime

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def broadcast_cpu(tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup):
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


def test_nvshmemx_getmem_with_scope():

    @triton.jit
    def _nvshmemx_getmem(ptr, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.getmem_nbi_block(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.getmem_nbi_warp(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    libshmem_device.getmem_nbi_thread(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.getmem_block(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.getmem_warp(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    libshmem_device.getmem_thread(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = pynvshmem.nvshmem_create_tensor((1024, ), torch.int8)

    for scope in ["block", "warp"]:
        for nbi in [True, False]:
            print(f"runing nvshmemx_getmem{'_nbi' if nbi else ''}_{scope}...")
            t.fill_(RANK + 1)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemx_getmem[(WORLD_SIZE, )](
                t,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=1 if scope == "warp" else 4,
            )
            pynvshmem.nvshmem_barrier_all()
            print(t.reshape(8, -1))


def test_nvshmemx_putmem_with_scope():

    @triton.jit
    def _nvshmemx_putmem(ptr, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_nbi_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_nbi_warp(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    libshmem_device.putmem_nbi_thread(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_warp(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    libshmem_device.putmem_thread(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = pynvshmem.nvshmem_create_tensor((1024, ), torch.int8)

    for scope in ["block", "warp"]:
        for nbi in [True, False]:
            print(f"runing nvshmemx_putmem{'_nbi' if nbi else ''}_{scope}...")
            t.fill_(RANK + 1)
            pynvshmem.nvshmem_barrier_all()
            _nvshmemx_putmem[(WORLD_SIZE, )](
                t,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=1 if scope == "warp" else 4,
            )
            pynvshmem.nvshmem_barrier_all()
            print(t.reshape(8, -1))


def test_nvshmem_signal():

    @triton.jit
    def _pingpong(t, iters):
        # pingpong for rank 0-1, 2-3, ...
        mype = libshmem_device.my_pe()
        thread_idx = tid(axis=0)
        if thread_idx == 0:
            for n in range(iters):
                if mype == 0:
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        1,
                    )
                elif mype == 1:
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        0,
                    )
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
        __syncthreads()

    print("test nvshmemx_signal with pingpong...")
    t = pynvshmem.nvshmem_create_tensor((1, ), torch.uint64)
    _pingpong[(1, )](t, 100, num_warps=1)
    pynvshmem.nvshmem_barrier_all()
    print(t)
    torch.cuda.synchronize()


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
    init_nvshmem_by_uniqueid(TP_GROUP)

    # test_nvshmemx_getmem_with_scope()
    # test_nvshmemx_putmem_with_scope()
    test_nvshmem_signal()
