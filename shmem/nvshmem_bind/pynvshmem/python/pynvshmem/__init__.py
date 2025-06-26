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
import sys
from typing import Sequence, List

import torch
import torch.distributed
from cuda import cudart

try:
    from triton._C._pynvshmem import nvshmem_malloc, nvshmem_free, nvshmem_ptr, nvshmem_team_n_pes, nvshmem_team_my_pe, nvshmem_my_pe, nvshmemx_get_uniqueid, nvshmemx_init_attr_with_uniqueid, nvshmem_barrier_all
    from triton._C._pynvshmem import *  # noqa: F403
except Exception as e:
    print(
        "please add NVSHMEM library path to LD_LIBRARY_PATH and try again",
        flush=True,
        file=sys.stderr,
    )
    raise e

# team node
NVSHMEM_TEAM_INVALID = -1
NVSHMEM_TEAM_WORLD = 0
NVSHMEM_TEAM_WORLD_INDEX = 0
NVSHMEM_TEAM_SHARED = 1
NVSHMEM_TEAM_SHARED_INDEX = 1
NVSHMEMX_TEAM_NODE = 2
NVSHMEM_TEAM_NODE_INDEX = 2
NVSHMEMX_TEAM_SAME_MYPE_NODE = 3
NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX = 3
NVSHMEMI_TEAM_SAME_GPU = 4
NVSHMEM_TEAM_SAME_GPU_INDEX = 4
NVSHMEMI_TEAM_GPU_LEADERS = 5
NVSHMEM_TEAM_GPU_LEADERS_INDEX = 5
NVSHMEM_TEAMS_MIN = 6
NVSHMEM_TEAM_INDEX_MAX = sys.maxsize

# class nvshmemi_cmp_type(Enum):
NVSHMEM_CMP_EQ = 0
NVSHMEM_CMP_NE = 1
NVSHMEM_CMP_GT = 2
NVSHMEM_CMP_LE = 3
NVSHMEM_CMP_LT = 4
NVSHMEM_CMP_GE = 5
NVSHMEM_CMP_SENTINEL = sys.maxsize

# class nvshmemi_amo_t(Enum):
NVSHMEMI_AMO_ACK = 1
NVSHMEMI_AMO_INC = 2
NVSHMEMI_AMO_SET = 3
NVSHMEMI_AMO_ADD = 4
NVSHMEMI_AMO_AND = 5
NVSHMEMI_AMO_OR = 6
NVSHMEMI_AMO_XOR = 7
NVSHMEMI_AMO_SIGNAL = 8
NVSHMEM_SIGNAL_SET = 9
NVSHMEM_SIGNAL_ADD = 10
NVSHMEMI_AMO_SIGNAL_SET = NVSHMEM_SIGNAL_SET  # Note - NVSHMEM_SIGNAL_SET == 9
NVSHMEMI_AMO_SIGNAL_ADD = NVSHMEM_SIGNAL_ADD  # Note - NVSHMEM_SIGNAL_ADD == 10
NVSHMEMI_AMO_END_OF_NONFETCH = 11  # end of nonfetch atomics
NVSHMEMI_AMO_FETCH = 12
NVSHMEMI_AMO_FETCH_INC = 13
NVSHMEMI_AMO_FETCH_ADD = 14
NVSHMEMI_AMO_FETCH_AND = 15
NVSHMEMI_AMO_FETCH_OR = 16
NVSHMEMI_AMO_FETCH_XOR = 17
NVSHMEMI_AMO_SWAP = 18
NVSHMEMI_AMO_COMPARE_SWAP = 19
NVSHMEMI_AMO_OP_SENTINEL = sys.maxsize


class SymmCudaBuffer:

    def __init__(self, ptr, nbytes, dtype: torch.dtype, own_data: bool = True):
        self.ptr = ptr
        self.nbytes = nbytes
        self.dtype = dtype
        self.own_data = own_data
        self._device = torch.cuda.current_device()
        # https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        self.__cuda_array_interface__ = {
            "data": (self.ptr, False),
            "shape": tuple((self.nbytes, )),
            "typestr": "<i1",  # uint8 data type
            "strides": None,  # Contiguous memory
            "version": 3,
        }
        # keep alive
        self.cudart = cudart
        self.nvshmem_free = nvshmem_free

    def __del__(self):
        if self.own_data:
            if cudart is None:
                print("⚠️ cudart is unloaded, strange things may happen when program exiting. take it easy...")

            err, device = self.cudart.cudaGetDevice()
            assert err == self.cudart.cudaError_t.cudaSuccess, f"cudaError: {err}"
            if device != self._device:
                err, = self.cudart.cudaSetDevice(self._device)
                assert err == self.cudart.cudaError_t.cudaSuccess, f"cudaError: {err}"
            # sync
            err, = self.cudart.cudaDeviceSynchronize()
            assert err == self.cudart.cudaError_t.cudaSuccess, f"cudaError: {err}"

            self.nvshmem_free(self.ptr)

            # sync
            err, = self.cudart.cudaDeviceSynchronize()
            assert err == self.cudart.cudaError_t.cudaSuccess
            # set device back
            if device != self._device:
                err, = self.cudart.cudaSetDevice(device)
                assert err == self.cudart.cudaError_t.cudaSuccess, f"cudaError: {err}"

            self.own_data = False


def symm_tensor(tensor: torch.Tensor, peer: int) -> torch.Tensor:
    """
        tensor.data_ptr() should be the nvshmem_malloc() pointer with no offset
    """
    assert getattr(tensor, "__symm_tensor__", False), "tensor is not a symm_tensor"
    if peer == nvshmem_my_pe():
        return tensor
    buffer = SymmCudaBuffer(ptr=nvshmem_ptr(tensor.data_ptr(), peer), nbytes=tensor.nbytes, dtype=tensor.dtype,
                            own_data=False)
    return torch.as_tensor(buffer, device="cuda").view(tensor.dtype).view(tensor.shape)


def nvshmem_create_tensor(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    nbytes = torch.Size(shape).numel() * dtype.itemsize
    torch.cuda.synchronize()
    buffer = SymmCudaBuffer(ptr=nvshmem_malloc(nbytes), nbytes=nbytes, dtype=dtype, own_data=True)
    t = torch.as_tensor(buffer, device="cuda").view(dtype).view(shape)
    setattr(t, "__symm_tensor__", True)  # only those who owns data is marked as __symm_tensor__
    return t


def nvshmem_create_tensor_list_intra_node(shape: Sequence[int], dtype: torch.dtype) -> List[torch.Tensor]:
    t = nvshmem_create_tensor(shape, dtype)
    local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)
    rank = nvshmem_my_pe()
    rank_offset = rank - local_rank
    return [symm_tensor(t, i + rank_offset) for i in range(nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE))]


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
        buffer: bytes = bytearray(nvshmemx_get_uniqueid())
        unique_id: torch.Tensor = torch.frombuffer(buffer, dtype=torch.uint8).cpu().clone()
    else:
        # the default device("cpu") may be modified by set_default_device
        unique_id: torch.Tensor = torch.empty(128, dtype=torch.uint8, device="cpu")

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.cpu().numpy().tobytes()
    nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)
    nvshmem_barrier_all()
    torch.cuda.synchronize()
