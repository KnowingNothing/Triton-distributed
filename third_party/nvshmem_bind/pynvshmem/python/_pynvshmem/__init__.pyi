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
from typing import List, Sequence

import numpy as np
import torch


def nvshmemx_cumodule_init(module: np.intp) -> None:
    ...


def nvshmemx_cumodule_finalize(module: np.intp) -> None:
    ...


def nvshmem_malloc(size: np.uint) -> np.intp:
    ...


def nvshmemx_get_uniqueid() -> bytes:
    ...


def nvshmemx_init_attr_with_uniqueid(rank: np.int32, nranks: np.int32, unique_id: bytes) -> None:
    ...


def nvshmem_int_p(ptr: np.intp, src: np.int32, dst: np.int32) -> None:
    ...


def nvshmem_barrier_all():
    ...


def nvshmem_barrier_all_on_stream():
    ...


# torch related
def nvshmem_create_tensor(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    ...


def nvshmem_create_tensor_list_intra_node(shape: Sequence[int], dtype: torch.dtype) -> List[torch.Tensor]:
    ...
