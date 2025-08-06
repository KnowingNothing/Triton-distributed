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
from typing import List, Any, Dict, Callable
from .task_base import TaskBase, DeviceProp
from .config import ConfigBase
from .task_base import TaskDependency, TaskIDManager, OutputTilingDesc, InputDependencyDesc
from triton_dist.models.utils import logger
import torch


class TaskBuilderBase:
    _create_config: Callable = None

    @classmethod
    def log(cls, msg: str, level: str = "debug"):
        logger.log(msg, level)

    @classmethod
    def get_task_id(cls, layer_id: int) -> int:
        return TaskIDManager.get_task_id(layer_id)

    @classmethod
    def _create_task(cls, layer_id: int, task_id: int, tile_id_or_start: int, num_tiles: int, config: ConfigBase,
                     dependency: TaskDependency, io_tensors: List['torch.Tensor'], extra_params: Dict[str, Any],
                     inputs_dep: Dict['torch.Tensor', InputDependencyDesc], outs_tile_mapping: Dict['torch.Tensor',
                                                                                                    OutputTilingDesc]):
        raise NotImplementedError()

    @classmethod
    def get_problem_size(cls, io_tensors: List['torch.Tensor'], extra_params: Dict[str, Any]):
        raise NotImplementedError

    @classmethod
    def create_config(cls, **kwargs) -> ConfigBase:
        if cls._create_config is None:
            raise RuntimeError("Config factory not initialized. Ensure the task is registered.")
        return cls._create_config(**kwargs)

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List['torch.Tensor'], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params)
