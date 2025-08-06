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
import torch
from typing import Any, Dict, List
from .utils import cdiv
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class ElementwiseConfig(ConfigBase):
    BLOCK_SIZE: int = 512


@dataclass
class ElementwiseTask(TaskBase):
    config: ElementwiseConfig


@dataclass
class AddTask(ElementwiseTask):
    pass


def elementwise_config_factory(**kwargs) -> ElementwiseConfig:
    return dataclasses.replace(ElementwiseConfig(), **kwargs)


def codegen_add(task: ElementwiseConfig) -> str:
    config: ElementwiseConfig = task.config

    code = f"""
add_task_compute(task_base_info, scoreboard, BLOCK_SIZE={config.BLOCK_SIZE})
"""
    return code


@registry.register_task(op_type="add", task_cls=AddTask, config_factory=elementwise_config_factory,
                        codegen_func=codegen_add)
class AddTaskBuilder(TaskBuilderBase):

    @classmethod
    def _create_task(cls, layer_id: int, task_id: int, tile_id_or_start: int, num_tiles: int, config: ElementwiseConfig,
                     dependency: TaskDependency, io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]):
        return AddTask(layer_id, task_id, tile_id_or_start, num_tiles, config, dependency, io_tensors, extra_params)

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        lhs, rhs = io_tensors[0]
        output = io_tensors[1][0]
        num_elements = output.numel()
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        num_tiles = cdiv(num_elements, kernel_config.BLOCK_SIZE)
        cls.log(f"Add Task: num_tiles = {num_tiles}, dependency = {dependency}")
        tasks = []
        for i in range(num_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks
