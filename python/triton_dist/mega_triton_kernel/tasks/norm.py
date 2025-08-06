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
from typing import Tuple, Any, Dict, List
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class QKNormRopeUpdateKVCacheConfig(ConfigBase):
    pass


class RMSNormConfig(ConfigBase):
    BLOCK_SIZE_N: int = 2048


@dataclass
class QKNormRopeUpdateKVCacheTask(TaskBase):
    config: QKNormRopeUpdateKVCacheConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps/rope_theta as constexpr in codegen
        return ()


@dataclass
class RMSNormTask(TaskBase):
    config: RMSNormConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps as constexpr in codegen
        return ()


def qk_norm_rope_update_kvcache_config_factory(**kwargs) -> QKNormRopeUpdateKVCacheConfig:
    return dataclasses.replace(QKNormRopeUpdateKVCacheConfig(), **kwargs)


def rms_norm_config_factory(**kwargs) -> RMSNormConfig:
    return dataclasses.replace(RMSNormConfig(), **kwargs)


def codegen_qk_norm_rope_update_kvcache(task: QKNormRopeUpdateKVCacheTask) -> str:
    qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = task.io_tensors[0]
    key_cache, value_cache, q_norm_rope = task.io_tensors[1]
    Q_HEAD_DIM = qkv.shape[-1]
    V_HEAD_DIM = value_cache.shape[-1]
    NUM_KV_HEADS = key_cache.shape[-2]
    NUM_Q_HEADS = qkv.shape[-2] - 2 * NUM_KV_HEADS
    PAGE_SIZE, NUM_KV_HEADS, V_HEAD_DIM = value_cache.shape[-3], value_cache.shape[-2], value_cache.shape[-1]
    MAX_NUM_BLOCKS_PER_SEQ = block_tables.shape[-1]
    code = f"""
rmsnorm_rope_update_kv_cache_task_compute(
    task_base_info, scoreboard, NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, Q_HEAD_DIM={Q_HEAD_DIM},
    V_HEAD_DIM={V_HEAD_DIM}, PAGE_SIZE={PAGE_SIZE}, MAX_NUM_BLOCKS_PER_SEQ={MAX_NUM_BLOCKS_PER_SEQ},
    Q_RMS_EPS={task.extra_params["q_rms_eps"]}, K_RMS_EPS={task.extra_params["k_rms_eps"]}
)
"""
    return code


def codegen_rms_norm(task: RMSNormTask) -> str:
    config: RMSNormConfig = task.config
    code = f"""
rmsnorm_task_compute(task_base_info, scoreboard, RMS_EPS={task.extra_params["rms_eps"]}, BLOCK_SIZE_N = {config.BLOCK_SIZE_N})
"""
    return code


@registry.register_task(op_type="qk_norm_rope_update_kvcache", task_cls=QKNormRopeUpdateKVCacheTask,
                        config_factory=qk_norm_rope_update_kvcache_config_factory,
                        codegen_func=codegen_qk_norm_rope_update_kvcache)
class QKNormRopeUpdateKVCacheTaskBuilder(TaskBuilderBase):

    @classmethod
    def _create_task(cls, layer_id: int, task_id: int, tile_id_or_start: int, num_tiles: int,
                     config: QKNormRopeUpdateKVCacheConfig, dependency: TaskDependency,
                     io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]):
        return QKNormRopeUpdateKVCacheTask(layer_id, task_id, tile_id_or_start, num_tiles, config, dependency,
                                           io_tensors, extra_params)

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = io_tensors[0]
        key_cache, value_cache, q_norm_rope = io_tensors[1]
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        assert len(qkv.shape) == 4
        batch, seq_len, num_qkv_heads, head_dim = qkv.shape
        num_kv_heads = key_cache.shape[-2]
        num_qk_heads = num_qkv_heads - num_kv_heads
        num_tiles = batch * seq_len * num_qk_heads
        cls.log(f"KNormRopeUpdateKVCache Task: num_tiles = {num_tiles}")
        tasks = []
        for i in range(num_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks


@registry.register_task(op_type="rms_norm", task_cls=RMSNormTask, config_factory=rms_norm_config_factory,
                        codegen_func=codegen_rms_norm)
class RMSNormTaskBuilder(TaskBuilderBase):

    @classmethod
    def _create_task(cls, layer_id: int, task_id: int, tile_id_or_start: int, num_tiles: int,
                     config: QKNormRopeUpdateKVCacheConfig, dependency: TaskDependency,
                     io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]):
        return RMSNormTask(layer_id, task_id, tile_id_or_start, num_tiles, config, dependency, io_tensors, extra_params)

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        input, weight = io_tensors[0]
        output = io_tensors[1][0]
        num_tiles = output.numel() // output.shape[-1]
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        cls.log(f"RMS Norm Task: num_tiles = {num_tiles}")
        tasks = []
        for i in range(num_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks
