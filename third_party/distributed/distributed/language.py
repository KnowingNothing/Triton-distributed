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

from triton.language import core as tl
from triton.language.semantic import cast, _str_to_sem, _str_to_scope, to_tensor, _convert_elem_to_ir_value
from triton.language.core import builtin

@builtin
def wait(barrierPtrs, numBarriers, scope: str, semantic: str, _builder=None):
    if not barrierPtrs.type.scalar.is_ptr():
        raise ValueError(
            f"Unsupported barrierPtrs type {barrierPtrs.type.__repr__()} in `distributed.language.wait`")
        
    scope = _str_to_scope(scope)
    semantic = _str_to_sem(semantic)
    return tl.tensor(
        _builder.create_distributed_wait(
            barrierPtrs.handle, to_tensor(numBarriers, _builder).handle, scope, semantic, tl.int32.to_ir(_builder)),
        tl.int32)

@builtin
def consume_token(value, token, _builder=None):
    assert token.type.scalar.is_int(), "token must be of int type"
    handle = _builder.create_distributed_consume_token(value.handle, token.handle)
    if isinstance(value, tl._experimental_tensor_descriptor):
        return tl._experimental_tensor_descriptor(handle, value.shape, value.strides, value.type)
    else:
        return tl.tensor(handle, value.type)


@builtin
def rank(axis=-1, _builder=None):
    axis = _convert_elem_to_ir_value(_builder, axis, require_i64=False)
    return tl.tensor(_builder.create_get_rank(axis), tl.int32)


@builtin
def num_ranks(axis=-1, _builder=None):
    axis = _convert_elem_to_ir_value(_builder, axis, require_i64=False)
    return tl.tensor(_builder.create_get_num_ranks(axis), tl.int32)


@builtin
def symm_at(ptr, rank, _builder=None):
    assert not ptr.type.is_block() and ptr.type.is_ptr(), "only support scalar pointer"
    rank = _convert_elem_to_ir_value(_builder, rank, require_i64=False)
    return tl.tensor(_builder.create_symm_at(ptr.handle, rank), ptr.type)
