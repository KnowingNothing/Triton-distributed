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
from triton.language.semantic import cast, _str_to_sem, _str_to_scope
from triton.language.core import builtin

@builtin
def wait(dataPtrs, barrierPtrs, scope: str, semantic: str, _builder=None):
    if not dataPtrs.type.scalar.is_ptr():
        raise ValueError(f"Unsupported dataPtrs type {dataPtrs.type.__repr__()} in `distributed.language.wait`")
    if not barrierPtrs.type.scalar.is_ptr():
        raise ValueError(f"Unsupported barrierPtrs type {barrierPtrs.type.__repr__()} in `distributed.language.wait`")
    ptr_ty = dataPtrs.type.scalar
    elt_ty = ptr_ty.element_ty
    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    is_bool = elt_ty == tl.int1
    if is_bool:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        dataPtrs = cast(dataPtrs, ptr_ty, _builder)
    # Create loaded result type `dst_ty`
    if dataPtrs.type.is_block():
        shape = dataPtrs.type.get_block_shapes()
        dst_ty = tl.block_type(tl.pointer_type(elt_ty, ptr_ty.address_space), shape)
    else:
        # Load by de-referencing the pointer of scalar
        dst_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        
    scope = _str_to_scope(scope)
    semantic = _str_to_sem(semantic)
    return tl.tensor(_builder.create_distributed_wait(dataPtrs.handle, barrierPtrs.handle, scope, semantic), dst_ty)