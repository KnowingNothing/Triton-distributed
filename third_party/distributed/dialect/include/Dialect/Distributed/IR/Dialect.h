//===- Dialect.h -------------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_DIALECT_DISTRIBUTED_IR_DIALECT_H_
#define TRITON_DIALECT_DISTRIBUTED_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"

// clang-format off
#include "distributed/dialect/include/Dialect/Distributed/IR/Dialect.h.inc"
#include "distributed/dialect/include/Dialect/Distributed/IR/DistributedEnums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "distributed/dialect/include/Dialect/Distributed/IR/DistributedAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "distributed/dialect/include/Dialect/Distributed/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace distributed {} // namespace distributed
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_DISTRIBUTED_IR_DIALECT_H_
