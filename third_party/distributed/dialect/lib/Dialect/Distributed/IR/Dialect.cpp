//===- Dialect.cpp ------------------------------------------------- C++ ---===//
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

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "Dialect/Distributed/IR/Dialect.h"
#include "Dialect/Distributed/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::distributed;

void mlir::triton::distributed::DistributedDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Distributed/IR/DistributedAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/Distributed/IR/Ops.cpp.inc"
      >();
}

#include "Dialect/Distributed/IR/DistributedEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Distributed/IR/DistributedAttrDefs.cpp.inc"

