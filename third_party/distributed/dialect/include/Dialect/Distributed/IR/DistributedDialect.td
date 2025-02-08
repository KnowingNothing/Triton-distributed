#ifndef DISTRIBUTED_DIALECT
#define DISTRIBUTED_DIALECT

include "mlir/IR/OpBase.td"

def Distributed_Dialect : Dialect {
  let name = "distributed";
  let cppNamespace = "::mlir::triton::distributed";

  let description = [{
    Distributed Dialect provides core ops for building third-party compiler-based
    distributed computation and communication support.
  }];

  let dependentDialects = [];
}

#endif
