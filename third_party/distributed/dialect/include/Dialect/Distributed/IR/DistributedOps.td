#ifndef DISTRIBUTED_OPS
#define DISTRIBUTED_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "DistributedDialect.td"
include "DistributedAttrDefs.td"

class TT_Distributed_Op<string mnemonic, list<Trait> traits = []> :
    Op<Distributed_Dialect, mnemonic, !listconcat(traits, [])> {
}


def TT_WaitOp : TT_Distributed_Op<"wait", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches data ptr type", "dataPtr", "result", "$_self">,
]> {
  let summary = "Wait for a signal to be ready (becomes 1)";

  let description = [{
    The operator creats a wait node that represents a spin lock waiting
    for a signal pointed by the `barrierPtr` to be set ready (set to 1).

    Example:

    ```mlir
    distributed.wait() {dataPtr, barrierPtr, scope = 1 : i32, semantic = 0 : i32}
    ```
  }];
  let arguments = (
    ins
    AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$dataPtr,
    AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$barrierPtr,

    DefaultValuedAttr<TT_MemSyncScopeAttr, "MemSyncScope::GPU">:$scope,
    DefaultValuedAttr<TT_MemSemanticAttr, "MemSemantic::ACQUIRE">:$semantic
  );

  let results = (outs TT_Type:$result);

  let assemblyFormat = [{
      $dataPtr `,` $barrierPtr `,`
      `scope` `=` $scope `semantic` `=` $semantic
      attr-dict `:` type($dataPtr) `,` type($barrierPtr) `->` type($result)
    }];
}

#endif // DISTRIBUTED_OPS
