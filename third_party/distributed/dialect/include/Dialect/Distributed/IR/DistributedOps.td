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
]> {
  let summary = "Wait for a signal to be ready (becomes 1) and produce a token.";

  let description = [{
    The operator creates a wait node that represents a spin lock waiting
    for a list of signals pointed by the `barrierPtr` to be set ready (set to 1).

    Example:

    ```mlir
    token = distributed.wait() {barrierPtr, num_barriers, scope = 1 : i32, semantic = 0 : i32}
    ```
  }];
  let arguments = (
    ins
    TT_PtrLike:$barrierPtr,
    TT_IntLike:$numBarriers,

    DefaultValuedAttr<TT_MemSyncScopeAttr, "MemSyncScope::GPU">:$scope,
    DefaultValuedAttr<TT_MemSemanticAttr, "MemSemantic::ACQUIRE">:$semantic
  );

  let results = (outs TT_IntLike:$token);

  let assemblyFormat = [{
      $barrierPtr `,` $numBarriers `,`
      `scope` `=` $scope `semantic` `=` $semantic
      attr-dict `:` type($barrierPtr) `,` type($numBarriers) `->` type($token)
    }];
}

def TT_ConsumeTokenOp : TT_Distributed_Op<"consume_token", [
  Elementwise,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches input type", "input", "result", "$_self">,
]> {
  let summary = "Consumes a token and builds data dependency between wait ops and input";

  let description = [{
    The operator creates a consume node that represents a data dependency between a wait node
    that produces a token and another node that uses the input data (e.g., load).

    Example:

    ```mlir
    distributed.consume_token() {input, token}
    ```
  }];
  let arguments = (
    ins
    AnyTypeOf<[TT_Type, TT_TensorDescType]>:$input,
    TT_IntLike:$token
  );

  let results = (outs AnyTypeOf<[TT_Type, TT_TensorDescType]>:$result);

  let assemblyFormat = [{
      $input `,` $token `,`
      attr-dict `:` type($input) `,` type($token) `->` type($result)
    }];
}


// TODO(zhengxuegui.0): device mesh support
def TT_GetRankOp : TT_Distributed_Op<"get_rank", [Pure]> {
  let summary = "Rank of the calling PE";

  let arguments = (ins I32:$axis);

  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}


def TT_GetNumRanksOp : TT_Distributed_Op<"get_num_ranks", [Pure]> {
  let summary = "Get the number of PEs runing in the program";

  let arguments = (ins I32:$axis);

  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}


def TT_SymmAtOp : TT_Distributed_Op<"symm_at", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"remoteAddr matches symmAddr type", "symmAddr", "remoteAddr", "$_self">,
]> {
  let summary = "obtain the symmetric address on the specified Rank.";

  let arguments = (ins
    TT_Ptr:$symmAddr,
    I32:$rank
  );

  let results = (outs TT_Ptr:$remoteAddr);

  let assemblyFormat = "$symmAddr `,` $rank `,` attr-dict `:` qualified(type($symmAddr))";
}
#endif // DISTRIBUTED_OPS
