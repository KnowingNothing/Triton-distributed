#ifndef DISTRIBUTED_ATTRDEFS
#define DISTRIBUTED_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "DistributedDialect.td"
include "mlir/IR/EnumAttr.td"

class Distributed_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<Distributed_Dialect, name, traits, baseCppClass> {
}


def TT_SignalOpAttr : I32EnumAttr<
    "SignalOp", "",
    [
        I32EnumAttrCase<"SET", 1, "set">,
        I32EnumAttrCase<"ADD", 2, "add">,
    ]> {
    let cppNamespace = "::mlir::triton::distributed";
}

def TT_CommScopeAttr : I32EnumAttr<
    "CommScope", "",
    [
      I32EnumAttrCase<"GPU", 1, "gpu">,
      I32EnumAttrCase<"INTRA_NODE", 2, "intra_node">,
      I32EnumAttrCase<"INTER_NODE", 3, "inter_node">,
    ]> {
    let cppNamespace = "::mlir::triton::distributed";
}
#endif // DISTRIBUTED_ATTRDEFS
