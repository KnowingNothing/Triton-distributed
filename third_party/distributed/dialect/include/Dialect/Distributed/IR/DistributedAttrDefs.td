#ifndef DISTRIBUTED_ATTRDEFS
#define DISTRIBUTED_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "DistributedDialect.td"

class Distributed_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<Distributed_Dialect, name, traits, baseCppClass> {
}

#endif // DISTRIBUTED_ATTRDEFS
