/*
 * Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
 */
#include <optional>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

namespace {

namespace py = pybind11;
using namespace mlir;
using namespace triton;

llvm::raw_fd_ostream &mlir_dumps() {
  std::error_code EC;
  static llvm::raw_fd_ostream S(::triton::tools::getStrEnv("MLIR_DUMP_PATH"),
                                EC, llvm::sys::fs::CD_CreateAlways);
  assert(!EC);
  return S;
}

llvm::raw_ostream &mlir_dumps_or_dbgs() {
  if (!::triton::tools::getStrEnv("MLIR_DUMP_PATH").empty()) {
    return mlir_dumps();
  } else {
    return llvm::dbgs();
  }
}

// A custom op builder that keeps track of the last location
class TritonOpBuilder {
public:
  TritonOpBuilder(MLIRContext *context) {
    builder = std::make_unique<OpBuilder>(context);
    lastLoc = std::make_unique<Location>(builder->getUnknownLoc());
  }

  OpBuilder &getBuilder() { return *builder; }
  MLIRContext *getContext() { return builder->getContext(); }

  bool isLineInfoEnabled() { return lineInfoEnabled; }

  void setLastLoc(Location loc) {
    if (lineInfoEnabled)
      lastLoc = std::make_unique<Location>(loc);
  }

  void setLastLoc(const std::string &fileName, int line, int column) {
    auto context = builder->getContext();
    setLastLoc(FileLineColLoc::get(context, fileName, line, column));
  }

  Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  void setInsertionPointToStart(Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(OpBuilder::InsertPoint pt) {
    if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
      setLastLoc(pt.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(pt);
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<Location> lastLoc;
  bool lineInfoEnabled = !triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO");
};

// Run the pass manager under a source manager diagnostic handler, which
// enables emitted MLIR diagnostics to directly reference Python source
// code. This diagnostic handler supports filtering diagnostic info by
// severity levels.
struct TritonSourceMgrDiagnosticHandler : public SourceMgrDiagnosticHandler {
  TritonSourceMgrDiagnosticHandler(MLIRContext *ctx,
                                   DiagnosticSeverity minSeverity)
      : SourceMgrDiagnosticHandler(sourceMgr, ctx, llvm::errs()) {
    setHandler([this, minSeverity](Diagnostic &diag) {
      auto severity = diag.getSeverity();
      switch (severity) {
      case DiagnosticSeverity::Error:
        break;
      case DiagnosticSeverity::Warning:
        if (minSeverity == DiagnosticSeverity::Error)
          return success();
        break;
      case DiagnosticSeverity::Remark:
        if (minSeverity == DiagnosticSeverity::Error ||
            minSeverity == DiagnosticSeverity::Warning)
          return success();
        break;
      case DiagnosticSeverity::Note:
        // notes are handled somewhere else.
        return failure();
      default:
        llvm_unreachable("Unknown diagnostic severity");
      }
      emitDiagnostic(diag);
      return success();
    });
  }

  llvm::SourceMgr sourceMgr;
};

std::string locationToString(Location loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  loc.print(os);
  os.flush(); // Make sure all the content is dumped into the 'str' string
  return str;
}

// Function to parse a comma-separated string into a vector of C-style strings
llvm::SmallVector<const char *, 3>
parseCommaSeparatedValues(const std::string &input,
                          llvm::SmallVector<std::string, 3> &storage) {
  llvm::SmallVector<StringRef, 3> split;
  llvm::SmallVector<const char *, 3> result;
  StringRef(input.c_str()).split(split, ',');
  llvm::transform(split, std::back_inserter(result), [&storage](StringRef str) {
    // StringRefs are not always null-terminated.
    // The purpose for this storage pattern is to
    // produce a collection of C-strings that are.
    storage.push_back(str.str());
    return storage.back().c_str();
  });
  return result;
}

void outputWarning(Location loc, const std::string &msg) {
  std::string locStr = locationToString(loc);

  PyErr_WarnEx(PyExc_UserWarning, (locStr + ": " + msg).c_str(),
               /*stack_level=*/2);
}

// Allow dump a reproducer in the console on crash.
struct ConsoleReproducerStream : public mlir::ReproducerStream {
  ~ConsoleReproducerStream() override {}

  StringRef description() override {
    return "std::errs, please share the reproducer above with Triton project.";
  }
  raw_ostream &os() override { return llvm::errs(); }
};

static ReproducerStreamFactory makeConsoleReproducer() {
  return [](std::string &error) -> std::unique_ptr<ReproducerStream> {
    return std::make_unique<ConsoleReproducerStream>();
  };
}

} // anonymous namespace

/*****************************************************************************/
/* Python bindings for ir                                                    */
/*****************************************************************************/

void init_triton_distributed_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::enum_<distributed::SignalOp>(m, "SIGNAL_OP", py::module_local())
      .value("SET", distributed::SignalOp::SET)
      .value("ADD", distributed::SignalOp::ADD)
      .export_values();

  py::enum_<distributed::CommScope>(m, "COMM_SCOPE", py::module_local())
      .value("GPU", distributed::CommScope::GPU)
      .value("INTRA_NODE", distributed::CommScope::INTRA_NODE)
      .value("INTER_NODE", distributed::CommScope::INTER_NODE)
      .export_values();

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry
        .insert<TritonDialect, ::mlir::triton::gpu::TritonGPUDialect,
                math::MathDialect, arith::ArithDialect, scf::SCFDialect,
                tensor::TensorDialect, ::mlir::gpu::GPUDialect,
                cf::ControlFlowDialect, ::mlir::triton::proton::ProtonDialect,
                ::mlir::triton::distributed::DistributedDialect,
                ::mlir::triton::simt::SIMTDialect, LLVM::LLVMDialect,
                mlir::ub::UBDialect>();
    mlir::LLVM::registerInlinerInterface(registry);
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    mlir::LLVM::registerInlinerInterface(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // simt ops
  py::class_<triton::simt::BlockYieldOp, OpState>(m, "BlockYieldOp",
                                                  py::module_local());
  py::class_<triton::simt::SIMTExecRegionOp, OpState>(m, "SIMTExecRegionOp",
                                                      py::module_local())
      .def(
          "get_simt_entry_block",
          [](triton::simt::SIMTExecRegionOp &self) -> Block * {
            return &self.getDefaultRegion().front();
          },
          ret::reference);

  // Since the builder is private, we inherit the original triton builder by
  // copying it and add the support of TritonDitributed. In this way, we can
  // directly use this new builder without maintaining two builders on the
  // Python side. Because the `InsertionPointer` needs to be consistent between
  // builders.
  py::class_<TritonOpBuilder>(m, "builder", py::module_local(),
                              py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      // getters
      .def("create_module",
           [](TritonOpBuilder &self) -> ModuleOp {
             return self.create<ModuleOp>();
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](TritonOpBuilder &self, Block &block) -> void {
             self.setInsertionPointToStart(block);
           })
      .def("set_insertion_point_to_end",
           [](TritonOpBuilder &self, Block &block) {
             self.setInsertionPointToEnd(block);
           })
      .def("set_insertion_point_after",
           [](TritonOpBuilder &self, Operation &op) {
             self.setInsertionPointAfter(op);
           })
      .def(
          "get_insertion_block",
          [](TritonOpBuilder &self) -> Block * {
            return self.getBuilder().getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point",
           [](TritonOpBuilder &self) {
             return self.getBuilder().saveInsertionPoint();
           })
      .def("restore_insertion_point",
           [](TritonOpBuilder &self, OpBuilder::InsertPoint pt) {
             self.restoreInsertionPoint(pt);
           })
      // Attr
      .def(
          "get_unit_attr",
          [](TritonOpBuilder &self) { return self.getBuilder().getUnitAttr(); })
      .def("get_bool_attr",
           [](TritonOpBuilder &self, bool value) {
             return self.getBuilder().getBoolAttr(value);
           })
      .def("get_int32_attr",
           [](TritonOpBuilder &self, int32_t value) {
             return self.getBuilder().getI32IntegerAttr(value);
           })
      // Use arith.ConstantOp to create constants
      // Constants
      .def("get_int1",
           [](TritonOpBuilder &self, bool v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI1Type()));
           })
      .def("get_int8",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_int16",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_int32",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_int64",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_uint8",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_uint16",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_uint32",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_uint64",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_bf16",
           [](TritonOpBuilder &self, float v) -> Value {
             auto type = self.getBuilder().getBF16Type();
             return self.create<arith::ConstantFloatOp>(
                 APFloat(type.getFloatSemantics(), std::to_string(v)), type);
           })
      .def("get_fp16",
           [](TritonOpBuilder &self, float v) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF16FloatAttr(v));
           })
      .def("get_fp32",
           [](TritonOpBuilder &self, float v) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF32FloatAttr(v));
           })
      .def("get_fp64",
           [](TritonOpBuilder &self, double v) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF64FloatAttr(v));
           })
      .def("get_null_value",
           [](TritonOpBuilder &self, Type type) -> Value {
             if (auto floatTy = dyn_cast<FloatType>(type))
               return self.create<arith::ConstantFloatOp>(
                   APFloat(floatTy.getFloatSemantics(), 0), floatTy);
             else if (auto intTy = dyn_cast<IntegerType>(type))
               return self.create<arith::ConstantIntOp>(0, intTy);
             else
               throw std::runtime_error("Not implemented");
           })
      .def("get_all_ones_value",
           [](TritonOpBuilder &self, Type type) -> Value {
             uint64_t val = 0xFFFFFFFFFFFFFFFF;
             if (auto intTy = dyn_cast<IntegerType>(type))
               return self.create<arith::ConstantIntOp>(val, intTy);
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getNoneType();
           })
      .def("get_int1_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI1Type();
           }) // or ret::copy?
      .def("get_int8_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI8Type();
           })
      .def("get_int16_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<IntegerType>(16);
           })
      .def("get_int32_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI32Type();
           })
      .def("get_int64_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI64Type();
           })
      .def("get_fp8e4nv_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E4M3FNType>();
           })
      .def("get_fp8e4b8_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E4M3FNUZType>();
           })
      .def("get_fp8e4b15_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI8Type();
           })
      .def("get_fp8e5_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E5M2Type>();
           })
      .def("get_fp8e5b16_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E5M2FNUZType>();
           })
      .def("get_half_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF16Type();
           })
      .def("get_bf16_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getBF16Type();
           })
      .def("get_float_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF32Type();
           })
      .def("get_double_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF64Type();
           })
      .def("get_ptr_ty",
           [](TritonOpBuilder &self, Type &type, int addrSpace) -> Type {
             return PointerType::get(type, addrSpace);
           })
      .def("get_block_ty",
           [](TritonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape) -> Type {
             return RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](TritonOpBuilder &self, std::vector<Type> inTypes,
              std::vector<Type> outTypes) -> Type {
             return self.getBuilder().getFunctionType(inTypes, outTypes);
           })
      // locs
      .def("set_loc",
           [](TritonOpBuilder &self, Location loc) { self.setLastLoc(loc); })
      .def("set_loc",
           [](TritonOpBuilder &self, const std::string &fileName, int line,
              int column) { self.setLastLoc(fileName, line, column); })
      .def("get_loc",
           [](TritonOpBuilder &self) -> Location { return self.getLastLoc(); })

      // Ops
      .def("get_or_insert_function",
           [](TritonOpBuilder &self, ModuleOp &module, std::string &funcName,
              Type &funcType, std::string &visibility,
              bool noinline) -> FuncOp {
             if (Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<FuncOp>(funcOperation);
             if (auto funcTy = dyn_cast<FunctionType>(funcType)) {
               llvm::SmallVector<NamedAttribute> attrs = {
                   NamedAttribute(
                       self.getBuilder().getStringAttr("sym_visibility"),
                       self.getBuilder().getStringAttr(visibility)),
                   NamedAttribute(self.getBuilder().getStringAttr("noinline"),
                                  self.getBuilder().getBoolAttr(noinline))};
               return self.create<FuncOp>(funcName, funcTy, attrs);
             }
             throw std::invalid_argument("invalid function type");
           })
      .def(
          "create_block",
          [](TritonOpBuilder &self) -> Block * {
            Region *parent = self.getBuilder().getBlock()->getParent();
            return self.getBuilder().createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](TritonOpBuilder &self, Region &parent,
             std::vector<Type> &argTypes) -> Block * {
            // TODO: update arg loc
            auto loc = self.getBuilder().getUnknownLoc();
            llvm::SmallVector<Location, 8> argLocs(argTypes.size(), loc);
            return self.getBuilder().createBlock(&parent, {}, argTypes,
                                                 argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](TritonOpBuilder &self) -> Block * { return new Block(); },
          ret::reference)
      // Function
      .def("ret",
           [](TritonOpBuilder &self, std::vector<Value> &vals) -> OpState {
             return self.create<ReturnOp>(vals);
           })
      .def("call",
           [](TritonOpBuilder &self, FuncOp &func, std::vector<Value> &args)
               -> OpState { return self.create<CallOp>(func, args); })
      // Unstructured control flow
      .def("create_cond_branch",
           [](TritonOpBuilder &self, Value condition, Block *trueDest,
              Block *falseDest) -> OpState {
             return self.create<cf::CondBranchOp>(condition, trueDest,
                                                  falseDest);
           })
      .def("create_branch",
           [](TritonOpBuilder &self, Block *dest, std::vector<Value> &args)
               -> OpState { return self.create<cf::BranchOp>(dest, args); })
      // Structured control flow
      .def("create_for_op",
           [](TritonOpBuilder &self, Value &lb, Value &ub, Value &step,
              std::vector<Value> &initArgs) -> scf::ForOp {
             return self.create<scf::ForOp>(lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](TritonOpBuilder &self, std::vector<Type> &retTypes,
              Value &condition, bool withElse) -> scf::IfOp {
             return self.create<scf::IfOp>(retTypes, condition, withElse);
           })
      .def("create_yield_op",
           [](TritonOpBuilder &self, std::vector<Value> &yields)
               -> scf::YieldOp { return self.create<scf::YieldOp>(yields); })
      .def("create_while_op",
           [](TritonOpBuilder &self, std::vector<Type> &retTypes,
              std::vector<Value> &initArgs) -> scf::WhileOp {
             return self.create<scf::WhileOp>(retTypes, initArgs);
           })
      .def("create_condition_op",
           [](TritonOpBuilder &self, Value &cond,
              std::vector<Value> &args) -> scf::ConditionOp {
             return self.create<scf::ConditionOp>(cond, args);
           })

      // miscellaneous
      .def("create_make_range",
           [](TritonOpBuilder &self, int start, int end) -> Value {
             auto retType = RankedTensorType::get(
                 {end - start}, self.getBuilder().getI32Type());
             return self.create<MakeRangeOp>(retType, start, end);
           })

      // Cast instructions
      // Conversions for custom FP types (FP8 and non-standard rounding modes)
      .def("create_fp_to_fp",
           [](TritonOpBuilder &self, Value &src, Type &dstType,
              std::optional<RoundingMode> roundingMode) -> Value {
             if (roundingMode.has_value())
               return self.create<FpToFpOp>(
                   dstType, src,
                   RoundingModeAttr::get(self.getBuilder().getContext(),
                                         roundingMode.value()));
             else
               return self.create<FpToFpOp>(dstType, src);
           })
      // Conversions for standard LLVM builtin types
      .def("create_bitcast",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<BitcastOp>(dstType, src);
           })
      .def("create_si_to_fp",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::SIToFPOp>(dstType, src);
           })
      .def("create_ui_to_fp",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::UIToFPOp>(dstType, src);
           })
      .def("create_fp_to_si",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::FPToSIOp>(dstType, src);
           })
      .def("create_fp_to_ui",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::FPToUIOp>(dstType, src);
           })
      .def("create_fp_ext",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::ExtFOp>(dstType, src);
           })
      .def("create_fp_trunc",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::TruncFOp>(dstType, src);
           })
      .def("create_int_cast",
           [](TritonOpBuilder &self, Value &src, Type &dstType,
              bool isSigned) -> Value {
             // get element type if necessary
             Type srcType = src.getType();
             auto srcTensorType = dyn_cast<RankedTensorType>(srcType);
             auto dstTensorType = dyn_cast<RankedTensorType>(dstType);
             Type srcEltType = srcType;
             Type dstEltType = dstType;
             if (dstTensorType && srcTensorType) {
               dstEltType = dstTensorType.getElementType();
               srcEltType = srcTensorType.getElementType();
             }
             unsigned srcWidth = srcEltType.getIntOrFloatBitWidth();
             unsigned dstWidth = dstEltType.getIntOrFloatBitWidth();
             if (srcWidth == dstWidth)
               return self.create<arith::BitcastOp>(dstType, src);
             else if (srcWidth > dstWidth)
               return self.create<arith::TruncIOp>(dstType, src);
             else if (isSigned)
               return self.create<arith::ExtSIOp>(dstType, src);
             else
               return self.create<arith::ExtUIOp>(dstType, src);
           })
      .def("create_fmul",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MulFOp>(lhs, rhs);
           })
      .def("create_fdiv",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivFOp>(lhs, rhs);
           })
      .def("create_frem",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemFOp>(lhs, rhs);
           })
      .def("create_fadd",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AddFOp>(lhs, rhs);
           })
      .def("create_fsub",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::SubFOp>(lhs, rhs);
           })
      .def("create_mul",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MulIOp>(lhs, rhs);
           })
      .def("create_umulhi",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<triton::MulhiUIOp>(lhs, rhs);
           })
      .def("create_sdiv",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivSIOp>(lhs, rhs);
           })
      .def("create_udiv",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivUIOp>(lhs, rhs);
           })
      .def("create_srem",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemSIOp>(lhs, rhs);
           })
      .def("create_urem",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemUIOp>(lhs, rhs);
           })
      .def("create_add",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AddIOp>(lhs, rhs);
           })
      .def("create_sub",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::SubIOp>(lhs, rhs));
           })
      .def("create_fma",
           [](TritonOpBuilder &self, Value &a, Value &b, Value &c) -> Value {
             return Value(self.create<math::FmaOp>(a, b, c));
           })
      .def("create_shl",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::ShLIOp>(lhs, rhs));
           })
      .def("create_lshr",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::ShRUIOp>(lhs, rhs));
           })
      .def("create_ashr",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::ShRSIOp>(lhs, rhs));
           })
      .def("create_minsi",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinSIOp>(lhs, rhs));
           })
      .def("create_minui",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinUIOp>(lhs, rhs));
           })
      // minimumf follows the torch.minimum convention and returns NaN if either
      // operand is NaN
      .def("create_minimumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinimumFOp>(lhs, rhs));
           })
      // minnumf follows the torch.fmin convention and returns the non-NaN
      // operand
      .def("create_minnumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinNumFOp>(lhs, rhs));
           })
      .def("create_maxsi",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaxSIOp>(lhs, rhs));
           })
      .def("create_maxui",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaxUIOp>(lhs, rhs));
           })
      // maximumf follows the torch.maximum convention and returns NaN if either
      // operand is NaN
      .def("create_maximumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaximumFOp>(lhs, rhs));
           })
      // maxnumf follows the torch.fmax convention and returns the non-NaN
      // operand
      .def("create_maxnumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaxNumFOp>(lhs, rhs));
           })
      .def("create_clampf",
           [](TritonOpBuilder &self, Value &input, Value &min, Value &max,
              PropagateNan propagateNan) -> Value {
             return Value(self.create<ClampFOp>(input, min, max, propagateNan));
           })
      .def("create_precise_sqrt",
           [](TritonOpBuilder &self, Value &input) -> Value {
             return Value(self.create<PreciseSqrtOp>(input));
           })
      .def("create_precise_divf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<PreciseDivFOp>(lhs, rhs));
           })
      // AddPtr (similar to GEP)
      .def("create_addptr",
           [](TritonOpBuilder &self, Value &ptr, Value &offset) -> Value {
             return self.create<AddPtrOp>(ptr.getType(), ptr, offset);
           })
      // Comparison (int)
      .def("create_icmpSLE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sle, lhs,
                                               rhs);
           })
      .def("create_icmpSLT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::slt, lhs,
                                               rhs);
           })
      .def("create_icmpSGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sge, lhs,
                                               rhs);
           })
      .def("create_icmpSGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, lhs,
                                               rhs);
           })
      .def("create_icmpULE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ule, lhs,
                                               rhs);
           })
      .def("create_icmpULT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ult, lhs,
                                               rhs);
           })
      .def("create_icmpUGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::uge, lhs,
                                               rhs);
           })
      .def("create_icmpUGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, lhs,
                                               rhs);
           })
      .def("create_icmpEQ",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs,
                                               rhs);
           })
      .def("create_icmpNE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ne, lhs,
                                               rhs);
           })
      // Comparison (float)
      .def("create_fcmpOLT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, lhs,
                                               rhs);
           })
      .def("create_fcmpOGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, lhs,
                                               rhs);
           })
      .def("create_fcmpOLE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OLE, lhs,
                                               rhs);
           })
      .def("create_fcmpOGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OGE, lhs,
                                               rhs);
           })
      .def("create_fcmpOEQ",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, lhs,
                                               rhs);
           })
      .def("create_fcmpONE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ONE, lhs,
                                               rhs);
           })
      .def("create_fcmpULT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ULT, lhs,
                                               rhs);
           })
      .def("create_fcmpUGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UGT, lhs,
                                               rhs);
           })
      .def("create_fcmpULE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ULE, lhs,
                                               rhs);
           })
      .def("create_fcmpUGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UGE, lhs,
                                               rhs);
           })
      .def("create_fcmpUEQ",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UEQ, lhs,
                                               rhs);
           })
      .def("create_fcmpUNE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UNE, lhs,
                                               rhs);
           })
      // // Logical
      .def("create_and",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AndIOp>(lhs, rhs);
           })
      .def("create_xor",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::XOrIOp>(lhs, rhs);
           })
      .def("create_or",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::OrIOp>(lhs, rhs);
           })
      // Input/Output
      .def("create_load",
           [](TritonOpBuilder &self, Value &ptrs, CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy, bool isVolatile) -> Value {
             return self.create<LoadOp>(ptrs, cacheModifier, evictionPolicy,
                                        isVolatile);
           })
      .def("create_store",
           [](TritonOpBuilder &self, Value &ptrs, Value &value,
              CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> void {
             self.create<StoreOp>(ptrs, value, cacheModifier, evictionPolicy);
           })
      .def("create_tensor_pointer_load",
           [](TritonOpBuilder &self, Value &ptr,
              std::vector<int32_t> &boundaryCheck,
              std::optional<PaddingOption> paddingOption,
              CacheModifier cacheModifier, EvictionPolicy evictionPolicy,
              bool isVolatile) -> Value {
             return self.create<LoadOp>(ptr, boundaryCheck, paddingOption,
                                        cacheModifier, evictionPolicy,
                                        isVolatile);
           })
      .def("create_tensor_pointer_store",
           [](TritonOpBuilder &self, Value &ptr, Value &val,
              std::vector<int32_t> &boundaryCheck, CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> void {
             self.create<StoreOp>(ptr, val, boundaryCheck, cacheModifier,
                                  evictionPolicy);
           })
      .def("create_masked_load",
           [](TritonOpBuilder &self, Value &ptrs, Value &mask,
              std::optional<Value> &other, CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy, bool isVolatile) -> Value {
             return self.create<LoadOp>(ptrs, mask, other.value_or(Value()),
                                        cacheModifier, evictionPolicy,
                                        isVolatile);
           })
      .def("create_masked_store",
           [](TritonOpBuilder &self, Value &ptrs, Value &val, Value &mask,
              CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> void {
             self.create<StoreOp>(ptrs, val, mask, cacheModifier,
                                  evictionPolicy);
           })
      .def("create_tensor_descriptor_type",
           [](TritonOpBuilder &self, Type blockTy) -> Type {
             auto ctx = self.getContext();
             return triton::TensorDescType::get(
                 ctx, cast<RankedTensorType>(blockTy));
           })
      .def("create_reinterpret_tensor_descriptor",
           [](TritonOpBuilder &self, Value desc_ptr, Type blockTy) -> Value {
             auto ctx = self.getContext();
             auto resultTy = triton::TensorDescType::get(
                 ctx, cast<RankedTensorType>(blockTy));
             return self.create<ReinterpretTensorDescOp>(resultTy, desc_ptr);
           })
      .def("create_descriptor_load",
           [](TritonOpBuilder &self, Value desc, std::vector<Value> &indices,
              CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> Value {
             auto descTy = cast<triton::TensorDescType>(desc.getType());
             auto resTy = descTy.getBlockType();
             return self.create<DescriptorLoadOp>(
                 resTy, desc, indices, cacheModifier, evictionPolicy);
           })
      .def("create_descriptor_gather",
           [](TritonOpBuilder &self, Value desc, Value x_indices, Value y_index,
              Type type) -> Value {
             return self.create<DescriptorGatherOp>(type, desc, x_indices,
                                                    y_index);
           })
      .def("create_descriptor_store",
           [](TritonOpBuilder &self, Value desc, Value value,
              std::vector<Value> &indices) -> void {
             self.create<DescriptorStoreOp>(desc, value, indices);
           })
      .def("create_descriptor_scatter",
           [](TritonOpBuilder &self, Value desc, Value value, Value x_indices,
              Value y_index) -> void {
             self.create<DescriptorScatterOp>(desc, x_indices, y_index, value);
           })
      .def("create_tensormap_create",
           [](TritonOpBuilder &self, Value desc_ptr, Value global_address,
              std::vector<Value> box_dim, std::vector<Value> global_dim,
              std::vector<Value> global_stride,
              std::vector<Value> element_stride, int32_t elem_type,
              int32_t interleave_layout, int32_t swizzle_mode,
              int32_t fill_mode) {
             self.create<ExperimentalTensormapCreateOp>(
                 desc_ptr, global_address, box_dim, global_dim, global_stride,
                 element_stride, elem_type, interleave_layout, swizzle_mode,
                 fill_mode);
           })
      .def("create_tensormap_fenceproxy_acquire",
           [](TritonOpBuilder &self, Value desc_ptr) {
             self.create<ExperimentalTensormapFenceproxyAcquireOp>(desc_ptr);
           })
      .def("create_reshape",
           [](TritonOpBuilder &self, Value &arg, std::vector<int64_t> &shape,
              bool allowReorder) -> Value {
             auto argType =
                 cast<RankedTensorType>(arg.getType()).getElementType();
             return self.create<ReshapeOp>(
                 RankedTensorType::get(shape, argType), arg, allowReorder);
           })
      .def("create_expand_dims",
           [](TritonOpBuilder &self, Value &arg, int axis) -> Value {
             auto argType = dyn_cast<RankedTensorType>(arg.getType());
             auto argEltType = argType.getElementType();
             std::vector<int64_t> retShape = argType.getShape();
             retShape.insert(retShape.begin() + axis, 1);
             return self.create<ExpandDimsOp>(
                 RankedTensorType::get(retShape, argEltType), arg, axis);
           })
      .def("create_cat",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
             auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
             if (!(lhsType.getShape().size() == 1 &&
                   rhsType.getShape().size() == 1))
               throw std::invalid_argument(
                   "shape not supported by cat. Expecting rank-1 inputs");
             std::vector<int64_t> shape{lhsType.getShape()[0] +
                                        rhsType.getShape()[0]};
             return self.create<CatOp>(
                 RankedTensorType::get(shape, lhsType.getElementType()), lhs,
                 rhs);
           })
      .def("create_join",
           [](TritonOpBuilder &self, Value &a, Value &b) -> Value {
             return self.create<JoinOp>(a, b);
           })
      .def("create_split",
           [](TritonOpBuilder &self, Value &a) -> std::vector<Value> {
             auto op = self.create<SplitOp>(a);
             return std::vector<Value>(op->result_begin(), op->result_end());
           })
      // Implements tl.trans and tl.permute.
      .def("create_trans",
           [](TritonOpBuilder &self, Value &arg,
              std::vector<int> &order) -> Value {
             auto argType = dyn_cast<RankedTensorType>(arg.getType());
             auto argEltType = argType.getElementType();
             auto retShape = applyPermutation(argType.getShape(), order);
             return self.create<TransOp>(
                 RankedTensorType::get(retShape, argEltType), arg, order);
           })
      .def("create_broadcast",
           [](TritonOpBuilder &self, Value &arg,
              std::vector<int64_t> &shape) -> Value {
             if (auto argType = dyn_cast<RankedTensorType>(arg.getType()))
               return self.createOrFold<BroadcastOp>(
                   RankedTensorType::get(shape, argType.getElementType()), arg);
             throw std::invalid_argument(
                 "arg is not of RankedTensorType, use create_splat");
           })
      .def("create_splat",
           [](TritonOpBuilder &self, Value &arg,
              std::vector<int64_t> &shape) -> Value {
             auto argType = arg.getType();
             auto ret = self.createOrFold<SplatOp>(
                 RankedTensorType::get(shape, argType), arg);
             return ret;
           })
      // // atomic
      .def("create_atomic_cas",
           [](TritonOpBuilder &self, Value &ptr, Value &cmp, Value &val,
              MemSemantic sem, MemSyncScope scope) -> Value {
             Type dstType;
             if (auto srcTensorType =
                     dyn_cast<RankedTensorType>(ptr.getType())) {
               Type dstElemType =
                   cast<PointerType>(srcTensorType.getElementType())
                       .getPointeeType();
               dstType =
                   RankedTensorType::get(srcTensorType.getShape(), dstElemType);
             } else {
               auto ptrType = cast<PointerType>(getElementTypeOrSelf(ptr));
               dstType = ptrType.getPointeeType();
             }
             return self.create<AtomicCASOp>(dstType, ptr, cmp, val, sem,
                                             scope);
           })
      .def("create_atomic_rmw",
           [](TritonOpBuilder &self, RMWOp rmwOp, Value &ptr, Value &val,
              Value &mask, MemSemantic sem, MemSyncScope scope) -> Value {
             Type dstType;
             if (auto srcTensorType =
                     dyn_cast<RankedTensorType>(ptr.getType())) {
               Type dstElemType =
                   cast<PointerType>(srcTensorType.getElementType())
                       .getPointeeType();
               dstType =
                   RankedTensorType::get(srcTensorType.getShape(), dstElemType);
             } else {
               auto ptrType = cast<PointerType>(getElementTypeOrSelf(ptr));
               dstType = ptrType.getPointeeType();
             }
             return self.create<AtomicRMWOp>(dstType, rmwOp, ptr, val, mask,
                                             sem, scope);
           })
      // External
      .def("create_extern_elementwise",
           [](TritonOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<Value> &argList, Type retType, bool isPure) -> Value {
             return self.create<ExternElementwiseOp>(retType, argList, libName,
                                                     libPath, symbol, isPure);
           })
      // Built-in instruction
      .def("create_get_program_id",
           [](TritonOpBuilder &self, int axis) -> Value {
             if (axis < 0 || axis > 3)
               throw pybind11::index_error("program_id must be in [0,3]");
             return self.create<GetProgramIdOp>(axis);
           })
      .def("create_get_num_programs",
           [](TritonOpBuilder &self, int axis) -> Value {
             if (axis < 0 || axis > 3)
               throw pybind11::index_error("program_id must be in [0,3]");
             return self.create<GetNumProgramsOp>(axis);
           })
      .def("create_dot",
           [](TritonOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, InputPrecision inputPrecision,
              int maxNumImpreciseAcc) -> mlir::Value {
             return self.create<DotOp>(c.getType(), a, b, c, inputPrecision,
                                       maxNumImpreciseAcc);
           })
      .def("create_dot_scaled",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              std::optional<mlir::Value> &lhs_scale,
              ScaleDotElemType lhs_format, mlir::Value &rhs,
              std::optional<mlir::Value> &rhs_scale,
              ScaleDotElemType rhs_format, bool fast_math,
              mlir::Value &c) -> mlir::Value {
             return self.create<DotScaledOp>(c.getType(), lhs, rhs, c,
                                             lhs_scale.value_or(Value()),
                                             rhs_scale.value_or(Value()),
                                             lhs_format, rhs_format, fast_math);
           })
      .def("create_floor",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::FloorOp>(val);
           })
      .def("create_ceil",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::CeilOp>(val);
           })
      .def("create_exp",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::ExpOp>(val);
           })
      .def("create_exp2",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::Exp2Op>(val);
           })
      .def("create_cos",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::CosOp>(val);
           })
      .def("create_sin",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::SinOp>(val);
           })
      .def("create_log",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::LogOp>(val);
           })
      .def("create_log2",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::Log2Op>(val);
           })
      .def("create_erf",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::ErfOp>(val);
           })
      .def("create_sqrt",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::SqrtOp>(val);
           })
      .def("create_rsqrt",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::RsqrtOp>(val);
           })
      .def("create_fabs",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::AbsFOp>(val);
           })
      .def("create_iabs",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::AbsIOp>(val);
           })
      .def("create_reduce",
           [](TritonOpBuilder &self, std::vector<Value> operands, int axis)
               -> OpState { return self.create<ReduceOp>(operands, axis); })
      .def("create_reduce_ret",
           [](TritonOpBuilder &self, py::args args) -> OpState {
             llvm::SmallVector<Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<Value>(arg));
             }
             return self.create<ReduceReturnOp>(return_values);
           })
      .def("create_scan",
           [](TritonOpBuilder &self, std::vector<Value> operands, int axis,
              bool reverse) -> OpState {
             return self.create<ScanOp>(operands, axis, reverse);
           })
      .def("create_scan_ret",
           [](TritonOpBuilder &self, py::args args) -> OpState {
             llvm::SmallVector<Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<Value>(arg));
             }
             return self.create<ScanReturnOp>(return_values);
           })
      .def("create_ptr_to_int",
           [](TritonOpBuilder &self, Value &val, Type &type) -> Value {
             return self.create<PtrToIntOp>(type, val);
           })
      .def("create_int_to_ptr",
           [](TritonOpBuilder &self, Value &val, Type &type) -> Value {
             return self.create<IntToPtrOp>(type, val);
           })
      .def("create_select",
           [](TritonOpBuilder &self, Value &condition, Value &trueValue,
              Value &falseValue) -> Value {
             return self.create<arith::SelectOp>(condition, trueValue,
                                                 falseValue);
           })
      .def("create_inline_asm",
           [](TritonOpBuilder &self, const std::string &inlineAsm,
              const std::string &constraints, const std::vector<Value> &values,
              const std::vector<Type> &types, bool isPure,
              int pack) -> OpState {
             return self.create<ElementwiseInlineAsmOp>(
                 types, inlineAsm, constraints, isPure, pack, values);
           })
      .def("create_print",
           [](TritonOpBuilder &self, const std::string &prefix, bool hex,
              const std::vector<Value> &values,
              const std::vector<int32_t> &isSigned) -> void {
             auto prefixAttr = StringAttr::get(self.getBuilder().getContext(),
                                               llvm::StringRef(prefix));
             self.create<PrintOp>(prefixAttr, hex, values, isSigned);
           })
      .def("create_assert",
           [](TritonOpBuilder &self, Value &condition,
              const std::string &message) -> void {
             auto messageAttr = StringAttr::get(self.getBuilder().getContext(),
                                                llvm::StringRef(message));
             self.create<AssertOp>(condition, messageAttr);
           })
      .def("create_assume",
           [](TritonOpBuilder &self, Value &condition) {
             self.create<LLVM::AssumeOp>(condition);
           })
      .def("create_poison",
           [](TritonOpBuilder &self, Type &type) -> Value {
             return self.create<ub::PoisonOp>(type);
           })
      .def("create_histogram",
           [](TritonOpBuilder &self, Value operand, int numBins) -> Value {
             return self.create<HistogramOp>(
                 RankedTensorType::get(
                     {static_cast<int64_t>(numBins)},
                     IntegerType::get(operand.getContext(), 32)),
                 operand);
           })
      .def("create_gather",
           [](TritonOpBuilder &self, Value src, Value indices, int axis)
               -> Value { return self.create<GatherOp>(src, indices, axis); })
      // Force GPU barrier
      .def("create_barrier",
           [](TritonOpBuilder &self) { self.create<mlir::gpu::BarrierOp>(); })
      // Make a block pointer (tensor pointer in Triton IR)
      .def("create_make_block_ptr",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &shape,
              std::vector<Value> &strides, std::vector<Value> &offsets,
              std::vector<int32_t> &tensorShape,
              std::vector<int32_t> &order) -> Value {
             return self.create<MakeTensorPtrOp>(base, shape, strides, offsets,
                                                 tensorShape, order);
           })
      // Advance a block pointer
      .def("create_advance",
           [](TritonOpBuilder &self, Value &ptr,
              std::vector<Value> &offsets) -> Value {
             return self.create<AdvanceOp>(ptr.getType(), ptr, offsets);
           })
      // Make a tensor descriptor
      .def("create_make_tensor_descriptor",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &shape,
              std::vector<Value> &strides,
              std::vector<int32_t> &tensorShape) -> Value {
             return self.create<MakeTensorDescOp>(base, shape, strides,
                                                  tensorShape);
           })
      // Proton Ops
      .def("create_proton_record",
           [](TritonOpBuilder &self, bool isStart, int32_t regionId) -> void {
             self.create<mlir::triton::proton::RecordOp>(isStart, regionId);
           })
      // SIMT ops
      .def("create_get_thread_id",
           [](TritonOpBuilder &self) -> Value {
             // triton only use 1D thread block
             Value tid = self.create<::mlir::gpu::ThreadIdOp>(
                 ::mlir::gpu::Dimension::x);
             Type ty_i32 = self.getBuilder().getIntegerType(32);
             tid = self.create<arith::IndexCastOp>(ty_i32, tid);
             return tid;
           })
      .def("create_get_block_size",
           [](TritonOpBuilder &self) -> Value {
             Value bs = self.create<::mlir::gpu::BlockDimOp>(
                 ::mlir::gpu::Dimension::x);
             Type ty_i32 = self.getBuilder().getIntegerType(32);
             bs = self.create<arith::IndexCastOp>(ty_i32, bs);
             return bs;
           })
      .def("create_simt_exec_region_op",
           [](TritonOpBuilder &self,
              std::vector<Value> &init_args) -> triton::simt::SIMTExecRegionOp {
             return self.create<mlir::triton::simt::SIMTExecRegionOp>(
                 init_args);
           })
      .def("create_block_yield_op",
           [](TritonOpBuilder &self,
              std::vector<Value> &yields) -> triton::simt::BlockYieldOp {
             return self.create<triton::simt::BlockYieldOp>(yields);
           })
      .def("create_extract",
           [](TritonOpBuilder &self, Value src,
              std::vector<Value> &indices) -> Value {
             std::vector<Value> to_index;
             for (size_t i = 0; i < indices.size(); ++i) {
               Value val = indices[i];
               if (!isa<IndexType>(val.getType())) {
                 to_index.push_back(self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), val));
               } else {
                 to_index.push_back(val);
               }
             }
             Value ret = self.create<tensor::ExtractOp>(src, to_index);
             return ret;
           })
      .def("create_insert",
           [](TritonOpBuilder &self, Value scalar, Value dest,
              std::vector<Value> &indices) -> Value {
             std::vector<Value> to_index;
             for (size_t i = 0; i < indices.size(); ++i) {
               Value val = indices[i];
               if (!isa<IndexType>(val.getType())) {
                 to_index.push_back(self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), val));
               } else {
                 to_index.push_back(val);
               }
             }
             return self.create<tensor::InsertOp>(scalar, dest, to_index);
           })

      // Distributed Ops
      .def("create_distributed_wait",
           [](TritonOpBuilder &self, Value &barrierPtrs, Value &numBarriers,
              Value &waitValue, MemSyncScope scope, MemSemantic semantic,
              Type &type) -> Value {
             return self.create<mlir::triton::distributed::WaitOp>(
                 type, barrierPtrs, numBarriers, waitValue, scope, semantic);
           })
      .def("create_distributed_consume_token",
           [](TritonOpBuilder &self, Value &input, Value &token) -> Value {
             return self.create<mlir::triton::distributed::ConsumeTokenOp>(
                 input, token);
           })
      .def("create_get_rank",
           [](TritonOpBuilder &self, Value axis) -> Value {
             return self.create<mlir::triton::distributed::GetRankOp>(axis);
           })
      .def("create_get_num_ranks",
           [](TritonOpBuilder &self, Value axis) -> Value {
             return self.create<mlir::triton::distributed::GetNumRanksOp>(axis);
           })
      .def("create_symm_at",
           [](TritonOpBuilder &self, Value ptr, Value rank) -> Value {
             return self.create<mlir::triton::distributed::SymmAtOp>(
                 ptr.getType(), ptr, rank);
           })
      .def("create_notify",
           [](TritonOpBuilder &self, Value ptr, Value signal, Value rank,
              distributed::SignalOp sigOp,
              distributed::CommScope commScope) -> void {
             self.create<mlir::triton::distributed::NotifyOp>(ptr, signal, rank,
                                                              sigOp, commScope);
           })
      .def("create_extern_call",
           [](TritonOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<Value> &argList, const std::vector<Type> &retTypes,
              bool isPure) -> OpState {
             return self.create<mlir::triton::distributed::ExternCallOp>(
                 retTypes, argList, libName, libPath, symbol, isPure);
           });
}

void init_triton_distributed_env_vars(py::module &m) {
  m.def("get_cache_invalidating_env_vars",
        []() -> std::map<std::string, std::string> {
          std::map<std::string, std::string> ret;
          for (const auto &envVar : CACHE_INVALIDATING_ENV_VARS) {
            auto strVal = triton::tools::getStrEnv(envVar);
            if (strVal.empty())
              continue;
            auto boolV = triton::tools::isEnvValueBool(strVal);
            if (boolV.has_value())
              ret[envVar] = boolV.value() ? "true" : "false";
            else
              ret[envVar] = strVal;
          }
          return ret;
        });
}
