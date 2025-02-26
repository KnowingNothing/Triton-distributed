//===- DistributedOpToLLVM.cpp ------------------------------------- C++---===//
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
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "third_party/distributed/dialect/include/Dialect/Distributed/IR/Dialect.h"

#include "Utility.h"
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {

Operation *CreateNVSHMEMOp(RewriterBase &rewriter, Operation *curOp,
                           const StringRef &symbol, StringRef libname,
                           StringRef libpath, ValueRange inputOperands,
                           Type retType) {
  auto loc = curOp->getLoc();
  Type funcType = mlir::triton::gpu::getFunctionType(retType, inputOperands);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, curOp, symbol, funcType, libname, libpath);
  auto op = LLVM::createLLVMCallOp(rewriter, loc, funcOp, inputOperands);
  return op;
}

template <typename DistOp>
class GenericOpToNVSHMEMDevice : public ConvertOpToLLVMPattern<DistOp> {
public:
  using OpAdaptor = typename DistOp::Adaptor;

  GenericOpToNVSHMEMDevice(const LLVMTypeConverter &converter,
                           const PatternBenefit &benefit, StringRef calleeName,
                           StringRef libname = "", StringRef libpath = "")
      : ConvertOpToLLVMPattern<DistOp>(converter, benefit),
        calleeName(calleeName), libname(libname), libpath(libpath) {}

  LogicalResult
  matchAndRewrite(DistOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (op->getNumResults() > 1)
      return failure();
    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
    auto newOperands = adaptor.getOperands();
    Type retType =
        op->getNumResults() == 0
            ? voidTy
            : this->getTypeConverter()->convertType(op->getResult(0).getType());
    auto nvshmemOp = CreateNVSHMEMOp(rewriter, op, calleeName, libname, libpath,
                                     newOperands, retType);
    auto newResult = nvshmemOp->getResult(0);
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, newResult);
    }

    return success();
  }

private:
  StringRef calleeName;
  StringRef libname;
  StringRef libpath;
};

template <typename... Args>
void registerGenericOpToNVSHMEMDevice(RewritePatternSet &patterns,
                                      LLVMTypeConverter &typeConverter,
                                      PatternBenefit benefit,
                                      StringRef calleeName, StringRef libname,
                                      StringRef libpath) {
  patterns.add<GenericOpToNVSHMEMDevice<Args>...>(typeConverter, benefit,
                                                  calleeName, libname, libpath);
}

struct WaitOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::WaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto type = op->getOperand(0).getType();
    assert(isa<triton::PointerType>(type) && "must be a pointer type");
    auto ptree_type = dyn_cast<triton::PointerType>(type).getPointeeType();
    auto intType = dyn_cast<mlir::IntegerType>(ptree_type);
    assert(intType && "barrier ptr must be integer type.");
    const size_t barrier_width = intType.getWidth();
    std::string scope = "";
    if (op.getScope() == triton::MemSyncScope::CTA) {
      scope = "cta";
    } else if (op.getScope() == triton::MemSyncScope::GPU) {
      scope = "gpu";
    } else if (op.getScope() == triton::MemSyncScope::SYSTEM) {
      scope = "sys";
    }
    std::string semantic = "";
    if (op.getSemantic() == triton::MemSemantic::ACQUIRE) {
      semantic = "acquire";
    } else if (op.getSemantic() == triton::MemSemantic::RELAXED) {
      semantic = "relaxed";
    } else if (op.getSemantic() == triton::MemSemantic::RELEASE) {
      semantic = "release";
    } else if (op.getSemantic() == triton::MemSemantic::ACQUIRE_RELEASE) {
      semantic = "acq_rel";
    }
    const std::string ld_ptx = "ld.global."s + semantic + "."s + scope + ".b"s +
                               std::to_string(barrier_width);
    const std::string bit_w = std::to_string(barrier_width);
    const std::string byte_w = std::to_string(barrier_width / 8);
    // TODO(zhengsize): how about more barriers?
    // we only consider warp sync now
    // so numBarriers should be <= WARP_SIZE
    // otherwise, the behavior is undefined
    const std::string ptx =
        "{                                                              \n\t"s +
        ".reg .pred %p<2>;                                              \n\t"s +
        ".reg .b32 %th<2>;                                              \n\t"s +
        ".reg .u64 %addr<2>;                                            \n\t"s +
        ".reg .b"s + bit_w + " %tmp<1>;                                 \n\t"s +
        "mov.u32 %th1, $1;                                              \n\t"s +
        "mov.u32 %th0, %tid.x;                                          \n\t"s +
        "rem.u32 %th0, %th0, 32;                                        \n\t"s +
        "mul.wide.s32 %addr1, %th0, "s + byte_w + ";                    \n\t"s +
        "add.u64 %addr0, $0, %addr1;                                    \n\t"s +
        "setp.lt.u32 %p0, %th0, %th1;                                   \n\t"s +
        "@!%p0 bra.uni skipLoop;                                        \n\t"s +
        "waitLoop:                                                      \n\t"s +
        "  "s + ld_ptx + " %tmp0, [%addr0];                             \n\t"s +
        "  setp.eq.b"s + bit_w + " %p0, %tmp0, 1;                       \n\t"s +
        "  @!%p0 bra.uni waitLoop;                                      \n\t"s +
        "skipLoop:                                                      \n\t"s +
        "bar.warp.sync 0xffffffff;                                      \n\t"s +
        "}                                                              \n\t"s;

    auto &waitOp = *ptxBuilder.create<>(ptx);
    waitOp({ptxBuilder.newOperand(adaptor.getBarrierPtr(), "l"),
            ptxBuilder.newOperand(adaptor.getNumBarriers(), "r")},
           /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConsumeTokenOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::ConsumeTokenOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::ConsumeTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

class NotifyOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::NotifyOp> {
public:
  NotifyOpConversion(const LLVMTypeConverter &converter,
                     const PatternBenefit &benefit, StringRef libname = "",
                     StringRef libpath = "")
      : ConvertOpToLLVMPattern<triton::distributed::NotifyOp>(converter,
                                                              benefit),
        libname(libname), libpath(libpath) {}

  LogicalResult
  matchAndRewrite(triton::distributed::NotifyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    bool isIntraNode =
        op.getCommScope() != ::mlir::triton::distributed::CommScope::INTER_NODE;
    auto signalType = op.getSigAddr().getType();
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto b = ::mlir::triton::TritonLLVMOpBuilder(loc, rewriter);
    Value threadId = rewriter.create<NVVM::ThreadIdXOp>(loc, i32_ty);
    Value pred = b.icmp_eq(threadId, b.i32_val(0));
    Block *prevBlock = op->getBlock();

    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, pred, ifBlock, thenBlock);
    rewriter.setInsertionPointToStart(ifBlock);

    rewriter.setInsertionPointToStart(ifBlock);
    if (isIntraNode) {
      // remote ptr
      bool isIntraRank =
          op.getCommScope() == ::mlir::triton::distributed::CommScope::GPU;
      Type retType = this->getTypeConverter()->convertType(signalType);
      Value remotePtr;
      if (isIntraRank) {
        remotePtr = adaptor.getSigAddr();
      } else {
        remotePtr =
            CreateNVSHMEMOp(rewriter, op, "nvshmem_ptr", libname, libpath,
                            {adaptor.getSigAddr(), adaptor.getRank()}, retType)
                ->getResult(0);
      }
      ::mlir::triton::PointerType sigalElemType =
          llvm::cast<::mlir::triton::PointerType>(signalType);
      const size_t signalWidth =
          sigalElemType.getPointeeType().getIntOrFloatBitWidth();
      std::string semantic = "relaxed";
      std::string stScope = isIntraRank ? "gpu" : "sys";
      std::string membarScope = isIntraRank ? "gl" : "sys";
      const std::string memBarPtx = "membar." + membarScope + ";\n\t";

      if (adaptor.getSigOp() == ::mlir::triton::distributed::SignalOp::SET) {
        std::string opType = "st";
        const std::string stSignalPtx =
            opType + "." + semantic + "." + stScope + ".global.b" +
            std::to_string(signalWidth) + " [$0], $1" + ";\n\t";
        const std::string ptx = memBarPtx + stSignalPtx;
        auto &notifyPtxOp = *ptxBuilder.create<>(ptx);
        notifyPtxOp({ptxBuilder.newOperand(remotePtr, "l"),
                     ptxBuilder.newOperand(adaptor.getSignalVal(), "l")}, // u64
                    /*onlyAttachMLIRArgs=*/true);
        auto voidTy = void_ty(op->getContext());
        ptxBuilder.launch(rewriter, loc, voidTy);
      } else {
        std::string opType = "add";
        // Operation .add requires .u32 or .s32 or .u64 or .f64 or f16 or f16x2
        // or .f32 or .bf16 or .bf16x2 type for instruction 'atom'
        const std::string stSignalPtx =
            "atom." + semantic + "." + stScope + ".global." + opType + ".u" +
            std::to_string(signalWidth) + " $0, [$1], $2" + ";\n\t";
        const std::string ptx = memBarPtx + stSignalPtx;
        auto &notifyPtxOp = *ptxBuilder.create<>(ptx);
        notifyPtxOp({ptxBuilder.newOperand("=l"),
                     ptxBuilder.newOperand(remotePtr, "l"),
                     ptxBuilder.newOperand(adaptor.getSignalVal(), "l")}, // u64
                    /*onlyAttachMLIRArgs=*/true);
        ptxBuilder.launch(rewriter, loc, retType);
      }
    } else {
      LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
      // NVSHMEM_SIGNAL_SET = 9
      // NVSHMEM_SIGNAL_ADD = 10
      int32_t v = -1;
      if (adaptor.getSigOp() == ::mlir::triton::distributed::SignalOp::SET) {
        v = 9;
      } else if (adaptor.getSigOp() ==
                 ::mlir::triton::distributed::SignalOp::ADD) {
        assert(0 && "unsupport sigOp.\n");
      }
      Value sigOp = mlir::LLVM::createConstantI32(loc, rewriter, v);
      auto nvshmemxSignalOp =
          CreateNVSHMEMOp(rewriter, op, "nvshmemx_signal_op", libname, libpath,
                          {adaptor.getSigAddr(), adaptor.getSignalVal(), sigOp,
                           adaptor.getRank()},
                          voidTy);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  StringRef libname;
  StringRef libpath;
};

} // namespace

void mlir::triton::NVIDIA::populateDistributedOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo,
    std::string NVSHMEMLibname, std::string NVSHMEMLibpath) {
  patterns.add<WaitOpConversion, ConsumeTokenOpConversion>(typeConverter,
                                                           benefit);

  // convert to nvshmem device func call
  registerGenericOpToNVSHMEMDevice<triton::distributed::GetRankOp>(
      patterns, typeConverter, benefit, "nvshmem_my_pe", NVSHMEMLibname,
      NVSHMEMLibpath);
  registerGenericOpToNVSHMEMDevice<triton::distributed::GetNumRanksOp>(
      patterns, typeConverter, benefit, "nvshmem_n_pes", NVSHMEMLibname,
      NVSHMEMLibpath);
  registerGenericOpToNVSHMEMDevice<triton::distributed::SymmAtOp>(
      patterns, typeConverter, benefit, "nvshmem_ptr", NVSHMEMLibname,
      NVSHMEMLibpath);

  patterns.add<NotifyOpConversion>(typeConverter, benefit);
}
