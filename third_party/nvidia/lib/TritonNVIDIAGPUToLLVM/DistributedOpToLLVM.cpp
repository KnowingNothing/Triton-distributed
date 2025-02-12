//===- DistributedOpToLLVM.cpp ------------------------------------- C++ ---===//
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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "third_party/distributed/dialect/include/Dialect/Distributed/IR/Dialect.h"

#include "Utility.h"
#include <string>


using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {


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
    const std::string ld_ptx = "ld.global."s + semantic + "."s + scope + ".b"s + std::to_string(barrier_width);
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

} // namespace

void mlir::triton::NVIDIA::populateDistributedOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<WaitOpConversion, ConsumeTokenOpConversion>(typeConverter, benefit);
}
