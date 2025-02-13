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

// TODO(zhengxuegui.0): automatically generate ptx
struct GetRankOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::GetRankOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

public:
  GetRankOpConversion(const LLVMTypeConverter &converter,
                      const PatternBenefit &benefit,
                      const NVIDIA::TargetInfo &targetInfo,
                      const bool enableInline = true)
      : ConvertOpToLLVMPattern<triton::distributed::GetRankOp>(converter,
                                                               benefit),
        targetInfo(targetInfo), enableInline(enableInline) {}

  LogicalResult
  matchAndRewrite(triton::distributed::GetRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ::mlir::triton::PTXBuilder ptxBuilder;

    const std::string funcBodyPtx =
        "{                                                              \n\t"s +
        " .reg.b32 %nvshmem_my_pe_wrapper_out;                          \n\t"s +
        " {                                                             \n\t"s +
        "   .reg .b32 %r<2>;                                            \n\t"s +
        "   ld.const.u32 %r1, [nvshmemi_device_state_d+4];              \n\t"s +
        "   mov.b32 %nvshmem_my_pe_wrapper_out, %r1;                    \n\t"s +
        " }                                                             \n\t"s +
        " mov.b32 $0, %nvshmem_my_pe_wrapper_out;                       \n\t"s +
        "}                                                              \n\t"s;

    const std::string funcCallPtx = "{                                \n\t"s +
                                    " .reg .b32 temp_param_reg;       \n\t"s +
                                    " .param .b32 retval0;            \n\t"s +
                                    " call.uni (retval0),             \n\t"s +
                                    " nvshmem_my_pe_wrapper,          \n\t"s +
                                    " (                               \n\t"s +
                                    " );                              \n\t"s +
                                    " ld.param.b32 $0, [retval0+0];   \n\t"s +
                                    "}                                \n\t"s;

    const std::string ptx = enableInline ? funcBodyPtx : funcCallPtx;
    auto &nvshmemMype = *ptxBuilder.create<>(ptx);
    nvshmemMype({ptxBuilder.newOperand("=r")},
                /*onlyAttachMLIRArgs=*/true);
    auto ptxResult =
        ptxBuilder.launch(rewriter, loc, op->getResult(0).getType());
    rewriter.replaceOp(op, ptxResult);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
  const bool enableInline;
};

struct GetNumRanksOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::GetNumRanksOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

public:
  GetNumRanksOpConversion(const LLVMTypeConverter &converter,
                          const PatternBenefit &benefit,
                          const NVIDIA::TargetInfo &targetInfo,
                          const bool enableInline = true)
      : ConvertOpToLLVMPattern<triton::distributed::GetNumRanksOp>(converter,
                                                                   benefit),
        targetInfo(targetInfo), enableInline(enableInline) {}

  LogicalResult
  matchAndRewrite(triton::distributed::GetNumRanksOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ::mlir::triton::PTXBuilder ptxBuilder;

    const std::string funcBodyPtx =
        "{                                                    \n\t"s +
        " .reg.b32 %nvshmem_n_pes_wrapper_out;                \n\t"s +
        " {                                                   \n\t"s +
        "   .reg .b32 %r<2>;                                  \n\t"s +
        "   ld.const.u32 %r1, [nvshmemi_device_state_d+8];    \n\t"s +
        "   mov.b32 %nvshmem_n_pes_wrapper_out, %r1;          \n\t"s +
        " }                                                   \n\t"s +
        " mov.b32 $0, %nvshmem_n_pes_wrapper_out;             \n\t"s +
        "}                                                    \n\t"s;

    const std::string funcCallPtx =
        "{                                                    \n\t"s +
        " .reg .b32 temp_param_reg;                           \n\t"s +
        " .param .b32 retval0;                                \n\t"s +
        " call.uni (retval0),                                 \n\t"s +
        " nvshmem_n_pes_wrapper,                              \n\t"s +
        " (                                                   \n\t"s +
        " );                                                  \n\t"s +
        " ld.param.b32 	$0, [retval0+0];                      \n\t"s +
        "}                                                    \n\t"s;
    const std::string ptx = enableInline ? funcBodyPtx : funcCallPtx;
    auto &nvshmemNPes = *ptxBuilder.create<>(ptx);
    nvshmemNPes({ptxBuilder.newOperand("=r")},
                /*onlyAttachMLIRArgs=*/true);
    auto ptxResult =
        ptxBuilder.launch(rewriter, loc, op->getResult(0).getType());
    rewriter.replaceOp(op, ptxResult);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
  const bool enableInline;
};

struct SymmAtOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::SymmAtOp> {
public:
  SymmAtOpConversion(const LLVMTypeConverter &converter,
                     const PatternBenefit &benefit,
                     const NVIDIA::TargetInfo &targetInfo,
                     const bool enableInline = true)
      : ConvertOpToLLVMPattern<triton::distributed::SymmAtOp>(converter,
                                                              benefit),
        targetInfo(targetInfo), enableInline(enableInline) {}

  LogicalResult
  matchAndRewrite(triton::distributed::SymmAtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ::mlir::triton::PTXBuilder ptxBuilder;

    const std::string funcBodyPtx =
        "{                                                        \n\t"s +
        " .reg.b64 %nvshmem_ptr_wrapper_in_0;                     \n\t"s +
        " .reg.b32 %nvshmem_ptr_wrapper_in_1;                     \n\t"s +
        " .reg.b64 %nvshmem_ptr_wrapper_out_0;                    \n\t"s +
        " mov.b64 %nvshmem_ptr_wrapper_in_0, $1;                  \n\t"s +
        " mov.b32 %nvshmem_ptr_wrapper_in_1, $2;                  \n\t"s +
        " {                                                       \n\t"s +
        "   .reg .pred  %p<5>;                                    \n\t"s +
        "   .reg .b32 %r<2>;                                      \n\t"s +
        "   .reg .b64 %rd<14>;                                    \n\t"s +
        "   mov.b64 %rd5, %nvshmem_ptr_wrapper_in_0;              \n\t"s +
        "   mov.b32 %r1, %nvshmem_ptr_wrapper_in_1;               \n\t"s +
        "   mov.u64 %rd13, 0;                                     \n\t"s +
        "   ld.const.u64  %rd6, [nvshmemi_device_state_d+40];     \n\t"s +
        "   sub.s64 %rd1, %rd5, %rd6;                             \n\t"s +
        "   setp.gt.u64 %p1, %rd6, %rd5;                          \n\t"s +
        "   ld.const.u64  %rd7, [nvshmemi_device_state_d+48];     \n\t"s +
        "   setp.ge.u64 %p2, %rd1, %rd7;                          \n\t"s +
        "   or.pred %p3, %p1, %p2;                                \n\t"s +
        "   @%p3 bra  L__BB6_2;                                   \n\t"s +
        "   ld.const.u64 %rd10, [nvshmemi_device_state_d+56];     \n\t"s +
        "   mul.wide.s32 %rd11, %r1, 8;                           \n\t"s +
        "   add.s64 %rd9, %rd10, %rd11;                           \n\t"s +
        "   // begin inline asm                                   \n\t"s +
        "   ld.global.nc.u64 %rd8, [%rd9];                        \n\t"s +
        "   // end inline asm                                     \n\t"s +
        "   setp.eq.s64 %p4, %rd8, 0;                             \n\t"s +
        "   add.s64 %rd12, %rd8, %rd1;                            \n\t"s +
        "   selp.b64 %rd13, %rd8, %rd12, %p4;                     \n\t"s +
        " L__BB6_2:                                               \n\t"s +
        "   mov.b64 %nvshmem_ptr_wrapper_out_0, %rd13;            \n\t"s +
        " }                                                       \n\t"s +
        " mov.b64 $0, %nvshmem_ptr_wrapper_out_0;                 \n\t"s +
        "}                                                        \n\t";

    const std::string funcCallPtx = "{                                  \n\t"s +
                                    " .reg .b32 temp_param_reg;         \n\t"s +
                                    " .param .b64 param0;               \n\t"s +
                                    " st.param.b64 	[param0+0], $1;     \n\t"s +
                                    " .param .b32 param1;               \n\t"s +
                                    " st.param.b32 	[param1+0], $2;     \n\t"s +
                                    " .param .b64 retval0;              \n\t"s +
                                    " call.uni (retval0),               \n\t"s +
                                    " nvshmem_ptr_wrapper,              \n\t"s +
                                    " (                                 \n\t"s +
                                    "   param0,                         \n\t"s +
                                    "   param1                          \n\t"s +
                                    " );                                \n\t"s +
                                    " ld.param.b64 	$0, [retval0+0];    \n\t"s +
                                    "}                                  \n\t";

    const std::string ptx = enableInline ? funcBodyPtx : funcCallPtx;
    auto &nvshmemPtr = *ptxBuilder.create<>(ptx);
    nvshmemPtr({ptxBuilder.newOperand("=l"),
                ptxBuilder.newOperand(adaptor.getSymmAddr(), "l"),
                ptxBuilder.newOperand(adaptor.getRank(), "r")},
               /*onlyAttachMLIRArgs=*/true);

    auto ttPtrTy = dyn_cast<triton::PointerType>(op.getSymmAddr().getType());
    if (auto ttPtrTy =
            dyn_cast<triton::PointerType>(op.getSymmAddr().getType())) {
      assert(ttPtrTy.getAddressSpace() == 1 && "Invalid addr space for SymmAt");
    }
    // addrspace = 1 means global memory
    auto ptxResult = ptxBuilder.launch(
        rewriter, loc, ptr_ty(rewriter.getContext(), /*addrspace=*/1));
    rewriter.replaceOp(op, ptxResult);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
  const bool enableInline;
};
} // namespace

void mlir::triton::NVIDIA::populateDistributedOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo, bool enableInline) {
  patterns.add<WaitOpConversion, ConsumeTokenOpConversion>(typeConverter, benefit);
  patterns.add<WaitOpConversion>(typeConverter, benefit);
  patterns.add<GetRankOpConversion>(typeConverter, benefit, targetInfo,
                                    enableInline);
  patterns.add<GetNumRanksOpConversion>(typeConverter, benefit, targetInfo,
                                        enableInline);
  patterns.add<SymmAtOpConversion>(typeConverter, benefit, targetInfo,
                                   enableInline);
}
