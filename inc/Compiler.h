#pragma once
#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

// Implementation details for Tensor utilities
namespace tensor_detail {
template <typename U>
struct is_character_or_byte_or_bool
    : std::bool_constant<
          std::is_same_v<U, char> || std::is_same_v<U, unsigned char> ||
          std::is_same_v<U, signed char> || std::is_same_v<U, std::byte> ||
          std::is_same_v<U, bool>> {};

static auto cToMLIRType = [](mlir::MLIRContext *ctx,
                             const std::type_info &type) -> mlir::Type {
  if (type == typeid(float)) {
    return mlir::Float32Type::get(ctx);
  } else if (type == typeid(double)) {
    return mlir::Float64Type::get(ctx);
  } else if (type == typeid(char)) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
  } else if (type == typeid(unsigned char)) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
  } else if (type == typeid(int32_t)) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
  } else if (type == typeid(int64_t)) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
  } else if (type == typeid(uint32_t)) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
  } else if (type == typeid(uint64_t)) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
  } else if (type == typeid(bool)) {
    return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Unsigned);
  } else {
    throw std::invalid_argument("Unsupported type for MLIR conversion");
  }
};

} // namespace tensor_detail

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include <iostream>

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"

// Define these to get the tablegen-generated pass declarations
#define GEN_PASS_DECL_SCFTOSPIRV
#define GEN_PASS_DECL_RECONCILEUNREALIZEDCASTSPASS
#include "mlir/Conversion/Passes.h"
#undef GEN_PASS_DECL_SCFTOSPIRV
#undef GEN_PASS_DECL_RECONCILEUNREALIZEDCASTSPASS

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"

#include "ScopedInserter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"

#include "mlir/IR/Verifier.h"

class PassPipelineConfigurator {
public:
  static void buildDefault(mlir::PassManager &pm) {
    mlir::bufferization::OneShotBufferizePassOptions opts;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertParallelLoopToGpuPass());
    
  }

  static void buildSPIRV(mlir::PassManager &pm) {   
    // Lower affine operations first
    pm.addPass(mlir::createLowerAffinePass());
    
    // Outline gpu.launch blocks into gpu.func within gpu.module
    pm.addPass(mlir::createGpuKernelOutliningPass());
  }
  
  static void buildSPIRVConversion(mlir::PassManager &pm) {
    // Map memref address spaces to SPIR-V storage classes
    pm.addPass(mlir::createMapMemRefStorageClassPass());
    
    // Now convert GPU module to SPIR-V module
    // Note: gpu.func needs spirv.entry_point_abi attribute - set via setEntryPointABIOnGPUFuncs before this
    pm.addPass(mlir::createConvertGPUToSPIRVPass());
    
    // Run canonicalization to clean up after GPU conversion
    pm.addPass(mlir::createCanonicalizerPass());
  }

  // Erase gpu.module ops after they've been converted to spirv.module
  static void eraseGPUModules(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::gpu::GPUModuleOp, 4> toErase;
    module.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
      toErase.push_back(gpuModule);
    });
    for (auto gpuModule : toErase) {
      // gpuModule.erase();
    }
  }

  // Set spirv.entry_point_abi attribute on all gpu.func operations
  static void setEntryPointABIOnGPUFuncs(mlir::ModuleOp module) {
    auto *ctx = module.getContext();
    module.walk([&](mlir::gpu::GPUFuncOp gpuFunc) {
      if (!gpuFunc->hasAttr(mlir::spirv::getEntryPointABIAttrName())) {
        // Get workgroup size from known_block_size if available, otherwise use [1, 1, 1]
        llvm::SmallVector<int32_t, 3> workgroupSize = {1, 1, 1};
        if (auto blockSize = gpuFunc.getKnownBlockSize()) {
          workgroupSize = llvm::SmallVector<int32_t, 3>(blockSize->begin(), blockSize->end());
        }
        auto abiAttr = mlir::spirv::getEntryPointABIAttr(ctx, workgroupSize);
        gpuFunc->setAttr(mlir::spirv::getEntryPointABIAttrName(), abiAttr);
      }
    });
  }

  static void spirvTargetRegister(mlir::MLIRContext* context, mlir::ModuleOp &module) {
     auto verCap = mlir::spirv::VerCapExtAttr::get(
        mlir::spirv::Version::V_1_3,
        {mlir::spirv::Capability::Shader},
        llvm::ArrayRef<mlir::spirv::Extension>(),
        context
     );

     auto limits = mlir::spirv::getDefaultResourceLimits(context);
     auto targetEnv = mlir::spirv::TargetEnvAttr::get(
        verCap,
        limits,
        mlir::spirv::ClientAPI::Vulkan,
        mlir::spirv::Vendor::Unknown,
        mlir::spirv::DeviceType::Unknown,
        0
     );

      module->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnv);   
  }

};

class FunctionFactory {
  mlir::OpBuilder &builder_;
  mlir::ModuleOp &module_;
  std::unordered_map<std::string, size_t> &counts_;

  std::string getUniqueFunctionName(const std::string &base) {
    size_t c = counts_[base]++;
    return c == 0 ? base : base + "_" + std::to_string(c);
  }

public:
  FunctionFactory(mlir::OpBuilder &builder, mlir::ModuleOp &module,
                  std::unordered_map<std::string, size_t> &counts)
      : builder_(builder), module_(module), counts_(counts) {}

  template <typename BodyFn>
  mlir::func::FuncOp createFunctionWithBody(llvm::StringRef baseName,
                                            llvm::ArrayRef<mlir::Type> inputs,
                                            llvm::ArrayRef<mlir::Type> results,
                                            BodyFn &&bodyFn,
                                            bool insertAtStart = true) {
    auto name = getUniqueFunctionName(baseName.str());
    auto fnType = builder_.getFunctionType(inputs, results);

    auto func = module_.lookupSymbol<mlir::func::FuncOp>(name);
    if (!func) {
      func = mlir::func::FuncOp::create(builder_.getUnknownLoc(), name, fnType);
      module_.push_back(func);
    }
    if (func.getBody().empty())
      func.addEntryBlock();

    auto &entry = func.getBody().front();
    vkml::ScopedInsertionPoint scope(
        builder_, &entry, insertAtStart ? entry.begin() : entry.end());
    bodyFn(builder_, func, entry);
    if (func && !func.getBody().empty()) {
      auto &block = func.getBody().front();
      if (block.empty() ||
          !block.back().mightHaveTrait<mlir::OpTrait::IsTerminator>())
        builder_.create<mlir::func::ReturnOp>(func.getLoc());
    }
    return func;
  }
};

class VariableFactory {
  mlir::OpBuilder &builder_;
  mlir::ModuleOp &module_;

public:
  VariableFactory(mlir::OpBuilder &builder, mlir::ModuleOp &module)
      : builder_(builder), module_(module) {}
};

namespace vkml {

class Compiler {
private:
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  mlir::func::FuncOp mainFunc_;
  mlir::PassManager pm_;
  static std::shared_ptr<Compiler> instance_;
  std::unordered_map<std::string, size_t> func_name_count_map_;

  std::unique_ptr<FunctionFactory> functionFactory_;
  std::unique_ptr<VariableFactory> variableFactory_;

  Compiler()
      : context_(), builder_(&context_), pm_(&context_), mainFunc_(nullptr) {
    // Create a dialect registry and register bufferization interfaces
    mlir::DialectRegistry registry;
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::ml_program::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::func::registerInlinerExtension(registry);
    context_.appendDialectRegistry(registry);

    // Load all the dialects
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::ml_program::MLProgramDialect>();
    context_.loadDialect<mlir::gpu::GPUDialect>();
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<mlir::scf::SCFDialect>();
    context_.loadDialect<mlir::tensor::TensorDialect>();
    context_.loadDialect<mlir::memref::MemRefDialect>();
    context_.loadDialect<mlir::bufferization::BufferizationDialect>();
    context_.loadDialect<mlir::linalg::LinalgDialect>();
    context_.loadDialect<mlir::math::MathDialect>();
    context_.loadDialect<mlir::spirv::SPIRVDialect>();
    context_.loadDialect<mlir::cf::ControlFlowDialect>();
    context_.loadDialect<mlir::DLTIDialect>();

    module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToStart(module_.getBody());

    functionFactory_ = std::make_unique<FunctionFactory>(builder_, module_,func_name_count_map_);
    variableFactory_ = std::make_unique<VariableFactory>(builder_, module_);

    mainFunc_ = functionFactory_->createFunctionWithBody(
        "main", {}, {},
        [&](mlir::OpBuilder &, mlir::func::FuncOp, mlir::Block &) {}, true);
  }

public:
  Compiler(const Compiler &) = delete;
  Compiler &operator=(const Compiler &) = delete;

  static std::shared_ptr<Compiler> getInstance() {
    if (instance_.get() == nullptr)
      instance_ = std::shared_ptr<Compiler>(new Compiler());
    return instance_;
  }

  mlir::MLIRContext *getContext() { return &context_; }
  mlir::OpBuilder &getBuilder() { return builder_; }
  mlir::ModuleOp getModule() { return module_; }
  mlir::Location getUnknownLoc() { return builder_.getUnknownLoc(); }

  // Scope helpers replacing old setInsertion* pattern.
  ScopedFunctionBeforeTerminator inMainBeforeTerminator() {
    return ScopedFunctionBeforeTerminator(mainFunc_, builder_);
  }
  ModuleScope inModuleStart() {
    return ModuleScope(module_, builder_, /*atStart=*/true);
  }
  ModuleScope inModuleEnd() {
    return ModuleScope(module_, builder_, /*atStart=*/false);
  }

  template <typename BodyFn>
  mlir::func::FuncOp createFunctionWithBody(llvm::StringRef baseName,
                                            llvm::ArrayRef<mlir::Type> inputs,
                                            llvm::ArrayRef<mlir::Type> results,
                                            BodyFn &&bodyFn,
                                            bool insertAtStart = true) {
    size_t c = func_name_count_map_[baseName.str()]++;
    std::string name = c == 0 ? baseName.str() : baseName.str() + "_" + std::to_string(c);
    auto fnType = builder_.getFunctionType(inputs, results);

    auto func = module_.lookupSymbol<mlir::func::FuncOp>(name);
    if (!func) {
      func = mlir::func::FuncOp::create(builder_.getUnknownLoc(), name, fnType);
      module_.push_back(func);
    }
    if (func.getBody().empty())
      func.addEntryBlock();

    auto &entry = func.getBody().front();
    ScopedInsertionPoint scope(
        builder_, &entry, insertAtStart ? entry.begin() : entry.end());
    bodyFn(builder_, func, entry);
    if (func && !func.getBody().empty()) {
      auto &block = func.getBody().front();
      if (block.empty() ||
          !block.back().mightHaveTrait<mlir::OpTrait::IsTerminator>())
        builder_.create<mlir::func::ReturnOp>(func.getLoc());
    }
    return func;
  }

  template<typename BodyFn>
  mlir::func::FuncOp createFunctionOp(llvm::StringRef baseName, 
                                      llvm::ArrayRef<mlir::Type> inputs,
                                      llvm::ArrayRef<mlir::Type> results,
                                      BodyFn &&bodyFn,
                                      bool insertAtStart = true) {
    
    return functionFactory_->createFunctionWithBody("func_" + baseName, inputs, results, bodyFn, insertAtStart);
  }

  void runLinalgToGPU() {
    mlir::PassManager pm(&context_);
    PassPipelineConfigurator::buildDefault(pm);
    if (mlir::failed(pm.run(module_))) {
      module_.dump();
      std::cerr << "Pipeline failed (partial lowering)\n";
      return;
    }
    llvm::outs() << "  ✓ Default pipeline completed\n";
    module_.print(llvm::outs());
    llvm::outs() << "\n";
    
    // Register SPIR-V target
    PassPipelineConfigurator::spirvTargetRegister(&context_, module_);
    
    // Phase 1: Outlining - creates gpu.module with gpu.func
    mlir::PassManager pm_outline(&context_);
    PassPipelineConfigurator::buildSPIRV(pm_outline);
    llvm::outs() << "\n Running outlining passes\n";

    if (mlir::failed(pm_outline.run(module_))) {
      module_.print(llvm::outs());
      std::cerr << "Outlining pipeline failed\n";
      return;
    }
    llvm::outs() << "  ✓ Outlining completed\n";
    module_.print(llvm::outs());
    llvm::outs() << "\n";
    
    // Phase 2: Set entry point ABI on all gpu.func ops
    PassPipelineConfigurator::setEntryPointABIOnGPUFuncs(module_);
    llvm::outs() << "  ✓ Entry point ABI set on gpu.func ops\n";
    module_.print(llvm::outs());
    llvm::outs() << "\n";
    
    // Phase 3: Convert gpu.module to spirv.module
    mlir::PassManager pm_spirv(&context_);
    PassPipelineConfigurator::buildSPIRVConversion(pm_spirv);
    llvm::outs() << "\n Running SPIR-V conversion passes\n";

    if (mlir::failed(pm_spirv.run(module_))) {
      module_.print(llvm::outs());
      std::cerr << "SPIR-V conversion pipeline failed\n";
      return;
    }
    llvm::outs() << "  ✓ SPIR-V conversion completed\n";
    module_.print(llvm::outs());
    llvm::outs() << "\n";
    
    // Phase 4: Erase gpu.module ops (now converted to spirv.module)
    PassPipelineConfigurator::eraseGPUModules(module_);
    llvm::outs() << "  ✓ GPU modules erased\n";
    module_.print(llvm::outs());
    llvm::outs() << "\n";
  }
};

std::shared_ptr<Compiler> Compiler::instance_ = nullptr;
inline void dump() {
  auto mod = Compiler::getInstance()->getModule();
  if (mlir::failed(mod.verify())) {
    mod.dump();
    throw std::runtime_error("Module verification failed");
  }
  if (mlir::failed(mod.verifyRegions())) {
    mod.dump();
    throw std::runtime_error("Module region verification failed");
  }
  if (mlir::failed(mod.verifyInvariants())) {
    mod.dump();
    throw std::runtime_error("Module type verification failed");
  }
  mod.walk([&](mlir::Operation *op) {
    if (mlir::failed(mlir::verify(op))) {
      mod.dump();
      throw std::runtime_error("Operation verification failed");
    }
  });

  mlir::OpPrintingFlags flags;
  mod.print(llvm::outs(), flags);
  llvm::outs() << "\n";
}

} // namespace vkml
