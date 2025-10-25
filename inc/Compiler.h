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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/PassManager.h"

#include <iostream>

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TosaToMLProgram/TosaToMLProgram.h"
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

#include "ScopedInserter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"

class PassPipelineConfigurator {
public:
  static void buildDefault(mlir::PassManager &pm) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createTosaToMLProgram());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToSCFPass());
    pm.addPass(mlir::createCanonicalizerPass());
    mlir::bufferization::OneShotBufferizePassOptions opts;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionBoundaryTypeConversion =
        mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertLinalgToParallelLoopsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertParallelLoopToGpuPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
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

  mlir::tosa::VariableOp createZeroInitialized(mlir::RankedTensorType type,
                                               llvm::ArrayRef<int64_t> shape,
                                               llvm::StringRef name) {
    builder_.setInsertionPointToStart(module_.getBody());
    auto loc = builder_.getUnknownLoc();
    // Shape stored as DenseElementsAttr (rank-1 tensor of i64 dims)
    auto shapeType = mlir::RankedTensorType::get({(int64_t)shape.size()},
                                                 builder_.getI64Type());
    auto shapeAttr = builder_.getIndexTensorAttr(shape);
    auto nameAttr = builder_.getStringAttr(name);
    mlir::Attribute zeroAttr, initAttr;
    auto elemType = type.getElementType();

    if (elemType.isF32())
      zeroAttr = builder_.getF32FloatAttr(0.0f);
    else if (elemType.isF64())
      zeroAttr = builder_.getF64FloatAttr(0.0);
    else if (llvm::isa<mlir::IntegerType>(elemType))
      zeroAttr = builder_.getIntegerAttr(elemType, 0);
    else
      zeroAttr = builder_.getF32FloatAttr(0.0f);
    initAttr = mlir::DenseElementsAttr::get(type, zeroAttr);

    return builder_.create<mlir::tosa::VariableOp>(loc, nameAttr.getValue(),
                                                   shapeAttr, type, initAttr);
  }
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
    context_.loadDialect<mlir::tosa::TosaDialect>();
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::ml_program::MLProgramDialect>();
    context_.loadDialect<mlir::gpu::GPUDialect>();
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<mlir::scf::SCFDialect>();
    context_.loadDialect<mlir::tensor::TensorDialect>();
    context_.loadDialect<mlir::memref::MemRefDialect>();
    context_.loadDialect<mlir::bufferization::BufferizationDialect>();
    context_.loadDialect<mlir::linalg::LinalgDialect>();

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

  mlir::tosa::VariableOp createVariable(mlir::RankedTensorType type,
                                        llvm::ArrayRef<int64_t> shape,
                                        llvm::StringRef name) {
    auto scope = inModuleStart();
    auto loc = builder_.getUnknownLoc();
    auto shapeAttr = builder_.getIndexTensorAttr(shape);
    auto nameAttr = builder_.getStringAttr(name);
    auto typeAttr = mlir::TypeAttr::get(type.getElementType());

    // Create a zero-initialized tensor as the initial value
    auto elemType = type.getElementType();
    mlir::Attribute zeroAttr;
    if (elemType.isF32()) {
      zeroAttr = builder_.getF32FloatAttr(0.0f);
    } else if (elemType.isF64()) {
      zeroAttr = builder_.getF64FloatAttr(0.0);
    } else if (elemType.isInteger(1)) {
      zeroAttr = builder_.getBoolAttr(false);
    } else if (llvm::isa<mlir::IntegerType>(elemType)) {
      zeroAttr = builder_.getIntegerAttr(elemType, 0);
    } else {
      zeroAttr = builder_.getF32FloatAttr(0.0f); // fallback
    }

    auto initialValueAttr = mlir::DenseElementsAttr::get(type, zeroAttr);

    return builder_.create<mlir::tosa::VariableOp>(loc, nameAttr, shapeAttr,
                                                   typeAttr, initialValueAttr);
  }

  // Overload that accepts actual data from a pointer
  template <typename T>
  mlir::tosa::VariableOp createVariableWithData(mlir::RankedTensorType type,
                                                llvm::ArrayRef<int64_t> shape,
                                                llvm::StringRef name,
                                                const T *dataPtr) {
    auto scope = inModuleStart();
    auto loc = builder_.getUnknownLoc();
    auto shapeAttr = builder_.getIndexTensorAttr(shape);
    auto nameAttr = builder_.getStringAttr(name);
    auto typeAttr = mlir::TypeAttr::get(type.getElementType());

    mlir::Attribute initialValueAttr;

    if (dataPtr != nullptr) {
      // Calculate total number of elements
      int64_t numElements = 1;
      for (auto dim : shape) {
        numElements *= dim;
      }

      // Create DenseElementsAttr from the raw data
      auto elemType = type.getElementType();

      if constexpr (std::is_same_v<T, float>) {
        if (elemType.isF32()) {
          llvm::ArrayRef<float> dataArray(dataPtr, numElements);
          initialValueAttr = mlir::DenseElementsAttr::get(type, dataArray);
        }
      } else if constexpr (std::is_same_v<T, double>) {
        if (elemType.isF64()) {
          llvm::ArrayRef<double> dataArray(dataPtr, numElements);
          initialValueAttr = mlir::DenseElementsAttr::get(type, dataArray);
        }
      } else if constexpr (std::is_integral_v<T>) {
        llvm::ArrayRef<T> dataArray(dataPtr, numElements);
        initialValueAttr = mlir::DenseElementsAttr::get(type, dataArray);
      }

      // Fallback to zero-initialization if type didn't match
      if (!initialValueAttr) {
        mlir::Attribute zeroAttr;
        if (elemType.isF32()) {
          zeroAttr = builder_.getF32FloatAttr(0.0f);
        } else if (elemType.isF64()) {
          zeroAttr = builder_.getF64FloatAttr(0.0);
        } else {
          zeroAttr = builder_.getIntegerAttr(elemType, 0);
        }
        initialValueAttr = mlir::DenseElementsAttr::get(type, zeroAttr);
      }
    } else {
      // No data provided, use zero initialization
      mlir::Attribute zeroAttr;
      auto elemType = type.getElementType();
      if (elemType.isF32()) {
        zeroAttr = builder_.getF32FloatAttr(0.1f);
      } else if (elemType.isF64()) {
        zeroAttr = builder_.getF64FloatAttr(0.1);
      } else if (elemType.isInteger(1)) {
        zeroAttr = builder_.getBoolAttr(false);
      } else if (llvm::isa<mlir::IntegerType>(elemType)) {
        zeroAttr = builder_.getIntegerAttr(elemType, 1);
      } else {
        zeroAttr = builder_.getF32FloatAttr(1.0f);
      }
      initialValueAttr = mlir::DenseElementsAttr::get(type, zeroAttr);
    }

    return builder_.create<mlir::tosa::VariableOp>(loc, nameAttr, shapeAttr,
                                                   typeAttr, initialValueAttr);
  }

  template<typename BodyFn>
  mlir::func::FuncOp createFunctionOp(llvm::StringRef baseName, 
                                      llvm::ArrayRef<mlir::Type> inputs,
                                      llvm::ArrayRef<mlir::Type> results,
                                      BodyFn &&bodyFn,
                                      bool insertAtStart = true) {
    
    return functionFactory_->createFunctionWithBody("func_" + baseName, inputs, results, bodyFn, insertAtStart);
  }

  void runTosaToGPU() {
    mlir::PassManager pm(&context_);
    PassPipelineConfigurator::buildDefault(pm);
    if (mlir::failed(pm.run(module_))) {
      module_.dump();
      std::cerr << "Pipeline failed (partial lowering)\n";
    }
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
