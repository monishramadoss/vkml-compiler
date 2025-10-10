#pragma once
#include <unordered_map>



#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"


// Implementation details for Tensor utilities
namespace tensor_detail {
    template<typename U>
    struct is_character_or_byte_or_bool : std::bool_constant<
        std::is_same_v<U, char> ||
        std::is_same_v<U, unsigned char> ||
        std::is_same_v<U, signed char> ||
        std::is_same_v<U, std::byte> ||
        std::is_same_v<U, bool>
    > {};

    
    static auto cToMLIRType = [](mlir::MLIRContext* ctx, const std::type_info& type) -> mlir::Type {
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
    

}

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"

#include <iostream>


#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"

#include "mlir/Conversion/TosaToMLProgram/TosaToMLProgram.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h"


#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"


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
        Compiler(): context_(), builder_(&context_), pm_(&context_), mainFunc_(nullptr) {
            // Create a dialect registry and register bufferization interfaces
            mlir::DialectRegistry registry;
            mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
            mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
            mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
            mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
            mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
            mlir::ml_program::registerBufferizableOpInterfaceExternalModels(registry);

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
            auto loc = builder_.getUnknownLoc();
            auto fnType = builder_.getFunctionType({}, {});
            mainFunc_ = builder_.create<mlir::func::FuncOp>(loc, "main", fnType);
            auto *entry = mainFunc_.addEntryBlock();
            mlir::OpBuilder::InsertionGuard g(builder_);
            builder_.setInsertionPointToStart(entry);
            builder_.create<mlir::func::ReturnOp>(loc);
            builder_.setInsertionPointToStart(module_.getBody());
        }

    public:
        
        Compiler(const Compiler&) = delete;
        Compiler& operator=(const Compiler&) = delete;

        static std::shared_ptr<Compiler> getInstance() {
            if(instance_.get() == nullptr)
                instance_ = std::shared_ptr<Compiler>(new Compiler());
            return instance_;
        }

        mlir::MLIRContext* getContext() { return &context_; }
        mlir::OpBuilder& getBuilder() { return builder_; }
        mlir::ModuleOp getModule() { return module_; }
        mlir::Location getUnknownLoc() { return builder_.getUnknownLoc(); }

       
        auto setInsertionIntoMain() {
            auto &block = mainFunc_.getBody().front();
            mlir::Operation *terminator = block.getTerminator();
            builder_.setInsertionPoint(terminator); 
            return builder_;
        }

        auto setInsertionGlobalModule() {
            builder_.setInsertionPointToStart(module_.getBody());
            return builder_;
        }

        mlir::tosa::VariableOp createVariable(mlir::RankedTensorType type,
                                            llvm::ArrayRef<int64_t> shape,
                                            llvm::StringRef name) {
            setInsertionGlobalModule();
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
            
            return builder_.create<mlir::tosa::VariableOp>(loc, nameAttr, shapeAttr, typeAttr, initialValueAttr);
        }

        // Overload that accepts actual data from a pointer
        template<typename T>
        mlir::tosa::VariableOp createVariableWithData(mlir::RankedTensorType type,
                                                      llvm::ArrayRef<int64_t> shape,
                                                      llvm::StringRef name,
                                                      const T* dataPtr) {
            setInsertionGlobalModule();
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
            
            return builder_.create<mlir::tosa::VariableOp>(loc, nameAttr, shapeAttr, typeAttr, initialValueAttr);
        }

        std::string getUniqueFunctionName(const std::string& baseName) {
            size_t count = func_name_count_map_[baseName]++;
            if (count == 0) {
                return baseName;
            } else {
                return baseName + "_" + std::to_string(count);
            }
        }

        void runTosaToGPU() {
            // ===== Stage 1: TOSA to Linalg =====
            pm_.addPass(mlir::createCanonicalizerPass());
            
            // Convert TOSA variables to MLProgram globals FIRST (module-level)
            pm_.addPass(mlir::createTosaToMLProgram());
            pm_.addPass(mlir::createCanonicalizerPass());
            
            // Function-level passes need to be added as nested passes
            // Convert TOSA to Linalg (this operates on FunctionOpInterface)
            pm_.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
            pm_.addPass(mlir::createCanonicalizerPass());
            
            // Convert TOSA to Arith (function-level)
            pm_.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass());
            
            // Convert TOSA control flow to SCF (function-level)
            pm_.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToSCFPass());
            
            // Convert TOSA tensor ops to Tensor dialect (function-level)
            pm_.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToTensorPass());
            
            pm_.addPass(mlir::createCanonicalizerPass());
            
            std::cerr << "Stage 1 (TOSA->Linalg) setup\n";
            
            // ===== Stage 2: Bufferization (Tensor -> MemRef) =====
            // Use the default One-Shot Bufferize pass
            // The pass doesn't bufferize function boundaries by default for simplicity
            // To enable function boundary bufferization, use the pass options or
            // use runOneShotBufferize directly with OneShotBufferizationOptions
            std::cerr << "Stage 2 (Bufferization) setup\n";
            
            mlir::bufferization::OneShotBufferizePassOptions options;
            options.bufferizeFunctionBoundaries = true;
            options.functionBoundaryTypeConversion =
                mlir::bufferization::LayoutMapOption::IdentityLayoutMap;

            pm_.addPass(mlir::bufferization::createOneShotBufferizePass(options));
            pm_.addPass(mlir::createCanonicalizerPass());
            
            
            // ===== Stage 3: Linalg to Loops =====
            // Convert Linalg operations to SCF loops
            pm_.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToParallelLoopsPass());
            pm_.addPass(mlir::createCanonicalizerPass());
            
            std::cerr << "Stage 3 (Linalg->Loops) setup\n";
            
            // ===== Stage 4: GPU Lowering =====
            // Map parallel loops to GPU operations
            pm_.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
            pm_.addPass(mlir::createCanonicalizerPass());
            
            std::cerr << "Stage 4 (GPU Lowering) setup\n";
            
            // ===== Stage 5: Cleanup =====
            pm_.addPass(mlir::createCanonicalizerPass());
            pm_.addPass(mlir::createCSEPass());
            
            std::cerr << "Running complete pipeline...\n";
            
            // Run the complete pipeline
            if (mlir::failed(pm_.run(module_))) {
                module_.dump();
                std::cerr << "Warning: Some operations were not fully lowered\n";
            }
            module_.dump();
            std::cerr << "Pipeline completed\n";
        }
        
        /// Example of how to bufferize function operands:
        /// 
        /// To bufferize function arguments/returns (tensor -> memref), you have two options:
        ///
        /// Option 1: Use the pass with command-line options
        /// ```
        /// mlir-opt --one-shot-bufferize="bufferize-function-boundaries=1 \
        ///          function-boundary-type-conversion=identity-layout-map" input.mlir
        /// ```
        ///
        /// Option 2: Programmatically configure the pass (requires creating BufferizationState)
        /// ```cpp
        /// mlir::bufferization::OneShotBufferizationOptions options;
        /// options.bufferizeFunctionBoundaries = true;
        /// options.setFunctionBoundaryTypeConversion(
        ///     mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
        /// 
        /// mlir::bufferization::BufferizationState state(module_, options);
        /// if (mlir::failed(mlir::bufferization::runOneShotBufferize(
        ///         module_, options, state))) {
        ///     // Handle error
        /// }
        /// ```
        ///
        /// For simpler cases (no function boundary bufferization needed),
        /// just use the default OneShotBufferizePass which this class uses in runTosaToGPU().
       
    };

    std::shared_ptr<Compiler> Compiler::instance_ = nullptr;
    inline void dump(){
        auto mod = Compiler::getInstance()->getModule();
        if(mlir::failed( mod.verify() )){
            mod.dump();
            throw std::runtime_error("Module verification failed");
        }
        if(mlir::failed(mod.verifyRegions())){
            mod.dump();
            throw std::runtime_error("Module region verification failed");
        }
        if(mlir::failed(mod.verifyInvariants())){
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
   
}
