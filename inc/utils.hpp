#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"


namespace vkml{
  namespace compiler_utils{
    struct globalBuilder {
        mlir::MLIRContext context_;
        mlir::OpBuilder builder_;
        globalBuilder() : builder_(&context_) {}
    };

    static globalBuilder& instance() {
        static globalBuilder instance;
        return instance;
    }    

    class ScopedInsertionPoint {
        mlir::OpBuilder builder_;
        mlir::OpBuilder::InsertPoint originalInsertPoint_;
    public:
        ScopedInsertionPoint(mlir::Block *block, mlir::Block::iterator insertPoint)
            : builder_(instance().builder_), originalInsertPoint_(instance().builder_.saveInsertionPoint()) {
            builder_.setInsertionPoint(block, insertPoint);
        }

        ScopedInsertionPoint(mlir::Operation *op)
            : builder_(instance().builder_), originalInsertPoint_(builder_.saveInsertionPoint()) {
            builder_.setInsertionPoint(op);
        }

        ScopedInsertionPoint(mlir::Block *block)
            : builder_(instance().builder_), originalInsertPoint_(builder_.saveInsertionPoint()) {
            builder_.setInsertionPointToStart(block);
        }

        ~ScopedInsertionPoint() { builder_.restoreInsertionPoint(originalInsertPoint_); }
    };

    class Module {
        mlir::ModuleOp module_;
        ScopedInsertionPoint guard_;
    public:
        Module(bool insertAtStart) : module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(&instance().context_))),
            guard_(module_.getBody(), insertAtStart ? module_.getBody()->begin() : module_.getBody()->end()) { }

        mlir::ModuleOp getModule() const { return module_; }
    };


    
    // class ScopedFunction {
    //     ScopedInsertionPoint guard_;
    // public:
    //     ScopedFunction(mlir::func::FuncOp func, mlir::OpBuilder &builder)
    //         : guard_(builder, &func.getBody().front()) {}       
    // };

    // class Function : public ScopedFunction {
    //     mlir::func::FuncOp func_;
    // public:
    //     Function(mlir::func::FuncOp func, mlir::OpBuilder &builder)
    //         : func_(func), ScopedFunction(func, builder) {

    //     } 
    // };

  }
}

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"


namespace compiler_utils {
  // void register_bufferization_op_interface(mlir::MLIRContext& registry){
  //     mlir::DialectRegistry dialectRegistry;
  //     mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  //     mlir::bufferization::func_ext::
  //         registerBufferizableOpInterfaceExternalModels(registry);
  //     mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  //     mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  //     mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  //     mlir::func::registerInlinerExtension(registry);
  //     context_.appendDialectRegistry(registry);
  // }      
} // namespace compiler_utils

} // namespace vkml