#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace vkml {
class ScopedInsertionPoint {
public:
  ScopedInsertionPoint(mlir::OpBuilder &builder, mlir::Block *block,
                       mlir::Block::iterator insertPoint)
      : builder_(builder), originalInsertPoint_(builder.saveInsertionPoint()) {
    builder.setInsertionPoint(block, insertPoint);
  }

  ScopedInsertionPoint(mlir::OpBuilder &builder, mlir::Operation *op)
      : builder_(builder), originalInsertPoint_(builder.saveInsertionPoint()) {
    builder.setInsertionPoint(op);
  }

  ScopedInsertionPoint(mlir::OpBuilder &builder, mlir::Block *block)
      : builder_(builder), originalInsertPoint_(builder.saveInsertionPoint()) {
    builder.setInsertionPointToStart(block);
  }

  ~ScopedInsertionPoint() { builder_.restoreInsertionPoint(originalInsertPoint_); }

private:
  mlir::OpBuilder &builder_;
  mlir::OpBuilder::InsertPoint originalInsertPoint_;
};


// RAII guard for inserting into a module body (start or end).
class ScopedModuleStartEnd {
public:
  ScopedModuleStartEnd(mlir::ModuleOp module, mlir::OpBuilder &builder, bool atStart)
      : guard_(builder, module.getBody(), atStart ? module.getBody()->begin() : module.getBody()->end()) {}
private:
  ScopedInsertionPoint guard_;
};

// RAII guard that ensures insertion at the start of the first block of a function.
class ScopedFunctionEntry {
public:
  ScopedFunctionEntry(mlir::func::FuncOp func, mlir::OpBuilder &builder)
      : guard_(builder, &func.getBody().front()) {}
private:
  ScopedInsertionPoint guard_;
};

// RAII guard that sets insertion *before* the terminator of the entry block if it exists,
// otherwise to the end of the block.
class ScopedFunctionBeforeTerminator {
public:
  ScopedFunctionBeforeTerminator(mlir::func::FuncOp func, mlir::OpBuilder &builder)
      : builder_(builder), original_(builder.saveInsertionPoint()) {
    mlir::Block &blk = func.getBody().front();
    if (auto *term = blk.getTerminator()) builder.setInsertionPoint(term); else builder.setInsertionPointToEnd(&blk);
  }
  ~ScopedFunctionBeforeTerminator() { builder_.restoreInsertionPoint(original_); }
private:
  mlir::OpBuilder &builder_;
  mlir::OpBuilder::InsertPoint original_;
};

// Convenience aliases matching earlier naming for external code migration.
using ModuleScope = ScopedModuleStartEnd;
using FunctionEntryScope = ScopedFunctionEntry;
using FunctionBeforeTerminatorScope = ScopedFunctionBeforeTerminator;
} // namespace vkml
