# Changes: Replacing TOSA Operations with Linalg and Tensor Dialect

## Overview
This document describes the changes made to replace TOSA operations with linalg generic and linalg named operations, and to use the tensor dialect instead of tosa.VariableOps for tensor read/write operations.

## Key Changes

### 1. Tensor Storage and Initialization (inc/Tensor.h)

**Before:**
- Used `mlir::tosa::VariableOp` to store tensor state
- Used `mlir::tosa::VariableReadOp` to read tensor values
- Used `mlir::tosa::VariableWriteOp` to write tensor values

**After:**
- Use `mlir::Value tensorValue_` to directly store tensor values
- Use `tensor.empty` to create uninitialized tensors
- Use `arith.constant` to create initialized constant tensors
- Direct value assignment instead of read/write operations

### 2. Arithmetic Operations

**Before:**
- `tosa.add` for addition
- `tosa.sub` for subtraction
- `tosa.mul` for multiplication (with scale parameter)
- `tosa.int_div` for integer division
- `tosa.reciprocal` + `tosa.mul` for floating-point division

**After:**
- `linalg.generic` with `arith.addi` for integer addition
- `linalg.generic` with `arith.addf` for floating-point addition
- `linalg.generic` with `arith.subi` for integer subtraction
- `linalg.generic` with `arith.subf` for floating-point subtraction
- `linalg.generic` with `arith.muli` for integer multiplication
- `linalg.generic` with `arith.mulf` for floating-point multiplication
- `linalg.generic` with `arith.divsi` for integer division
- `linalg.generic` with `arith.divf` for floating-point division

### 3. Unary Operations

**Before:**
- `tosa.abs` for absolute value
- `tosa.bitwise_not` for bitwise NOT
- `tosa.logical_not` for logical NOT

**After:**
- `linalg.generic` with `math.absi` for integer absolute value
- `linalg.generic` with `math.absf` for floating-point absolute value
- `linalg.generic` with `arith.xori` (with all-ones constant) for bitwise NOT
- `linalg.generic` with comparison to zero for logical NOT

### 4. Bitwise Operations

**Before:**
- `tosa.bitwise_and`, `tosa.bitwise_or`, `tosa.bitwise_xor`
- `tosa.logical_left_shift`, `tosa.logical_right_shift`

**After:**
- `linalg.generic` with `arith.andi`, `arith.ori`, `arith.xori`
- `linalg.generic` with `arith.shli`, `arith.shrui`

### 5. Comparison Operations

**Before:**
- `tosa.equal`, `tosa.greater`, `tosa.greater_equal`

**After:**
- `linalg.generic` with `arith.cmpi` (eq, sgt, sge predicates) for integers
- `linalg.generic` with `arith.cmpf` (oeq, ogt, oge predicates) for floats

### 6. Tensor Slicing

**Before:**
- `tosa.slice` with `tosa.const_shape` for start and size parameters

**After:**
- `tensor.extract_slice` with OpFoldResult offsets, sizes, and strides

### 7. Type Casting

**Before:**
- `tosa.cast` with type inference

**After:**
- `linalg.generic` with `arith.sitofp` for int-to-float conversion
- `linalg.generic` with `arith.fptosi` for float-to-int conversion

### 8. Constant Creation

**Before:**
- `tosa.const` for constant tensors

**After:**
- `arith.constant` with DenseElementsAttr

## Implementation Details

### Helper Functions

Two new template helper functions were added:

1. `linalgBinaryOp<ArithOp, U, V>`: Creates a `linalg.generic` operation for binary arithmetic operations with broadcasting support.

2. `linalgUnaryOp<ArithOp, V>`: Creates a `linalg.generic` operation for unary operations.

These helpers:
- Create empty output tensors using `tensor.empty`
- Set up affine maps for iteration
- Define parallel iterator types
- Build the operation body using the specified ArithOp

### Compiler.h Changes

Removed the following functions as they're no longer needed:
- `createVariable()`
- `createVariableWithData()`
- `VariableFactory::createZeroInitialized()`

The `VariableFactory` class is now empty and can be removed in future cleanup.

## Benefits

1. **Standard MLIR Dialects**: Uses standard linalg and tensor dialects instead of TOSA-specific operations
2. **Better Optimization**: Linalg operations have more optimization passes available
3. **Cleaner Semantics**: Direct value usage instead of variable read/write operations
4. **Type Safety**: Explicit handling of integer vs. floating-point operations
5. **Broadcasting**: Proper broadcasting semantics in binary operations

## Testing

The existing API remains unchanged. Code like:
```cpp
Tensor<float> tensor_0({2, 3});
Tensor<float> tensor_1({1, 3});
auto result = tensor_0 + tensor_1;
```

Will now generate linalg.generic operations instead of TOSA operations, while maintaining the same high-level behavior.

## Future Work

1. Add comprehensive unit tests
2. Remove unused VariableFactory class
3. Add support for more complex operations (matmul, conv, etc.) using linalg named operations
4. Optimize broadcasting implementation
5. Add better error handling for shape mismatches
