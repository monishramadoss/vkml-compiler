# Linalg Named Operations

This document describes the linalg named operations added to the Tensor class that go beyond the C++ operator overloads.

## Matrix Operations

### `matmul(other)`
**Description**: 2D matrix multiplication (C = A @ B)
- **Input shapes**: A: [M, K], B: [K, N]
- **Output shape**: C: [M, N]
- **MLIR operation**: `linalg.matmul`
- **Example**:
```cpp
Tensor<float> A({2, 3});  // 2x3 matrix
Tensor<float> B({3, 4});  // 3x4 matrix
auto C = A.matmul(B);     // Result: 2x4 matrix
```

### `dot(other)`
**Description**: 1D dot product (scalar = dot(A, B))
- **Input shapes**: A: [N], B: [N]
- **Output shape**: scalar (rank 0)
- **MLIR operation**: `linalg.dot`
- **Example**:
```cpp
Tensor<float> a({5});
Tensor<float> b({5});
auto result = a.dot(b);  // Result: scalar
```

### `matvec(vec)`
**Description**: Matrix-vector multiplication (y = A @ x)
- **Input shapes**: A: [M, N], x: [N]
- **Output shape**: y: [M]
- **MLIR operation**: `linalg.matvec`
- **Example**:
```cpp
Tensor<float> A({3, 4});
Tensor<float> x({4});
auto y = A.matvec(x);    // Result: vector of size 3
```

### `batch_matmul(other)`
**Description**: Batched matrix multiplication (C = A @ B, batched)
- **Input shapes**: A: [B, M, K], B: [B, K, N]
- **Output shape**: C: [B, M, N]
- **MLIR operation**: `linalg.batch_matmul`
- **Example**:
```cpp
Tensor<float> A({2, 3, 4});  // 2 batches of 3x4 matrices
Tensor<float> B({2, 4, 5});  // 2 batches of 4x5 matrices
auto C = A.batch_matmul(B);  // Result: 2 batches of 3x5 matrices
```

### `vecmat(matrix)`
**Description**: Vector-matrix multiplication (y = x @ A)
- **Input shapes**: x: [N], A: [N, M]
- **Output shape**: y: [M]
- **MLIR operation**: `linalg.vecmat`
- **Example**:
```cpp
Tensor<float> x({4});
Tensor<float> A({4, 5});
auto y = x.vecmat(A);    // Result: vector of size 5
```

## Structural Operations

### `transpose()`
**Description**: Transpose last two dimensions
- **For 2D**: [M, N] → [N, M]
- **For higher rank**: [..., M, N] → [..., N, M]
- **MLIR operation**: `linalg.transpose`
- **Example**:
```cpp
Tensor<float> A({3, 4});
auto B = A.transpose();      // Result: 4x3 matrix

Tensor<float> T({2, 3, 4});
auto T2 = T.transpose();     // Result: [2, 4, 3]
```

### `fill(shape, value)` (static)
**Description**: Create a tensor filled with a scalar value
- **MLIR operation**: `linalg.fill`
- **Example**:
```cpp
auto zeros = Tensor<float>::fill({3, 4}, 0.0f);
auto ones = Tensor<int>::fill({5, 5}, 1);
```

### `copy()`
**Description**: Create a copy of the tensor
- **MLIR operation**: `linalg.copy`
- **Example**:
```cpp
Tensor<float> original({3, 4});
auto copied = original.copy();
```

### `map<UnaryOp>()`
**Description**: Apply a unary operation to each element
- **Uses**: `linalg.generic` with the specified operation
- **Example**:
```cpp
Tensor<float> tensor({2, 3});
auto abs_tensor = tensor.map<mlir::math::AbsFOp>();
```

## Reduction Operations

### `reduce<ReduceOp>()`
**Description**: Generic reduction along the last dimension
- **Uses**: `linalg.generic` with reduction iterator type
- **Example**:
```cpp
Tensor<float> tensor({3, 4});
auto reduced = tensor.reduce<mlir::arith::AddFOp>();  // Sum reduction
```

### `sum()`
**Description**: Sum reduction along the last dimension
- **Input shape**: [..., N]
- **Output shape**: [...]
- **Example**:
```cpp
Tensor<float> tensor({3, 4});
auto result = tensor.sum();  // Result: [3]

Tensor<float> vec({5});
auto scalar = vec.sum();     // Result: scalar
```

### `max()`
**Description**: Maximum reduction along the last dimension
- **Input shape**: [..., N]
- **Output shape**: [...]
- **Example**:
```cpp
Tensor<float> tensor({2, 5});
auto result = tensor.max();  // Result: [2]
```

### `min()`
**Description**: Minimum reduction along the last dimension
- **Input shape**: [..., N]
- **Output shape**: [...]
- **Example**:
```cpp
Tensor<int32_t> tensor({4, 3});
auto result = tensor.min();  // Result: [4]
```

## Type Support

All operations support:
- **Floating-point types**: `float`, `double`
- **Integer types**: `int32_t`, `int64_t`, `uint32_t`, `uint64_t`
- **Mixed types**: Operations between different types use C++ `std::common_type_t` for result type

## Error Handling

All operations perform shape validation and throw `std::runtime_error` for:
- Rank mismatches
- Incompatible dimensions
- Invalid operations (e.g., transpose on 1D tensor)

## Implementation Notes

1. All operations use **linalg named operations** from MLIR, not `linalg.generic`
2. Operations integrate seamlessly with existing C++ operator overloads
3. All operations properly handle tensor construction and value management
4. Empty output tensors are created with `tensor.empty` before operations
5. Reduction operations properly initialize output with identity values
