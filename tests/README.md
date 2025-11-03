# VKML Compiler Tests

This directory contains unit tests for the VKML compiler project using Google Test framework.

## Building and Running Tests

### Prerequisites
- CMake 3.16 or higher
- C++20 compatible compiler
- LLVM/MLIR dependencies (initialized via git submodules)

### Build Tests

```bash
# Configure the project with tests enabled
cmake --preset x64-debug-linux

# Build the test executable
cmake --build build/x64-debug-linux --target tensor_tests

# Run all tests
cd build/x64-debug-linux
ctest --output-on-failure --verbose

# Or run the test executable directly
./tests/tensor_tests
```

### Running Specific Tests

```bash
# Run tests matching a pattern
./tests/tensor_tests --gtest_filter=TensorTest.CreateTensorWithShape

# List all available tests
./tests/tensor_tests --gtest_list_tests
```

## Continuous Integration

The tests are automatically run on GitHub Actions for:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop` branches

See `.github/workflows/tests.yml` for the CI configuration.

## Test Coverage

Current tests cover:
- Tensor creation with various shapes (1D, 2D, 3D)
- Different data types (float, double, int32_t, uint32_t, uint64_t)
- **Arithmetic operations:**
  - Addition (+)
  - Subtraction (-)
  - Multiplication (*)
  - Division (/ for both float and int)
  - Modulo (%)
- **Bitwise operations:**
  - AND (&)
  - OR (|)
  - XOR (^)
  - NOT (~)
  - Left shift (<<)
  - Right shift (>>)
- **Logical operations:**
  - Logical AND (&&)
  - Logical OR (||)
  - Logical NOT (!)
- **Comparison operations:**
  - Equal (==)
  - Not equal (!=)
  - Greater than (>)
  - Greater than or equal (>=)
  - Less than (<)
  - Less than or equal (<=)
- **Unary operations:**
  - Unary plus/abs (+)
  - Prefix increment (++)
  - Postfix increment (++)
  - Prefix decrement (--)
  - Postfix decrement (--)
- **Indexing:**
  - Subscript operator ([])
- Broadcasting operations
- Symbolic ID generation

## Adding New Tests

To add new tests:

1. Add test cases to `Tensor_tests.cpp` or create new test files
2. If creating new test files, update `tests/CMakeLists.txt` to include them
3. Follow Google Test conventions for test naming and structure
4. Ensure tests are isolated and don't depend on execution order
