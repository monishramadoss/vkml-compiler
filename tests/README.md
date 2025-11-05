# VKML Compiler Tests

This directory contains unit tests for the VKML compiler project using Google Test framework.

## Test Structure

The tests are organized by operator category for better maintainability and focused testing:

- **ArithmeticOps_tests.cpp**: Tests for arithmetic operators (+, -, *, /, %, ++, --, unary +) - Uses CTest
- **BitwiseOps_tests.cpp**: Tests for bitwise operators (&, |, ^, ~, <<, >>) - Uses CTest
- **ComparisonOps_tests.cpp**: Tests for comparison operators (==, !=, >, >=, <, <=) - Uses CTest
- **LogicalOps_tests.cpp**: Tests for logical operators (&&, ||, !) - Uses CTest
- **OtherOps_tests.cpp**: Tests for tensor creation, subscript operator, and broadcasting - Uses CTest
- **Tensor_tests.cpp**: Original comprehensive test suite (uses Google Test for backward compatibility)
- **test_utils.h**: Simple test framework for CTest-based tests

## Shape Generator

The `ShapeGenerator.h` utility provides automated generation of test shapes including:

### ShapeGenerator
Generates a comprehensive set of edge cases and unusual tensor shapes:
- 1D shapes (scalars, small, large, prime numbers, powers of 2)
- 2D shapes (squares, rectangles, very wide/tall, prime dimensions)
- 3D shapes (cubes, common ML shapes, asymmetric)
- 4D shapes (image batches, feature maps)
- 5D+ shapes (stress testing)
- Broadcastable shapes (dimensions with size 1)

### BroadcastShapeGenerator
Generates shape pairs specifically designed for testing broadcasting:
- Same shape pairs
- Scalar broadcasting
- Row and column broadcasting
- Different rank broadcasting
- Complex multi-dimensional broadcasting

### RandomShapeGenerator
Generates random shapes with configurable parameters:
- Configurable rank range
- Configurable dimension size range
- Useful for fuzz testing

## Building and Running Tests

### Prerequisites
- CMake 3.16 or higher
- C++20 compatible compiler
- LLVM/MLIR dependencies (initialized via git submodules)

### Build Tests

```bash
# Configure the project with tests enabled
cmake --preset x64-debug-linux

# Build all test executables
cmake --build build/x64-debug-linux --target arithmetic_ops_tests
cmake --build build/x64-debug-linux --target bitwise_ops_tests
cmake --build build/x64-debug-linux --target comparison_ops_tests
cmake --build build/x64-debug-linux --target logical_ops_tests
cmake --build build/x64-debug-linux --target other_ops_tests
cmake --build build/x64-debug-linux --target tensor_tests

# Or build all tests at once
cmake --build build/x64-debug-linux

# Run all tests
cd build/x64-debug-linux
ctest --output-on-failure --verbose

# Or run a specific test executable directly
./tests/arithmetic_ops_tests
./tests/bitwise_ops_tests
./tests/comparison_ops_tests
./tests/logical_ops_tests
./tests/other_ops_tests
./tests/tensor_tests
```

### Running Specific Tests

```bash
# Run tests matching a pattern in a specific suite
./tests/arithmetic_ops_tests --gtest_filter=ArithmeticOpsTest.AdditionVariousShapes

# List all available tests in a suite
./tests/arithmetic_ops_tests --gtest_list_tests

# Run tests with verbose output
./tests/arithmetic_ops_tests --gtest_print_time=1

# Original test suite is still available
./tests/tensor_tests --gtest_filter=TensorTest.CreateTensorWithShape
```

## Continuous Integration

The tests are automatically run on GitHub Actions for:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop` branches

See `.github/workflows/tests.yml` for the CI configuration.

## Test Coverage

The tests are now organized into focused suites by operator category. Each suite includes:

- **Basic tests**: Verify operators work with simple shapes
- **VariousShapes tests**: Test operators with a wide range of edge-case shapes using ShapeGenerator
- **Broadcasting tests**: Test operators with broadcastable shape pairs using BroadcastShapeGenerator
- **Type-specific tests**: Verify operators work with different data types (float, int, uint, etc.)

Current operator coverage:
- Tensor creation with various shapes (1D, 2D, 3D, 4D, 5D+)
- Different data types (float, double, int32_t, uint32_t, uint64_t)
- **Arithmetic operations** (ArithmeticOps_tests.cpp):
  - Addition (+)
  - Subtraction (-)
  - Multiplication (*)
  - Division (/ for both float and int)
  - Modulo (%)
  - Unary plus/abs (+)
  - Prefix/postfix increment (++)
  - Prefix/postfix decrement (--)
- **Bitwise operations** (BitwiseOps_tests.cpp):
  - AND (&)
  - OR (|)
  - XOR (^)
  - NOT (~)
  - Left shift (<<)
  - Right shift (>>)
- **Logical operations** (LogicalOps_tests.cpp):
  - Logical AND (&&)
  - Logical OR (||)
  - Logical NOT (!)
- **Comparison operations** (ComparisonOps_tests.cpp):
  - Equal (==)
  - Not equal (!=)
  - Greater than (>)
  - Greater than or equal (>=)
  - Less than (<)
  - Less than or equal (<=)
- **Other operations** (OtherOps_tests.cpp):
  - Subscript operator ([])
  - Broadcasting operations
  - Symbolic ID generation
  - Tensor creation with various shapes

## Adding New Tests

To add new tests:

1. Choose the appropriate test file based on operator category:
   - ArithmeticOps_tests.cpp for arithmetic operators
   - BitwiseOps_tests.cpp for bitwise operators
   - ComparisonOps_tests.cpp for comparison operators
   - LogicalOps_tests.cpp for logical operators
   - OtherOps_tests.cpp for other operations
2. Add test cases following the existing pattern (Basic, VariousShapes, Broadcasting)
3. Utilize ShapeGenerator and BroadcastShapeGenerator for comprehensive shape testing
4. Follow Google Test conventions for test naming and structure
5. Ensure tests are isolated and don't depend on execution order
6. Run the specific test suite to verify your changes
