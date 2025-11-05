# Testing Guide

This guide explains the new test structure for the VKML Compiler project.

## Overview

The test suite has been refactored to:
1. **Separate tests by operator category** for better organization and maintainability
2. **Use a shape generator** to brute-force test operators with strange and abnormal shapes
3. **Use CTest** as the testing framework (except for the original test file which retains Google Test)

## Test Files

### Operator-Specific Tests (CTest-based)

- **ArithmeticOps_tests.cpp**: Addition, subtraction, multiplication, division, modulo, increment, decrement
- **BitwiseOps_tests.cpp**: Bitwise AND, OR, XOR, NOT, left shift, right shift
- **ComparisonOps_tests.cpp**: Equality, inequality, greater than, less than, etc.
- **LogicalOps_tests.cpp**: Logical AND, OR, NOT
- **OtherOps_tests.cpp**: Tensor creation, subscript operator, broadcasting

### Legacy Test (Google Test)

- **Tensor_tests.cpp**: Original comprehensive test suite (kept for backward compatibility)

## Shape Generator

The `ShapeGenerator.h` provides three generators:

### 1. ShapeGenerator
Generates ~50+ edge-case shapes including:
- 1D shapes: scalars (1), small (2-7), primes (7, 11, 13), powers of 2 (16, 1024)
- 2D shapes: squares, rectangles, very wide/tall, prime dimensions
- 3D shapes: cubes, image-like, different orderings
- 4D shapes: batch images, feature maps
- 5D+ shapes: stress testing
- Broadcastable shapes: dimensions with size 1

**Usage:**
```cpp
ShapeGenerator gen;
for (const auto& shape : gen) {
    Tensor<float> tensor(shape);
    // Test with this shape
}
```

### 2. BroadcastShapeGenerator
Generates ~20 shape pairs designed for broadcast testing:
- Same shapes
- Scalar broadcasting
- Row/column broadcasting
- Different rank broadcasting
- Complex multi-dimensional broadcasting

**Usage:**
```cpp
BroadcastShapeGenerator gen;
for (const auto& pair : gen) {
    Tensor<float> t1(pair.shape1);
    Tensor<float> t2(pair.shape2);
    auto result = t1 + t2; // Test broadcasting
}
```

### 3. RandomShapeGenerator
Generates random shapes with configurable parameters:

**Usage:**
```cpp
RandomShapeGenerator gen(42); // seed
auto shape = gen.generate(
    1,    // minRank
    4,    // maxRank
    1,    // minDim
    10    // maxDim
);
```

## Test Pattern

Each operator test follows this pattern:

1. **Basic test**: Simple fixed shape (e.g., {2, 3})
2. **VariousShapes test**: Iterate through all ShapeGenerator shapes
3. **Broadcasting test**: Iterate through BroadcastShapeGenerator pairs (for binary ops)
4. **Type-specific tests**: Test with different data types (float, int, uint)

## Running Tests

### Build and Run All Tests
```bash
cd /home/runner/work/vkml-compiler/vkml-compiler
cmake --preset x64-debug-linux
cmake --build build/x64-debug-linux
cd build/x64-debug-linux
ctest --output-on-failure --verbose
```

### Run Specific Test Suite
```bash
cd build/x64-debug-linux
./tests/arithmetic_ops_tests
./tests/bitwise_ops_tests
./tests/comparison_ops_tests
./tests/logical_ops_tests
./tests/other_ops_tests
```

### Run Individual CTest
```bash
cd build/x64-debug-linux
ctest -R ArithmeticOps --verbose
```

## Test Output

CTest-based tests output:
```
PASSED: Addition Basic
PASSED: Addition Various Shapes
PASSED: Addition Broadcasting
...
========================================
Tests run: 13
Failures: 0
All tests PASSED!
========================================
```

Failures show file, line, and diagnostic info:
```
FAILED: Division Int Basic
  /path/to/test.cpp:123: Expected 2 but got 3
```

## Adding New Tests

### For Existing Operator Categories

Add a new test function to the appropriate file:

```cpp
void test_new_feature() {
    TEST_BEGIN("New Feature");
    
    // Your test code
    Tensor<float> tensor({2, 3});
    auto result = some_operation(tensor);
    
    ASSERT_EQ(result.getShape().size(), 2);
    TEST_END();
}

int main() {
    // ... existing tests
    test_new_feature();  // Add here
    return TestRunner::report();
}
```

### For New Operator Categories

1. Create new file `NewCategory_tests.cpp`
2. Follow the pattern in existing files
3. Add to `CMakeLists.txt`:
```cmake
add_executable(new_category_tests NewCategory_tests.cpp)
target_link_libraries(new_category_tests PRIVATE vkml-compiler)
target_include_directories(new_category_tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME NewCategory COMMAND new_category_tests)
```

## Benefits of This Approach

1. **Comprehensive Coverage**: Shape generator tests ~50+ edge cases per operator automatically
2. **Organized**: Easy to find and maintain tests for specific operators
3. **Fast Iteration**: Run only the tests you need during development
4. **Broadcasting Validation**: Dedicated generator for broadcast scenarios
5. **CTest Integration**: Standard CMake test runner, no external dependencies
6. **Scalable**: Easy to add new operators or shape patterns

## Shape Examples

The generator includes unusual shapes that often reveal bugs:

- **{1}**: Scalar-like
- **{7, 11}**: Prime dimensions
- **{1, 100}**: Very wide matrix
- **{100, 1}**: Very tall matrix
- **{1, 3, 1}**: Broadcasting middle dimension
- **{2, 2, 2, 2, 2}**: 5D small cube
- **{1, 2, 3, 4, 5}**: Ascending dimensions
- **{5, 7, 11, 13}**: All different primes

These shapes stress-test:
- Boundary conditions (size 1)
- Memory layout (prime numbers)
- Broadcasting logic (mixed 1s)
- High-rank operations (5D+)
- Asymmetric operations (wide vs tall)
