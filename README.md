# VKML Compiler

[![Unit Tests](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml/badge.svg)](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml)

An MLIR-based compiler leveraging linalg and tensor dialects for tensor operations and GPU acceleration.

## Features

- C++ Template-based Tensor API
- Linalg dialect integration with MLIR
- GPU transformation pipeline (Linalg → Bufferization → GPU)
- Type-safe tensor operations with compile-time shape inference
- Comprehensive linalg named operations:
  - Matrix operations: `matmul`, `dot`, `matvec`, `vecmat`, `batch_matmul`
  - Reductions: `sum`, `max`, `min`
  - Structural: `transpose`, `fill`, `copy`, `map`
  - Element-wise: arithmetic, bitwise, logical, comparison operators

## Building

### Prerequisites

- CMake 3.16 or higher
- C++20 compatible compiler
- Ninja build system
- ccache (optional, for faster rebuilds)
- mold (optional, for faster linking)

### Build Instructions

```bash
# Clone with submodules
git clone --recursive https://github.com/monishramadoss/vkml-compiler.git
cd vkml-compiler

# Configure
cmake --preset x64-debug-linux

# Build
cmake --build build/x64-debug-linux
```

## Testing

The project includes comprehensive unit tests (50+ test cases) using Google Test framework, covering all tensor operations including arithmetic, bitwise, logical, comparison operations, linalg named operations, and more.

```bash
# Build tests
cmake --build build/x64-debug-linux --target tensor_tests

# Run tests
cd build/x64-debug-linux
ctest --output-on-failure --verbose
```

For more details, see [tests/README.md](tests/README.md).

## API Documentation

For detailed documentation on linalg named operations, see [LINALG_OPS.md](LINALG_OPS.md).

## Continuous Integration

Unit tests are automatically run on GitHub Actions for all pull requests and pushes to main/develop branches.

## Project Structure

```
.
├── inc/              # Header files
│   ├── Compiler.h    # Main compiler interface
│   ├── Tensor.h      # Tensor template class
│   └── ScopedInserter.h
├── lib/              # Implementation files
│   └── compiler.cpp
├── tests/            # Unit tests
│   └── Tensor_tests.cpp
├── tools/            # MLIR tools
├── LINALG_OPS.md     # Linalg operations documentation
└── llvm/             # LLVM/MLIR submodule
```

## License

See LICENSE file for details.
