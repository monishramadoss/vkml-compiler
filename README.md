# VKML Compiler

[![Unit Tests](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml/badge.svg)](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml)

An MLIR-based compiler leveraging the TOSA (Tensor Operator Set Architecture) dialect for tensor operations and GPU acceleration.

## Features

- C++ Template-based Tensor API
- TOSA dialect integration with MLIR
- GPU transformation pipeline (TOSA → Linalg → GPU)
- Type-safe tensor operations with compile-time shape inference

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

The project includes comprehensive unit tests (35+ test cases) using Google Test framework, covering all tensor operations including arithmetic, bitwise, logical, comparison operations, and more.

```bash
# Build tests
cmake --build build/x64-debug-linux --target tensor_tests

# Run tests
cd build/x64-debug-linux
ctest --output-on-failure --verbose
```

For more details, see [tests/README.md](tests/README.md).

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
└── llvm/             # LLVM/MLIR submodule
```

## License

See LICENSE file for details.
