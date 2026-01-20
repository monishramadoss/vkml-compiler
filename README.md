# VKML Compiler

[![Unit Tests](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml/badge.svg)](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml)

An MLIR-based compiler leveraging linalg and tensor dialects for tensor operations and GPU acceleration.

## Features

- C++ Template-based Tensor API
- Linalg dialect integration with MLIR
- Complete pipeline: Linalg → Bufferization → GPU → SPIR-V
- SPIR-V binary generation for Vulkan integration
- VulkanPipeline wrapper for shader modules and descriptor sets
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

- [LINALG_OPS.md](LINALG_OPS.md) - Linalg named operations
- [SPIRV_PIPELINE.md](SPIRV_PIPELINE.md) - Linalg to SPIR-V pipeline for Vulkan
- [SPIRV_TARGET_ENV.md](SPIRV_TARGET_ENV.md) - SPIR-V target environment configuration

## Examples

See the [examples/](examples/) directory for usage examples, including:
- `vulkan_pipeline_example.cpp` - Complete pipeline from linalg to SPIR-V for Vulkan
- `spirv_validation_example.cpp` - SPIR-V validation using spirv-val

## Scripts

See the [scripts/](scripts/) directory for utility scripts:
- `validate_spirv.py` - Python script to validate SPIR-V binaries with spirv-val
- `validate_spirv.sh` - Bash script to validate SPIR-V binaries

## Continuous Integration

Unit tests are automatically run on GitHub Actions for all pull requests and pushes to main/develop branches.

## Project Structure

```
.
├── inc/                  # Header files
│   ├── Compiler.h        # Main compiler interface
│   ├── Tensor.h          # Tensor template class
│   ├── VulkanPipeline.h  # Vulkan integration wrapper
│   ├── SPIRVTargetEnv.h  # SPIR-V target environment
│   └── ScopedInserter.h
├── lib/                  # Implementation files
│   └── compiler.cpp
├── examples/             # Usage examples
│   ├── vulkan_pipeline_example.cpp
│   └── spirv_validation_example.cpp
├── scripts/              # Utility scripts
│   ├── validate_spirv.py
│   └── validate_spirv.sh
├── tests/                # Unit tests
│   └── Tensor_tests.cpp
├── tools/                # MLIR tools
├── LINALG_OPS.md         # Linalg operations documentation
├── SPIRV_PIPELINE.md     # SPIR-V pipeline documentation
├── SPIRV_TARGET_ENV.md   # Target environment documentation
└── llvm/                 # LLVM/MLIR submodule
```

## License

See LICENSE file for details.
