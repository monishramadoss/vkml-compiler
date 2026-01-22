# VKML Compiler

[![Unit Tests](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml/badge.svg)](https://github.com/monishramadoss/vkml-compiler/actions/workflows/tests.yml)

An MLIR-based compiler leveraging linalg and tensor dialects for tensor operations and GPU acceleration.

## Features

- C++ Template-based Tensor API
- Linalg dialect integration with MLIR
- GPU transformation pipeline (Linalg → Bufferization → GPU)
- **SPIR-V compilation pipeline with JIT support for Vulkan**
- **Vulkan-compatible binary buffer generation**
- **SPIR-V validation with spirv-val (optional)**
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

The project includes comprehensive unit tests (50+ test cases) using a custom test framework, covering all tensor operations including arithmetic, bitwise, logical, comparison operations, linalg named operations, and SPIR-V validation.

```bash
# Build tests
cmake --build build/x64-debug-linux --target tensor_tests

# Run tests
cd build/x64-debug-linux
ctest --output-on-failure --verbose
```

### SPIR-V Validation Tests

The project includes SPIR-V validation tests that verify the generated SPIR-V is valid for Vulkan. These tests require [SPIR-V Tools](https://github.com/KhronosGroup/SPIRV-Tools) to be installed:

```bash
# Install SPIR-V Tools (Ubuntu/Debian)
sudo apt-get install spirv-tools

# Run SPIR-V validation tests
./spirv_validation_tests
```

For more details, see [tests/README.md](tests/README.md).

## SPIR-V Pipeline Usage

The VKML compiler provides a complete pipeline to convert linalg operations to SPIR-V binary buffers that can be used directly with Vulkan:

```cpp
#include "Tensor.h"
#include "Compiler.h"
#include "SPIRVPipeline.h"

// Define tensor operations
Tensor<float> inputA({32, 32});
Tensor<float> inputB({32, 32});
auto result = inputA + inputB;

// Get compiler and run linalg to GPU conversion
auto compiler = vkml::Compiler::getInstance();
compiler->runLinalgToGPU();

// Convert to SPIR-V and serialize
vkml::SPIRVPipeline spirvPipeline(compiler->getContext(), compiler->getModule());
spirvPipeline.compileToSPIRV();
spirvPipeline.serializeToBinary();

// Get binary buffer for Vulkan
const vkml::VulkanBuffer* buffer = spirvPipeline.getBinary();

// Use with Vulkan
VkShaderModuleCreateInfo createInfo{};
createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
createInfo.codeSize = buffer->sizeInBytes();
createInfo.pCode = buffer->data();
vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);

// Validate SPIR-V (optional, requires spirv-val)
bool isValid = spirvPipeline.validate();
```

Run the example:
```bash
./spirv-example
```

## Compilation Pipeline

The VKML compiler uses a multi-stage transformation pipeline:

1. **Linalg Operations** - High-level tensor operations
2. **Bufferization** - Convert tensors to memrefs
3. **GPU Dialect** - Parallel operations mapped to GPU
4. **SPIR-V** - Vulkan-compatible binary shader code

```
Linalg → Bufferization → GPU → SPIR-V Binary
```

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
│   ├── SPIRVPipeline.h  # SPIR-V compilation pipeline
│   └── ScopedInserter.h
├── lib/              # Implementation files
│   ├── compiler.cpp
│   └── SPIRVPipeline.cpp
├── tests/            # Unit tests
│   ├── Tensor_tests.cpp
│   └── SPIRVValidation_tests.cpp
├── tools/            # MLIR tools
├── spirv_example.cpp # SPIR-V pipeline example
├── LINALG_OPS.md     # Linalg operations documentation
└── llvm/             # LLVM/MLIR submodule
```

## License

See LICENSE file for details.
