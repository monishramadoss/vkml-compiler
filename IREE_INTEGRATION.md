# IREE Integration

This document describes the IREE (Intermediate Representation Execution Environment) integration into the vkml-compiler project.

## What is IREE?

IREE is an MLIR-based end-to-end compiler and runtime that lowers Machine Learning models to a unified IR. It's designed to scale from datacenter to mobile and edge deployments.

## Integration Details

### Submodule Setup

IREE has been integrated as a git submodule:
- Repository: https://github.com/iree-org/iree.git
- Branch: main
- Location: `iree/` directory

To initialize the submodule and its dependencies:
```bash
git submodule update --init --recursive iree
```

### Build Configuration

The following IREE build options are configured in `CMakeLists.txt`:

- `IREE_BUILD_COMPILER`: OFF - Compiler is disabled to avoid conflicts with the existing LLVM/MLIR build
- `IREE_BUILD_TESTS`: OFF - Tests are disabled
- `IREE_BUILD_DOCS`: OFF - Documentation build is disabled
- `IREE_BUILD_SAMPLES`: OFF - Sample projects are disabled
- `IREE_BUILD_PYTHON_BINDINGS`: OFF - Python bindings are disabled
- `IREE_BUILD_TRACY`: OFF - Tracy profiling is disabled
- `IREE_ENABLE_RUNTIME_TRACING`: OFF - Runtime tracing is disabled
- `IREE_ENABLE_COMPILER_TRACING`: OFF - Compiler tracing is disabled

### Runtime Components

The IREE runtime is built and available for use. Key components include:

- **HAL Drivers**: local-sync, local-task, null, vulkan
- **Executable Loaders**: embedded-elf, system-library, vmvx-module
- **Runtime API**: Available through `iree/runtime/api.h`

### CMake Requirements

- Minimum CMake version: 3.21 (required by IREE)
- Generator: Ninja (recommended)
- Build type: Debug or Release

### Building with IREE

To build the project with IREE:

```bash
# Configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build the runtime
ninja -C build iree_runtime_unified

# Build the test application
ninja -C build test-iree-integration
```

### Testing the Integration

A test program `test_iree_integration.cpp` demonstrates basic IREE runtime usage:

```bash
# Run the integration test
./build/test-iree-integration
```

Expected output:
```
IREE runtime instance created successfully!
```

## Library Usage

To use IREE in your code:

1. Include the runtime API:
```cpp
#include "iree/runtime/api.h"
```

2. Link against the IREE runtime library:
```cmake
target_link_libraries(your_target PRIVATE iree_runtime_unified)
```

3. Add the include directory:
```cmake
target_include_directories(your_target PRIVATE ${CMAKE_SOURCE_DIR}/iree/runtime/src)
```

## Architecture

The integration follows this structure:
- vkml-compiler builds its own LLVM/MLIR (required for the compiler)
- IREE builds only its runtime (no compiler to avoid duplicate LLVM builds)
- Both can coexist and be used together in the same project

## Future Enhancements

Potential future improvements:
- Enable IREE compiler support with shared LLVM/MLIR
- Add IREE dialect support to vkml-compiler
- Integrate IREE code generation backends
- Add support for IREE's HAL (Hardware Abstraction Layer) in compilation pipeline

## References

- [IREE Project](https://iree.dev/)
- [IREE GitHub](https://github.com/iree-org/iree)
- [IREE Documentation](https://iree.dev/developers/)
