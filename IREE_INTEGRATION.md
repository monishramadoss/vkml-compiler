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

- `IREE_BUILD_COMPILER`: ON - Compiler components are enabled
- `IREE_BUILD_BUNDLED_LLVM`: OFF - Uses the existing vkml-compiler LLVM/MLIR build
- `IREE_BUILD_TESTS`: OFF - Tests are disabled
- `IREE_BUILD_DOCS`: OFF - Documentation build is disabled
- `IREE_BUILD_SAMPLES`: OFF - Sample projects are disabled
- `IREE_BUILD_PYTHON_BINDINGS`: OFF - Python bindings are disabled
- `IREE_BUILD_TRACY`: OFF - Tracy profiling is disabled
- `IREE_ENABLE_RUNTIME_TRACING`: OFF - Runtime tracing is disabled
- `IREE_ENABLE_COMPILER_TRACING`: OFF - Compiler tracing is disabled

### Compiler Components

The IREE compiler is built and available for use. Key components include:

- **IREE Compiler Dialects**: Access to IREE's MLIR dialects and transformations
- **Code Generation**: IREE's codegen pipelines for various backends
- **Compiler API**: Available through IREE compiler headers

### CMake Requirements

- Minimum CMake version: 3.21 (required by IREE)
- Generator: Ninja (recommended)
- Build type: Debug or Release

### Building with IREE

To build the project with IREE:

```bash
# Configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build the compiler components
ninja -C build
```

## Library Usage

To use IREE compiler components in your code:

1. Include the compiler API:
```cpp
#include "iree/compiler/..." // Specific headers as needed
```

2. Link against the IREE compiler libraries:
```cmake
target_link_libraries(your_target PRIVATE iree_compiler_...)
```

3. Add the include directory:
```cmake
target_include_directories(your_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler/src)
```

## Architecture

The integration follows this structure:
- vkml-compiler builds its own LLVM/MLIR
- IREE compiler uses the existing LLVM/MLIR (no duplicate builds)
- IREE compiler components are available for code generation and transformation

## Future Enhancements

Potential future improvements:
- Add IREE dialect support to vkml-compiler
- Integrate IREE code generation backends
- Add support for IREE's compiler transformations in the compilation pipeline

## References

- [IREE Project](https://iree.dev/)
- [IREE GitHub](https://github.com/iree-org/iree)
- [IREE Documentation](https://iree.dev/developers/)
