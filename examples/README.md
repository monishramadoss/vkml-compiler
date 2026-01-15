# VKML Compiler Examples

This directory contains example programs demonstrating various features of the VKML compiler.

## Examples

### vulkan_pipeline_example.cpp

Demonstrates the complete pipeline from linalg operations to SPIR-V binary suitable for Vulkan:

1. **Tensor Creation**: Creates tensor operations using the Tensor API
2. **Linalg Operations**: Shows the initial MLIR representation in linalg dialect
3. **Pipeline Execution**: Runs the transformation pipeline (Linalg → GPU → SPIR-V)
4. **SPIR-V Binary**: Extracts the compiled SPIR-V binary
5. **Vulkan Integration**: Shows how to use the binary with Vulkan API

The example demonstrates:
- Creating a VulkanPipeline from linalg operations
- Extracting SPIR-V binary data for VkShaderModuleCreateInfo
- Configuring descriptor set layouts
- Preparing data for Vulkan compute pipeline

## Building

To build the examples, ensure the project is configured with examples enabled:

```bash
# Configure
cmake --preset x64-debug-linux

# Build the example
cmake --build build/x64-debug-linux --target vulkan_pipeline_example

# Run the example
./build/x64-debug-linux/examples/vulkan_pipeline_example
```

## Usage

The examples show how the VKML compiler enables:

1. **High-level tensor operations** - Write operations using C++ tensor API
2. **Automatic lowering** - Pipeline automatically converts to GPU code
3. **SPIR-V generation** - JIT compilation to SPIR-V binary format
4. **Vulkan integration** - Binary is ready for use with Vulkan API

This allows developers to write high-level tensor computations that are automatically compiled to efficient GPU code for Vulkan.
