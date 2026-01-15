# Linalg to SPIR-V Pipeline

This document describes the pipeline that converts linalg operations to SPIR-V binaries suitable for Vulkan integration.

## Overview

The VKML compiler provides a complete pipeline that transforms high-level tensor operations written in C++ into SPIR-V binary code that can be used with Vulkan compute shaders. This enables developers to write tensor computations at a high level while generating efficient GPU code.

The pipeline supports configurable target environments, allowing drivers to specify hardware capabilities and generate optimized SPIR-V code. See [SPIRV_TARGET_ENV.md](SPIRV_TARGET_ENV.md) for details on target environment configuration.

## Pipeline Stages

### 1. Linalg Operations (Input)

The pipeline starts with linalg operations created through the Tensor API:

```cpp
Tensor<float> a({4, 4});
Tensor<float> b({4, 4});
auto result = a + b;  // Creates linalg operations
```

At this stage, the MLIR module contains operations in the:
- **linalg dialect** - High-level linear algebra operations
- **tensor dialect** - Tensor types and operations
- **arith dialect** - Arithmetic operations

### 2. Bufferization

The first transformation converts tensor operations to buffer operations:
- `tensor` types → `memref` types
- Operations work on memory buffers instead of abstract tensors
- Enables in-place updates and memory reuse

### 3. Linalg to Parallel Loops

Linalg operations are lowered to parallel loop constructs:
- `linalg.generic` → `scf.parallel` loops
- Exposes parallelism for GPU mapping

### 4. Parallel Loops to GPU

Parallel loops are mapped to GPU execution:
- `scf.parallel` → GPU kernel launches
- Work divided into blocks and threads
- Creates `gpu.launch` operations

### 5. GPU Kernel Outlining

GPU kernels are extracted into separate modules:
- Kernel code moved to `gpu.module` operations
- Prepares for target-specific code generation

### 6. GPU to SPIR-V

GPU operations are converted to SPIR-V dialect:
- `gpu.module` → `spirv.module`
- GPU memory spaces → SPIR-V storage classes
- GPU builtin values → SPIR-V builtin variables

### 7. SPIR-V Serialization (Output)

Finally, the SPIR-V module is serialized to binary format:
- SPIR-V text representation → binary words (uint32_t)
- Binary is ready for Vulkan shader module creation
- Can be passed directly to `VkShaderModuleCreateInfo`

## API Usage

### Basic Pipeline

```cpp
#include "Compiler.h"
#include "VulkanPipeline.h"

// Create operations
Tensor<float> a({4, 4});
Tensor<float> b({4, 4});
auto result = a + b;

// Get compiler and run pipeline
auto compiler = vkml::Compiler::getInstance();
auto vulkanPipeline = compiler->createVulkanPipeline();
```

### Advanced Pipeline Control

For more control, you can run pipeline stages individually:

```cpp
auto compiler = vkml::Compiler::getInstance();

// Run linalg to GPU pipeline
compiler->runLinalgToGPU();

// Dump intermediate representation
vkml::dump();

// Continue to SPIR-V
compiler->runLinalgToSPIRV();

// Serialize SPIR-V
auto spirvBinary = compiler->serializeSPIRV();
```

## Vulkan Integration

### Creating Shader Modules

The VulkanPipeline class provides data in the format needed by Vulkan:

```cpp
auto vulkanPipeline = compiler->createVulkanPipeline();

// Create Vulkan shader module
VkShaderModuleCreateInfo createInfo{};
createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
createInfo.codeSize = vulkanPipeline->getShaderModuleSize();
createInfo.pCode = vulkanPipeline->getShaderModuleData();

VkShaderModule shaderModule;
vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
```

### Descriptor Sets

The VulkanPipeline can store descriptor set layout information:

```cpp
// Add descriptor set layout
vulkanPipeline->addDescriptorSetLayout(0, {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}
});

// Create Vulkan descriptor set layout
for (const auto& setLayout : vulkanPipeline->getDescriptorSetLayouts()) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const auto& binding : setLayout.bindings) {
        VkDescriptorSetLayoutBinding vkBinding{};
        vkBinding.binding = binding.binding;
        vkBinding.descriptorType = (VkDescriptorType)binding.descriptorType;
        vkBinding.descriptorCount = binding.descriptorCount;
        vkBinding.stageFlags = binding.stageFlags;
        bindings.push_back(vkBinding);
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();
    
    VkDescriptorSetLayout vkLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &vkLayout);
}
```

### Creating Compute Pipeline

```cpp
VkPipelineShaderStageCreateInfo shaderStage{};
shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
shaderStage.module = shaderModule;
shaderStage.pName = vulkanPipeline->getEntryPoint().c_str();

VkComputePipelineCreateInfo pipelineInfo{};
pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
pipelineInfo.stage = shaderStage;
pipelineInfo.layout = pipelineLayout;

VkPipeline computePipeline;
vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, 
                        nullptr, &computePipeline);
```

## VulkanPipeline Class Reference

### Methods

#### getShaderModuleData()
Returns a pointer to the SPIR-V binary data (`const uint32_t*`)
- Use with `VkShaderModuleCreateInfo::pCode`

#### getShaderModuleSize()
Returns the size of the SPIR-V binary in bytes (`size_t`)
- Use with `VkShaderModuleCreateInfo::codeSize`

#### getShaderModuleWordCount()
Returns the number of 32-bit words in the SPIR-V binary

#### getSPIRVBinary()
Returns the full SPIR-V binary as `std::vector<uint32_t>`

#### getDescriptorSetLayouts()
Returns all descriptor set layouts as `std::vector<DescriptorSetLayout>`

#### getEntryPoint()
Returns the entry point name (default: "main")

#### addDescriptorSetLayout(set, bindings)
Manually adds a descriptor set layout

#### isValid()
Checks if the pipeline contains a valid SPIR-V binary

## Example Workflow

1. **Write high-level code** using Tensor API
2. **Run pipeline** with `createVulkanPipeline()`
3. **Create Vulkan shader module** using SPIR-V binary
4. **Configure descriptor sets** for input/output buffers
5. **Create compute pipeline** linking shader and layouts
6. **Execute** on GPU via Vulkan command buffers

## Benefits

- **High-level API**: Write tensor operations in C++
- **Automatic optimization**: MLIR applies optimizations during lowering
- **GPU acceleration**: Automatically generates parallel GPU code
- **Vulkan integration**: Binary is directly usable with Vulkan
- **JIT compilation**: Generate code at runtime based on tensor shapes

## Supported Operations

All Tensor operations are supported:
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`
- Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`
- Linalg named ops: `matmul`, `dot`, `transpose`, `fill`, etc.

## Technical Details

### SPIR-V Format

The output SPIR-V binary follows the standard format:
- Word 0: Magic number (0x07230203)
- Word 1: Version number
- Word 2: Generator ID
- Word 3: Bound (maximum ID used)
- Word 4: Reserved (0)
- Word 5+: Instructions

### Memory Layout

The pipeline assumes:
- Row-major memory layout
- Contiguous buffer storage
- GPU accessible memory (device or shared)

### Shader Entry Point

The default entry point is "main". This can be changed:
```cpp
vulkanPipeline->setEntryPoint("compute_main");
```

## See Also

- [LINALG_OPS.md](../LINALG_OPS.md) - Linalg operations documentation
- [examples/vulkan_pipeline_example.cpp](../examples/vulkan_pipeline_example.cpp) - Complete example
- [README.md](../README.md) - Project overview
