# Pipeline Diagram

## Complete Flow: Linalg to SPIR-V to Vulkan

```
┌─────────────────────────────────────────────────────────────────┐
│                         C++ Tensor API                          │
│                                                                 │
│  Tensor<float> a({4, 4});                                      │
│  Tensor<float> b({4, 4});                                      │
│  auto result = a + b;                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Linalg Operations                            │
│                    (linalg dialect)                             │
│                                                                 │
│  func.func @main() {                                           │
│    %0 = tensor.empty() : tensor<4x4xf32>                       │
│    %1 = tensor.empty() : tensor<4x4xf32>                       │
│    %2 = linalg.generic {...} ins(%0, %1)                       │
│    return                                                       │
│  }                                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ runLinalgToGPU()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Bufferization                                │
│                    (memref dialect)                             │
│                                                                 │
│  tensor<4x4xf32> → memref<4x4xf32>                             │
│  Abstract tensors → Concrete memory buffers                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Parallel Loops                                  │
│                 (scf.parallel)                                  │
│                                                                 │
│  scf.parallel (%i, %j) = (0, 0) to (4, 4) {                    │
│    // parallel iteration space                                  │
│  }                                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GPU Operations                                │
│                   (gpu.launch)                                  │
│                                                                 │
│  gpu.launch blocks(...) threads(...) {                         │
│    // GPU execution model                                       │
│  }                                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                GPU Kernel Outlining                             │
│                                                                 │
│  gpu.module @kernel_module {                                   │
│    gpu.func @kernel(...) {                                     │
│      // extracted kernel code                                   │
│    }                                                            │
│  }                                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ runLinalgToSPIRV()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SPIR-V Operations                              │
│                  (spirv dialect)                                │
│                                                                 │
│  spirv.module {                                                │
│    spirv.func @kernel(...) {                                   │
│      // SPIR-V instructions                                     │
│    }                                                            │
│    spirv.EntryPoint "GLCompute" @kernel                        │
│  }                                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ serializeSPIRV()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SPIR-V Binary                                  │
│               (vector<uint32_t>)                                │
│                                                                 │
│  [0x07230203, 0x00010000, 0x000d000a, ...]                     │
│   ^magic      ^version   ^generator                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ createVulkanPipeline()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VulkanPipeline                                 │
│                                                                 │
│  • getSPIRVBinary()                                            │
│  • getShaderModuleData()  ──────────┐                         │
│  • getShaderModuleSize()            │                         │
│  • getDescriptorSetLayouts()        │                         │
│  • getEntryPoint()                  │                         │
└─────────────────────────────────────┼───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vulkan API                                   │
│                                                                 │
│  VkShaderModuleCreateInfo createInfo{};                        │
│  createInfo.codeSize = vulkanPipeline->getShaderModuleSize(); │
│  createInfo.pCode = vulkanPipeline->getShaderModuleData();    │
│  vkCreateShaderModule(device, &createInfo, nullptr, &module);  │
│                                                                 │
│  VkComputePipelineCreateInfo pipelineInfo{};                   │
│  pipelineInfo.stage.module = module;                           │
│  pipelineInfo.stage.pName = "main";                            │
│  vkCreateComputePipelines(device, ..., &pipeline);             │
│                                                                 │
│  vkCmdBindPipeline(cmdBuf, ..., pipeline);                     │
│  vkCmdDispatch(cmdBuf, ...);  ← Execute on GPU!                │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### Compiler Methods

- `runLinalgToGPU()` - Linalg → Bufferization → GPU
- `runLinalgToSPIRV()` - Complete pipeline to SPIR-V
- `serializeSPIRV()` - Extract binary from SPIR-V module
- `createVulkanPipeline()` - One-step pipeline + wrapper

### VulkanPipeline Class

Wraps SPIR-V binary with Vulkan metadata:
- Shader module data (binary pointer & size)
- Entry point name
- Descriptor set layouts
- Push constant ranges

### Usage Pattern

```cpp
// 1. Write high-level code
Tensor<float> a({4, 4}), b({4, 4});
auto result = a + b;

// 2. Compile to SPIR-V
auto vulkanPipeline = vkml::Compiler::getInstance()
                        ->createVulkanPipeline();

// 3. Use with Vulkan
VkShaderModuleCreateInfo info{};
info.codeSize = vulkanPipeline->getShaderModuleSize();
info.pCode = vulkanPipeline->getShaderModuleData();
vkCreateShaderModule(device, &info, nullptr, &shaderModule);
```

## Transformation Passes

1. **Canonicalization** (×3) - Simplify IR
2. **One-Shot Bufferization** - tensor → memref
3. **Linalg to Parallel Loops** - Extract parallelism
4. **GPU Mapping** - Map to GPU execution model
5. **Parallel Loop to GPU** - Create GPU operations
6. **GPU Kernel Outlining** - Extract kernels
7. **Inlining** (×2) - Inline functions
8. **CSE** - Common subexpression elimination
9. **GPU to SPIR-V** - Convert to SPIR-V dialect
10. **Serialization** - Binary encoding

## Benefits

✓ High-level tensor API in C++
✓ Automatic GPU code generation
✓ Direct Vulkan integration
✓ Type-safe operations
✓ JIT compilation
✓ Standard MLIR pipeline
