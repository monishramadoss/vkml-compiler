# SPIR-V Target Environment Configuration

## Overview

The SPIR-V target environment configuration allows drivers to specify hardware capabilities and target ABI settings when compiling linalg operations to SPIR-V. This ensures the generated SPIR-V code is compatible with the target hardware and uses appropriate features.

## SPIRVTargetEnv Structure

The `SPIRVTargetEnv` structure encapsulates all target environment settings:

```cpp
struct SPIRVTargetEnv {
  // SPIR-V and Vulkan versions
  uint32_t spirvVersion;
  uint32_t vulkanVersion;
  
  // Device capabilities
  struct Capabilities {
    bool supportsFloat16;
    bool supportsFloat64;
    bool supportsInt8;
    bool supportsInt16;
    bool supportsInt64;
    bool supportsStorageBuffer16BitAccess;
    bool supportsStorageBuffer8BitAccess;
    bool supportsShaderSubgroupExtendedTypes;
    bool supportsVariablePointers;
    bool supportsVariablePointersStorageBuffer;
  } capabilities;
  
  // Resource limits
  struct Limits {
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSizeX;
    uint32_t maxComputeWorkGroupSizeY;
    uint32_t maxComputeWorkGroupSizeZ;
    uint32_t subgroupSize;
  } limits;
  
  // Extensions to enable
  std::vector<std::string> extensions;
};
```

## Usage

### 1. Using Predefined Configurations

```cpp
auto compiler = vkml::Compiler::getInstance();

// Set Vulkan 1.0 target (most conservative)
compiler->setVulkan1_0Target();

// Set Vulkan 1.2 target (moderate features)
compiler->setVulkan1_2Target();

// Set Vulkan 1.3 target (latest features)
compiler->setVulkan1_3Target();
```

### 2. Custom Configuration from Hardware

```cpp
auto compiler = vkml::Compiler::getInstance();

// Query hardware capabilities (pseudocode)
VkPhysicalDeviceProperties deviceProps;
VkPhysicalDeviceFeatures deviceFeatures;
vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

// Create custom target environment based on hardware
auto targetEnv = vkml::SPIRVTargetEnv::getVulkan1_2();
targetEnv.capabilities.supportsFloat64 = deviceFeatures.shaderFloat64;
targetEnv.capabilities.supportsInt64 = deviceFeatures.shaderInt64;
targetEnv.limits.maxComputeWorkGroupSizeX = deviceProps.limits.maxComputeWorkGroupSize[0];
targetEnv.limits.maxComputeWorkGroupSizeY = deviceProps.limits.maxComputeWorkGroupSize[1];
targetEnv.limits.maxComputeWorkGroupSizeZ = deviceProps.limits.maxComputeWorkGroupSize[2];
targetEnv.limits.maxComputeWorkGroupInvocations = deviceProps.limits.maxComputeWorkGroupInvocations;

// Set the custom configuration
compiler->setSPIRVTargetEnv(targetEnv);
```

### 3. Modifying Existing Configuration

```cpp
auto compiler = vkml::Compiler::getInstance();

// Get current configuration
auto& targetEnv = compiler->getSPIRVTargetEnv();

// Modify specific capabilities
targetEnv.capabilities.supportsFloat64 = true;
targetEnv.limits.subgroupSize = 64;

// Configuration is automatically used
```

### 4. Accessing Target Environment from Pipeline

```cpp
auto vulkanPipeline = compiler->createVulkanPipeline();

// Get target environment used for compilation
const auto& env = vulkanPipeline->getTargetEnv();

std::cout << "Compiled for SPIR-V version: " 
          << std::hex << env.spirvVersion << std::dec << "\n";
std::cout << "Compiled for Vulkan version: " 
          << std::hex << env.vulkanVersion << std::dec << "\n";
std::cout << "Float64 support: " 
          << (env.capabilities.supportsFloat64 ? "yes" : "no") << "\n";
```

## Predefined Configurations

### Vulkan 1.0 (Default)
- **SPIR-V Version**: 1.0 (0x00010000)
- **Vulkan Version**: 1.0 (0x00100000)
- **Features**: Basic compute shader support
- **Use case**: Maximum compatibility

### Vulkan 1.2
- **SPIR-V Version**: 1.5 (0x00010500)
- **Vulkan Version**: 1.2 (0x00102000)
- **Features**:
  - Float16 and Int8 support
  - Storage buffer 8-bit and 16-bit access
  - Shader subgroup extended types
- **Use case**: Modern hardware with enhanced features

### Vulkan 1.3
- **SPIR-V Version**: 1.6 (0x00010600)
- **Vulkan Version**: 1.3 (0x00103000)
- **Features**:
  - All Vulkan 1.2 features
  - Float64, Int16, and Int64 support
  - Variable pointers
- **Use case**: Latest hardware with full feature set

## Complete Example

```cpp
#include "Compiler.h"
#include "VulkanPipeline.h"
#include "SPIRVTargetEnv.h"

// Query Vulkan device capabilities
VkPhysicalDeviceProperties deviceProps;
VkPhysicalDeviceFeatures deviceFeatures;
VkPhysicalDeviceVulkan12Features vulkan12Features{};
vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

VkPhysicalDeviceFeatures2 features2{};
features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
features2.pNext = &vulkan12Features;

vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

// Configure SPIR-V target environment from device capabilities
auto compiler = vkml::Compiler::getInstance();
auto targetEnv = vkml::SPIRVTargetEnv::getDefault();

// Set version based on device API version
if (deviceProps.apiVersion >= VK_API_VERSION_1_3) {
    targetEnv = vkml::SPIRVTargetEnv::getVulkan1_3();
} else if (deviceProps.apiVersion >= VK_API_VERSION_1_2) {
    targetEnv = vkml::SPIRVTargetEnv::getVulkan1_2();
} else {
    targetEnv = vkml::SPIRVTargetEnv::getVulkan1_0();
}

// Fill in actual device capabilities
targetEnv.capabilities.supportsFloat16 = vulkan12Features.shaderFloat16;
targetEnv.capabilities.supportsFloat64 = deviceFeatures.shaderFloat64;
targetEnv.capabilities.supportsInt8 = vulkan12Features.shaderInt8;
targetEnv.capabilities.supportsInt16 = deviceFeatures.shaderInt16;
targetEnv.capabilities.supportsInt64 = deviceFeatures.shaderInt64;
targetEnv.capabilities.supportsStorageBuffer16BitAccess = 
    vulkan12Features.storageBuffer16BitAccess;
targetEnv.capabilities.supportsStorageBuffer8BitAccess = 
    vulkan12Features.storageBuffer8BitAccess;

// Fill in device limits
targetEnv.limits.maxComputeWorkGroupInvocations = 
    deviceProps.limits.maxComputeWorkGroupInvocations;
targetEnv.limits.maxComputeWorkGroupSizeX = 
    deviceProps.limits.maxComputeWorkGroupSize[0];
targetEnv.limits.maxComputeWorkGroupSizeY = 
    deviceProps.limits.maxComputeWorkGroupSize[1];
targetEnv.limits.maxComputeWorkGroupSizeZ = 
    deviceProps.limits.maxComputeWorkGroupSize[2];

// Set the configured environment
compiler->setSPIRVTargetEnv(targetEnv);

// Create tensor operations
Tensor<float> a({1024, 1024});
Tensor<float> b({1024, 1024});
auto result = a * b;

// Compile to SPIR-V with hardware-appropriate settings
auto vulkanPipeline = compiler->createVulkanPipeline();

// Use the compiled shader
VkShaderModuleCreateInfo createInfo{};
createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
createInfo.codeSize = vulkanPipeline->getShaderModuleSize();
createInfo.pCode = vulkanPipeline->getShaderModuleData();

VkShaderModule shaderModule;
vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
```

## Benefits

1. **Hardware Optimization**: Compiler can generate code optimized for specific hardware capabilities
2. **Version Compatibility**: Ensures SPIR-V is compatible with target Vulkan version
3. **Feature Detection**: Enables/disables features based on hardware support
4. **Resource Limits**: Respects device limits for work group sizes and invocations
5. **Explicit Control**: Driver has full control over target environment

## Future Enhancements

- Automatic capability detection from SPIR-V module
- Extension management and validation
- Profile-based configurations (mobile, desktop, high-performance)
- Optimization level control per capability
- Dynamic feature selection based on operation requirements
