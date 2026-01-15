#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace vkml {

// SPIR-V target environment configuration for hardware capabilities
struct SPIRVTargetEnv {
  // SPIR-V version (e.g., 0x00010000 for 1.0, 0x00010300 for 1.3, 0x00010600 for 1.6)
  uint32_t spirvVersion = 0x00010000;
  
  // Vulkan version (e.g., VK_API_VERSION_1_0, VK_API_VERSION_1_2, VK_API_VERSION_1_3)
  uint32_t vulkanVersion = 0x00100000; // VK_API_VERSION_1_0
  
  // Device capabilities
  struct Capabilities {
    bool supportsFloat16 = false;
    bool supportsFloat64 = false;
    bool supportsInt8 = false;
    bool supportsInt16 = false;
    bool supportsInt64 = false;
    bool supportsStorageBuffer16BitAccess = false;
    bool supportsStorageBuffer8BitAccess = false;
    bool supportsShaderSubgroupExtendedTypes = false;
    bool supportsVariablePointers = false;
    bool supportsVariablePointersStorageBuffer = false;
  } capabilities;
  
  // Resource limits
  struct Limits {
    uint32_t maxComputeWorkGroupInvocations = 128;
    uint32_t maxComputeWorkGroupSizeX = 128;
    uint32_t maxComputeWorkGroupSizeY = 128;
    uint32_t maxComputeWorkGroupSizeZ = 64;
    uint32_t subgroupSize = 32;
  } limits;
  
  // Extensions to enable
  std::vector<std::string> extensions;
  
  // Default constructor with conservative settings
  SPIRVTargetEnv() = default;
  
  // Helper to set common configurations
  static SPIRVTargetEnv getDefault() {
    return SPIRVTargetEnv();
  }
  
  static SPIRVTargetEnv getVulkan1_0() {
    SPIRVTargetEnv env;
    env.spirvVersion = 0x00010000; // SPIR-V 1.0
    env.vulkanVersion = 0x00100000; // Vulkan 1.0
    return env;
  }
  
  static SPIRVTargetEnv getVulkan1_2() {
    SPIRVTargetEnv env;
    env.spirvVersion = 0x00010500; // SPIR-V 1.5
    env.vulkanVersion = 0x00102000; // Vulkan 1.2
    env.capabilities.supportsFloat16 = true;
    env.capabilities.supportsInt8 = true;
    env.capabilities.supportsStorageBuffer16BitAccess = true;
    env.capabilities.supportsStorageBuffer8BitAccess = true;
    env.capabilities.supportsShaderSubgroupExtendedTypes = true;
    return env;
  }
  
  static SPIRVTargetEnv getVulkan1_3() {
    SPIRVTargetEnv env;
    env.spirvVersion = 0x00010600; // SPIR-V 1.6
    env.vulkanVersion = 0x00103000; // Vulkan 1.3
    env.capabilities.supportsFloat16 = true;
    env.capabilities.supportsFloat64 = true;
    env.capabilities.supportsInt8 = true;
    env.capabilities.supportsInt16 = true;
    env.capabilities.supportsInt64 = true;
    env.capabilities.supportsStorageBuffer16BitAccess = true;
    env.capabilities.supportsStorageBuffer8BitAccess = true;
    env.capabilities.supportsShaderSubgroupExtendedTypes = true;
    env.capabilities.supportsVariablePointers = true;
    env.capabilities.supportsVariablePointersStorageBuffer = true;
    return env;
  }
};

} // namespace vkml
