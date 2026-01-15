#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "SPIRVTargetEnv.h"

namespace vkml {

// Descriptor binding information for Vulkan descriptor sets
struct DescriptorBinding {
  uint32_t binding;
  uint32_t descriptorType; // VkDescriptorType (buffer=6, image=1, etc.)
  uint32_t descriptorCount;
  uint32_t stageFlags; // VkShaderStageFlags
};

// Descriptor set layout information
struct DescriptorSetLayout {
  uint32_t set;
  std::vector<DescriptorBinding> bindings;
};

// Push constant range information
struct PushConstantRange {
  uint32_t stageFlags;
  uint32_t offset;
  uint32_t size;
};

// Vulkan pipeline wrapper for SPIR-V binaries
class VulkanPipeline {
private:
  std::vector<uint32_t> spirvBinary_;
  std::vector<DescriptorSetLayout> descriptorSetLayouts_;
  std::vector<PushConstantRange> pushConstantRanges_;
  std::string entryPoint_;
  SPIRVTargetEnv targetEnv_;
  
  // Parse SPIR-V binary to extract descriptor set and push constant info
  void parseSPIRVReflection();

public:
  VulkanPipeline() : entryPoint_("main"), targetEnv_(SPIRVTargetEnv::getDefault()) {}
  
  // Initialize with SPIR-V binary
  explicit VulkanPipeline(const std::vector<uint32_t> &spirvBinary)
      : spirvBinary_(spirvBinary), entryPoint_("main"), 
        targetEnv_(SPIRVTargetEnv::getDefault()) {
    parseSPIRVReflection();
  }
  
  // Initialize with SPIR-V binary and target environment
  VulkanPipeline(const std::vector<uint32_t> &spirvBinary, const SPIRVTargetEnv &env)
      : spirvBinary_(spirvBinary), entryPoint_("main"), targetEnv_(env) {
    parseSPIRVReflection();
  }

  // Get SPIR-V binary data for VkShaderModuleCreateInfo
  const uint32_t* getShaderModuleData() const {
    return spirvBinary_.data();
  }

  // Get SPIR-V binary size in bytes
  size_t getShaderModuleSize() const {
    return spirvBinary_.size() * sizeof(uint32_t);
  }

  // Get SPIR-V binary size in uint32_t words
  size_t getShaderModuleWordCount() const {
    return spirvBinary_.size();
  }

  // Get the full SPIR-V binary
  const std::vector<uint32_t>& getSPIRVBinary() const {
    return spirvBinary_;
  }

  // Get descriptor set layouts
  const std::vector<DescriptorSetLayout>& getDescriptorSetLayouts() const {
    return descriptorSetLayouts_;
  }

  // Get push constant ranges
  const std::vector<PushConstantRange>& getPushConstantRanges() const {
    return pushConstantRanges_;
  }

  // Get entry point name
  const std::string& getEntryPoint() const {
    return entryPoint_;
  }

  // Set entry point name
  void setEntryPoint(const std::string &name) {
    entryPoint_ = name;
  }

  // Add a descriptor set layout manually (if not parsed from SPIR-V)
  void addDescriptorSetLayout(uint32_t set, const std::vector<DescriptorBinding> &bindings) {
    DescriptorSetLayout layout;
    layout.set = set;
    layout.bindings = bindings;
    descriptorSetLayouts_.push_back(layout);
  }

  // Add a push constant range manually
  void addPushConstantRange(uint32_t stageFlags, uint32_t offset, uint32_t size) {
    PushConstantRange range;
    range.stageFlags = stageFlags;
    range.offset = offset;
    range.size = size;
    pushConstantRanges_.push_back(range);
  }

  // Check if pipeline is valid
  bool isValid() const {
    return !spirvBinary_.empty() && spirvBinary_.size() >= 5; // Minimum SPIR-V header size
  }
  
  // Get target environment information
  const SPIRVTargetEnv& getTargetEnv() const {
    return targetEnv_;
  }
  
  // Set target environment information
  void setTargetEnv(const SPIRVTargetEnv &env) {
    targetEnv_ = env;
  }
};

// Implementation of SPIR-V reflection parsing
inline void VulkanPipeline::parseSPIRVReflection() {
  // Basic validation
  if (spirvBinary_.size() < 5) {
    return; // Invalid SPIR-V
  }

  // Check SPIR-V magic number
  if (spirvBinary_[0] != 0x07230203) {
    return; // Invalid magic number
  }

  // For now, we provide a basic implementation that doesn't parse the full SPIR-V
  // In a production system, you would parse the SPIR-V instructions to extract:
  // - OpDecorate with Binding and DescriptorSet decorations
  // - OpVariable with storage classes (Uniform, StorageBuffer, etc.)
  // - OpEntryPoint for entry point names
  
  // This is a placeholder - users can manually add descriptor set layouts
  // using addDescriptorSetLayout() if needed, or implement full SPIR-V parsing
}

} // namespace vkml
