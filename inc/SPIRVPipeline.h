#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <optional>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace vkml {

// Represents a Vulkan-compatible SPIR-V binary buffer
struct VulkanBuffer {
  std::vector<uint32_t> spirvBinary;  // SPIR-V binary words
  size_t sizeInBytes() const { return spirvBinary.size() * sizeof(uint32_t); }
  const uint32_t* data() const { return spirvBinary.data(); }
};

// Descriptor set binding information
struct DescriptorBinding {
  uint32_t binding;
  uint32_t descriptorType;  // Storage buffer, uniform buffer, etc.
  uint32_t descriptorCount;
};

struct DescriptorSetInfo {
  uint32_t setNumber;
  std::vector<DescriptorBinding> bindings;
};

// Class to handle SPIR-V compilation pipeline
class SPIRVPipeline {
private:
  mlir::MLIRContext* context_;
  mlir::ModuleOp module_;
  std::optional<VulkanBuffer> compiledBinary_;
  std::vector<DescriptorSetInfo> descriptorSets_;

  // Configure the GPU to SPIR-V pass pipeline
  void configureGPUToSPIRVPasses(mlir::PassManager& pm);
  
  // Extract descriptor set information from the module
  void extractDescriptorSetInfo();

public:
  SPIRVPipeline(mlir::MLIRContext* context, mlir::ModuleOp module);

  // Run the compilation pipeline: GPU dialect → SPIR-V dialect
  bool compileToSPIRV();

  // Serialize SPIR-V to binary format
  bool serializeToBinary();

  // Get the compiled SPIR-V binary buffer
  const VulkanBuffer* getBinary() const;

  // Get descriptor set information
  const std::vector<DescriptorSetInfo>& getDescriptorSets() const;

  // Validate the SPIR-V binary (requires spirv-val)
  bool validate() const;

  // Get the SPIR-V module for inspection
  mlir::ModuleOp getModule() const { return module_; }
};

} // namespace vkml
