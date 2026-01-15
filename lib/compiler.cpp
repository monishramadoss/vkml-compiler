#include "Compiler.h"
#include "VulkanPipeline.h"

namespace vkml {

std::shared_ptr<VulkanPipeline> Compiler::createVulkanPipeline() {
  // Run the full pipeline to SPIR-V
  runLinalgToSPIRV();
  
  // Serialize the SPIR-V binary
  auto spirvBinary = serializeSPIRV();
  
  // Create and return the VulkanPipeline
  return std::make_shared<VulkanPipeline>(spirvBinary);
}

} // namespace vkml
