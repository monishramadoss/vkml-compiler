#include "SPIRVPipeline.h"

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/Passes.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <array>
#include <unistd.h>

namespace vkml {

SPIRVPipeline::SPIRVPipeline(mlir::MLIRContext* context, mlir::ModuleOp module)
    : context_(context), module_(module) {
  // Ensure SPIR-V dialect is loaded
  context_->loadDialect<mlir::spirv::SPIRVDialect>();
}

void SPIRVPipeline::configureGPUToSPIRVPasses(mlir::PassManager& pm) {
  // Add canonicalization before conversion
  pm.addPass(mlir::createCanonicalizerPass());

  // Convert GPU dialect to SPIR-V dialect
  // Note: createConvertGPUToSPIRVPass() operates on builtin.module and internally
  // walks gpu.module ops, so we use addPass() not addNestedPass()
  pm.addPass(mlir::createConvertGPUToSPIRVPass());

  // Add canonicalization and cleanup after conversion
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  
  // Lower ABI attributes and update SPIR-V version/capability/extension
  pm.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVUpdateVCEPass());
}

bool SPIRVPipeline::compileToSPIRV() {
  // Create a pass manager for GPU to SPIR-V conversion
  mlir::PassManager pm(context_);
  configureGPUToSPIRVPasses(pm);

  // Run the pipeline
  if (mlir::failed(pm.run(module_))) {
    llvm::errs() << "Failed to convert GPU to SPIR-V\n";
    return false;
  }

  return true;
}

bool SPIRVPipeline::serializeToBinary() {
  if (!compiledBinary_) {
    compiledBinary_ = VulkanBuffer();
  }

  // Find SPIR-V modules in the MLIR module
  bool foundSpvModule = false;
  module_.walk([&](mlir::spirv::ModuleOp spvModule) {
    foundSpvModule = true;
    
    // Serialize the SPIR-V module to binary
    llvm::SmallVector<uint32_t, 0> binary;
    if (mlir::succeeded(mlir::spirv::serialize(spvModule, binary))) {
      compiledBinary_->spirvBinary.assign(binary.begin(), binary.end());
    } else {
      llvm::errs() << "Failed to serialize SPIR-V module\n";
    }
  });

  if (!foundSpvModule) {
    llvm::errs() << "No SPIR-V module found in the MLIR module\n";
    return false;
  }

  return !compiledBinary_->spirvBinary.empty();
}

void SPIRVPipeline::extractDescriptorSetInfo() {
  // Walk through the SPIR-V module to extract descriptor set information
  module_.walk([&](mlir::spirv::ModuleOp spvModule) {
    spvModule.walk([&](mlir::spirv::GlobalVariableOp globalVar) {
      // Check if this is a descriptor binding
      if (auto descriptorSet = globalVar->getAttrOfType<mlir::IntegerAttr>("descriptor_set")) {
        if (auto binding = globalVar->getAttrOfType<mlir::IntegerAttr>("binding")) {
          uint32_t setNum = descriptorSet.getInt();
          uint32_t bindingNum = binding.getInt();
          
          // Find or create descriptor set
          auto it = std::find_if(descriptorSets_.begin(), descriptorSets_.end(),
                                 [setNum](const DescriptorSetInfo& dsi) {
                                   return dsi.setNumber == setNum;
                                 });
          
          if (it == descriptorSets_.end()) {
            descriptorSets_.push_back({setNum, {}});
            it = descriptorSets_.end() - 1;
          }
          
          // Add binding information
          DescriptorBinding desc;
          desc.binding = bindingNum;
          desc.descriptorType = 7; // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER (typical for compute)
          desc.descriptorCount = 1;
          it->bindings.push_back(desc);
        }
      }
    });
  });
}

const VulkanBuffer* SPIRVPipeline::getBinary() const {
  if (compiledBinary_) {
    return &(*compiledBinary_);
  }
  return nullptr;
}

const std::vector<DescriptorSetInfo>& SPIRVPipeline::getDescriptorSets() const {
  return descriptorSets_;
}

bool SPIRVPipeline::validate() const {
  if (!compiledBinary_ || compiledBinary_->spirvBinary.empty()) {
    llvm::errs() << "No SPIR-V binary to validate\n";
    return false;
  }

  // Create a unique temporary file using mkstemp for security
  char tmpFileTemplate[] = "/tmp/spirv_validation_XXXXXX";
  int fd = mkstemp(tmpFileTemplate);
  if (fd == -1) {
    llvm::errs() << "Failed to create temporary file for validation: " 
                 << std::strerror(errno) << "\n";
    return false;
  }

  // Write binary to the temporary file
  ssize_t bytesWritten = write(fd, compiledBinary_->spirvBinary.data(), 
                                compiledBinary_->spirvBinary.size() * sizeof(uint32_t));
  close(fd);
  
  if (bytesWritten < 0) {
    llvm::errs() << "Failed to write SPIR-V binary: " << std::strerror(errno) << "\n";
    std::remove(tmpFileTemplate);
    return false;
  }

  // Run spirv-val on the binary with proper quoting to prevent injection
  // Using array to avoid shell injection
  std::array<const char*, 5> args = {
    "/bin/sh", "-c", nullptr, nullptr, nullptr
  };
  
  std::string cmd = "spirv-val --target-env vulkan1.0 '";
  cmd += tmpFileTemplate;
  cmd += "' 2>&1";
  args[2] = cmd.c_str();
  
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    llvm::errs() << "Failed to run spirv-val (is it installed?)\n";
    std::remove(tmpFileTemplate);
    return false;
  }

  char buffer[256];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe)) {
    result += buffer;
  }
  
  int exitCode = pclose(pipe);
  std::remove(tmpFileTemplate);

  if (exitCode != 0) {
    llvm::errs() << "SPIR-V validation failed:\n" << result << "\n";
    return false;
  }

  return true;
}

} // namespace vkml
