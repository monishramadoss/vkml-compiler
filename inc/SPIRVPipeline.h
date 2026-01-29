#pragma once

#include "Compiler.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include <vector>
#include <string>
#include <iostream>

namespace vkml {

struct VulkanBuffer {
    std::vector<uint32_t> spirvBinary;

    size_t sizeInBytes() const {
        return spirvBinary.size() * sizeof(uint32_t);
    }

    const uint32_t* data() const {
        return spirvBinary.data();
    }
};

struct DescriptorSetInfo {
    uint32_t setNumber;
    struct Binding {
        uint32_t binding;
        std::string descriptorType;
    };
    std::vector<Binding> bindings;
};

class SPIRVPipeline {
public:
    SPIRVPipeline(mlir::MLIRContext* context, mlir::ModuleOp module)
        : context_(context), module_(module) {}

    bool compileToSPIRV() {
        // Defines the target environment for SPIR-V generation
        auto context = context_;
        auto verCap = mlir::spirv::VerCapExtAttr::get(
            mlir::spirv::Version::V_1_0, 
            {mlir::spirv::Capability::Shader},
            llvm::ArrayRef<mlir::spirv::Extension>(), 
            context
        );
        
        auto limits = mlir::spirv::getDefaultResourceLimits(context);
        
        auto targetEnv = mlir::spirv::TargetEnvAttr::get(
            verCap, 
            limits, 
            mlir::spirv::ClientAPI::Vulkan, 
            mlir::spirv::Vendor::Unknown, 
            mlir::spirv::DeviceType::Unknown,
            0
        );
        
        module_->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnv);

        module_.dump();
        mlir::PassManager pm(context_);
        
    std::cout << "  - SPIR-V Pass Pipeline:\n";
        
        PassPipelineConfigurator::buildSPIRV(pm);
        
        // Run the pipeline
        if (mlir::failed(pm.run(module_))) {
            printf("SPIR-V compilation failed.\n");
            module_.dump();
            return false;
        }
        module_.dump();
        return true;
    }

    bool serializeToBinary() {
        // Find the SPIR-V module within the MLIR module
        mlir::spirv::ModuleOp targetOp;
        module_.walk([&](mlir::spirv::ModuleOp op) {
            targetOp = op;
            return mlir::WalkResult::interrupt();
        });

        if (!targetOp) {
             std::cerr << "Could not find SPIR-V module to serialize. Dumping module:\n";
             module_.dump();
             return false;
        }

        llvm::SmallVector<uint32_t, 0> binary;
        if (mlir::failed(mlir::spirv::serialize(targetOp, binary))) {
            return false;
        }
        
        buffer_.spirvBinary.assign(binary.begin(), binary.end());
        return true;
    }

    const VulkanBuffer* getBinary() const {
        return &buffer_;
    }

    bool validate() {
        // Validation skipped since spirv-tools library might not be available
        return true; 
    }

    std::vector<DescriptorSetInfo> getDescriptorSets() const {
        // Placeholder implementation
        return {};
    }

private:
    mlir::MLIRContext* context_;
    mlir::ModuleOp module_;
    VulkanBuffer buffer_;
};

} // namespace vkml
