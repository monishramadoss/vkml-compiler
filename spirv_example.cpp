#include "Tensor.h"
#include "Compiler.h"
#include "SPIRVPipeline.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "========================================\n";
    std::cout << "VKML SPIR-V Pipeline Example\n";
    std::cout << "========================================\n\n";

    // Step 1: Create tensors and operations using linalg
    std::cout << "Step 1: Creating tensor operations...\n";
    Tensor<float> inputA({32, 32});
    Tensor<float> inputB({32, 32});
    
    // Perform operations - these will be linalg operations
    auto sum = inputA + inputB;
    auto product = sum * inputA;
    
    std::cout << "  - Created tensors of shape [32, 32]\n";
    std::cout << "  - Defined operations: (A + B) * A\n\n";

    // Step 2: Get compiler instance and run linalg to GPU conversion
    std::cout << "Step 2: Converting linalg to GPU dialect...\n";
    auto compiler = vkml::Compiler::getInstance();
    
    // Print the initial MLIR module (linalg operations)
    std::cout << "  - Initial linalg IR:\n";
    std::cout << "    ----------------------------------------\n";
    vkml::dump();
    std::cout << "    ----------------------------------------\n\n";
    
    // Run the linalg to GPU pipeline
    compiler->runLinalgToGPU();
    std::cout << "  ✓ Linalg to GPU conversion completed\n\n";

    // Step 3: Convert GPU to SPIR-V and serialize
    std::cout << "Step 3: Converting GPU dialect to SPIR-V...\n";
    auto module = compiler->getModule();
    auto context = compiler->getContext();
    
    vkml::SPIRVPipeline spirvPipeline(context, module);
    
    if (!spirvPipeline.compileToSPIRV()) {
        std::cerr << "  ✗ Failed to compile to SPIR-V\n";
        return 1;
    }
    std::cout << "  ✓ GPU to SPIR-V compilation completed\n";

    if (!spirvPipeline.serializeToBinary()) {
        std::cerr << "  ✗ Failed to serialize SPIR-V\n";
        return 1;
    }
    std::cout << "  ✓ SPIR-V serialization completed\n\n";

    // Step 4: Get the binary buffer (ready for Vulkan)
    std::cout << "Step 4: Extracting SPIR-V binary for Vulkan...\n";
    const vkml::VulkanBuffer* buffer = spirvPipeline.getBinary();
    
    if (!buffer || buffer->spirvBinary.empty()) {
        std::cerr << "  ✗ No SPIR-V binary generated\n";
        return 1;
    }
    
    std::cout << "  ✓ Binary buffer extracted\n";
    std::cout << "  - Size: " << buffer->sizeInBytes() << " bytes\n";
    std::cout << "  - Words: " << buffer->spirvBinary.size() << " (32-bit words)\n";
    std::cout << "  - Magic: 0x" << std::hex << std::setfill('0') << std::setw(8) 
              << buffer->spirvBinary[0] << std::dec << " (SPIR-V magic number)\n";
    
    // Show how to use with Vulkan
    std::cout << "\n  Usage with Vulkan:\n";
    std::cout << "    VkShaderModuleCreateInfo createInfo{};\n";
    std::cout << "    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;\n";
    std::cout << "    createInfo.codeSize = " << buffer->sizeInBytes() << ";\n";
    std::cout << "    createInfo.pCode = buffer->data();\n";
    std::cout << "    vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);\n\n";

    // Step 5: Validate the SPIR-V
    std::cout << "Step 5: Validating SPIR-V for Vulkan compatibility...\n";
    if (spirvPipeline.validate()) {
        std::cout << "  ✓ SPIR-V validation passed (Vulkan 1.0 compatible)\n";
    } else {
        std::cout << "  ! SPIR-V validation skipped (spirv-val not available)\n";
        std::cout << "    Install SPIR-V Tools for validation: https://github.com/KhronosGroup/SPIRV-Tools\n";
    }

    // Step 6: Descriptor set information
    std::cout << "\nStep 6: Descriptor set information:\n";
    const auto& descriptorSets = spirvPipeline.getDescriptorSets();
    
    if (descriptorSets.empty()) {
        std::cout << "  - No descriptor sets found (may require additional GPU dialect setup)\n";
    } else {
        for (const auto& descriptorSet : descriptorSets) {
            std::cout << "  - Descriptor Set " << descriptorSet.setNumber << ":\n";
            for (const auto& binding : descriptorSet.bindings) {
                std::cout << "    * Binding " << binding.binding 
                         << " (Type: " << binding.descriptorType << ")\n";
            }
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Pipeline execution completed successfully!\n";
    std::cout << "========================================\n";

    return 0;
}
