#include "Tensor.h"
#include "Compiler.h"
#include "VulkanPipeline.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== VKML SPIR-V Pipeline Example ===\n\n";

    // Create tensors and perform linalg operations
    std::cout << "1. Creating tensors and operations...\n";
    Tensor<float> tensor_a({4, 4});
    Tensor<float> tensor_b({4, 4});
    
    // Perform operations (will generate linalg operations)
    auto result = tensor_a + tensor_b;
    auto result2 = result * tensor_a;
    
    std::cout << "   Created tensor operations\n\n";

    // Get the compiler instance
    auto compiler = vkml::Compiler::getInstance();
    
    // Dump the initial MLIR module (linalg dialect)
    std::cout << "2. Initial MLIR Module (Linalg dialect):\n";
    std::cout << "   ----------------------------------------\n";
    vkml::dump();
    std::cout << "\n";

    // Create a VulkanPipeline (this runs the full pipeline and serializes)
    std::cout << "3. Running pipeline: Linalg -> GPU -> SPIR-V...\n";
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    if (!vulkanPipeline->isValid()) {
        std::cerr << "   ERROR: Failed to create valid SPIR-V binary\n";
        return 1;
    }
    std::cout << "   Pipeline completed successfully!\n\n";

    // Display SPIR-V information
    std::cout << "4. SPIR-V Binary Information:\n";
    std::cout << "   Binary size: " << vulkanPipeline->getShaderModuleSize() 
              << " bytes (" << vulkanPipeline->getShaderModuleWordCount() << " words)\n";
    std::cout << "   Entry point: " << vulkanPipeline->getEntryPoint() << "\n";
    
    // Display first few words of SPIR-V binary (header)
    const auto& binary = vulkanPipeline->getSPIRVBinary();
    if (binary.size() >= 5) {
        std::cout << "   SPIR-V Header:\n";
        std::cout << "     Magic:   0x" << std::hex << std::setfill('0') << std::setw(8) << binary[0] << std::dec << "\n";
        std::cout << "     Version: " << ((binary[1] >> 16) & 0xFF) << "." << ((binary[1] >> 8) & 0xFF) << "\n";
        std::cout << "     Generator: 0x" << std::hex << std::setfill('0') << std::setw(8) << binary[2] << std::dec << "\n";
        std::cout << "     Bound: " << binary[3] << "\n";
    }
    std::cout << "\n";

    // Show how to use with Vulkan (pseudocode)
    std::cout << "5. Vulkan Integration Usage:\n";
    std::cout << "   ----------------------------------------\n";
    std::cout << "   // Create Vulkan shader module:\n";
    std::cout << "   VkShaderModuleCreateInfo createInfo{};\n";
    std::cout << "   createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;\n";
    std::cout << "   createInfo.codeSize = vulkanPipeline->getShaderModuleSize();\n";
    std::cout << "   createInfo.pCode = vulkanPipeline->getShaderModuleData();\n";
    std::cout << "   vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);\n\n";
    
    std::cout << "   // Descriptor sets:\n";
    const auto& descriptorSets = vulkanPipeline->getDescriptorSetLayouts();
    if (descriptorSets.empty()) {
        std::cout << "   (No descriptor sets found - add manually if needed)\n";
    } else {
        for (const auto& setLayout : descriptorSets) {
            std::cout << "   Set " << setLayout.set << ": " << setLayout.bindings.size() << " bindings\n";
        }
    }
    std::cout << "\n";

    // Demonstrate manual descriptor set configuration
    std::cout << "6. Example: Manual Descriptor Set Configuration:\n";
    std::cout << "   ----------------------------------------\n";
    // In a real scenario, you might add descriptor bindings like this:
    vulkanPipeline->addDescriptorSetLayout(0, {
        {0, 6, 1, 0x00000001}, // binding 0: storage buffer, compute stage
        {1, 6, 1, 0x00000001}, // binding 1: storage buffer, compute stage
        {2, 6, 1, 0x00000001}  // binding 2: storage buffer, compute stage
    });
    
    std::cout << "   Added descriptor set layout with 3 storage buffer bindings\n";
    std::cout << "   Descriptor sets: " << vulkanPipeline->getDescriptorSetLayouts().size() << "\n\n";

    std::cout << "=== Pipeline Complete ===\n";
    std::cout << "\nThe VulkanPipeline object can now be used to:\n";
    std::cout << "  - Create Vulkan shader modules (VkShaderModule)\n";
    std::cout << "  - Configure descriptor set layouts (VkDescriptorSetLayout)\n";
    std::cout << "  - Set up compute pipeline (VkComputePipeline)\n";
    std::cout << "  - Execute GPU compute shaders via Vulkan API\n";

    return 0;
}
