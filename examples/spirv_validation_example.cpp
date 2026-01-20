#include "Tensor.h"
#include "Compiler.h"
#include "VulkanPipeline.h"
#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "=== SPIR-V Validation Example ===\n\n";

    // Create tensors and operations
    std::cout << "1. Creating tensor operations...\n";
    Tensor<float> a({8, 8});
    Tensor<float> b({8, 8});
    auto result = a + b;
    std::cout << "   Created addition operation\n\n";

    // Configure compiler
    auto compiler = vkml::Compiler::getInstance();
    compiler->setVulkan1_2Target();
    
    // Compile to SPIR-V
    std::cout << "2. Compiling to SPIR-V...\n";
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    if (!vulkanPipeline->isValid()) {
        std::cerr << "   ERROR: Failed to create valid SPIR-V binary\n";
        return 1;
    }
    std::cout << "   Compilation successful\n";
    std::cout << "   Binary size: " << vulkanPipeline->getShaderModuleSize() << " bytes\n\n";

    // Save SPIR-V to file
    std::cout << "3. Saving SPIR-V to file...\n";
    std::string spirvFile = "output.spv";
    if (vulkanPipeline->saveSPIRVToFile(spirvFile)) {
        std::cout << "   Saved to: " << spirvFile << "\n\n";
    } else {
        std::cerr << "   ERROR: Failed to save SPIR-V file\n";
        return 1;
    }

    // Validate using spirv-val
    std::cout << "4. Validating SPIR-V with spirv-val...\n";
    
    // Determine target environment based on compilation settings
    const auto& targetEnv = vulkanPipeline->getTargetEnv();
    std::string targetEnvStr = "vulkan1.0";
    if (targetEnv.vulkanVersion >= 0x00103000) {
        targetEnvStr = "vulkan1.3";
    } else if (targetEnv.vulkanVersion >= 0x00102000) {
        targetEnvStr = "vulkan1.2";
    } else if (targetEnv.vulkanVersion >= 0x00101000) {
        targetEnvStr = "vulkan1.1";
    }
    
    // Try Python script first, then bash script
    std::string cmd = "../scripts/validate_spirv.py -t " + targetEnvStr + " " + spirvFile;
    std::cout << "   Running: " << cmd << "\n\n";
    
    int result_code = std::system(cmd.c_str());
    
    // If Python script fails, try bash script
    if (result_code != 0) {
        cmd = "../scripts/validate_spirv.sh -t " + targetEnvStr + " " + spirvFile;
        std::cout << "   Trying bash script: " << cmd << "\n\n";
        result_code = std::system(cmd.c_str());
    }
    
    if (result_code == 0) {
        std::cout << "\n=== Validation Complete ===\n";
        std::cout << "SPIR-V binary is valid and ready for Vulkan!\n";
    } else {
        std::cerr << "\n=== Validation Failed ===\n";
        std::cerr << "Please check the output above for errors.\n";
        std::cerr << "Note: If spirv-val is not installed, run with -i flag:\n";
        std::cerr << "  ../scripts/validate_spirv.py -i " << spirvFile << "\n";
    }

    return result_code;
}
