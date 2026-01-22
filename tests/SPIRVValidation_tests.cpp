#include "Tensor.h"
#include "Compiler.h"
#include "SPIRVPipeline.h"
#include "test_utils.h"

#include <fstream>
#include <cstdio>
#include <unistd.h>

// Helper function to check if spirv-val is available
bool isSPIRVValAvailable() {
    // Check if spirv-val exists in PATH using access() for safety
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) return false;
    
    std::string pathStr(pathEnv);
    size_t pos = 0;
    std::string delimiter = ":";
    
    while ((pos = pathStr.find(delimiter)) != std::string::npos) {
        std::string dir = pathStr.substr(0, pos);
        std::string fullPath = dir + "/spirv-val";
        if (access(fullPath.c_str(), X_OK) == 0) {
            return true;
        }
        pathStr.erase(0, pos + delimiter.length());
    }
    
    // Check last directory
    if (!pathStr.empty()) {
        std::string fullPath = pathStr + "/spirv-val";
        if (access(fullPath.c_str(), X_OK) == 0) {
            return true;
        }
    }
    
    return false;
}

// Helper function to compile and validate SPIR-V
bool compileAndValidate(const std::string& testName) {
    auto compiler = vkml::Compiler::getInstance();
    auto module = compiler->getModule();
    auto context = compiler->getContext();
    
    // Run linalg to GPU pipeline first
    compiler->runLinalgToGPU();
    
    // Create SPIR-V pipeline
    vkml::SPIRVPipeline spirvPipeline(context, module);
    
    // Compile to SPIR-V
    if (!spirvPipeline.compileToSPIRV()) {
        std::cerr << testName << ": Failed to compile to SPIR-V\n";
        return false;
    }
    
    // Serialize to binary
    if (!spirvPipeline.serializeToBinary()) {
        std::cerr << testName << ": Failed to serialize SPIR-V\n";
        return false;
    }
    
    // Get the binary
    const vkml::VulkanBuffer* buffer = spirvPipeline.getBinary();
    if (!buffer || buffer->spirvBinary.empty()) {
        std::cerr << testName << ": No SPIR-V binary generated\n";
        return false;
    }
    
    std::cout << testName << ": Generated " << buffer->sizeInBytes() << " bytes of SPIR-V\n";
    
    // Validate if spirv-val is available
    if (isSPIRVValAvailable()) {
        if (!spirvPipeline.validate()) {
            std::cerr << testName << ": SPIR-V validation failed\n";
            return false;
        }
        std::cout << testName << ": SPIR-V validation passed\n";
    } else {
        std::cout << testName << ": spirv-val not available, skipping validation\n";
    }
    
    return true;
}

void test_spirv_basic_addition() {
    TEST_BEGIN("SPIR-V Basic Addition");
    
    Tensor<float> tensor1({16, 16});
    Tensor<float> tensor2({16, 16});
    auto result = tensor1 + tensor2;
    
    ASSERT_TRUE(compileAndValidate("Basic Addition"));
    TEST_END();
}

void test_spirv_subtraction() {
    TEST_BEGIN("SPIR-V Subtraction");
    
    Tensor<float> tensor1({32, 32});
    Tensor<float> tensor2({32, 32});
    auto result = tensor1 - tensor2;
    
    ASSERT_TRUE(compileAndValidate("Subtraction"));
    TEST_END();
}

void test_spirv_multiplication() {
    TEST_BEGIN("SPIR-V Multiplication");
    
    Tensor<float> tensor1({8, 8});
    Tensor<float> tensor2({8, 8});
    auto result = tensor1 * tensor2;
    
    ASSERT_TRUE(compileAndValidate("Multiplication"));
    TEST_END();
}

void test_spirv_division() {
    TEST_BEGIN("SPIR-V Division");
    
    Tensor<float> tensor1({16, 16});
    Tensor<float> tensor2({16, 16});
    auto result = tensor1 / tensor2;
    
    ASSERT_TRUE(compileAndValidate("Division"));
    TEST_END();
}

void test_spirv_chained_operations() {
    TEST_BEGIN("SPIR-V Chained Operations");
    
    Tensor<float> a({8, 8});
    Tensor<float> b({8, 8});
    Tensor<float> c({8, 8});
    
    auto result = (a + b) * c - a;
    
    ASSERT_TRUE(compileAndValidate("Chained Operations"));
    TEST_END();
}

void test_spirv_integer_operations() {
    TEST_BEGIN("SPIR-V Integer Operations");
    
    Tensor<int32_t> tensor1({16, 16});
    Tensor<int32_t> tensor2({16, 16});
    auto result = tensor1 + tensor2;
    
    ASSERT_TRUE(compileAndValidate("Integer Operations"));
    TEST_END();
}

void test_spirv_binary_buffer() {
    TEST_BEGIN("SPIR-V Binary Buffer");
    
    Tensor<float> tensor1({4, 4});
    Tensor<float> tensor2({4, 4});
    auto result = tensor1 + tensor2;
    
    auto compiler = vkml::Compiler::getInstance();
    auto module = compiler->getModule();
    auto context = compiler->getContext();
    
    compiler->runLinalgToGPU();
    
    vkml::SPIRVPipeline spirvPipeline(context, module);
    
    ASSERT_TRUE(spirvPipeline.compileToSPIRV());
    ASSERT_TRUE(spirvPipeline.serializeToBinary());
    
    const vkml::VulkanBuffer* buffer = spirvPipeline.getBinary();
    ASSERT_TRUE(buffer != nullptr);
    ASSERT_TRUE(!buffer->spirvBinary.empty());
    
    // Check SPIR-V magic number (0x07230203)
    ASSERT_EQ(buffer->spirvBinary[0], 0x07230203);
    
    // Verify buffer size is reasonable
    ASSERT_GT(buffer->sizeInBytes(), 0);
    
    std::cout << "Binary size: " << buffer->sizeInBytes() << " bytes\n";
    std::cout << "Word count: " << buffer->spirvBinary.size() << " words\n";
    
    TEST_END();
}

void test_spirv_vulkan_compatibility() {
    TEST_BEGIN("SPIR-V Vulkan Compatibility");
    
    if (!isSPIRVValAvailable()) {
        std::cout << "spirv-val not available, skipping test\n";
        TEST_END();
        return;
    }
    
    Tensor<float> tensor1({16, 16});
    Tensor<float> tensor2({16, 16});
    auto result = tensor1 * tensor2;
    
    auto compiler = vkml::Compiler::getInstance();
    auto module = compiler->getModule();
    auto context = compiler->getContext();
    
    compiler->runLinalgToGPU();
    
    vkml::SPIRVPipeline spirvPipeline(context, module);
    
    ASSERT_TRUE(spirvPipeline.compileToSPIRV());
    ASSERT_TRUE(spirvPipeline.serializeToBinary());
    ASSERT_TRUE(spirvPipeline.validate());
    
    TEST_END();
}

int main() {
    std::cout << "========================================\n";
    std::cout << "SPIR-V Validation Tests\n";
    std::cout << "========================================\n\n";
    
    if (!isSPIRVValAvailable()) {
        std::cout << "WARNING: spirv-val not found. Validation tests will be skipped.\n";
        std::cout << "Install SPIR-V Tools to enable validation.\n\n";
    }
    
    test_spirv_basic_addition();
    test_spirv_subtraction();
    test_spirv_multiplication();
    test_spirv_division();
    test_spirv_chained_operations();
    test_spirv_integer_operations();
    test_spirv_binary_buffer();
    test_spirv_vulkan_compatibility();
    
    return TestRunner::report();
}
