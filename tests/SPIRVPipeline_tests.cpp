#include "test_utils.h"
#include "Tensor.h"
#include "Compiler.h"
#include "VulkanPipeline.h"

// Test basic SPIR-V serialization
void test_SPIRVSerialization() {
    TEST_BEGIN("SPIRVSerialization");
    
    // Create tensors and operations
    Tensor<float> tensor1({2, 2});
    Tensor<float> tensor2({2, 2});
    auto result = tensor1 + tensor2;
    
    auto compiler = vkml::Compiler::getInstance();
    
    // Run the pipeline to SPIR-V
    compiler->runLinalgToSPIRV();
    
    // Serialize SPIR-V
    auto spirvBinary = compiler->serializeSPIRV();
    
    // Check that we got a binary
    ASSERT_GT(spirvBinary.size(), 0);
    
    TEST_END();
}

// Test VulkanPipeline creation
void test_VulkanPipelineCreation() {
    TEST_BEGIN("VulkanPipelineCreation");
    
    // Reset compiler for clean state
    // Note: In actual tests, each test should have isolated state
    
    // Create tensors and operations
    Tensor<float> a({4, 4});
    Tensor<float> b({4, 4});
    auto c = a * b;
    
    auto compiler = vkml::Compiler::getInstance();
    
    // Create VulkanPipeline
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    // Check that pipeline is valid
    ASSERT_TRUE(vulkanPipeline != nullptr);
    ASSERT_TRUE(vulkanPipeline->isValid());
    
    TEST_END();
}

// Test SPIR-V binary properties
void test_SPIRVBinaryProperties() {
    TEST_BEGIN("SPIRVBinaryProperties");
    
    // Create simple operation
    Tensor<float> x({3, 3});
    Tensor<float> y({3, 3});
    auto z = x + y;
    
    auto compiler = vkml::Compiler::getInstance();
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    // Check binary properties
    ASSERT_TRUE(vulkanPipeline->isValid());
    ASSERT_GT(vulkanPipeline->getShaderModuleSize(), 0);
    ASSERT_GT(vulkanPipeline->getShaderModuleWordCount(), 0);
    ASSERT_TRUE(vulkanPipeline->getShaderModuleData() != nullptr);
    
    // Check that size in bytes = word count * 4
    ASSERT_EQ(vulkanPipeline->getShaderModuleSize(), 
              vulkanPipeline->getShaderModuleWordCount() * sizeof(uint32_t));
    
    // Check SPIR-V magic number (first word should be 0x07230203)
    const auto& binary = vulkanPipeline->getSPIRVBinary();
    ASSERT_GT(binary.size(), 4);
    ASSERT_EQ(binary[0], 0x07230203);
    
    TEST_END();
}

// Test VulkanPipeline entry point
void test_VulkanPipelineEntryPoint() {
    TEST_BEGIN("VulkanPipelineEntryPoint");
    
    Tensor<int32_t> a({2, 2});
    Tensor<int32_t> b({2, 2});
    auto c = a - b;
    
    auto compiler = vkml::Compiler::getInstance();
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    // Default entry point should be "main"
    ASSERT_TRUE(vulkanPipeline->getEntryPoint() == "main");
    
    // Test setting custom entry point
    vulkanPipeline->setEntryPoint("compute_kernel");
    ASSERT_TRUE(vulkanPipeline->getEntryPoint() == "compute_kernel");
    
    TEST_END();
}

// Test VulkanPipeline descriptor set layout
void test_VulkanPipelineDescriptorSets() {
    TEST_BEGIN("VulkanPipelineDescriptorSets");
    
    Tensor<float> input({10, 10});
    Tensor<float> weights({10, 10});
    auto output = input * weights;
    
    auto compiler = vkml::Compiler::getInstance();
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    // Initially no descriptor sets
    ASSERT_EQ(vulkanPipeline->getDescriptorSetLayouts().size(), 0);
    
    // Add a descriptor set layout
    vulkanPipeline->addDescriptorSetLayout(0, {
        {0, 6, 1, 0x00000001}, // binding 0: storage buffer
        {1, 6, 1, 0x00000001}, // binding 1: storage buffer
        {2, 6, 1, 0x00000001}  // binding 2: storage buffer
    });
    
    ASSERT_EQ(vulkanPipeline->getDescriptorSetLayouts().size(), 1);
    
    const auto& layouts = vulkanPipeline->getDescriptorSetLayouts();
    ASSERT_EQ(layouts[0].set, 0);
    ASSERT_EQ(layouts[0].bindings.size(), 3);
    ASSERT_EQ(layouts[0].bindings[0].binding, 0);
    ASSERT_EQ(layouts[0].bindings[1].binding, 1);
    ASSERT_EQ(layouts[0].bindings[2].binding, 2);
    
    TEST_END();
}

// Test VulkanPipeline push constants
void test_VulkanPipelinePushConstants() {
    TEST_BEGIN("VulkanPipelinePushConstants");
    
    Tensor<float> data({5, 5});
    auto processed = data + data;
    
    auto compiler = vkml::Compiler::getInstance();
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    // Initially no push constants
    ASSERT_EQ(vulkanPipeline->getPushConstantRanges().size(), 0);
    
    // Add push constant range
    vulkanPipeline->addPushConstantRange(0x00000001, 0, 16); // compute stage, offset 0, 16 bytes
    
    ASSERT_EQ(vulkanPipeline->getPushConstantRanges().size(), 1);
    
    const auto& ranges = vulkanPipeline->getPushConstantRanges();
    ASSERT_EQ(ranges[0].stageFlags, 0x00000001);
    ASSERT_EQ(ranges[0].offset, 0);
    ASSERT_EQ(ranges[0].size, 16);
    
    TEST_END();
}

// Test multiple operations in pipeline
void test_MultipleOperationsPipeline() {
    TEST_BEGIN("MultipleOperationsPipeline");
    
    Tensor<float> a({8, 8});
    Tensor<float> b({8, 8});
    Tensor<float> c({8, 8});
    
    // Multiple operations
    auto temp1 = a + b;
    auto temp2 = temp1 * c;
    auto result = temp2 - a;
    
    auto compiler = vkml::Compiler::getInstance();
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    ASSERT_TRUE(vulkanPipeline->isValid());
    ASSERT_GT(vulkanPipeline->getShaderModuleWordCount(), 0);
    
    TEST_END();
}

// Test with integer tensors
void test_IntegerTensorPipeline() {
    TEST_BEGIN("IntegerTensorPipeline");
    
    Tensor<int32_t> x({4, 4});
    Tensor<int32_t> y({4, 4});
    auto z = x + y;
    
    auto compiler = vkml::Compiler::getInstance();
    auto vulkanPipeline = compiler->createVulkanPipeline();
    
    ASSERT_TRUE(vulkanPipeline->isValid());
    ASSERT_GT(vulkanPipeline->getShaderModuleSize(), 0);
    
    // Verify SPIR-V header
    const auto& binary = vulkanPipeline->getSPIRVBinary();
    ASSERT_GT(binary.size(), 4);
    ASSERT_EQ(binary[0], 0x07230203); // SPIR-V magic
    
    TEST_END();
}

int main() {
    // Run all tests
    test_SPIRVSerialization();
    test_VulkanPipelineCreation();
    test_SPIRVBinaryProperties();
    test_VulkanPipelineEntryPoint();
    test_VulkanPipelineDescriptorSets();
    test_VulkanPipelinePushConstants();
    test_MultipleOperationsPipeline();
    test_IntegerTensorPipeline();
    
    return TestRunner::report();
}
