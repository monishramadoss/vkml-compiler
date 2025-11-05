#include <gtest/gtest.h>
#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"

// Test fixture for Logical operator tests
class LogicalOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the compiler instance for each test
    }
};

// ========== Logical AND Tests ==========

TEST_F(LogicalOpsTest, LogicalAndBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 && tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(LogicalOpsTest, LogicalAndVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 && tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(LogicalOpsTest, LogicalAndBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 && tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Logical OR Tests ==========

TEST_F(LogicalOpsTest, LogicalOrBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 || tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(LogicalOpsTest, LogicalOrVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 || tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(LogicalOpsTest, LogicalOrBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 || tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Logical NOT Tests ==========

TEST_F(LogicalOpsTest, LogicalNotBasic) {
    Tensor<float> tensor({2, 3});
    
    auto result = !tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(LogicalOpsTest, LogicalNotVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor(shape);
        
        auto result = !tensor;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}
