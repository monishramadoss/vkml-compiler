#include <gtest/gtest.h>
#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"

// Test fixture for Bitwise operator tests
class BitwiseOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the compiler instance for each test
    }
};

// ========== Bitwise AND Tests ==========

TEST_F(BitwiseOpsTest, BitwiseAndBasic) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 & tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(BitwiseOpsTest, BitwiseAndVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<uint32_t> tensor1(shape);
        Tensor<uint32_t> tensor2(shape);
        
        auto result = tensor1 & tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(BitwiseOpsTest, BitwiseAndBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<uint32_t> tensor1(pair.shape1);
        Tensor<uint32_t> tensor2(pair.shape2);
        
        auto result = tensor1 & tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Bitwise OR Tests ==========

TEST_F(BitwiseOpsTest, BitwiseOrBasic) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 | tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(BitwiseOpsTest, BitwiseOrVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<uint32_t> tensor1(shape);
        Tensor<uint32_t> tensor2(shape);
        
        auto result = tensor1 | tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(BitwiseOpsTest, BitwiseOrBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<uint32_t> tensor1(pair.shape1);
        Tensor<uint32_t> tensor2(pair.shape2);
        
        auto result = tensor1 | tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Bitwise XOR Tests ==========

TEST_F(BitwiseOpsTest, BitwiseXorBasic) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 ^ tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(BitwiseOpsTest, BitwiseXorVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<uint32_t> tensor1(shape);
        Tensor<uint32_t> tensor2(shape);
        
        auto result = tensor1 ^ tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(BitwiseOpsTest, BitwiseXorBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<uint32_t> tensor1(pair.shape1);
        Tensor<uint32_t> tensor2(pair.shape2);
        
        auto result = tensor1 ^ tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Bitwise NOT Tests ==========

TEST_F(BitwiseOpsTest, BitwiseNotBasic) {
    Tensor<uint32_t> tensor({2, 3});
    
    auto result = ~tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(BitwiseOpsTest, BitwiseNotVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<uint32_t> tensor(shape);
        
        auto result = ~tensor;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

// ========== Left Shift Tests ==========

TEST_F(BitwiseOpsTest, LeftShiftBasic) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 << tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(BitwiseOpsTest, LeftShiftVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<uint32_t> tensor1(shape);
        Tensor<uint32_t> tensor2(shape);
        
        auto result = tensor1 << tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(BitwiseOpsTest, LeftShiftBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<uint32_t> tensor1(pair.shape1);
        Tensor<uint32_t> tensor2(pair.shape2);
        
        auto result = tensor1 << tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Right Shift Tests ==========

TEST_F(BitwiseOpsTest, RightShiftBasic) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 >> tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(BitwiseOpsTest, RightShiftVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<uint32_t> tensor1(shape);
        Tensor<uint32_t> tensor2(shape);
        
        auto result = tensor1 >> tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(BitwiseOpsTest, RightShiftBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<uint32_t> tensor1(pair.shape1);
        Tensor<uint32_t> tensor2(pair.shape2);
        
        auto result = tensor1 >> tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}
