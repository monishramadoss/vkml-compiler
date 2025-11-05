#include <gtest/gtest.h>
#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"

// Test fixture for Other operator tests (indexing, tensor creation, etc.)
class OtherOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the compiler instance for each test
    }
};

// ========== Tensor Creation Tests ==========

TEST_F(OtherOpsTest, CreateTensorWithShape) {
    Tensor<float> tensor({2, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(OtherOpsTest, Create1DTensor) {
    Tensor<float> tensor({5});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 5);
}

TEST_F(OtherOpsTest, CreateIntTensor) {
    Tensor<int32_t> tensor({3, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(OtherOpsTest, Create3DTensor) {
    Tensor<float> tensor({2, 3, 4});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

TEST_F(OtherOpsTest, CreateDoubleTensor) {
    Tensor<double> tensor({2, 2});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
}

TEST_F(OtherOpsTest, CreateUInt64Tensor) {
    Tensor<uint64_t> tensor({2, 2});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
}

TEST_F(OtherOpsTest, CreateVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor(shape);
        auto resultShape = tensor.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

// ========== Symbolic ID Tests ==========

TEST_F(OtherOpsTest, SymbolicIDGeneration) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto id1 = tensor1.getSymbolicId();
    auto id2 = tensor2.getSymbolicId();
    
    // IDs should be different
    EXPECT_NE(id1, id2);
    
    // IDs should start with "tensor_"
    EXPECT_EQ(id1.substr(0, 7), "tensor_");
    EXPECT_EQ(id2.substr(0, 7), "tensor_");
}

// ========== Subscript Operator Tests ==========

TEST_F(OtherOpsTest, SubscriptBasic) {
    Tensor<float> tensor({3, 4});
    
    auto slice = tensor[0];
    
    // Should reduce rank by 1
    auto shape = slice.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 4);
}

TEST_F(OtherOpsTest, SubscriptVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        if (shape.size() < 2) continue; // Skip 1D tensors for subscript test
        
        Tensor<float> tensor(shape);
        
        auto slice = tensor[0];
        auto resultShape = slice.getShape();
        
        // Should reduce rank by 1
        ASSERT_EQ(resultShape.size(), shape.size() - 1);
        for (size_t i = 0; i < resultShape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i + 1]);
        }
    }
}

TEST_F(OtherOpsTest, SubscriptMultipleDimensions) {
    Tensor<float> tensor({5, 7, 11});
    
    auto slice = tensor[2];
    auto shape = slice.getShape();
    
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 7);
    EXPECT_EQ(shape[1], 11);
}

// ========== Broadcasting Tests ==========

TEST_F(OtherOpsTest, BroadcastOperation) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({1, 3});
    
    // This should work with broadcasting rules
    auto result = tensor1 + tensor2;
    
    // Result should exist
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(OtherOpsTest, BroadcastVariousShapePairs) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        // Test with addition (any binary op would work)
        auto result = tensor1 + tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}
