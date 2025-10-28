#include <gtest/gtest.h>
#include "Tensor.h"
#include "Compiler.h"

// Test fixture for Tensor tests
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the compiler instance for each test
        // Note: The Compiler uses a singleton pattern, so we reuse the same instance
    }
};

// Test tensor creation with different shapes
TEST_F(TensorTest, CreateTensorWithShape) {
    Tensor<float> tensor({2, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test tensor creation with 1D shape
TEST_F(TensorTest, Create1DTensor) {
    Tensor<float> tensor({5});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 5);
}

// Test tensor creation with different data types
TEST_F(TensorTest, CreateIntTensor) {
    Tensor<int32_t> tensor({3, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
}

// Test tensor addition operation
TEST_F(TensorTest, TensorAddition) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    // This should compile and create the operation
    auto result = tensor1 + tensor2;
    
    // Check that result has the correct shape
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test tensor subtraction operation
TEST_F(TensorTest, TensorSubtraction) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 - tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test tensor multiplication operation
TEST_F(TensorTest, TensorMultiplication) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 * tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test symbolic ID generation
TEST_F(TensorTest, SymbolicIDGeneration) {
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

// Test 3D tensor creation
TEST_F(TensorTest, Create3DTensor) {
    Tensor<float> tensor({2, 3, 4});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

// Test double precision tensor
TEST_F(TensorTest, CreateDoubleTensor) {
    Tensor<double> tensor({2, 2});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
}

// Test broadcasting-like operations (different shapes)
TEST_F(TensorTest, BroadcastOperation) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({1, 3});
    
    // This should work with TOSA broadcasting rules
    auto result = tensor1 + tensor2;
    
    // Result should exist
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
