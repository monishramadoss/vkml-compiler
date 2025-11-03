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

// Test tensor division operation (floating point)
TEST_F(TensorTest, TensorDivisionFloat) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test tensor division operation (integer)
TEST_F(TensorTest, TensorDivisionInt) {
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test tensor modulo operation
TEST_F(TensorTest, TensorModulo) {
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 % tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test bitwise AND operation
TEST_F(TensorTest, TensorBitwiseAnd) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 & tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test bitwise OR operation
TEST_F(TensorTest, TensorBitwiseOr) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 | tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test bitwise XOR operation
TEST_F(TensorTest, TensorBitwiseXor) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 ^ tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test bitwise NOT operation
TEST_F(TensorTest, TensorBitwiseNot) {
    Tensor<uint32_t> tensor({2, 3});
    
    auto result = ~tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test left shift operation
TEST_F(TensorTest, TensorLeftShift) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 << tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test right shift operation
TEST_F(TensorTest, TensorRightShift) {
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 >> tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test logical AND operation
TEST_F(TensorTest, TensorLogicalAnd) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 && tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test logical OR operation
TEST_F(TensorTest, TensorLogicalOr) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 || tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test logical NOT operation
TEST_F(TensorTest, TensorLogicalNot) {
    Tensor<float> tensor({2, 3});
    
    auto result = !tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test equality comparison
TEST_F(TensorTest, TensorEqual) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 == tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test inequality comparison
TEST_F(TensorTest, TensorNotEqual) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 != tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test greater than comparison
TEST_F(TensorTest, TensorGreaterThan) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 > tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test greater than or equal comparison
TEST_F(TensorTest, TensorGreaterEqual) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 >= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test less than comparison
TEST_F(TensorTest, TensorLessThan) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 < tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test less than or equal comparison
TEST_F(TensorTest, TensorLessEqual) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 <= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// Test unary plus (abs) operation
TEST_F(TensorTest, TensorUnaryPlus) {
    Tensor<float> tensor({2, 3});
    
    auto result = +tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test prefix increment operation
TEST_F(TensorTest, TensorPrefixIncrement) {
    Tensor<float> tensor({2, 3});
    
    ++tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test postfix increment operation
TEST_F(TensorTest, TensorPostfixIncrement) {
    Tensor<float> tensor({2, 3});
    
    tensor++;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test prefix decrement operation
TEST_F(TensorTest, TensorPrefixDecrement) {
    Tensor<float> tensor({2, 3});
    
    --tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test postfix decrement operation
TEST_F(TensorTest, TensorPostfixDecrement) {
    Tensor<float> tensor({2, 3});
    
    tensor--;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test subscript operator
TEST_F(TensorTest, TensorSubscript) {
    Tensor<float> tensor({3, 4});
    
    auto slice = tensor[0];
    
    // Should reduce rank by 1
    auto shape = slice.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 4);
}

// Test uint64 tensor type
TEST_F(TensorTest, CreateUInt64Tensor) {
    Tensor<uint64_t> tensor({2, 2});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
