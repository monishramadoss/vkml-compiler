#include "test_utils.h"
#include "Tensor.h"
#include "Compiler.h"

// Test tensor creation with different shapes
void test_CreateTensorWithShape() {
    TEST_BEGIN("CreateTensorWithShape");
    Tensor<float> tensor({2, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test tensor creation with 1D shape
void test_Create1DTensor() {
    TEST_BEGIN("Create1DTensor");
    Tensor<float> tensor({5});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 5);
    TEST_END();
}

// Test tensor creation with different data types
void test_CreateIntTensor() {
    TEST_BEGIN("CreateIntTensor");
    Tensor<int32_t> tensor({3, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test tensor addition operation
void test_TensorAddition() {
    TEST_BEGIN("TensorAddition");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    // This should compile and create the operation
    auto result = tensor1 + tensor2;
    
    // Check that result has the correct shape
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test tensor subtraction operation
void test_TensorSubtraction() {
    TEST_BEGIN("TensorSubtraction");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 - tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test tensor multiplication operation
void test_TensorMultiplication() {
    TEST_BEGIN("TensorMultiplication");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 * tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test symbolic ID generation
void test_SymbolicIDGeneration() {
    TEST_BEGIN("SymbolicIDGeneration");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto id1 = tensor1.getSymbolicId();
    auto id2 = tensor2.getSymbolicId();
    
    // IDs should be different
    EXPECT_NE(id1, id2);
    
    // IDs should start with "tensor_"
    EXPECT_EQ(id1.substr(0, 7), "tensor_");
    EXPECT_EQ(id2.substr(0, 7), "tensor_");
    TEST_END();
}

// Test 3D tensor creation
void test_Create3DTensor() {
    TEST_BEGIN("Create3DTensor");
    Tensor<float> tensor({2, 3, 4});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    TEST_END();
}

// Test double precision tensor
void test_CreateDoubleTensor() {
    TEST_BEGIN("CreateDoubleTensor");
    Tensor<double> tensor({2, 2});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
    TEST_END();
}

// Test broadcasting-like operations (different shapes)
void test_BroadcastOperation() {
    TEST_BEGIN("BroadcastOperation");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({1, 3});
    
    // This should work with TOSA broadcasting rules
    auto result = tensor1 + tensor2;
    
    // Result should exist
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test tensor division operation (floating point)
void test_TensorDivisionFloat() {
    TEST_BEGIN("TensorDivisionFloat");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test tensor division operation (integer)
void test_TensorDivisionInt() {
    TEST_BEGIN("TensorDivisionInt");
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test tensor modulo operation
void test_TensorModulo() {
    TEST_BEGIN("TensorModulo");
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 % tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test bitwise AND operation
void test_TensorBitwiseAnd() {
    TEST_BEGIN("TensorBitwiseAnd");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 & tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test bitwise OR operation
void test_TensorBitwiseOr() {
    TEST_BEGIN("TensorBitwiseOr");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 | tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test bitwise XOR operation
void test_TensorBitwiseXor() {
    TEST_BEGIN("TensorBitwiseXor");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 ^ tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test bitwise NOT operation
void test_TensorBitwiseNot() {
    TEST_BEGIN("TensorBitwiseNot");
    Tensor<uint32_t> tensor({2, 3});
    
    auto result = ~tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test left shift operation
void test_TensorLeftShift() {
    TEST_BEGIN("TensorLeftShift");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 << tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test right shift operation
void test_TensorRightShift() {
    TEST_BEGIN("TensorRightShift");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 >> tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test logical AND operation
void test_TensorLogicalAnd() {
    TEST_BEGIN("TensorLogicalAnd");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 && tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test logical OR operation
void test_TensorLogicalOr() {
    TEST_BEGIN("TensorLogicalOr");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 || tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test logical NOT operation
void test_TensorLogicalNot() {
    TEST_BEGIN("TensorLogicalNot");
    Tensor<float> tensor({2, 3});
    
    auto result = !tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test equality comparison
void test_TensorEqual() {
    TEST_BEGIN("TensorEqual");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 == tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test inequality comparison
void test_TensorNotEqual() {
    TEST_BEGIN("TensorNotEqual");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 != tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test greater than comparison
void test_TensorGreaterThan() {
    TEST_BEGIN("TensorGreaterThan");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 > tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test greater than or equal comparison
void test_TensorGreaterEqual() {
    TEST_BEGIN("TensorGreaterEqual");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 >= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test less than comparison
void test_TensorLessThan() {
    TEST_BEGIN("TensorLessThan");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 < tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test less than or equal comparison
void test_TensorLessEqual() {
    TEST_BEGIN("TensorLessEqual");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 <= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

// Test unary plus (abs) operation
void test_TensorUnaryPlus() {
    TEST_BEGIN("TensorUnaryPlus");
    Tensor<float> tensor({2, 3});
    
    auto result = +tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test prefix increment operation
void test_TensorPrefixIncrement() {
    TEST_BEGIN("TensorPrefixIncrement");
    Tensor<float> tensor({2, 3});
    
    ++tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test postfix increment operation
void test_TensorPostfixIncrement() {
    TEST_BEGIN("TensorPostfixIncrement");
    Tensor<float> tensor({2, 3});
    
    tensor++;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test prefix decrement operation
void test_TensorPrefixDecrement() {
    TEST_BEGIN("TensorPrefixDecrement");
    Tensor<float> tensor({2, 3});
    
    --tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test postfix decrement operation
void test_TensorPostfixDecrement() {
    TEST_BEGIN("TensorPostfixDecrement");
    Tensor<float> tensor({2, 3});
    
    tensor--;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test subscript operator
void test_TensorSubscript() {
    TEST_BEGIN("TensorSubscript");
    Tensor<float> tensor({3, 4});
    
    auto slice = tensor[0];
    
    // Should reduce rank by 1
    auto shape = slice.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 4);
    TEST_END();
}

// Test uint64 tensor type
void test_CreateUInt64Tensor() {
    TEST_BEGIN("CreateUInt64Tensor");
    Tensor<uint64_t> tensor({2, 2});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
    TEST_END();
}



int main() {
    test_CreateTensorWithShape();
    test_Create1DTensor();
    test_CreateIntTensor();
    test_TensorAddition();
    test_TensorSubtraction();
    test_TensorMultiplication();
    test_SymbolicIDGeneration();
    test_Create3DTensor();
    test_CreateDoubleTensor();
    test_BroadcastOperation();
    test_TensorDivisionFloat();
    test_TensorDivisionInt();
    test_TensorModulo();
    test_TensorBitwiseAnd();
    test_TensorBitwiseOr();
    test_TensorBitwiseXor();
    test_TensorBitwiseNot();
    test_TensorLeftShift();
    test_TensorRightShift();
    test_TensorLogicalAnd();
    test_TensorLogicalOr();
    test_TensorLogicalNot();
    test_TensorEqual();
    test_TensorNotEqual();
    test_TensorGreaterThan();
    test_TensorGreaterEqual();
    test_TensorLessThan();
    test_TensorLessEqual();
    test_TensorUnaryPlus();
    test_TensorPrefixIncrement();
    test_TensorPostfixIncrement();
    test_TensorPrefixDecrement();
    test_TensorPostfixDecrement();
    test_TensorSubscript();
    test_CreateUInt64Tensor();
    
    return TestRunner::report();
}
