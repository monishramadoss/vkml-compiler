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

// ========== Tests for Linalg Named Operations ==========

// Test matrix multiplication
void test_MatrixMultiplication() {
    TEST_BEGIN("MatrixMultiplication");
    Tensor<float> matrixA({2, 3});  // 2x3 matrix
    Tensor<float> matrixB({3, 4});  // 3x4 matrix
    
    auto result = matrixA.matmul(matrixB);
    
    // Result should be 2x4
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);
    TEST_END();
}

// Test matrix multiplication with square matrices
void test_SquareMatrixMultiplication() {
    TEST_BEGIN("SquareMatrixMultiplication");
    Tensor<float> matrixA({3, 3});
    Tensor<float> matrixB({3, 3});
    
    auto result = matrixA.matmul(matrixB);
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test dot product
void test_DotProduct() {
    TEST_BEGIN("DotProduct");
    Tensor<float> vectorA({5});
    Tensor<float> vectorB({5});
    
    auto result = vectorA.dot(vectorB);
    
    // Result should be scalar (rank 0)
    auto shape = result.getShape();
    EXPECT_EQ(shape.size(), 0);
    TEST_END();
}

// Test matrix-vector multiplication
void test_MatrixVectorMultiplication() {
    TEST_BEGIN("MatrixVectorMultiplication");
    Tensor<float> matrix({3, 4});
    Tensor<float> vector({4});
    
    auto result = matrix.matvec(vector);
    
    // Result should be 1D vector of size 3
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 3);
    TEST_END();
}

// Test transpose operation
void test_Transpose2D() {
    TEST_BEGIN("Transpose2D");
    Tensor<float> matrix({3, 4});
    
    auto result = matrix.transpose();
    
    // Shape should be swapped: 3x4 -> 4x3
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test transpose with 3D tensor
void test_Transpose3D() {
    TEST_BEGIN("Transpose3D");
    Tensor<float> tensor({2, 3, 4});
    
    auto result = tensor.transpose();
    
    // Last two dimensions should be swapped: 2x3x4 -> 2x4x3
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 3);
    TEST_END();
}

// Test fill operation with float
void test_FillFloat() {
    TEST_BEGIN("FillFloat");
    auto tensor = Tensor<float>::fill({2, 3}, 3.14f);
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test fill operation with integer
void test_FillInt() {
    TEST_BEGIN("FillInt");
    auto tensor = Tensor<int32_t>::fill({3, 3}, 42);
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

// Test sum reduction
void test_SumReduction() {
    TEST_BEGIN("SumReduction");
    Tensor<float> tensor({3, 4});
    
    auto result = tensor.sum();
    
    // Reduces last dimension: 3x4 -> 3
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 3);
    TEST_END();
}

// Test sum reduction to scalar
void test_SumReductionToScalar() {
    TEST_BEGIN("SumReductionToScalar");
    Tensor<float> vector({5});
    
    auto result = vector.sum();
    
    // Reduces to scalar: 5 -> []
    auto shape = result.getShape();
    EXPECT_EQ(shape.size(), 0);
    TEST_END();
}

// Test max reduction
void test_MaxReduction() {
    TEST_BEGIN("MaxReduction");
    Tensor<float> tensor({2, 5});
    
    auto result = tensor.max();
    
    // Reduces last dimension: 2x5 -> 2
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 2);
    TEST_END();
}

// Test min reduction
void test_MinReduction() {
    TEST_BEGIN("MinReduction");
    Tensor<int32_t> tensor({4, 3});
    
    auto result = tensor.min();
    
    // Reduces last dimension: 4x3 -> 4
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 4);
    TEST_END();
}

// Test integer matrix multiplication
void test_IntMatrixMultiplication() {
    TEST_BEGIN("IntMatrixMultiplication");
    Tensor<int32_t> matrixA({2, 3});
    Tensor<int32_t> matrixB({3, 2});
    
    auto result = matrixA.matmul(matrixB);
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
    TEST_END();
}

// Test integer dot product
void test_IntDotProduct() {
    TEST_BEGIN("IntDotProduct");
    Tensor<int32_t> vectorA({10});
    Tensor<int32_t> vectorB({10});
    
    auto result = vectorA.dot(vectorB);
    
    auto shape = result.getShape();
    EXPECT_EQ(shape.size(), 0);
    TEST_END();
}

// Test fill operation with 1D tensor
void test_Fill1D() {
    TEST_BEGIN("Fill1D");
    auto tensor = Tensor<float>::fill({10}, 1.5f);
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 10);
    TEST_END();
}

// Test batched matrix multiplication
void test_BatchMatrixMultiplication() {
    TEST_BEGIN("BatchMatrixMultiplication");
    Tensor<float> matrixA({2, 3, 4});  // batch=2, 3x4 matrices
    Tensor<float> matrixB({2, 4, 5});  // batch=2, 4x5 matrices
    
    auto result = matrixA.batch_matmul(matrixB);
    
    // Result should be [2, 3, 5]
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 5);
    TEST_END();
}

// Test vector-matrix multiplication
void test_VectorMatrixMultiplication() {
    TEST_BEGIN("VectorMatrixMultiplication");
    Tensor<float> vector({4});
    Tensor<float> matrix({4, 5});
    
    auto result = vector.vecmat(matrix);
    
    // Result should be 1D vector of size 5
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 5);
    TEST_END();
}

// Test copy operation
void test_CopyOperation() {
    TEST_BEGIN("CopyOperation");
    Tensor<float> original({3, 4});
    
    auto copied = original.copy();
    
    auto shape = copied.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    TEST_END();
}

// Test map operation with abs
void test_MapOperationAbs() {
    TEST_BEGIN("MapOperationAbs");
    Tensor<float> tensor({2, 3});
    
    auto result = tensor.map<mlir::math::AbsFOp>();
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
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
    
    // Linalg named operations tests
    test_MatrixMultiplication();
    test_SquareMatrixMultiplication();
    test_DotProduct();
    test_MatrixVectorMultiplication();
    test_Transpose2D();
    test_Transpose3D();
    test_FillFloat();
    test_FillInt();
    test_SumReduction();
    test_SumReductionToScalar();
    test_MaxReduction();
    test_MinReduction();
    test_IntMatrixMultiplication();
    test_IntDotProduct();
    test_Fill1D();
    test_BatchMatrixMultiplication();
    test_VectorMatrixMultiplication();
    test_CopyOperation();
    test_MapOperationAbs();
    
    return TestRunner::report();
}
