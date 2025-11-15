#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"
#include "test_utils.h"

void test_addition_basic() {
    TEST_BEGIN("Addition Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 + tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_addition_various_shapes() {
    TEST_BEGIN("Addition Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 + tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
    TEST_END();
}

void test_addition_broadcasting() {
    TEST_BEGIN("Addition Broadcasting");
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 + tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
    TEST_END();
}

void test_subtraction_basic() {
    TEST_BEGIN("Subtraction Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 - tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_subtraction_various_shapes() {
    TEST_BEGIN("Subtraction Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 - tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
    TEST_END();
}

void test_multiplication_basic() {
    TEST_BEGIN("Multiplication Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 * tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_multiplication_various_shapes() {
    TEST_BEGIN("Multiplication Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 * tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
    TEST_END();
}

void test_division_float_basic() {
    TEST_BEGIN("Division Float Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_division_int_basic() {
    TEST_BEGIN("Division Int Basic");
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_modulo_basic() {
    TEST_BEGIN("Modulo Basic");
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 % tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_unary_plus_basic() {
    TEST_BEGIN("Unary Plus Basic");
    Tensor<float> tensor({2, 3});
    
    auto result = +tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_increment_basic() {
    TEST_BEGIN("Increment Basic");
    Tensor<float> tensor({2, 3});
    
    ++tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_decrement_basic() {
    TEST_BEGIN("Decrement Basic");
    Tensor<float> tensor({2, 3});
    
    --tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

int main() {
    test_addition_basic();
    test_addition_various_shapes();
    test_addition_broadcasting();
    test_subtraction_basic();
    test_subtraction_various_shapes();
    test_multiplication_basic();
    test_multiplication_various_shapes();
    test_division_float_basic();
    test_division_int_basic();
    test_modulo_basic();
    test_unary_plus_basic();
    test_increment_basic();
    test_decrement_basic();
    
    return TestRunner::report();
}
