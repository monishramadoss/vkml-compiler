#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"
#include "test_utils.h"

void test_bitwise_and_basic() {
    TEST_BEGIN("Bitwise AND Basic");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 & tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_bitwise_and_various_shapes() {
    TEST_BEGIN("Bitwise AND Various Shapes");
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
    TEST_END();
}

void test_bitwise_or_basic() {
    TEST_BEGIN("Bitwise OR Basic");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 | tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_bitwise_or_various_shapes() {
    TEST_BEGIN("Bitwise OR Various Shapes");
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
    TEST_END();
}

void test_bitwise_xor_basic() {
    TEST_BEGIN("Bitwise XOR Basic");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 ^ tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_bitwise_xor_various_shapes() {
    TEST_BEGIN("Bitwise XOR Various Shapes");
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
    TEST_END();
}

void test_bitwise_not_basic() {
    TEST_BEGIN("Bitwise NOT Basic");
    Tensor<uint32_t> tensor({2, 3});
    
    auto result = ~tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_bitwise_not_various_shapes() {
    TEST_BEGIN("Bitwise NOT Various Shapes");
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
    TEST_END();
}

void test_left_shift_basic() {
    TEST_BEGIN("Left Shift Basic");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 << tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_left_shift_various_shapes() {
    TEST_BEGIN("Left Shift Various Shapes");
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
    TEST_END();
}

void test_right_shift_basic() {
    TEST_BEGIN("Right Shift Basic");
    Tensor<uint32_t> tensor1({2, 3});
    Tensor<uint32_t> tensor2({2, 3});
    
    auto result = tensor1 >> tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_right_shift_various_shapes() {
    TEST_BEGIN("Right Shift Various Shapes");
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
    TEST_END();
}

int main() {
    test_bitwise_and_basic();
    test_bitwise_and_various_shapes();
    test_bitwise_or_basic();
    test_bitwise_or_various_shapes();
    test_bitwise_xor_basic();
    test_bitwise_xor_various_shapes();
    test_bitwise_not_basic();
    test_bitwise_not_various_shapes();
    test_left_shift_basic();
    test_left_shift_various_shapes();
    test_right_shift_basic();
    test_right_shift_various_shapes();
    
    return TestRunner::report();
}
