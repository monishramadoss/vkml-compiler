#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"
#include "test_utils.h"

void test_logical_and_basic() {
    TEST_BEGIN("Logical AND Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 && tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_logical_and_various_shapes() {
    TEST_BEGIN("Logical AND Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 && tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

void test_logical_or_basic() {
    TEST_BEGIN("Logical OR Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 || tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_logical_or_various_shapes() {
    TEST_BEGIN("Logical OR Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 || tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

void test_logical_not_basic() {
    TEST_BEGIN("Logical NOT Basic");
    Tensor<float> tensor({2, 3});
    
    auto result = !tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_logical_not_various_shapes() {
    TEST_BEGIN("Logical NOT Various Shapes");
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
    TEST_END();
}

int main() {
    test_logical_and_basic();
    test_logical_and_various_shapes();
    test_logical_or_basic();
    test_logical_or_various_shapes();
    test_logical_not_basic();
    test_logical_not_various_shapes();
    
    return TestRunner::report();
}
