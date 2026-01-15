#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"
#include "test_utils.h"

void test_equal_basic() {
    TEST_BEGIN("Equal Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 == tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_equal_various_shapes() {
    TEST_BEGIN("Equal Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 == tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

void test_not_equal_basic() {
    TEST_BEGIN("Not Equal Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 != tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_not_equal_various_shapes() {
    TEST_BEGIN("Not Equal Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 != tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

void test_greater_than_basic() {
    TEST_BEGIN("Greater Than Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 > tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_greater_than_various_shapes() {
    TEST_BEGIN("Greater Than Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 > tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

void test_greater_equal_basic() {
    TEST_BEGIN("Greater Equal Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 >= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_greater_equal_various_shapes() {
    TEST_BEGIN("Greater Equal Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 >= tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

void test_less_than_basic() {
    TEST_BEGIN("Less Than Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 < tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_less_than_various_shapes() {
    TEST_BEGIN("Less Than Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 < tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

void test_less_equal_basic() {
    TEST_BEGIN("Less Equal Basic");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 <= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_less_equal_various_shapes() {
    TEST_BEGIN("Less Equal Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 <= tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
    TEST_END();
}

int main() {
    test_equal_basic();
    test_equal_various_shapes();
    test_not_equal_basic();
    test_not_equal_various_shapes();
    test_greater_than_basic();
    test_greater_than_various_shapes();
    test_greater_equal_basic();
    test_greater_equal_various_shapes();
    test_less_than_basic();
    test_less_than_various_shapes();
    test_less_equal_basic();
    test_less_equal_various_shapes();
    
    return TestRunner::report();
}
