#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"
#include "test_utils.h"

void test_create_tensor_with_shape() {
    TEST_BEGIN("Create Tensor With Shape");
    Tensor<float> tensor({2, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_create_1d_tensor() {
    TEST_BEGIN("Create 1D Tensor");
    Tensor<float> tensor({5});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 5);
    TEST_END();
}

void test_create_int_tensor() {
    TEST_BEGIN("Create Int Tensor");
    Tensor<int32_t> tensor({3, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
    TEST_END();
}

void test_create_3d_tensor() {
    TEST_BEGIN("Create 3D Tensor");
    Tensor<float> tensor({2, 3, 4});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    TEST_END();
}

void test_create_various_shapes() {
    TEST_BEGIN("Create Various Shapes");
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor(shape);
        auto resultShape = tensor.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
    TEST_END();
}

void test_symbolic_id_generation() {
    TEST_BEGIN("Symbolic ID Generation");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto id1 = tensor1.getSymbolicId();
    auto id2 = tensor2.getSymbolicId();
    
    // IDs should be different
    EXPECT_NE(id1.compare(id2), 0);
    
    // IDs should start with "tensor_"
    ASSERT_TRUE(id1.substr(0, 7) == "tensor_");
    ASSERT_TRUE(id2.substr(0, 7) == "tensor_");
    TEST_END();
}

void test_subscript_basic() {
    TEST_BEGIN("Subscript Basic");
    Tensor<float> tensor({3, 4});
    
    auto slice = tensor[0];
    
    // Should reduce rank by 1
    auto shape = slice.getShape();
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 4);
    TEST_END();
}

void test_subscript_various_shapes() {
    TEST_BEGIN("Subscript Various Shapes");
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
    TEST_END();
}

void test_broadcast_operation() {
    TEST_BEGIN("Broadcast Operation");
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({1, 3});
    
    // This should work with broadcasting rules
    auto result = tensor1 + tensor2;
    
    // Result should exist
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
    TEST_END();
}

void test_broadcast_various_shape_pairs() {
    TEST_BEGIN("Broadcast Various Shape Pairs");
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        // Test with addition (any binary op would work)
        auto result = tensor1 + tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
    TEST_END();
}

int main() {
    test_create_tensor_with_shape();
    test_create_1d_tensor();
    test_create_int_tensor();
    test_create_3d_tensor();
    test_create_various_shapes();
    test_symbolic_id_generation();
    test_subscript_basic();
    test_subscript_various_shapes();
    test_broadcast_operation();
    test_broadcast_various_shape_pairs();
    
    return TestRunner::report();
}
