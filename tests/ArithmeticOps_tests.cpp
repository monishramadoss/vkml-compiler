#include <gtest/gtest.h>
#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"

// Test fixture for Arithmetic operator tests
class ArithmeticOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the compiler instance for each test
    }
};

// ========== Addition Tests ==========

TEST_F(ArithmeticOpsTest, AdditionBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 + tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, AdditionVariousShapes) {
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
}

TEST_F(ArithmeticOpsTest, AdditionBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        // This should work with broadcasting
        auto result = tensor1 + tensor2;
        
        // Result should have a valid shape
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

TEST_F(ArithmeticOpsTest, AdditionIntType) {
    Tensor<int32_t> tensor1({3, 3});
    Tensor<int32_t> tensor2({3, 3});
    
    auto result = tensor1 + tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
}

// ========== Subtraction Tests ==========

TEST_F(ArithmeticOpsTest, SubtractionBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 - tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, SubtractionVariousShapes) {
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
}

TEST_F(ArithmeticOpsTest, SubtractionBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 - tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Multiplication Tests ==========

TEST_F(ArithmeticOpsTest, MultiplicationBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 * tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, MultiplicationVariousShapes) {
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
}

TEST_F(ArithmeticOpsTest, MultiplicationBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 * tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Division Tests ==========

TEST_F(ArithmeticOpsTest, DivisionFloatBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, DivisionFloatVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 / tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(ArithmeticOpsTest, DivisionIntBasic) {
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 / tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, DivisionIntVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<int32_t> tensor1(shape);
        Tensor<int32_t> tensor2(shape);
        
        auto result = tensor1 / tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

// ========== Modulo Tests ==========

TEST_F(ArithmeticOpsTest, ModuloBasic) {
    Tensor<int32_t> tensor1({2, 3});
    Tensor<int32_t> tensor2({2, 3});
    
    auto result = tensor1 % tensor2;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, ModuloVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<int32_t> tensor1(shape);
        Tensor<int32_t> tensor2(shape);
        
        auto result = tensor1 % tensor2;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

// ========== Unary Plus (Abs) Tests ==========

TEST_F(ArithmeticOpsTest, UnaryPlusBasic) {
    Tensor<float> tensor({2, 3});
    
    auto result = +tensor;
    
    auto shape = result.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, UnaryPlusVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor(shape);
        
        auto result = +tensor;
        auto resultShape = result.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

// ========== Increment/Decrement Tests ==========

TEST_F(ArithmeticOpsTest, PrefixIncrementBasic) {
    Tensor<float> tensor({2, 3});
    
    ++tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, PrefixIncrementVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor(shape);
        
        ++tensor;
        auto resultShape = tensor.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(ArithmeticOpsTest, PostfixIncrementBasic) {
    Tensor<float> tensor({2, 3});
    
    tensor++;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, PrefixDecrementBasic) {
    Tensor<float> tensor({2, 3});
    
    --tensor;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(ArithmeticOpsTest, PrefixDecrementVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor(shape);
        
        --tensor;
        auto resultShape = tensor.getShape();
        
        ASSERT_EQ(resultShape.size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(resultShape[i], shape[i]);
        }
    }
}

TEST_F(ArithmeticOpsTest, PostfixDecrementBasic) {
    Tensor<float> tensor({2, 3});
    
    tensor--;
    
    auto shape = tensor.getShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}
