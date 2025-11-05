#include <gtest/gtest.h>
#include "Tensor.h"
#include "Compiler.h"
#include "ShapeGenerator.h"

// Test fixture for Comparison operator tests
class ComparisonOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the compiler instance for each test
    }
};

// ========== Equality Tests ==========

TEST_F(ComparisonOpsTest, EqualBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 == tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(ComparisonOpsTest, EqualVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 == tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(ComparisonOpsTest, EqualBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 == tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

TEST_F(ComparisonOpsTest, EqualIntType) {
    Tensor<int32_t> tensor1({3, 3});
    Tensor<int32_t> tensor2({3, 3});
    
    auto result = tensor1 == tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

// ========== Not Equal Tests ==========

TEST_F(ComparisonOpsTest, NotEqualBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 != tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(ComparisonOpsTest, NotEqualVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 != tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(ComparisonOpsTest, NotEqualBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 != tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Greater Than Tests ==========

TEST_F(ComparisonOpsTest, GreaterThanBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 > tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(ComparisonOpsTest, GreaterThanVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 > tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(ComparisonOpsTest, GreaterThanBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 > tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Greater Equal Tests ==========

TEST_F(ComparisonOpsTest, GreaterEqualBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 >= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(ComparisonOpsTest, GreaterEqualVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 >= tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(ComparisonOpsTest, GreaterEqualBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 >= tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Less Than Tests ==========

TEST_F(ComparisonOpsTest, LessThanBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 < tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(ComparisonOpsTest, LessThanVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 < tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(ComparisonOpsTest, LessThanBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 < tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}

// ========== Less Equal Tests ==========

TEST_F(ComparisonOpsTest, LessEqualBasic) {
    Tensor<float> tensor1({2, 3});
    Tensor<float> tensor2({2, 3});
    
    auto result = tensor1 <= tensor2;
    
    auto shape = result.getShape();
    EXPECT_GT(shape.size(), 0);
}

TEST_F(ComparisonOpsTest, LessEqualVariousShapes) {
    ShapeGenerator gen;
    
    for (const auto& shape : gen) {
        Tensor<float> tensor1(shape);
        Tensor<float> tensor2(shape);
        
        auto result = tensor1 <= tensor2;
        auto resultShape = result.getShape();
        
        EXPECT_GT(resultShape.size(), 0);
    }
}

TEST_F(ComparisonOpsTest, LessEqualBroadcasting) {
    BroadcastShapeGenerator gen;
    
    for (const auto& pair : gen) {
        Tensor<float> tensor1(pair.shape1);
        Tensor<float> tensor2(pair.shape2);
        
        auto result = tensor1 <= tensor2;
        
        auto shape = result.getShape();
        EXPECT_GT(shape.size(), 0);
    }
}
