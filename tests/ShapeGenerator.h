#pragma once

#include <vector>
#include <cstdint>
#include <random>

/**
 * @brief Generator that yields strange and abnormal shapes for testing tensor operations
 * 
 * This generator produces various edge cases and unusual shapes to stress-test
 * tensor operations, including:
 * - Empty dimensions (size 0)
 * - Single element dimensions (size 1)
 * - Very large dimensions
 * - High-rank tensors
 * - Asymmetric shapes
 * - Prime number dimensions
 */
class ShapeGenerator {
public:
    /**
     * @brief Iterator for generating test shapes
     */
    class Iterator {
    private:
        size_t index_;
        std::vector<std::vector<int64_t>> shapes_;
        
    public:
        Iterator(size_t index, const std::vector<std::vector<int64_t>>& shapes)
            : index_(index), shapes_(shapes) {}
        
        const std::vector<int64_t>& operator*() const {
            return shapes_[index_];
        }
        
        Iterator& operator++() {
            ++index_;
            return *this;
        }
        
        bool operator!=(const Iterator& other) const {
            return index_ != other.index_;
        }
    };
    
    ShapeGenerator() {
        generateShapes();
    }
    
    Iterator begin() const {
        return Iterator(0, shapes_);
    }
    
    Iterator end() const {
        return Iterator(shapes_.size(), shapes_);
    }
    
    const std::vector<std::vector<int64_t>>& getShapes() const {
        return shapes_;
    }
    
private:
    std::vector<std::vector<int64_t>> shapes_;
    
    void generateShapes() {
        // 1D shapes - edge cases
        shapes_.push_back({1});           // Scalar-like
        shapes_.push_back({2});           // Minimal non-trivial
        shapes_.push_back({3});           // Small odd
        shapes_.push_back({7});           // Prime number
        shapes_.push_back({16});          // Power of 2
        shapes_.push_back({100});         // Larger dimension
        shapes_.push_back({1024});        // Large power of 2
        
        // 2D shapes - standard cases
        shapes_.push_back({1, 1});        // Single element
        shapes_.push_back({1, 5});        // Row vector
        shapes_.push_back({5, 1});        // Column vector
        shapes_.push_back({2, 2});        // Small square
        shapes_.push_back({3, 3});        // Odd square
        shapes_.push_back({2, 3});        // Small rectangular
        shapes_.push_back({3, 2});        // Transposed rectangular
        shapes_.push_back({4, 5});        // Rectangular
        shapes_.push_back({5, 4});        // Transposed
        
        // 2D shapes - edge cases
        shapes_.push_back({1, 100});      // Very wide
        shapes_.push_back({100, 1});      // Very tall
        shapes_.push_back({7, 11});       // Prime dimensions
        shapes_.push_back({16, 16});      // Power of 2 square
        shapes_.push_back({32, 64});      // Large power of 2
        shapes_.push_back({13, 17});      // Larger primes
        
        // 3D shapes
        shapes_.push_back({1, 1, 1});     // Single element 3D
        shapes_.push_back({2, 2, 2});     // Small cube
        shapes_.push_back({1, 3, 4});     // Batch of matrices
        shapes_.push_back({2, 3, 4});     // Common 3D shape
        shapes_.push_back({3, 2, 4});     // Different ordering
        shapes_.push_back({4, 3, 2});     // Reversed
        shapes_.push_back({1, 16, 16});   // Image-like with batch
        shapes_.push_back({5, 7, 11});    // All prime dimensions
        shapes_.push_back({8, 8, 8});     // Cube power of 2
        
        // 4D shapes (common in ML)
        shapes_.push_back({1, 1, 1, 1});  // Single element 4D
        shapes_.push_back({1, 3, 28, 28}); // Small image batch
        shapes_.push_back({2, 3, 32, 32}); // Batch of images
        shapes_.push_back({4, 64, 7, 7}); // Feature maps
        shapes_.push_back({1, 1, 5, 5});  // Single channel feature
        
        // 5D and higher - stress test
        shapes_.push_back({1, 1, 1, 1, 1}); // 5D single element
        shapes_.push_back({2, 2, 2, 2, 2}); // 5D small
        shapes_.push_back({1, 2, 3, 4, 5}); // 5D ascending
        shapes_.push_back({1, 1, 1, 1, 1, 1}); // 6D
        
        // Asymmetric shapes with broadcasting potential
        shapes_.push_back({1, 3});        // Broadcastable row
        shapes_.push_back({3, 1});        // Broadcastable column
        shapes_.push_back({1, 1, 3});     // Broadcastable 3D
        shapes_.push_back({3, 1, 1});     // Different broadcasting
        shapes_.push_back({1, 3, 1});     // Middle dim broadcast
        
        // Odd size combinations
        shapes_.push_back({7, 13, 5});    // All different primes
        shapes_.push_back({11, 1, 7});    // Prime with broadcast
        shapes_.push_back({1, 11, 13, 1}); // Multiple broadcast dims
    }
};

/**
 * @brief Generator for shapes suitable for broadcasting operations
 */
class BroadcastShapeGenerator {
public:
    struct ShapePair {
        std::vector<int64_t> shape1;
        std::vector<int64_t> shape2;
    };
    
    class Iterator {
    private:
        size_t index_;
        const std::vector<ShapePair>& pairs_;
        
    public:
        Iterator(size_t index, const std::vector<ShapePair>& pairs)
            : index_(index), pairs_(pairs) {}
        
        const ShapePair& operator*() const {
            return pairs_[index_];
        }
        
        Iterator& operator++() {
            ++index_;
            return *this;
        }
        
        bool operator!=(const Iterator& other) const {
            return index_ != other.index_;
        }
    };
    
    BroadcastShapeGenerator() {
        generatePairs();
    }
    
    Iterator begin() const {
        return Iterator(0, pairs_);
    }
    
    Iterator end() const {
        return Iterator(pairs_.size(), pairs_);
    }
    
private:
    std::vector<ShapePair> pairs_;
    
    void generatePairs() {
        // Same shape pairs
        pairs_.push_back({{2, 3}, {2, 3}});
        pairs_.push_back({{4, 5}, {4, 5}});
        
        // Broadcasting with scalars (1D)
        pairs_.push_back({{1}, {5}});
        pairs_.push_back({{5}, {1}});
        
        // 2D broadcasting cases
        pairs_.push_back({{1, 3}, {2, 3}});  // Row broadcast
        pairs_.push_back({{2, 1}, {2, 3}});  // Column broadcast
        pairs_.push_back({{1, 1}, {3, 4}});  // Full broadcast
        pairs_.push_back({{3, 4}, {1, 4}});  // Row broadcast
        pairs_.push_back({{3, 4}, {3, 1}});  // Column broadcast
        
        // Different rank broadcasting
        pairs_.push_back({{3}, {2, 3}});     // 1D to 2D
        pairs_.push_back({{2, 3}, {3}});     // 2D to 1D
        pairs_.push_back({{1, 3}, {2, 1, 3}}); // 2D to 3D
        pairs_.push_back({{2, 3, 4}, {4}});  // 3D to 1D
        pairs_.push_back({{2, 3, 4}, {3, 4}}); // 3D to 2D
        
        // Complex broadcasting scenarios
        pairs_.push_back({{1, 5, 1, 7}, {3, 1, 4, 1}});
        pairs_.push_back({{2, 1, 3}, {1, 4, 3}});
        pairs_.push_back({{1, 1, 5}, {3, 4, 1}});
        
        // Edge cases
        pairs_.push_back({{1, 1, 1}, {5, 7, 11}});  // All broadcast
        pairs_.push_back({{8, 1, 6, 1}, {7, 1, 5}});  // Partial overlap
    }
};

/**
 * @brief Generator for random shapes with configurable parameters
 */
class RandomShapeGenerator {
public:
    RandomShapeGenerator(int seed = 42) : gen_(seed) {}
    
    /**
     * @brief Generate a random shape with given constraints
     * @param minRank Minimum tensor rank
     * @param maxRank Maximum tensor rank
     * @param minDim Minimum dimension size
     * @param maxDim Maximum dimension size
     */
    std::vector<int64_t> generate(int minRank = 1, int maxRank = 4, 
                                   int64_t minDim = 1, int64_t maxDim = 10) {
        std::uniform_int_distribution<int> rankDist(minRank, maxRank);
        std::uniform_int_distribution<int64_t> dimDist(minDim, maxDim);
        
        int rank = rankDist(gen_);
        std::vector<int64_t> shape;
        shape.reserve(rank);
        
        for (int i = 0; i < rank; ++i) {
            shape.push_back(dimDist(gen_));
        }
        
        return shape;
    }
    
    /**
     * @brief Generate multiple random shapes
     */
    std::vector<std::vector<int64_t>> generateMultiple(int count, int minRank = 1, 
                                                       int maxRank = 4,
                                                       int64_t minDim = 1, 
                                                       int64_t maxDim = 10) {
        std::vector<std::vector<int64_t>> shapes;
        shapes.reserve(count);
        
        for (int i = 0; i < count; ++i) {
            shapes.push_back(generate(minRank, maxRank, minDim, maxDim));
        }
        
        return shapes;
    }
    
private:
    std::mt19937 gen_;
};
