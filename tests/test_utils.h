#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>

// Simple test framework for CTest
class TestRunner {
public:
    static int failures;
    static int tests;
    static std::string current_test;
    
    static void begin_test(const std::string& name) {
        current_test = name;
        tests++;
    }
    
    static void assert_true(bool condition, const std::string& message, 
                           const char* file, int line) {
        if (!condition) {
            std::cerr << "FAILED: " << current_test << "\n";
            std::cerr << "  " << file << ":" << line << ": " << message << "\n";
            failures++;
        }
    }
    
    static void assert_equal(long long expected, long long actual,
                           const char* file, int line) {
        if (expected != actual) {
            std::cerr << "FAILED: " << current_test << "\n";
            std::cerr << "  " << file << ":" << line << ": Expected " << expected 
                     << " but got " << actual << "\n";
            failures++;
        }
    }
    
    static void assert_not_equal(long long val1, long long val2,
                                const char* file, int line) {
        if (val1 == val2) {
            std::cerr << "FAILED: " << current_test << "\n";
            std::cerr << "  " << file << ":" << line << ": Values should not be equal: " 
                     << val1 << "\n";
            failures++;
        }
    }
    
    // Overloads for string comparison
    static void assert_equal(const std::string& expected, const std::string& actual,
                           const char* file, int line) {
        if (expected != actual) {
            std::cerr << "FAILED: " << current_test << "\n";
            std::cerr << "  " << file << ":" << line << ": Expected \"" << expected 
                     << "\" but got \"" << actual << "\"\n";
            failures++;
        }
    }
    
    static void assert_not_equal(const std::string& val1, const std::string& val2,
                                const char* file, int line) {
        if (val1 == val2) {
            std::cerr << "FAILED: " << current_test << "\n";
            std::cerr << "  " << file << ":" << line << ": Values should not be equal: \"" 
                     << val1 << "\"\n";
            failures++;
        }
    }
    
    static void pass_test() {
        std::cout << "PASSED: " << current_test << "\n";
    }
    
    static int report() {
        std::cout << "\n========================================\n";
        std::cout << "Tests run: " << tests << "\n";
        std::cout << "Failures: " << failures << "\n";
        if (failures == 0) {
            std::cout << "All tests PASSED!\n";
        } else {
            std::cout << "Some tests FAILED!\n";
        }
        std::cout << "========================================\n";
        return failures > 0 ? 1 : 0;
    }
};

// Initialize static members
int TestRunner::failures = 0;
int TestRunner::tests = 0;
std::string TestRunner::current_test = "";

#define TEST_BEGIN(name) TestRunner::begin_test(name)
#define TEST_END() TestRunner::pass_test()
#define ASSERT_TRUE(condition) TestRunner::assert_true((condition), #condition, __FILE__, __LINE__)
#define ASSERT_FALSE(condition) TestRunner::assert_true(!(condition), "!(" #condition ")", __FILE__, __LINE__)
#define ASSERT_EQ(expected, actual) TestRunner::assert_equal((expected), (actual), __FILE__, __LINE__)
#define ASSERT_NE(val1, val2) TestRunner::assert_not_equal((val1), (val2), __FILE__, __LINE__)
#define ASSERT_GT(val1, val2) TestRunner::assert_true((val1) > (val2), #val1 " > " #val2, __FILE__, __LINE__)
#define EXPECT_EQ(expected, actual) ASSERT_EQ(expected, actual)
#define EXPECT_NE(val1, val2) ASSERT_NE(val1, val2)
#define EXPECT_GT(val1, val2) ASSERT_GT(val1, val2)
