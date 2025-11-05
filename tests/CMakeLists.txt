# Google Test setup
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Create test executable for original tests
add_executable(tensor_tests
  Tensor_tests.cpp
)

target_link_libraries(tensor_tests
  PRIVATE
    vkml-compiler
    GTest::gtest_main
)

target_include_directories(tensor_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
)

# Create test executables for operator-specific tests (using CTest)
add_executable(arithmetic_ops_tests
  ArithmeticOps_tests.cpp
)

target_link_libraries(arithmetic_ops_tests
  PRIVATE
    vkml-compiler
)

target_include_directories(arithmetic_ops_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(bitwise_ops_tests
  BitwiseOps_tests.cpp
)

target_link_libraries(bitwise_ops_tests
  PRIVATE
    vkml-compiler
)

target_include_directories(bitwise_ops_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(comparison_ops_tests
  ComparisonOps_tests.cpp
)

target_link_libraries(comparison_ops_tests
  PRIVATE
    vkml-compiler
)

target_include_directories(comparison_ops_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(logical_ops_tests
  LogicalOps_tests.cpp
)

target_link_libraries(logical_ops_tests
  PRIVATE
    vkml-compiler
)

target_include_directories(logical_ops_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(other_ops_tests
  OtherOps_tests.cpp
)

target_link_libraries(other_ops_tests
  PRIVATE
    vkml-compiler
)

target_include_directories(other_ops_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(tensor_tests)

add_test(NAME ArithmeticOps COMMAND arithmetic_ops_tests)
add_test(NAME BitwiseOps COMMAND bitwise_ops_tests)
add_test(NAME ComparisonOps COMMAND comparison_ops_tests)
add_test(NAME LogicalOps COMMAND logical_ops_tests)
add_test(NAME OtherOps COMMAND other_ops_tests)
