<<<<<<< Updated upstream
enable_testing()

# Create test executable for original tests (now using CTest)
add_executable(tensor_tests
  Tensor_tests.cpp
)

target_link_libraries(tensor_tests
  PRIVATE
    vkml-compiler
)

target_include_directories(tensor_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
  ${CMAKE_CURRENT_SOURCE_DIR}
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
add_test(NAME TensorTests COMMAND tensor_tests)
add_test(NAME ArithmeticOps COMMAND arithmetic_ops_tests)
add_test(NAME BitwiseOps COMMAND bitwise_ops_tests)
add_test(NAME ComparisonOps COMMAND comparison_ops_tests)
add_test(NAME LogicalOps COMMAND logical_ops_tests)
add_test(NAME OtherOps COMMAND other_ops_tests)
=======
cmake_minimum_required(VERSION 3.16)
# Simplified test CMakeLists: test executables (e.g., vkml-empty-count) are defined in the root CMakeLists.
# We rely solely on CTest without GoogleTest. Register test targets here with add_test(NAME <name> COMMAND <target>)
# only if they are defined in the root CMakeLists, guarded by BUILD_TESTING.

>>>>>>> Stashed changes
