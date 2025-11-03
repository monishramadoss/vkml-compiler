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

# Create test executable
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

include(GoogleTest)
gtest_discover_tests(tensor_tests)
