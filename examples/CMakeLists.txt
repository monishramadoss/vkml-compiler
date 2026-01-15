# Examples for VKML Compiler

add_executable(vulkan_pipeline_example vulkan_pipeline_example.cpp)
target_link_libraries(vulkan_pipeline_example PRIVATE vkml-compiler)
target_include_directories(vulkan_pipeline_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../inc)
