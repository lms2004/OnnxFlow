cmake_minimum_required(VERSION 3.14)
project(ONNXFLOW)

# Set C++ standard to 17 (required by GoogleTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download GoogleTest using FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # For Windows compatibility
FetchContent_MakeAvailable(googletest)



set(ONNXFLOW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/core/include")


# --------------------------------- 主程序配置 ---------------------------------
add_executable(ONNXFLOW main.cpp)
target_include_directories(ONNXFLOW
  PRIVATE
    ${ONNXFLOW_INCLUDE_DIR}
)


# --------------------------------- 测试程序配置 ---------------------------------
enable_testing()

# 添加测试可执行文件（test_alloc）
add_executable(test_alloc test/test_alloc.cpp)

target_include_directories(test_alloc
  PRIVATE
    ${ONNXFLOW_INCLUDE_DIR}
)

target_link_libraries(test_alloc PRIVATE GTest::gtest_main)  # 注意小写

# 自动发现并注册测试用例
include(GoogleTest)
gtest_discover_tests(test_alloc)