cmake_minimum_required(VERSION 3.14)
project(ONNXFLOW)

# 设置 C++ 标准为 17（GoogleTest 要求）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)  # 启用调试模式

# 使用 FetchContent 下载 GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true  # 设置此选项来避免警告
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # 兼容 Windows
FetchContent_MakeAvailable(googletest)

# --------------------------------- 设置路径 ---------------------------------
set(ONNXFLOW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/core/include")
set(ONNXFLOW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/core/source")
set(ONNXFLOW_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")


# --------------------------------- 全局配置 ---------------------------------
# 自动查找 CUDA 包
find_package(CUDA)

# 如果没有找到 CUDA，退出并显示错误
if (NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA 未找到！请确保 CUDA 已安装并可用。")
endif()

# 获取源代码和测试的 .cpp 文件
file(GLOB SOURCES CONFIGURE_DEPENDS "${ONNXFLOW_SOURCE_DIR}/*.cpp")



# --------------------------------- 主程序配置 ---------------------------------
add_executable(ONNXFLOW main.cpp ${SOURCES})
target_include_directories(ONNXFLOW
  PRIVATE
    ${ONNXFLOW_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}  # 自动设置 CUDA 头文件目录
)

# --------------------------------- 测试程序配置 ---------------------------------
enable_testing()

# 添加测试可执行文件（test_alloc）
add_executable(test_alloc test/test_alloc.cpp ${SOURCES})
target_include_directories(test_alloc
  PRIVATE
    ${ONNXFLOW_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}  # 自动设置 CUDA 头文件目录
)


# --------------------------------- 链接配置 ---------------------------------

# 链接 CUDA 库
target_link_libraries(ONNXFLOW PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(test_alloc PRIVATE ${CUDA_LIBRARIES} GTest::gtest_main)

# 自动发现并注册测试用例
include(GoogleTest)
gtest_discover_tests(test_alloc)
