cmake_minimum_required(VERSION 3.14)
project(ONNXFLOW)

# 设置 C++ 标准为 17（GoogleTest 要求）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)  # 启用调试模式

# set(TEST_FUNC ON)  # 启用测试 myfunc


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

# 获取 base 目录下的所有源文件
file(GLOB SOURCES CONFIGURE_DEPENDS "${ONNXFLOW_SOURCE_DIR}/base/*.cpp")

# --------------------------------- 主程序配置 ---------------------------------
add_executable(ONNXFLOW main.cpp ${SOURCES})
target_include_directories(ONNXFLOW
  PRIVATE
    ${ONNXFLOW_INCLUDE_DIR}/base
    ${CUDA_INCLUDE_DIRS}  # 自动设置 CUDA 头文件目录
)
target_link_libraries(ONNXFLOW PRIVATE ${CUDA_LIBRARIES})  # 新增这行

# --------------------------------- 测试程序配置 ---------------------------------
enable_testing()

# 获取 test 目录下的所有 .cpp 文件
file(GLOB TEST_SOURCES "${ONNXFLOW_TEST_DIR}/*.cpp")
file(GLOB TEST_MYFUNC_SOURCES "${ONNXFLOW_TEST_DIR}/test_func/*.cpp")

# 为每个测试文件创建一个独立的测试可执行文件
foreach(TEST_FILE ${TEST_SOURCES})
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  add_executable(test_${TEST_NAME} ${TEST_FILE} ${SOURCES})
  target_include_directories(test_${TEST_NAME}
    PRIVATE
      ${ONNXFLOW_INCLUDE_DIR}/base
      ${CUDA_INCLUDE_DIRS}  # 自动设置 CUDA 头文件目录
  )
  target_link_libraries(test_${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} GTest::gtest_main)

  # 自动发现并注册测试用例
  include(GoogleTest)
  gtest_discover_tests(test_${TEST_NAME})
endforeach()


# 判断 TEST_FUNC 是否启用
if(TEST_FUNC)
  # 为每个 test_func 目录下的测试文件创建一个独立的测试可执行文件
  foreach(TEST_FILE ${TEST_MYFUNC_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(test_${TEST_NAME} ${TEST_FILE} ${SOURCES})
    target_include_directories(test_${TEST_NAME}
      PRIVATE
        ${ONNXFLOW_INCLUDE_DIR}/base
        ${CUDA_INCLUDE_DIRS}  # 自动设置 CUDA 头文件目录
    )
    target_link_libraries(test_${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} GTest::gtest_main)

    # 自动发现并注册测试用例
    include(GoogleTest)
    gtest_discover_tests(test_${TEST_NAME})
  endforeach()
else()
    message(STATUS "TEST_FUNC is disabled. Skipping test creation.")
endif()


