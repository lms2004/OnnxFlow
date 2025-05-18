cmake_minimum_required(VERSION 3.14)
project(ONNXFLOW)

# 设置 C++ 标准为 17（GoogleTest 要求）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型为 Debug（可以根据需要调整）
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

# 使用 FetchContent 下载 GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true  # 设置此选项来避免警告
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # 兼容 Windows
FetchContent_MakeAvailable(googletest)

# 启用测试
enable_testing()

# 包含子目录的 CMakeLists.txt
add_subdirectory(core)
add_subdirectory(test)