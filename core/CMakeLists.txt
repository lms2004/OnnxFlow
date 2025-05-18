# 设置 core 目录下的路径
set(ONNXFLOW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ONNXFLOW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source")

# 自动查找 CUDA 包
find_package(CUDA REQUIRED)

# 使用 GLOB_RECURSE 收集所有源文件
file(GLOB_RECURSE SOURCES "${ONNXFLOW_SOURCE_DIR}/**/*.cpp")

# 获取所有头文件并提取其所在目录（如果需要）
file(GLOB_RECURSE HEADER_FILES 
    "${ONNXFLOW_INCLUDE_DIR}/*.h" 
    "${ONNXFLOW_INCLUDE_DIR}/*.hpp"
    "${ONNXFLOW_INCLUDE_DIR}/*.cuh"  # 如果有 CUDA 头文件的话
)
# 提取目录路径并去重
set(INCLUDE_DIRS "")
foreach(header ${HEADER_FILES})
    get_filename_component(dir ${header} DIRECTORY)
    list(APPEND INCLUDE_DIRS ${dir})
endforeach()
list(REMOVE_DUPLICATES INCLUDE_DIRS)

# 创建静态库（包含所有源文件）
add_library(ONNXFLOW_CORE STATIC 
    ${SOURCES}
)
target_include_directories(ONNXFLOW_CORE
    PUBLIC
        ${INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(ONNXFLOW_CORE PRIVATE ${CUDA_LIBRARIES})

# 如果需要主程序，可以在这里定义（但建议单独处理）
add_executable(ONNXFLOW 
    "${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp" 
)
target_include_directories(ONNXFLOW
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(ONNXFLOW PRIVATE ONNXFLOW_CORE)