include(GoogleTest)
enable_testing()

# 设置测试相关的路径
set(ONNXFLOW_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# 使用 GLOB_RECURSE 收集测试源文件
file(GLOB_RECURSE TEST_SOURCESS "${ONNXFLOW_TEST_DIR}/*.cpp")
file(GLOB_RECURSE TEST_MYFUNC_SOURCES "${ONNXFLOW_TEST_DIR}/test_func/*.cpp")

# 定义一个函数来创建测试可执行文件
function(add_test_executable TEST_NAME TEST_SOURCES)
    # 确保 TEST_SOURCES 是有效的
    if(NOT EXISTS ${TEST_SOURCES})
        message(FATAL_ERROR "Test source file not found: ${TEST_SOURCES}")
    endif()

    add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_include_directories(${TEST_NAME}
        PRIVATE
            "${CMAKE_SOURCE_DIR}/core/include"
            ${CUDA_INCLUDE_DIRS}
    )
    target_link_libraries(${TEST_NAME} PRIVATE ONNXFLOW_CORE GTest::gtest_main)

    # 添加测试到 CTest
    include(GoogleTest)
    gtest_discover_tests(${TEST_NAME})
    message(STATUS "Test executable created: ${TEST_NAME}")
endfunction()

# 为每个测试文件创建一个独立的测试可执行文件
foreach(TEST_FILE ${TEST_SOURCESS})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_test_executable(${TEST_NAME} ${TEST_FILE})
endforeach()

# 判断 TEST_FUNC 是否启用
option(TEST_FUNC "Enable test_func tests" OFF)
if(TEST_FUNC)
    foreach(TEST_FILE ${TEST_MYFUNC_SOURCES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_test_executable(${TEST_NAME} ${TEST_FILE})
    endforeach()
else()
    message(STATUS "TEST_FUNC is disabled. Skipping test creation.")
endif()