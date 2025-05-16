#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include "myfunc.h"

// 测试 Memcpy 的函数
TEST(MemcpyTest, HandlesNullDest) {
    // 捕获 stderr 输出
    ::testing::internal::CaptureStderr();
    
    char source[] = "Hello, World!";
    void* dest = NULL;
    Memcpy(dest, source, 13);  // 应该打印错误日志
    
    // 获取捕获的日志内容
    std::string output = ::testing::internal::GetCapturedStderr();

    // 验证错误日志内容
    EXPECT_TRUE(output.find("memcpy dest or src is NULL") != std::string::npos);
}

TEST(MemcpyTest, HandlesNullSrc) {
    // 捕获 stderr 输出
    ::testing::internal::CaptureStderr();
    
    char* source = NULL;
    char dest[20];
    Memcpy(dest, source, 13);  // 应该打印错误日志
    
    // 获取捕获的日志内容
    std::string output = ::testing::internal::GetCapturedStderr();

    // 验证错误日志内容
    EXPECT_TRUE(output.find("memcpy dest or src is NULL") != std::string::npos);
}

TEST(MemcpyTest, HandlesZeroCount) {
    // 捕获 stderr 输出
    ::testing::internal::CaptureStderr();
    
    char source[] = "Hello, World!";
    char dest[20];
    Memcpy(dest, source, 0);  // 应该打印错误日志
    
    // 获取捕获的日志内容
    std::string output = ::testing::internal::GetCapturedStderr();
    
    // 验证错误日志内容
    EXPECT_TRUE(output.find("memcpy count is 0") != std::string::npos);  // 确保输出包含该日志
}

TEST(MemcpyTest, HandlesValidMemcpy) {
    char source[] = "Hello, World!";
    char dest[20] = {0};
    Memcpy(dest, source, 13);  // 应该成功复制

    EXPECT_STREQ(dest, "Hello, World!");  // 验证复制结果
}

