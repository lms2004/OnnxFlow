#include <gtest/gtest.h>
#include <cstring>  // For memcpy
#include <iostream>
#include "myfunc.h"

// Memcpy 测试类
class MemcpyTest : public ::testing::Test {
protected:
    // 如果需要设置其他内容
    void SetUp() override {}

    void TearDown() override {}
};

// 测试正常的 memcpy 操作
TEST_F(MemcpyTest, CopyDataSuccessfully) {
    const char* src = "Hello, World!";
    char dest[20];

    // 执行 memcpy 操作
    void* result = Memcpy(dest, src, strlen(src) + 1);  // +1 是为了复制字符串的结束符
    EXPECT_NE(result, nullptr);  // 确保返回值不为 nullptr
    EXPECT_STREQ(dest, src);  // 确保 dest 的内容和 src 一致
}

// 测试目标指针为 NULL 的情况
TEST_F(MemcpyTest, DestinationIsNull) {
    const char* src = "Hello, World!";
    void* result = Memcpy(nullptr, src, strlen(src) + 1);
    EXPECT_EQ(result, nullptr);  // 确保返回值为 nullptr
}

// 测试源指针为 NULL 的情况
TEST_F(MemcpyTest, SourceIsNull) {
    char dest[20];
    void* result = Memcpy(dest, nullptr, strlen(dest) + 1);
    EXPECT_EQ(result, nullptr);  // 确保返回值为 nullptr
}

// 测试复制字节数为 0 的情况
TEST_F(MemcpyTest, CountIsZero) {
    char dest[20];
    const char* src = "Hello, World!";
    void* result = Memcpy(dest, src, 0);
    EXPECT_EQ(result, nullptr);  // 确保返回值为 nullptr
}
