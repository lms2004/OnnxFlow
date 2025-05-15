#include <gtest/gtest.h>
#include "alloc.h"
#include <iostream>

// 测试 CPUAllocator 的分配和释放功能
TEST(CPUAllocatorTest, AllocateAndDeallocate) {
    // 获取 CPUAllocator 的实例
    std::shared_ptr<DeviceAllocator> allocator = CPUAllocatorSingleton::getInstance();
    
    // 分配内存
    void* ptr = allocator->allocate(26);
    ASSERT_NE(ptr, nullptr);  // 确保内存分配成功
    
    // 将分配的内存填充字符并进行验证
    char a = 'a';
    for (int i = 0; i < 26; i++) {
        ((char*)ptr)[i] = a + i;  // 设置字符值
        ASSERT_EQ(((char*)ptr)[i], a + i);  // 验证每个字符值
    }

    // 释放内存
    allocator->deallocate(&ptr);

    ASSERT_EQ(ptr, nullptr);  // 确保内存释放成功
}
