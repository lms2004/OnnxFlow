#include <gtest/gtest.h>
#include "alloc.h"
#include <iostream>

// 测试 CPUAllocator 的分配和释放功能
TEST(CPUAllocatorTest, AllocateAndDeallocate) {
    // 获取 CPUAllocator 的实例
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    
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

TEST(CPUAllocatorTest, AllocateErrorValue){
    // 获取 CPUAllocator 的实例
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    
    // 分配零字节内存
    void* ptr = allocator->allocate(-1);
    ASSERT_EQ(ptr, nullptr);  // 确保分配零字节内存返回 nullptr
}


// 测试 CUDADeviceAllocator 的分配和释放功能
TEST(CUDADeviceAllocatorTest, AllocateAndDeallocate) {
    // 获取 CUDADeviceAllocator 的实例
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);
    
    // 分配内存
    void* ptr = allocator->allocate(26);
    ASSERT_NE(ptr, nullptr);  // 确保内存分配成功
    
    // 释放内存
    allocator->deallocate(&ptr);
    ASSERT_EQ(ptr, nullptr);  // 确保内存被释放并且指针被置为空
}

TEST(CUDADeviceAllocatorTest, AllocateErrorValue){
    // 获取 CUDADeviceAllocator 的实例
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);
    
    // 分配零字节内存
    void* ptr = allocator->allocate(-1);
    ASSERT_EQ(ptr, nullptr);  // 确保分配零字节内存返回 nullptr
}


// 测试 DeviceAllocator 的 memcpy 功能
TEST(DeviceAllocatorTest, MemcpyTest) {
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);

    // 分配源和目标内存
    size_t byte_size = 26;
    void* src_ptr = allocator->allocate(byte_size);
    void* dest_ptr = allocator->allocate(byte_size);
    
    // 初始化源内存
    char a = 'a';
    for (size_t i = 0; i < byte_size; i++) {
        ((char*)src_ptr)[i] = a + i;  // 设置字符值
    }

    // 使用 memcpy 进行内存复制
    allocator->memcpy(src_ptr, dest_ptr, byte_size, MemcpyKind::kMemcpyCPU2CPU, nullptr, false);
    
    // 验证目标内存内容
    for (size_t i = 0; i < byte_size; i++) {
        ASSERT_EQ(((char*)dest_ptr)[i], a + i);  // 验证每个字符值
    }

    // 释放内存
    allocator->deallocate(&src_ptr);
    allocator->deallocate(&dest_ptr);
}

// 测试 DeviceAllocator 的 memset_zero 功能
TEST(DeviceAllocatorTest, MemsetZeroTest) {
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);

    // 分配内存
    size_t byte_size = 26;
    void* ptr = allocator->allocate(byte_size);
    
    // 使用 memset_zero 清空内存
    allocator->memset_zero(ptr, byte_size, nullptr, false);
    
    // 验证内存是否被清空
    for (size_t i = 0; i < byte_size; i++) {
        ASSERT_EQ(((char*)ptr)[i], 0);  // 验证每个字节是否为 0
    }

    // 释放内存
    allocator->deallocate(&ptr);
}
