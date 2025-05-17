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

// 测试 DeviceAllocator 的 memcpy 功能（CPU到CPU，同步）
TEST(DeviceAllocatorTest, MemcpyCPU2CPUSync) {
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);

    // 分配源和目标内存
    size_t byte_size = 26;
    void* src_ptr = allocator->allocate(byte_size);
    void* dest_ptr = allocator->allocate(byte_size);
    
    // 初始化源内存
    char a = 'a';
    for (size_t i = 0; i < byte_size; i++) {
        ((char*)src_ptr)[i] = a + i;
    }

    // 使用 memcpy 进行同步内存复制
    allocator->memcpy(src_ptr, dest_ptr, byte_size, MemcpyKind::kMemcpyCPU2CPU, nullptr, true);
    
    // 验证目标内存内容
    for (size_t i = 0; i < byte_size; i++) {
        ASSERT_EQ(((char*)dest_ptr)[i], a + i);
    }

    // 释放内存
    allocator->deallocate(&src_ptr);
    allocator->deallocate(&dest_ptr);
}

// 测试 DeviceAllocator 的 memcpy 功能（CUDA相关，需CUDA环境）
TEST(DeviceAllocatorTest, MemcpyCUDA2CUDA) {
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);

    // 分配源和目标内存
    size_t byte_size = 26;
    void* src_ptr = allocator->allocate(byte_size);
    void* dest_ptr = allocator->allocate(byte_size);
    
    // 初始化源内存（在主机上准备数据，然后复制到设备）
    char* host_src = new char[byte_size];
    char a = 'a';
    for (size_t i = 0; i < byte_size; i++) {
        host_src[i] = a + i;
    }
    
    // 将数据从主机复制到设备
    allocator->memcpy(host_src, src_ptr, byte_size, MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

    // 使用 memcpy 进行设备到设备的复制
    allocator->memcpy(src_ptr, dest_ptr, byte_size, MemcpyKind::kMemcpyCUDA2CUDA, nullptr, true);
    
    // 将结果复制回主机进行验证
    char* host_dest = new char[byte_size];
    allocator->memcpy(dest_ptr, host_dest, byte_size, MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
    
    // 验证目标内存内容
    for (size_t i = 0; i < byte_size; i++) {
        ASSERT_EQ(host_dest[i], a + i);
    }

    // 释放内存
    allocator->deallocate(&src_ptr);
    allocator->deallocate(&dest_ptr);
    delete[] host_src;
    delete[] host_dest;
}

// 测试 DeviceAllocator 的 memcpy 功能（零字节复制）
TEST(DeviceAllocatorTest, MemcpyZeroByte) {
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);

    // 分配源和目标内存
    void* src_ptr = allocator->allocate(26);
    void* dest_ptr = allocator->allocate(26);
    
    // 使用 memcpy 进行零字节复制
    allocator->memcpy(src_ptr, dest_ptr, 0, MemcpyKind::kMemcpyCPU2CPU, nullptr, false);
    
    // 无需验证内容，因为没有数据被复制

    // 释放内存
    allocator->deallocate(&src_ptr);
    allocator->deallocate(&dest_ptr);
}

// 测试 DeviceAllocator 的 memcpy 功能（无效memcpy类型）
TEST(DeviceAllocatorTest, MemcpyInvalidKind) {
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);

    // 分配源和目标内存
    size_t byte_size = 26;
    void* src_ptr = allocator->allocate(byte_size);
    void* dest_ptr = allocator->allocate(byte_size);
    
    // 初始化源内存
    char a = 'a';
    for (size_t i = 0; i < byte_size; i++) {
        ((char*)src_ptr)[i] = a + i;
    }

    // 使用无效的 memcpy 类型
    allocator->memcpy(src_ptr, dest_ptr, byte_size, static_cast<MemcpyKind>(999), nullptr, false);
    
    // 验证目标内存未被修改（假设目标内存初始为零或其他值）
    for (size_t i = 0; i < byte_size; i++) {
        ASSERT_NE(((char*)dest_ptr)[i], a + i);
    }

    // 释放内存
    allocator->deallocate(&src_ptr);
    allocator->deallocate(&dest_ptr);
}

// 测试 DeviceAllocator 的 memcpy 功能（异步复制，CUDA环境）
TEST(DeviceAllocatorTest, MemcpyCUDA2CUDAAync) {
    std::shared_ptr<DeviceAllocator> allocator = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 分配源和目标内存
    size_t byte_size = 26;
    void* src_ptr = allocator->allocate(byte_size);
    void* dest_ptr = allocator->allocate(byte_size);
    
    // 初始化源内存（在主机上准备数据，然后复制到设备）
    char* host_src = new char[byte_size];
    char a = 'a';
    for (size_t i = 0; i < byte_size; i++) {
        host_src[i] = a + i;
    }
    
    // 将数据从主机异步复制到设备
    allocator->memcpy(host_src, src_ptr, byte_size, MemcpyKind::kMemcpyCPU2CUDA, stream, false);

    // 使用 memcpy 进行设备到设备的异步复制
    allocator->memcpy(src_ptr, dest_ptr, byte_size, MemcpyKind::kMemcpyCUDA2CUDA, stream, false);
    
    // 将结果异步复制回主机
    char* host_dest = new char[byte_size];
    allocator->memcpy(dest_ptr, host_dest, byte_size, MemcpyKind::kMemcpyCUDA2CPU, stream, true);
    
    // 验证目标内存内容
    for (size_t i = 0; i < byte_size; i++) {
        ASSERT_EQ(host_dest[i], a + i);
    }

    // 释放内存
    allocator->deallocate(&src_ptr);
    allocator->deallocate(&dest_ptr);
    delete[] host_src;
    delete[] host_dest;
    
    // 销毁流
    cudaStreamDestroy(stream);
}
