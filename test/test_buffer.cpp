#include <gtest/gtest.h>
#include "buffer.h"
#include <thread>
#define USE_CUDA

// 测试固件类
class BufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
        cuda_alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);
    }

    std::shared_ptr<DeviceAllocator> cpu_alloc;
    std::shared_ptr<DeviceAllocator> cuda_alloc;
    const size_t test_size = 1024; // 1KB测试数据
};

// 测试用例1：Buffer创建逻辑
TEST_F(BufferTest, CreateAndBasicProperties) {
    // 正常创建测试
    auto buffer = Buffer::create(test_size, cpu_alloc, nullptr, false);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->byte_size(), test_size);
    EXPECT_EQ(buffer->device_type(), DeviceType::kDeviceCPU);
    EXPECT_FALSE(buffer->is_external());

    // 外部指针测试
    void* external_ptr = malloc(test_size);
    auto ext_buffer = Buffer::create(test_size, cpu_alloc, external_ptr, true);
    ASSERT_NE(ext_buffer, nullptr);
    EXPECT_TRUE(ext_buffer->is_external());
    free(external_ptr);
}

// 测试用例2：内存分配功能
TEST_F(BufferTest, MemoryAllocation) {
    // 内部分配测试
    auto buffer = Buffer::create(test_size, cpu_alloc, nullptr, false);
    buffer->allocate();
    EXPECT_NE(buffer->ptr(), nullptr);

    // 外部分配验证
    void* external_ptr = malloc(test_size);
    auto ext_buffer = Buffer::create(test_size, cpu_alloc, external_ptr, true);
    ext_buffer->allocate(); 
    EXPECT_NE(ext_buffer->ptr(), external_ptr);
    free(external_ptr);
}

// 测试用例3：数据拷贝功能
TEST_F(BufferTest, DataCopyOperations) {
    // 初始化源缓冲区
    auto src = Buffer::create(test_size, cpu_alloc, nullptr, false);
    src->allocate();
    memset(src->ptr(), 0xAA, test_size); // 填充测试模式

    // 目标缓冲区
    auto dest = Buffer::create(test_size, cpu_alloc, nullptr, false);
    dest->allocate();
    
    // 执行拷贝
    dest->copy_from(src.get()); 

    // 验证数据一致性
    EXPECT_EQ(memcmp(src->ptr(), dest->ptr(), test_size), 0);
}

// 测试用例4：设备类型设置
TEST_F(BufferTest, DeviceTypeHandling) {
    auto buffer = Buffer::create(test_size, cpu_alloc, nullptr, false);
    
    // 初始设备类型验证
    EXPECT_EQ(buffer->device_type(), DeviceType::kDeviceCPU);

    // 修改设备类型
    buffer->set_device_type(DeviceType::kDeviceCUDA);
    EXPECT_EQ(buffer->device_type(), DeviceType::kDeviceCUDA);
}

// 测试用例5：enable_shared_from_this功能验证
TEST_F(BufferTest, SharedPointerLifecycle) {
    // 验证共享指针的引用计数
    auto buffer = Buffer::create(test_size, cpu_alloc, nullptr, false);
    auto shared_ptr = buffer->getptr();
    ASSERT_EQ(buffer.use_count(), 2);  // 原始指针+共享指针
    
    // 验证跨作用域释放
    {
        auto local_ptr = buffer->getptr();
        EXPECT_EQ(buffer.use_count(), 3);
    }
    EXPECT_EQ(buffer.use_count(), 2);
    
    // 验证指针有效性
    buffer.reset();
    EXPECT_EQ(shared_ptr.use_count(), 1);
    EXPECT_NE(shared_ptr->ptr(), nullptr);  // 网页6提到的智能指针有效性验证
}

// 测试用例6：异常场景下的资源安全
//  考虑信号捕获 todo{}
// TEST_F(BufferTest, ExceptionSafety) {
//     // 构造非法参数测试
//     EXPECT_THROW(Buffer::create(0, cpu_alloc, nullptr, false), std::bad_alloc);
    
//     // 验证外部指针异常场景
//     void* invalid_ptr = reinterpret_cast<void*>(0xDEADBEEF);
//     EXPECT_THROW({
//         auto bad_buffer = Buffer::create(test_size, cpu_alloc, invalid_ptr, true);
//         bad_buffer->allocate();  // 网页8提到的异常安全策略
//     }, std::runtime_error);
    
//     // 验证拷贝时设备不匹配异常
//     auto cuda_buffer = Buffer::create(test_size, cuda_alloc, nullptr, false);
//     auto cpu_buffer = Buffer::create(test_size, cpu_alloc, nullptr, false);
//     EXPECT_THROW(cpu_buffer->copy_from(cuda_buffer.get()), std::logic_error);
// }

// 测试用例7：多线程访问验证
TEST_F(BufferTest, ConcurrentAccess) {
    auto shared_buffer = Buffer::create(test_size, cpu_alloc, nullptr, false);
    
    std::vector<std::thread> workers;
    const int thread_count = 4;
    
    for (int i = 0; i < thread_count; ++i) {
        workers.emplace_back([&] {
            auto local_ptr = shared_buffer->getptr();
            memset(local_ptr->ptr(), rand()%256, test_size);  // 网页6建议的线程安全访问
        });
    }
    
    for (auto& t : workers) t.join();
    EXPECT_NE(shared_buffer->ptr(), nullptr);  // 网页7的RAII验证
}



#ifdef USE_CUDA
// 测试用例8：CUDA异步拷贝验证
TEST_F(BufferTest, CudaAsyncCopy) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto src = Buffer::create(test_size, cuda_alloc, nullptr, false);
    auto dest = Buffer::create(test_size, cuda_alloc, nullptr, false);
    src->allocate();
    dest->allocate();

    // 异步拷贝测试
    cudaMemcpyAsync(dest->ptr(), src->ptr(), test_size, cudaMemcpyDeviceToDevice, stream);
    
    // 验证操作完成
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);  // 网页4的事件验证
    
    cudaStreamDestroy(stream);
}

// 测试用例9：设备内存压力测试
TEST_F(BufferTest, CudaMemoryStress) {
    const size_t large_size = 1 << 28; // 256MB
    std::vector<std::shared_ptr<Buffer>> buffers;
    
    for(int i=0; i<10; ++i){
        auto buf = Buffer::create(large_size, cuda_alloc, nullptr, false);
        ASSERT_NO_THROW(buf->allocate());  // 网页8的异常安全验证
        buffers.push_back(buf);
    }
    
    // 验证内存释放
    buffers.clear();
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    EXPECT_GT(free_mem, total_mem*0.8);  // 内存应释放超过80%
}
#endif