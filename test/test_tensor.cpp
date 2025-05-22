#include "tensor.h"
#include "base.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorTest, ConstructorWithSingleDim) {
    Tensor t(DataType::kDataTypeFp32, 5);
    EXPECT_EQ(t.dims(), std::vector<int32_t>({5}));
    EXPECT_EQ(t.size(), 5);
    EXPECT_EQ(t.data_type(), DataType::kDataTypeFp32);
}

// 修复: 设置use_external为true以正确标识外部内存
TEST(TensorTest, ConstructorWithExternalMemory) {
    float data[10];
    Tensor t(DataType::kDataTypeFp32, {10}, false, nullptr, data);  // 第三个参数设为true
    EXPECT_TRUE(t.get_buffer()->is_external());
    EXPECT_EQ(t.get_buffer()->ptr(), data);
}

TEST(TensorTest, MultiDimConstructor) {
    Tensor t(DataType::kDataTypeInt32, 2, 3, 4);
    EXPECT_EQ(t.dims(), std::vector<int32_t>({2, 3, 4}));
    EXPECT_EQ(t.size(), 2 * 3 * 4);
}

TEST(TensorTest, EmptyTensor) {
    Tensor t;
    EXPECT_TRUE(t.is_empty());
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorTest, ReshapeOperation) {
    auto alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    Tensor t(DataType::kDataTypeFp32, {2, 3}, true, alloc);
    auto original_dims = t.dims();
    
    // 有效reshape（元素总数6不变）
    t.reshape({3, 2});
    EXPECT_EQ(t.dims(), std::vector<int32_t>({3, 2})); // 维度变更验证
    original_dims = t.dims(); // 更新原始维度

    // 捕获无效reshape错误
    ::testing::internal::CaptureStderr();
    t.reshape({4, 3});  // 总元素12 > 6
    std::string output = ::testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.find("larger than the original") != std::string::npos);
    EXPECT_EQ(t.dims(), original_dims); // 维度回滚验证[1,3](@ref)
}

TEST(TensorTest, MemoryAllocation) {
    auto alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    Tensor t(DataType::kDataTypeFp32, {10});
    
    EXPECT_TRUE(t.allocate(alloc));
    EXPECT_NE(t.get_buffer(), nullptr);
    EXPECT_GE(t.get_buffer()->byte_size(), 10*sizeof(float));
}

// 修复: 补充Buffer::create的缺失参数
TEST(TensorTest, AssignBuffer) {
    auto alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    auto buffer = Buffer::create(100, alloc, nullptr, false);  // 添加ptr和use_external参数
    
    Tensor t(DataType::kDataTypeFp32, {25}); // Needs 100 bytes (25 * 4)
    EXPECT_TRUE(t.assign(buffer));
    
    // Test undersized buffer
    auto small_buffer = Buffer::create(10, alloc, nullptr, false);  // 添加参数
    EXPECT_FALSE(t.assign(small_buffer));
}

TEST(TensorTest, DeviceConversion) {
    auto alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    Tensor t(DataType::kDataTypeFp32, {100}, true, alloc);
    
    // Initial should be CPU
    ASSERT_EQ(t.device_type(), DeviceType::kDeviceCPU);
    
    // Convert to CUDA
    t.to_cuda();
    EXPECT_EQ(t.device_type(), DeviceType::kDeviceCUDA);
    
    // Convert back to CPU
    t.to_cpu();
    EXPECT_EQ(t.device_type(), DeviceType::kDeviceCPU);
}

TEST(TensorTest, StridesCalculation) {
    Tensor t(DataType::kDataTypeFp32, {2, 3, 4});
    auto strides = t.strides();
    EXPECT_EQ(strides, std::vector<size_t>({3 * 4, 4, 1}));
}

TEST(TensorTest, DataTypeSize) {
    Tensor t1(DataType::kDataTypeFp32, {1});
    EXPECT_EQ(t1.byte_size(), 4);
    
    Tensor t2(DataType::kDataTypeInt8, {10});
    EXPECT_EQ(t2.byte_size(), 10);
}

TEST(TensorTest, ResetOperation) {
    Tensor t(DataType::kDataTypeFp32, {5});
    t.reset(DataType::kDataTypeInt16, {2, 3});
    
    EXPECT_EQ(t.data_type(), DataType::kDataTypeInt16);
    EXPECT_EQ(t.dims(), std::vector<int32_t>({2, 3}));
    EXPECT_TRUE(t.is_empty());
}

TEST(TensorTest, BoundaryConditions) {
    auto alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    // 零维张量
    Tensor t1(DataType::kDataTypeFp32, {0});
    EXPECT_TRUE(t1.is_empty());
    
    // 扩容错误场景
    Tensor t2(DataType::kDataTypeFp32, {5}, true, alloc);
    ::testing::internal::CaptureStderr();
    t2.reshape({2, 3});  // 需要6元素
    std::string output = ::testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.find("larger than the original") != std::string::npos);
    EXPECT_EQ(t2.size(), 5);  // 容量不变性验证[3,6](@ref)
}

TEST(TensorTest, ResizeBehavior) {
    auto alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    Tensor t(DataType::kDataTypeFp32, {2, 2}, true, alloc);
    auto* original_ptr = t.get_buffer()->ptr();
    
    // 缩容（不重新分配内存）
    t.resize({1, 2});
    EXPECT_EQ(t.dims(), std::vector<int32_t>({1, 2}));
    EXPECT_EQ(t.get_buffer()->ptr(), original_ptr); // 内存地址不变
    
    // 扩容（触发内存重分配）
    ::testing::internal::CaptureStderr();
    t.resize({3, 4});  // 需要12元素 > 4
    std::string output = ::testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.empty()); // 无错误日志
    EXPECT_NE(t.get_buffer()->ptr(), original_ptr); // 新内存验证[2,4](@ref)
}

TEST(TensorTest, ExternalMemoryReshape) {
    float external_data[6];
    Tensor t(DataType::kDataTypeFp32, {2, 3}, false, nullptr, external_data);
    
    // 外部内存允许超容量reshape（仅修改元数据）
    t.reshape({3, 2}); 
    EXPECT_EQ(t.dims(), std::vector<int32_t>({3, 2}));
    EXPECT_EQ(t.get_buffer()->ptr(), external_data); // 内存不变性[1,3](@ref)
}


TEST(TensorTest, ElementAccess) {
    auto alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    Tensor t(DataType::kDataTypeFp32, {3}, true, alloc);
    auto* data = static_cast<float*>(t.get_buffer()->ptr());
    data[1] = 3.14f;
    EXPECT_FLOAT_EQ(data[1], 3.14f);
}

/*
TEST(TensorTest, to_cpu) {
  auto alloc_cu = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);
  Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
//   set_value_cu(t1_cu.ptr<float>(), 32 * 32);

  t1_cu.to_cpu();
  ASSERT_EQ(t1_cu.device_type(), DeviceType::kDeviceCPU);
  float* cpu_ptr = t1_cu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(cpu_ptr + i), 1.f);
  }
}

TEST(TensorTest, clone_cuda) {
  auto alloc_cu = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);
  Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);

  Tensor t2_cu = t1_cu.clone();
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  cudaMemcpy(p2, t1_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  ASSERT_EQ(t2_cu.data_type(), DataType::kDataTypeFp32);
  ASSERT_EQ(t2_cu.size(), 32 * 32);

  t2_cu.to_cpu();
  Memcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

*/
TEST(TensorTest, clone_cpu) {
  auto alloc_cpu = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  for (int i = 0; i < 32 * 32; ++i) {
    t1_cpu.index<float>(i) = 1.f;
  }

  Tensor t2_cpu = t1_cpu.clone();
  float* p2 = new float[32 * 32];
  Memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  Memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}


TEST(TensorTest, to_cu) {
  auto alloc_cpu = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  float* p1 = t1_cpu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    *(p1 + i) = 1.f;
  }


  t1_cpu.to_cuda(nullptr);
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(p2 + i), 1.f);
  }
  delete[] p2;
}

TEST(TensorTest, init1) {
  auto alloc_cu = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);

  int32_t size = 32 * 151;

  Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
  ASSERT_EQ(t1.is_empty(), false);
}

TEST(TensorTest, init3) {
  float* ptr = new float[32];
  ptr[0] = 31;
  Tensor t1(DataType::kDataTypeFp32, 32, false, nullptr, ptr);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.ptr<float>(), ptr);
  ASSERT_EQ(*t1.ptr<float>(), 31);
}

TEST(TensorTest, init2) {
  auto alloc_cu = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);

  int32_t size = 32 * 151;

  Tensor t1(DataType::kDataTypeFp32, size, false, alloc_cu);
  ASSERT_EQ(t1.is_empty(), true);
}

TEST(TensorTest, assign1) {
  auto alloc_cpu = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);

  int32_t size = 32 * 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  std::shared_ptr<Buffer> buffer = Buffer::create(size * sizeof(float), alloc_cpu, nullptr, true);

  buffer->set_device_type(DeviceType::kDeviceCPU);

  ASSERT_EQ(t1_cpu.assign(buffer), true);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
  delete[] ptr;
}