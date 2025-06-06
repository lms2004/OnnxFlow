#pragma once  // 确保头文件只会被包含一次
#include <map>
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#include "base.h"
#include "myfunc.h"
using mybase::DeviceType;
using mybase::DataType;
using mybase::MemcpyKind;

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual void* allocate(size_t n) const = 0;
    virtual void deallocate(void** p) = 0;
    
    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                        bool need_sync = false) const;
    
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

    DeviceType device_type() const {
        return device_type_;
    }

    ~DeviceAllocator() = default;
private:
    DeviceType device_type_;
};

/* ----- CPU ------ */
class CPUAllocator : public DeviceAllocator {
public:
    explicit CPUAllocator(): DeviceAllocator(DeviceType::kDeviceCPU) {};

    void* allocate(size_t n) const override;
    void deallocate(void** p) override;
};


/* ----- GPU ------ */
class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {};

  void* allocate(size_t byte_size) const override;

  void deallocate(void** p) override;
};


class DeviceAllocatorSingleton {
 public:
  // 创建实例，使用懒加载模式，确保只有一个实例被创建
  static std::shared_ptr<DeviceAllocator> getInstance(DeviceType device_type_) {
    if (device_type_ == DeviceType::kDeviceCPU) {
      if (!cpu_instance) {
        cpu_instance = std::make_shared<CPUAllocator>();
      }
      return cpu_instance;
    } else if (device_type_ == DeviceType::kDeviceCUDA) {
      if (!cuda_instance) {
        cuda_instance = std::make_shared<CUDADeviceAllocator>();
      }
      return cuda_instance;
    }
    return nullptr;  // 返回 nullptr 如果设备类型未知
  }

 private:
  static std::shared_ptr<DeviceAllocator> cpu_instance;
  static std::shared_ptr<DeviceAllocator> cuda_instance;
};