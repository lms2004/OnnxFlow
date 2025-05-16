#pragma once  // 确保头文件只会被包含一次
#include <map>
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#include "base.h"
#include "myfunc.h"

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual void* allocate(size_t n) const = 0;
    virtual void deallocate(void** p) = 0;
    
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
  static std::shared_ptr<DeviceAllocator> getInstance(DeviceType device_type_) {
    if (device_type_ == DeviceType::kDeviceCPU) {
      if (!instance) {
        instance = std::make_shared<CPUAllocator>();
      }
    } else if (device_type_ == DeviceType::kDeviceCUDA) {
      if (!instance) {
        instance = std::make_shared<CUDADeviceAllocator>();
      }
    }
    return instance;
  }

 private:
  static std::shared_ptr<DeviceAllocator> instance;
};