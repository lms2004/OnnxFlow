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
    explicit DeviceAllocator(DeviceType device_type){};

    virtual void* allocate(size_t n) const = 0;
    virtual void deallocate(void** p) = 0;
    ~DeviceAllocator() = default;
};

/* ----- CPU ------ */
class CPUAllocator : public DeviceAllocator {
public:
    explicit CPUAllocator(DeviceType device_type) : DeviceAllocator(device_type) {}

    void* allocate(size_t n) const override;
    void deallocate(void** p) override;
};

class CPUAllocatorSingleton {   
public:
    static std::shared_ptr<CPUAllocator> getInstance() {
        if(instance == nullptr) {
            instance = std::make_shared<CPUAllocator>(DeviceType::kDeviceCPU);
        }
        return instance;
    }

private:
    static std::shared_ptr<CPUAllocator> instance;
};

/* ----- GPU ------ */
class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {};

  void* allocate(size_t byte_size) const override;

  void deallocate(void** p) override;
};


class CUDADeviceAllocatorSingleton {
 public:
  static std::shared_ptr<CUDADeviceAllocator> getInstance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};