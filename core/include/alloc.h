#pragma once  // 确保头文件只会被包含一次

#include <memory>
#include <cstdlib>
#include <cstdio>
#include "base.h"
#include "myfunc.h"

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type){};

    virtual void* allocate(size_t n) = 0;
    virtual void deallocate(void** p) = 0;
    ~DeviceAllocator() = default;
};

class CPUAllocator : public DeviceAllocator {
public:
    explicit CPUAllocator(DeviceType device_type) : DeviceAllocator(device_type) {}

    void* allocate(size_t n) override;
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
