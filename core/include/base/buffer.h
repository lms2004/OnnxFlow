#ifndef __BUFFER_H__
#define __BUFFER_H__
#include "base.h"
#include "alloc.h"
/*
    RALL:禁用拷贝、使用智能指针管理生命周期

*/
class buffer : public NoCopyable, std::enable_shared_from_this<buffer>{
public:
    buffer() = default;
    
    ~buffer() {
        if (ptr_ && !use_external_) {
        allocator_->deallocate(&ptr_);
        }
    }
    



private:
    buffer(size_t byte_size, DeviceType device_type = DeviceType::kDeviceCPU)
        : byte_size_(byte_size), device_type_(device_type) {
        if (byte_size_ > 0) {
            allocator_ = std::make_shared<DeviceAllocator>(device_type_);
            ptr_ = allocator_->allocate(byte_size_);
        }
    }

    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;
};




#endif
