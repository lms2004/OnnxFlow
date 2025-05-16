#ifndef __Buffer_H__
#define __Buffer_H__
#include "base.h"
#include "alloc.h"
/*
 * RAII -> 解决单个控制块 被多个 std::shared_ptr 管理时的所有权共享问题
 * 1. 禁用拷贝语义（通过 NoCopyable 基类）
 * 2. 资源生命周期绑定对象：构造时分配内存，析构时按需释放
 * 3. 支持共享指针安全获取（enable_shared_from_this）
 *
 * 工厂模式实现要点：
 * 1. 强制通过 create() 创建对象，确保生命周期由 shared_ptr 管理
 * 2. 参数化构造逻辑，支持动态资源分配策略
 */
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer>{
public:
    ~Buffer();

    //  静态工厂函数：唯一创建入口
    static std::shared_ptr<Buffer> create(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
                bool use_external){
                    return std::shared_ptr<Buffer>(new Buffer(byte_size, allocator, ptr, use_external));
    }

    std::shared_ptr<Buffer> getptr() {
        return shared_from_this(); // 返回 weak_ptr
    }



    void copy_from(const Buffer& buffer) const;
    void copy_from(const Buffer* buffer) const;



    // --- getter func ---
    
    size_t byte_size() const;
    void* ptr();             // 适用于读写访问
    const void* ptr() const; // 适用于只读访问
    bool is_external() const;
    DeviceType device_type() const;

    // --- setter func ---

    void set_device_type(DeviceType device_type);

private:
    Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
                bool use_external);

    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;
};




#endif
