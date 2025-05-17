#include "buffer.h"

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
               bool use_external)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      use_external_(use_external) {
  if (!ptr_ && allocator_) {
    device_type_ = allocator_->device_type();
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size);
  }
}

Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->deallocate(&ptr_);
    }
  }
}

void Buffer::allocate() {
    if (!(allocator_ && byte_size_ != 0)) {
        Error("无效的 allocator 或 byte_size 为零。");
        return;
    }

    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size_);
    if (!ptr_) {
        Error("Buffer 分配失败：allocator 返回了空指针。");
        return;
    }
}


// ---------------------- core func ------------------------------

void Buffer::copy_from(const Buffer& buffer) const {
  Assert(buffer.ptr_ != nullptr || allocator_ != nullptr,
         "Buffer::copy_from: buffer is nullptr"); // copy from 需要保证 dst: buffer 不为空， src: allocator_ 也不为空

  size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
  const DeviceType& buffer_device = buffer.device_type();
  const DeviceType& current_device = this->device_type();

  Assert(buffer_device!= DeviceType::kDeviceUnknown || current_device!= DeviceType::kDeviceUnknown,
         "Buffer::copy_from: buffer_device or current_device is unknown");

  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CUDA);
  }
}

void Buffer::copy_from(const Buffer* buffer) const {
  Assert(buffer!= nullptr || buffer->ptr_!= nullptr || allocator_!= nullptr,
         "Buffer::copy_from: buffer is nullptr");

  size_t dest_size = byte_size_;
  size_t src_size = buffer->byte_size_;
  size_t byte_size = src_size < dest_size ? src_size : dest_size;

  const DeviceType& buffer_device = buffer->device_type();
  const DeviceType& current_device = this->device_type();

  Assert(buffer_device!= DeviceType::kDeviceUnknown || current_device!= DeviceType::kDeviceUnknown,
         "Buffer::copy_from: buffer_device or current_device is unknown");

  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CUDA);
  }
}


// ---------------------- setter func ------------------------------
void Buffer::set_device_type(DeviceType device_type) {
  device_type_ = device_type;
}

// ---------------------- getter func ------------------------------
void* Buffer::ptr() {
  return ptr_;
}

const void* Buffer::ptr() const {
  return ptr_;
}

size_t Buffer::byte_size() const {
  return byte_size_;
}
DeviceType Buffer::device_type() const {
  return device_type_;
}
bool Buffer::is_external() const {
  return use_external_;
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
  return allocator_;
}