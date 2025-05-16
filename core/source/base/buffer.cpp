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

// ---------------------- copy func ------------------------------
void Buffer::copy_from(const Buffer& buffer) const {
  ;
}

void Buffer::copy_from(const Buffer* buffer) const{
  ;
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