#include "tensor.h"
#include <numeric>
// ------------------ utility func ------------------



// 计算一个表示张量维度的范围（由 begin 和 end 指定）所对应的元素总数
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}


// ------------------ getters ------------------
size_t Tensor::size() const { return this->size_; }

const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

DataType Tensor::data_type() const{
    return this->data_type_;
}

DeviceType Tensor::device_type() const{
    Assert(this->buffer_, "Tensor::device_type: buffer_ is nullptr");
    return this->buffer_->device_type();
}

std::shared_ptr<Buffer> Tensor::get_buffer() const { return buffer_; }

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

// -------- attribute func --------
bool Tensor::is_empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

int32_t Tensor::get_dim(int32_t idx) const {
    Assert(idx >= 0, "Tensor::get_dim: idx should be >= 0");
    Assert(idx < static_cast<int32_t>(this->dims_.size()),
                "Tensor::get_dim: idx should be < dims_.size()");
    return this->dims_.at(idx);
}

size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

// ------------------ setter ------------------
void Tensor::set_device_type(DeviceType device_type) const {
    Assert(this->buffer_, "Tensor::set_device_type: buffer_ is nullptr");
    this->buffer_->set_device_type(device_type);
}



// ------------------ core functions ------------------

// 重置 Tensor 的属性，数据置空
void Tensor::reset(DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
  this->buffer_ = nullptr;
}



// 修整 Tensor 的维度，不对原始多维数组进行修改；
// 例如，将一个 3x4 的矩阵 reshape 为 2x6 的矩阵，不会改变原始矩阵的形状。
void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
  // 外部内存，直接修改dims_和size_
  if (!buffer_ || buffer_->is_external()) {
    this->dims_ = dims;
    this->size_ = size;
    return;
  }

  if (size > size_) {
    Error("Tensor::reshape: The size of the new dims is larger than the original one.");
    return;
  }

  this->dims_ = dims;
  this->size_ = size;
}
// 修整 Tensor 的维度，对原始多维数组进行修改；
// 例如，将一个 3x4 resize 为 2x6 的矩阵，改变原始矩阵的形状。
// 例如，将一个 3x4 resize 为 9x4 的矩阵，扩容原始矩阵空间
void Tensor::resize(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
  // 外部内存，直接修改dims_和size_
  if (!buffer_ || buffer_->is_external()) {
    this->dims_ = dims;
    this->size_ = size;
    return;
  }

  if (size > size_) {
      auto new_buffer = Buffer::create(size * DataTypeSize(data_type_), buffer_->allocator(), nullptr, false);
      if(!new_buffer) {
        Error("Tensor::resize: Failed to create new buffer.");
      }
      size_t byte_size = this->byte_size();
      new_buffer->copy_from(buffer_.get());
      this->buffer_ = new_buffer;
  }

  this->dims_ = dims;
  this->size_ = size;
}



// 为Tensor分配内存空间，支持按需重新分配。
bool Tensor::allocate(std::shared_ptr<DeviceAllocator> allocator, bool need_realloc) {
  if (!allocator) {
    Error("Tensor::allocate: allocator is nullptr");
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    Error("Tensor::allocate: byte_size is 0");
    return false;
  }

  if (buffer_ && byte_size <= buffer_->byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }

  buffer_ = Buffer::create(byte_size, allocator, nullptr, false);
  if (!buffer_->ptr()) {
    Error("Tensor::allocate: buffer_ is nullptr");
    return false;
  }
  return true;
}

// 将已有的内存资源（Buffer）绑定到当前Tensor，不涉及内存分配。
bool Tensor::assign(std::shared_ptr<Buffer> buffer) {
  if (!buffer) {
    Error("Tensor::assign: buffer is nullptr");
    return false;
  }
  if (buffer_) {
    if (buffer_->device_type() != buffer->device_type()) {
      Error("Tensor::assign: The device type of the new buffer is different from the original one.");
    }
  }

  size_t byte_size = this->byte_size();
  if (byte_size > buffer->byte_size()) {
    Error("Tensor::assign: The size of buffer is too small for the tensor!");
    return false;
  }
  buffer_ = buffer;
  return true;
}


void Tensor::init_buffer(std::shared_ptr<DeviceAllocator> alloc, DataType data_type,
                         bool need_alloc, void* ptr) {
  // 不需要分配内存，没有分配器 -> 外部内存 -> 直接使用传入的指针
  if (!alloc && !need_alloc && ptr != nullptr) {
    this->buffer_ = Buffer::create(0, nullptr, ptr, true);
    return;
  }else {
    allocate(alloc, true);
  }
}

// 当前的 Tensor 数据从 CPU 内存迁移到 GPU 内存(注意是：迁移，而不是分配新的内存)
void Tensor::to_cuda(cudaStream_t stream) {
  Assert(buffer_ != nullptr, "Tensor::to_cuda: buffer_ is nullptr");
  const DeviceType device_type = this->device_type();
  if (device_type == DeviceType::kDeviceUnknown) {
    Error("Tensor::to_cuda: The device type of the tensor is unknown.");
  } else if (device_type == DeviceType::kDeviceCPU) {
    size_t byte_size = this->byte_size();
    auto cu_alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCUDA);
    auto cu_buffer = Buffer::create(byte_size, cu_alloc, nullptr, false);
    cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, MemcpyKind::kMemcpyCPU2CUDA,
                     stream, true);
    this->buffer_ = cu_buffer;
    set_device_type(DeviceType::kDeviceCUDA);
  } else {
    Error("Tensor::to_cuda: The device type of the tensor is already cuda.");
  }
}

void Tensor::to_cpu() {
  Assert(buffer_!= nullptr, "Tensor::to_cpu: buffer_ is nullptr");
  const DeviceType device_type = this->device_type();

  if (device_type == DeviceType::kDeviceUnknown) {
    Error("Tensor::to_cpu: The device type of the tensor is unknown.");
  } else if (device_type == DeviceType::kDeviceCUDA) {
    size_t byte_size = this->byte_size();
    auto cpu_alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
    auto cpu_buffer = Buffer::create(byte_size, cpu_alloc, nullptr, false);
    cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                      MemcpyKind::kMemcpyCUDA2CPU);
    this->buffer_ = cpu_buffer;
    set_device_type(DeviceType::kDeviceCPU);
  } else {
    Error("Tensor::to_cpu: The device type of the tensor is already cpu.");
  }
}

Tensor Tensor::clone() const {
  Tensor new_tensor = *this; // 赋值成员变量

  // 分配新的 buffer
  size_t byte_size = this->byte_size();

  auto allocator = buffer_->allocator();
  new_tensor.buffer_ = Buffer::create(byte_size, allocator, nullptr, false);
  new_tensor.buffer_->copy_from(buffer_.get());
  return new_tensor;
}


// ----------- Constructor ------------------

Tensor::Tensor(DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else if (ptr != nullptr){
    Assert(need_alloc == false, "Tensor::Tensor: ptr should be nullptr when need_alloc is true");
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<DeviceAllocator> alloc, void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}
