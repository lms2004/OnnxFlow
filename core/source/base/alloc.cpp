#include "alloc.h"


std::shared_ptr<DeviceAllocator> DeviceAllocatorSingleton::cpu_instance = nullptr;
std::shared_ptr<DeviceAllocator> DeviceAllocatorSingleton::cuda_instance = nullptr;


void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
    Assert(src_ptr != nullptr, "src_ptr is null");
    Assert(dest_ptr != nullptr, "dest_ptr is null");

    if (byte_size == 0) return;  // 如果没有要复制的字节，提前返回

    cudaStream_t stream_ = (stream) ? static_cast<CUstream_st*>(stream) : nullptr;

    // 根据 memcpy 类型决定操作
    cudaMemcpyKind kind;
    switch (memcpy_kind) {
        case MemcpyKind::kMemcpyCPU2CPU:
            kind = cudaMemcpyHostToHost;
            break;
        case MemcpyKind::kMemcpyCPU2CUDA:
            kind = cudaMemcpyHostToDevice;
            break;
        case MemcpyKind::kMemcpyCUDA2CPU:
            kind = cudaMemcpyDeviceToHost;
            break;
        case MemcpyKind::kMemcpyCUDA2CUDA:
            kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            Error("Invalid memcpy kind");
            return;  // 如果 memcpy 类型无效，直接返回
    }

    // 执行 memcpy 操作，如果有流，则使用异步复制
    if (stream_) {
        CudaMemcpyAsync(dest_ptr, src_ptr, byte_size, kind, stream_);
    } else {
        CudaMemcpy(dest_ptr, src_ptr, byte_size, kind);
    }

    // 如果需要，进行同步
    if (need_sync) {
        CudaDeviceSynchronize();
    }
}


void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  Assert(ptr != nullptr, "ptr is null");
  if (device_type_ == DeviceType::kDeviceCPU) {
    Memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      CudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      CudaMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      CudaDeviceSynchronize();
    }
  }
}

// ----- 派生类实现 -----


void* CPUAllocator::allocate(size_t n)const {
    return Malloc(n);
}

void CPUAllocator::deallocate(void** ptr){
  if (*ptr) {
    free(*ptr);
    *ptr = nullptr;  // 显式置空，避免后续误用
  }
}


void* CUDADeviceAllocator::allocate(size_t n)const {
  void* ptr = nullptr; // 注意：不要使用 void** ptr = nullptr;
  CudaMalloc(&ptr, n);
  return ptr;
}

void CUDADeviceAllocator::deallocate(void** ptr){
  if (*ptr) {
    CudaFree(*ptr);
    *ptr = nullptr;  // 显式置空，避免后续误用
  }
}