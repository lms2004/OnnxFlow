#include "alloc.h"

std::shared_ptr<CPUAllocator> CPUAllocatorSingleton::instance = nullptr;  // 定义静态成员变量
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorSingleton::instance = nullptr;  // 定义静态成员变量



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
    cudaFree(*ptr);
    *ptr = nullptr;  // 显式置空，避免后续误用
  }
}