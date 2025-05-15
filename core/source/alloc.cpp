#include "alloc.h"
std::shared_ptr<CPUAllocator> CPUAllocatorSingleton::instance = nullptr;  // 定义静态成员变量

void* CPUAllocator::allocate(size_t n) {
    return Malloc(n);
}

void CPUAllocator::deallocate(void** ptr) {
  if (*ptr) {
    free(*ptr);
    *ptr = nullptr;  // 显式置空，避免后续误用
  }
}