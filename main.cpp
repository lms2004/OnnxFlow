#include "alloc.h"
#include <memory>

int main(){
    log_fp = fopen("log.txt", "w");
    error_fp = fopen("error.txt", "w");
    // 获取 CPUAllocator 的实例
    std::shared_ptr<DeviceAllocator> allocator = CPUAllocatorSingleton::getInstance();
    
    // 分配零字节内存
    void* ptr = allocator->allocate(-1);
}