#include "myfunc.h"
#include <cuda_runtime.h>
#include <cstring>
/* ---------- CPU ---------- */
// BEGIN CPU Function
void* Malloc(size_t size){
    void* ptr = NULL;

    if(size == 0){
        Error("malloc size is 0");
        return NULL;
    }

    ptr = malloc(size);
    
    if(ptr == NULL){
        Error("malloc failed");

    }
    return ptr;
}

void Memcpy(void* dest, const void* src, std::size_t count) {
    if (dest == NULL || src == NULL) {
        Error("memcpy dest or src is NULL");
        return;  // 立即返回，避免继续执行
    }
    if (count == 0) {
        Error("memcpy count is 0");
        return;  // 立即返回，避免继续执行
    }

    std::memcpy(dest, src, count);
}

void Memset(void* dest, int value, std::size_t count) {
    if (dest == NULL) {
        Error("memset dest is NULL");
        return;  // 立即返回，避免继续执行
    }
    if (count == 0) {
        Error("memset count is 0");
        return;  // 立即返回，避免继续执行
    }

    std::memset(dest, value, count);
}

/* ----------- GPU ---------- */
void CudaMalloc(void** _devPtr, size_t _size){
    int id = -1;
    CudaGetDevice(&id);
    
    cudaError_t Error = cudaMalloc(_devPtr, _size);
    if(Error != cudaSuccess){
        Error("cudaMalloc failed");
        return;
    }

    if(_devPtr == NULL){
        Error("cudaMalloc _devPtr is NULL");
        return;
    }
    if(_size == 0){
        Error("cudaMalloc _size is 0");
        return;
    }
}

void CudaGetDevice(int *device){
    cudaError_t state = cudaGetDevice(device);
    if(state!= cudaSuccess){
        Error("cudaGetDevice failed");
        return;
    }
}

void CudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind){
    cudaError_t Error = cudaMemcpy(dst, src, count, kind);
    if(Error!= cudaSuccess){
        Error("cudaMemcpy failed");
        return;
    }
}

void CudaMemcpyAsync (void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream){
    cudaError_t Error = cudaMemcpyAsync(dst, src, count, kind, stream);
    if(Error!= cudaSuccess){
        Error("cudaMemcpyAsync failed");
        return;
    }
}

void CudaFree(void* _devPtr){
    cudaError_t Error = cudaFree(_devPtr);
    if(Error!= cudaSuccess){
        Error("cudaFree failed");
        return;
    }
}
void CudaDeviceSynchronize(){
    cudaError_t Error = cudaDeviceSynchronize();
    if(Error!= cudaSuccess){
        Error("cudaDeviceSynchronize failed");
        return;
    }
}

void CudaMemset(void* ptr, int value, size_t count){
    cudaError_t Error = cudaMemset(ptr, value, count);
    if(Error!= cudaSuccess){
        Error("cudaMemset failed");
        return;
    }
}
void CudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream){
    cudaError_t Error = cudaMemsetAsync(ptr, value, count, stream);
    if(Error!= cudaSuccess){
        Error("cudaMemsetAsync failed");
        return;
    }
}

