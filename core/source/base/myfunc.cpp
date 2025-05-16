#include "myfunc.h"
#include <cuda_runtime.h>

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
