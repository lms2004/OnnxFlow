#include "myfunc.h"
#include <cuda_runtime.h>

/* ---------- Log ---------- */
// BEGIN Log Function

void printError(const char* errorMessage, const char* fileName, int lineNumber) {
    if (errorMessage == nullptr) {
        std::cout << "\033[31mError: Invalid error message\033[0m"
                  << " at " << fileName << " line " << lineNumber << std::endl;
        return;
    }
    
    // ANSI 转义码：\033[31m 将文本颜色设置为红色，\033[0m 重置文本颜色为默认
    std::cout << "\033[31mError: " << errorMessage 
              << " at " << fileName << " line " << lineNumber << "\033[0m" << std::endl;
}

// 使用示例（可以用文件和行号信息调用此函数）
#define PRINT_ERROR(msg) printError(msg, __FILE__, __LINE__)


// END Log Function


/* ---------- CPU ---------- */
// BEGIN CPU Function
void* Malloc(size_t size){
    void* ptr = NULL;

    if(size == 0){
        PRINT_ERROR("malloc size is 0");
        return NULL;
    }

    ptr = malloc(size);
    
    if(ptr == NULL){
        PRINT_ERROR("malloc failed");

    }
    return ptr;
}


/* ----------- GPU ---------- */
void CudaMalloc(void** _devPtr, size_t _size){
    int id = -1;
    CudaGetDevice(&id);
    
    cudaError_t error = cudaMalloc(_devPtr, _size);
    if(error != cudaSuccess){
        PRINT_ERROR("cudaMalloc failed");
        return;
    }

    if(_devPtr == NULL){
        PRINT_ERROR("cudaMalloc _devPtr is NULL");
        return;
    }
    if(_size == 0){
        PRINT_ERROR("cudaMalloc _size is 0");
        return;
    }
}

void CudaGetDevice(int *device){
    cudaError_t state = cudaGetDevice(device);
    if(state!= cudaSuccess){
        PRINT_ERROR("cudaGetDevice failed");
        return;
    }
}
