#pragma once  // 确保头文件只会被包含一次
#include "debug.h"


/* ---------- CPU ---------- */

// Malloc 函数声明
void* Malloc(size_t size);


/* ----------- GPU ---------- */

void CudaMalloc(void** _devPtr, size_t _size);
void CudaGetDevice(int *device);