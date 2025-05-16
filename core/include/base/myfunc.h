#pragma once  // 确保头文件只会被包含一次
#ifndef __MYFUNC_H__
#define __MYFUNC_H__
#include "debug.h"
#include <cuda_runtime.h>

/* ---------- CPU ---------- */

// Malloc 函数声明
void* Malloc(size_t size);
void Memcpy( void* dest, const void* src, std::size_t count );
void Memset( void* dest, int value, std::size_t count );
/* ----------- GPU ---------- */

void CudaMalloc(void** _devPtr, size_t _size);
void CudaGetDevice(int *device);
void CudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
void CudaMemcpyAsync (void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
void CudaFree(void* _devPtr);
void CudaDeviceSynchronize();
void CudaMemset(void* ptr, int value, size_t count);
void CudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream = 0);
#endif// __MYFUNC_H__