#pragma once  // 确保头文件只会被包含一次
#ifndef __MYFUNC_H__
#define __MYFUNC_H__
#include "debug.h"


/* ---------- CPU ---------- */

// Malloc 函数声明
void* Malloc(size_t size);
void Memcpy( void* dest, const void* src, std::size_t count );

/* ----------- GPU ---------- */

void CudaMalloc(void** _devPtr, size_t _size);
void CudaGetDevice(int *device);

#endif// __MYFUNC_H__