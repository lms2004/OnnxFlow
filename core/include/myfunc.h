#pragma once  // 确保头文件只会被包含一次

#include <stdlib.h>
#include <cstdio>
#include <iostream>

/* ---------- Log ---------- */

// 错误打印函数声明
void printError(const char* errorMessage, const char* fileName, int lineNumber);

/* ---------- CPU ---------- */

// Malloc 函数声明
void* Malloc(size_t size);


/* ----------- GPU ---------- */

void CudaMalloc(void** _devPtr, size_t _size);
void CudaGetDevice(int *device);