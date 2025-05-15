#pragma once  // 确保头文件只会被包含一次

#include <stdlib.h>
#include <cstdio>
#include <iostream>

// 错误打印函数声明
void printError(const char* errorMessage);

// Malloc 函数声明
void* Malloc(size_t size);
