#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>  // 用于文件日志输出

// 文件日志指针，用于将日志信息输出到文件
extern FILE* log_fp;
extern FILE* error_fp;

/* ---------- Logging Macros ---------- */

// Log宏：用于打印普通的日志信息，包括文件名、行号、函数名和日志内容
// 示例使用：
// Log("This is a test log message");
// Log("The value of x is %d", x);
#define Log(format, ...) \
    do { \
        fprintf(stdout, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
        if (log_fp) fprintf(log_fp, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
    } while (0)

// Error宏：用于打印错误日志信息，包括文件名、行号、函数名和错误信息
// 示例使用：
// Error("Error occurred while opening file");
// Error("Failed to read data from file: %s", file_name);
#define Error(format, ...) \
    do { \
        fprintf(stderr, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
        if (error_fp) fprintf(error_fp, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
    } while (0)

// Assert宏：用于条件判断，如果条件不成立，打印错误信息并触发断言失败
// 示例使用：
// Assert(x > 0, "x must be greater than 0, but got %d", x);
// 如果 x <= 0，程序将输出错误信息并终止执行
#define Assert(cond, format, ...) \
  do { \
    if (!(cond)) { \
      fprintf(stderr, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
      if (error_fp) fprintf(error_fp, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
      fflush(stderr); \
      assert(cond); \
    } \
  } while (0)

// Panic宏：用于触发断言失败并打印格式化的错误信息，通常用于程序发生严重错误时
// 示例使用：
// panic("Critical error: memory allocation failed");
// 该宏会停止程序执行，通常用于无法恢复的错误情况
#define panic(format, ...) Assert(0, format, ## __VA_ARGS__)

#endif   // __DEBUG_H__
