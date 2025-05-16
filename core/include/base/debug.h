#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>  // 用于文件日志输出

// 文件日志指针
extern FILE* log_fp;
extern FILE* error_fp;


/* ---------- Logging Macros ---------- */

// 打印日志信息，格式化输出包括文件名、行号、函数名
#define Log(format, ...) \
    do { \
        fprintf(stdout, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
        if (log_fp) fprintf(log_fp, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
    } while (0)

// 打印错误日志信息，格式化输出包括文件名、行号、函数名
#define Error(format, ...) \
    do { \
        fprintf(stderr, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
        if (error_fp) fprintf(error_fp, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
    } while (0)

// 断言宏，如果条件不成立，则打印错误信息并触发断言失败
#define Assert(cond, format, ...) \
  do { \
    if (!(cond)) { \
      fprintf(stderr, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
      if (error_fp) fprintf(error_fp, "[%s:%d %s] " format "\n", __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
      fflush(stdout); \
      extern void assert_fail_msg(); \
      assert_fail_msg(); \
      assert(cond); \
    } \
  } while (0)

// Panic 宏，用于触发断言失败并打印格式化信息
#define panic(format, ...) Assert(0, format, ## __VA_ARGS__)

#endif
