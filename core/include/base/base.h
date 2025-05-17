#pragma once  // 确保头文件只会被包含一次
#ifndef __BASE_H__
#define __BASE_H__


#include <cstdint>
#include <cstddef>
/*
  成员（前缀 k 通常用于表示常量或枚举值( Google 的编程风格指南）：
    - kDeviceUnknown：未知设备。
    - kDeviceCPU：CPU 设备。
    - kDeviceCUDA：CUDA 设备。
*/
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32,
  kDataTypeInt8,
  kDataTypeInt32,
  kDataTypeInt64,
  kDataTypeInt16 ,
  kDataTypeUint8,
  kDataTypeUint16,
  kDataTypeUint32,
  kDataTypeUint64,
  kDataTypeBool,
};

inline size_t DataTypeSize(DataType data_type) {
  switch (data_type) {
    case DataType::kDataTypeFp32:
      return sizeof(float);
    case DataType::kDataTypeInt8:
      return sizeof(int8_t);
    case DataType::kDataTypeInt32:
      return sizeof(int32_t);
    case DataType::kDataTypeInt64:
      return sizeof(int64_t);
    case DataType::kDataTypeInt16:
      return sizeof(int16_t);
    case DataType::kDataTypeUint8:
      return sizeof(uint8_t);
    case DataType::kDataTypeUint16:
      return sizeof(uint16_t);
    case DataType::kDataTypeUint32:
      return sizeof(uint32_t);
    case DataType::kDataTypeUint64:
      return sizeof(uint64_t);
    case DataType::kDataTypeBool:
      return sizeof(bool);
    default:
      return 0;  // Unknown data type
  }
}


enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

// 这个类的作用是禁止对象的拷贝构造和赋值操作
// 参考： https://stackoverflow.com/questions/2173746/how-do-i-make-this-c-object-non-copyable
class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};

#endif