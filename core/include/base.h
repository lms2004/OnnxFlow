#pragma once  // 确保头文件只会被包含一次

#include <cstdint>

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