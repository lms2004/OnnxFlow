#include <base/base.h>
#include "kernels_interface.h"
namespace kernel {
// AddKernel get_add_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return add_kernel_cpu;
//   } else if (device_type == DeviceType::kDeviceCUDA) {
//     return add_kernel_cu;
//   } else {
//     Error("Unknown device type for get an add kernel.");
//     return nullptr;
//   }
// }

// EmbeddingKernel get_emb_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return emb_kernel_normal;
//   } else if (device_type == DeviceType::kDeviceCUDA) {
//     return emb_kernel_cu;
//   } else {
    
//     return nullptr;
//   }
// }

// MatmulKernel get_matmul_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return matmul_kernel_cpu;
//   } else if (device_type == DeviceType::kDeviceCUDA) {
//     return matmul_kernel_cu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get an matmul kernel.";
//     return nullptr;
//   }
// }

// MatmulKernelQuant get_matmul_kernel_quant8(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCUDA) {
//     return matmul_kernel_cu_qint8;
//   } else {
//     LOG(FATAL) << "Unknown device type for get an matmul kernel.";
//     return nullptr;
//   }
// }

// MHAKernel get_mha_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return mha_kernel;
//   } else if (device_type == DeviceType::kDeviceCUDA) {
//     return mha_kernel_cu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get an mha kernel.";
//     return nullptr;
//   }
// }

// RoPEKernel get_rope_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return rope_kernel_cpu;
//   } else if (device_type == DeviceType::kDeviceCUDA) {
//     return rope_kernel_cu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get a rope kernel.";
//     return nullptr;
//   }
// }

// ScaleKernel get_scale_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return scale_inplace_cpu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get a rope kernel.";
//     return nullptr;
//   }
// }

// SoftmaxInplaceKernel get_softmax_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return softmax_inplace_cpu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get an softmax kernel.";
//     return nullptr;
//   }
// }

// SwigluKernel get_swiglu_kernel(DeviceType device_type, void* stream) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return swiglu_kernel_cpu;
//   } else if (device_type == DeviceType::kDeviceCUDA) {
//     return swiglu_kernel_cu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
//     return nullptr;
//   }
// }

// RMSNormKernel get_rmsnorm_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return rmsnorm_kernel_cpu;
//   } else if (device_type == DeviceType::kDeviceCUDA) {
//     return rmsnorm_kernel_cu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get an rmsnorm kernel.";
//     return nullptr;
//   }
// }

// ScaleSumKernel get_scale_sum_kernel(DeviceType device_type) {
//   if (device_type == DeviceType::kDeviceCPU) {
//     return scale_sum_kernel_cpu;
//   } else {
//     LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
//     return nullptr;
//   }
// }

}  // namespace kernel
