#ifndef OP_LAYER_H
#define OP_LAYER_H

#include "base.h"
#include <string>

enum class LayerType {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};


class BaseLayer {
 public:
  explicit BaseLayer(DeviceType device_type, LayerType layer_type,
                     DataType data_type, std::string layer_name = "");

  // -- 不带参数（权重）输入输出部分 --
  // virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  // virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  // virtual size_t input_size() const = 0;

  // virtual size_t output_size() const = 0;

  // virtual base::Status check() const = 0;

  // virtual tensor::Tensor& get_input(int32_t idx) = 0;

  // virtual tensor::Tensor& get_output(int32_t idx) = 0;

  // virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  // virtual const tensor::Tensor& get_output(int32_t idx) const = 0;



  // -- getters and setters --

  DataType data_type() const;

  LayerType layer_type() const;

  DeviceType device_type() const; // 返回层的设备类型

  const std::string& get_layer_name() const; // 返回层的名字

  void set_layer_name(const std::string& layer_name); // 设置层的名称

  void set_device_type(DeviceType device_type); // 设置层的设备类型

 protected:
  std::string layer_name_; // 层名
  LayerType layer_type_ = LayerType::kLayerUnknown; // 层类型
  DataType data_type_ = DataType::kDataTypeUnknown; // 层数据类型
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

#endif// OP_LAYER_H