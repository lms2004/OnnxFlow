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
    explicit BaseLayer(DeviceType device_type, LayerType layer_type, DataType data_type,
                        std::string layer_name = "")
            : device_type_(device_type),
            layer_type_(layer_type),
            data_type_(data_type),
            layer_name_(layer_name) {}

    virtual ~BaseLayer() = default;




// 允许派生类访问基类的成员变量
protected:
  std::string layer_name_;
  LayerType layer_type_ = LayerType::kLayerUnknown;
  DataType data_type_ = DataType::kDataTypeUnknown;
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};


#endif// OP_LAYER_H