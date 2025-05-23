#ifndef OP_LAYER_H
#define OP_LAYER_H

#include "tensor.h"
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

  virtual void forward() = 0;

  // --------- getter func ---------                   
  DataType data_type() const;
  LayerType layer_type() const;
  DeviceType device_type() const; // 返回层的设备类型

  const std::string& get_layer_name() const; // 返回层的名字

  // -------- setter func ---------
  void set_layer_name(const std::string& layer_name); // 设置层的名称
  void set_device_type(DeviceType device_type); // 设置层的设备类型

 protected:
  std::string layer_name_; // 层名
  LayerType layer_type_ = LayerType::kLayerUnknown; // 层类型
  DataType data_type_ = DataType::kDataTypeUnknown; // 层数据类型
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class Layer : public BaseLayer {
 public:
  explicit Layer(DeviceType device_type, LayerType layer_type,
                     DataType data_type, std::string layer_name = "");
  void forward() override;

  // --------- getter and  setter func of in/out ---------
  void set_input(int32_t idx, const Tensor& input);
  void set_output(int32_t idx, const Tensor& output);

  const Tensor& get_input(int32_t idx) const ;
  const Tensor& get_output(int32_t idx) const;
  Tensor& get_input(int32_t idx);
  Tensor& get_output(int32_t idx);

  size_t input_size() const;
  size_t output_size() const;

  void reset_input_size(int size);
  void reset_output_size(int size);

 protected:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};


class LayerParam : public Layer {
 public:
    explicit LayerParam(DeviceType device_type, LayerType layer_type,
                     DataType data_type, std::string layer_name = "", bool is_quant_layer = false);
    
    // --------- core func of LayerParam ---------
    void forward() override;

    // --------- getter func of weight ---------
    size_t weight_size() const;

    Tensor& get_weight(int32_t idx);
    const Tensor& get_weight(int32_t idx) const;
    
    bool is_quant_layer() const;

    // --------- setter func of weight ---------
    void set_weight(int32_t idx, const Tensor& weight);
    // void set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
    //                         DeviceType device_type = DeviceType::kDeviceUnknown);

    void reset_weight_size(int size);
 protected:
    bool is_quant_layer_ = false;
    std::vector<Tensor> weights_;
};




#endif// OP_LAYER_H