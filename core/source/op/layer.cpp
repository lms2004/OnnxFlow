#include "layer.h"
#include "base.h"

static Tensor default_tensor;  // 返回一个静态的默认 Tensor

/* --------------  BaseLayer class -------------- */
BaseLayer::BaseLayer(DeviceType device_type, LayerType layer_type,
                     DataType data_type, std::string layer_name)
    : layer_name_(layer_name),
      layer_type_(layer_type),
      data_type_(data_type),
      device_type_(device_type) {
    // Constructor implementation
}


// -- getters and setters for BaseLayer --

DataType BaseLayer::data_type() const{
    return data_type_;
}

LayerType BaseLayer::layer_type() const{
    return layer_type_;
}

DeviceType BaseLayer::device_type() const{
    return device_type_;
}

const std::string& BaseLayer::get_layer_name() const{
    return layer_name_;
}

void BaseLayer::set_layer_name(const std::string& layer_name){
    layer_name_ = layer_name;
}

void BaseLayer::set_device_type(DeviceType device_type){
    device_type_ = device_type;
}

/* --------------  Layer class -------------- */
Layer::Layer(DeviceType device_type, LayerType layer_type, 
             DataType data_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, data_type, layer_name) {
}

// -- core func of Layer --
void Layer::forward() {
    return;
}

void Layer::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (inputs.size() != inputs_.size()) {
        Error("Layer::forward: input size mismatch");
        return;
    }
    if (outputs.size() != outputs_.size()) {
        Error("Layer::forward: output size mismatch");
        return;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        this->set_input(i, inputs[i]);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        this->set_output(i, outputs[i]);
    }
    this->forward();
}

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

// -- getter func of Layer --
const Tensor& Layer::get_input(int32_t idx) const {
    if(idx < 0 || idx >= inputs_.size()) {
        Error("Layer::get_input: index out of bounds");
        return default_tensor;  // 如果索引越界，返回默认 Tensor
    }
    return inputs_.at(idx);
}

const Tensor& Layer::get_output(int32_t idx) const {
    if(idx < 0 || idx >= outputs_.size()) {
        Error("Layer::get_output: index out of bounds");
        return default_tensor;  // 如果索引越界，返回默认 Tensor
    }
    return outputs_.at(idx);
}

Tensor& Layer::get_input(int32_t idx) {
    if(idx < 0 || idx >= inputs_.size()) {
        Error("Layer::get_input: index out of bounds");
        return default_tensor;  // 如果索引越界，返回默认 Tensor
    }
    return inputs_.at(idx);
}

Tensor& Layer::get_output(int32_t idx) {
    if(idx < 0 || idx >= outputs_.size()) {
        Error("Layer::get_output: index out of bounds");
        return default_tensor;  // 如果索引越界，返回默认 Tensor
    }
    return outputs_.at(idx);
}

size_t Layer::input_size() const {
    return inputs_.size();
}

size_t Layer::output_size() const {
    return outputs_.size();
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

// -- setter func of Layer --
void Layer::reset_input_size(int size) {
    if(size < 0) {
        Error("Layer::reset_input_size: size must be >= 0");
        return;
    }
    inputs_.resize(size);
}

void Layer::reset_output_size(int size) {
    if(size < 0) {
        Error("Layer::reset_output_size: size must be >= 0");
        return;
    }
    outputs_.resize(size);
}


void Layer::set_input(int32_t idx, const Tensor& input) {
    if(idx < 0 || idx >= inputs_.size()) {
        Error("Layer::set_input: index out of bounds");
        return;
    }
    inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const Tensor& output) {
    if(idx < 0 || idx >= outputs_.size()) {
        Error("Layer::set_output: index out of bounds");
        return;
    }
    outputs_.at(idx) = output;
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
    if(!config) {
        Log("Layer::set_cuda_config: config is null");
    }
  this->cuda_config_ = config;
}

/* --------------  LayerParam class -------------- */
LayerParam::LayerParam(DeviceType device_type, LayerType layer_type,
                     DataType data_type, std::string layer_name, bool is_quant_layer)
    : Layer(device_type, layer_type, data_type, layer_name),
      is_quant_layer_(is_quant_layer) {
}

// -- core func of LayerParam --
void LayerParam::forward() {
    return;
}

void LayerParam::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    return;
}



void LayerParam::to_cuda() {
    Layer::to_cuda();
    for (auto& weight : weights_) {
        weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
    if (!scales_.is_empty()) {
        scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
}

//  getter func of LayerParam
size_t LayerParam::weight_size() const {
    return weights_.size();
}

Tensor& LayerParam::get_weight(int32_t idx) {
    if(idx < 0 || idx >= weights_.size()) {
        Error("LayerParam::get_weight: index out of bounds");
        return default_tensor;  // 如果索引越界，返回默认 Tensor
    }
    return weights_.at(idx);
}

const Tensor& LayerParam::get_weight(int32_t idx) const {
    if(idx < 0 || idx >= weights_.size()) {
        Error("LayerParam::get_weight: index out of bounds");
        return default_tensor;  // 如果索引越界，返回默认 Tensor
    }
    return weights_.at(idx);
}

const Tensor& LayerParam::get_scales() const {
    if (scales_.is_empty()) {
        Error("LayerParam::get_scales: scales is empty");
        return default_tensor;  // 如果索引越界，返回默认 Tensor
    }
    return scales_;
}
int32_t LayerParam::get_scale_num() const {
  if (scales_.is_empty()) {
    Error("LayerParam::get_scale_num: scales is empty");
    return 0;
  }
  return static_cast<int32_t>(scales_.size());
}

bool LayerParam::is_quant_layer() const {
    return is_quant_layer_;
}

// setter func of LayerParam
void LayerParam::set_weight(int32_t idx, const Tensor& weight) {
    if(idx < 0 || idx >= weights_.size()) {
        Error("LayerParam::set_weight: index out of bounds");
        return;
    }
    if(weight.data_type() != data_type_) {
        Error("LayerParam::set_weight: weight data type mismatch");
        return;
    }
    if(weight.device_type() != device_type_) {
        Error("LayerParam::set_weight: weight device type mismatch");
        return;
    }
    weights_.at(idx) = weight;
}

void LayerParam::reset_weight_size(int size) {
    if(size < 0) {
        Error("LayerParam::reset_weight_size: size must be >= 0");
        return;
    }
    weights_.resize(size);
}


void LayerParam::set_scales(const Tensor& scales) {
  if (scales.is_empty()) {
        Error("LayerParam::set_scales: scales is empty");
        return;
  }
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

