#include "layer.h"
#include "base.h"


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


// ----------- getter and setter func of Layer -----------
void Layer::set_input(int32_t idx, const Tensor& input) {
    inputs_[idx] = input;
}

void Layer::set_output(int32_t idx, const Tensor& output) {
    outputs_[idx] = output;
}

const Tensor& Layer::get_input(int32_t idx) const {
    return inputs_[idx];
}

const Tensor& Layer::get_output(int32_t idx) const {
    return outputs_[idx];
}

Tensor& Layer::get_input(int32_t idx) {
    return inputs_[idx];
}

Tensor& Layer::get_output(int32_t idx) {
    return outputs_[idx];
}

size_t Layer::input_size() const {
    return inputs_.size();
}

size_t Layer::output_size() const {
    return outputs_.size();
}

void Layer::reset_input_size(size_t size) {
    inputs_.resize(size);
}

void Layer::reset_output_size(size_t size) {
    outputs_.resize(size);
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


//  getter and setter func of LayerParam
size_t LayerParam::weight_size() const {
    return weights_.size();
}

void LayerParam::reset_weight_size(size_t size) {
    weights_.resize(size);
}

Tensor& LayerParam::get_weight(int32_t idx) {
    return weights_[idx];
}

const Tensor& LayerParam::get_weight(int32_t idx) const {
    return weights_[idx];
}

void LayerParam::set_weight(int32_t idx, const Tensor& weight) {
    weights_[idx] = weight;
}


