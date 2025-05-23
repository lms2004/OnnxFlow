#include <gtest/gtest.h>
#include "layer.h"
#include "base.h"

namespace mybase {
namespace {

// 测试基础枚举和工具函数
TEST(BaseEnumsTest, BasicEnumValues) {
  EXPECT_EQ(static_cast<uint8_t>(DeviceType::kDeviceCPU), 1);
  EXPECT_EQ(static_cast<uint8_t>(DataType::kDataTypeFp32), 1);
  EXPECT_EQ(DataTypeSize(DataType::kDataTypeInt8), sizeof(int8_t));
}

// 添加 MockBaseLayer 类
class MockBaseLayer : public BaseLayer {
public:
    MockBaseLayer(DeviceType device_type, LayerType layer_type, DataType data_type, const char* name)
        : BaseLayer(device_type, layer_type, data_type, name) {}

    void forward() override {}  // 空实现
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {}
    void to_cuda() override {}  // 空实现
};

// 修改 BaseLayerTest 测试夹具
class BaseLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        base_layer_ = std::make_unique<MockBaseLayer>(  // 使用具体派生类
            DeviceType::kDeviceCUDA, 
            LayerType::kLayerUnknown,
            DataType::kDataTypeFp32,
            "test_layer");
    }
    std::unique_ptr<MockBaseLayer> base_layer_;
};

TEST_F(BaseLayerTest, ConstructorAndGetters) {
  EXPECT_EQ(base_layer_->device_type(), DeviceType::kDeviceCUDA);
  EXPECT_EQ(base_layer_->data_type(), DataType::kDataTypeFp32);
  EXPECT_EQ(base_layer_->get_layer_name(), "test_layer");
}

TEST_F(BaseLayerTest, SettersWork) {
  base_layer_->set_layer_name("new_name");
  EXPECT_EQ(base_layer_->get_layer_name(), "new_name");

  base_layer_->set_device_type(DeviceType::kDeviceCPU);
  EXPECT_EQ(base_layer_->device_type(), DeviceType::kDeviceCPU);
}

// Layer 测试夹具
class LayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    layer_ = std::make_unique<Layer>(
        DeviceType::kDeviceCPU,
        LayerType::kLayerLinear,  // 假设有具体的LayerType
        DataType::kDataTypeInt8,
        "test_layer");
  }

  std::unique_ptr<Layer> layer_;
  Tensor dummy_tensor_;  // 假设Tensor有默认构造函数
};

TEST_F(LayerTest, InputOutputManagement) {
  // 测试初始大小
  EXPECT_EQ(layer_->input_size(), 0);
  EXPECT_EQ(layer_->output_size(), 0);

  // 调整大小并设置值
  layer_->reset_input_size(2);
  layer_->reset_output_size(1);
  
  layer_->set_input(0, dummy_tensor_);
  layer_->set_output(0, dummy_tensor_);

  EXPECT_EQ(layer_->input_size(), 2);
  EXPECT_EQ(layer_->output_size(), 1);
  
  // 测试非const版本
  layer_->get_input(0) = dummy_tensor_;
  layer_->get_output(0) = dummy_tensor_;
}

// LayerParam 测试夹具
class LayerParamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    layer_param_ = std::make_unique<LayerParam>(
        DeviceType::kDeviceCPU,
        LayerType::kLayerLinear,
        DataType::kDataTypeInt8,
        "conv_layer",
        true);
  }

  std::unique_ptr<LayerParam> layer_param_;
  Tensor dummy_tensor_;
};

TEST_F(LayerParamTest, WeightSizeManagement) {
  // 初始状态验证
  EXPECT_EQ(layer_param_->weight_size(), 0);

  // 正常尺寸重置
  layer_param_->reset_weight_size(5);
  EXPECT_EQ(layer_param_->weight_size(), 5); // [2,4](@ref)

  // 边界值：重置为0
  layer_param_->reset_weight_size(0);
  EXPECT_EQ(layer_param_->weight_size(), 0);
  
  // 捕获 stderr 输出
  ::testing::internal::CaptureStderr();
  
  // 异常场景：负数尺寸
  layer_param_->reset_weight_size(-1);
  
  // 获取捕获的日志内容
  std::string output = ::testing::internal::GetCapturedStderr();
  // 验证错误日志内容
  EXPECT_TRUE(output.find("size must be >= 0") != std::string::npos);
}

TEST_F(LayerParamTest, WeightReadWrite) {
  layer_param_->reset_weight_size(3);
  std::shared_ptr<DeviceAllocator> alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor test_tensor(DataType::kDataTypeInt8, 1, 2, true, alloc);

  // 合法索引访问
  layer_param_->set_weight(0, test_tensor);
  EXPECT_TRUE(layer_param_->get_weight(0) == test_tensor);

  // 非法索引访问（索引越界）(1)
  ::testing::internal::CaptureStderr();
  layer_param_->set_weight(3, test_tensor);
  std::string output1 = ::testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output1.find("index out of bounds") != std::string::npos);

  // 非法索引访问（索引越界）(2)
  ::testing::internal::CaptureStderr();
  layer_param_->get_weight(-1);
  std::string output2 = ::testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output2.find("index out of bounds") != std::string::npos);

  // 常量版本访问验证
  const auto& const_param = *layer_param_;
  EXPECT_NO_THROW(const_param.get_weight(0)); // [6](@ref)
}


TEST_F(LayerParamTest, QuantizationFlag) {
  EXPECT_TRUE(layer_param_->is_quant_layer());
}

// 测试 Layer 的 forward 方法及异常场景
TEST_F(LayerTest, ForwardFunctionality) {
  layer_->reset_input_size(2);
  layer_->reset_output_size(1);

  // 创建测试用 Tensor
  std::shared_ptr<DeviceAllocator> alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor t1(DataType::kDataTypeInt8, 1, 2, true, alloc);
  Tensor t2(DataType::kDataTypeInt8, 1, 2, true, alloc);
  Tensor out_tensor;

  std::vector<Tensor> inputs = {t1, t2};
  std::vector<Tensor> outputs = {out_tensor};

  // 正常调用测试
  testing::internal::CaptureStderr();
  layer_->forward(inputs, outputs);
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.empty()) << "Unexpected error output: " << output;

  // 输入尺寸不匹配测试
  inputs.push_back(Tensor());  // 添加第三个输入
  testing::internal::CaptureStderr();
  layer_->forward(inputs, outputs);
  output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.find("input size mismatch") != std::string::npos);
}

// 测试 CUDA 转换功能
TEST_F(LayerTest, CudaConversion) {
  // 设置 CUDA 配置
  auto cuda_config = std::make_shared<kernel::CudaConfig>();
  cuda_config->stream = nullptr;  // 实际使用时应为有效 CUDA 流
  
  // 配置输入输出 Tensor
  layer_->reset_input_size(1);
  layer_->reset_output_size(1);
  std::shared_ptr<DeviceAllocator> alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor cpu_tensor(DataType::kDataTypeFp32, 1, 2, true, alloc);
  
  layer_->set_input(0, cpu_tensor);
  layer_->set_output(0, cpu_tensor);
  layer_->set_cuda_config(cuda_config);

  // 执行转换并验证
  testing::internal::CaptureStderr();
  layer_->to_cuda();
  std::string output = testing::internal::GetCapturedStderr();
  
  EXPECT_TRUE(output.empty()) << "Unexpected error output: " << output;
  EXPECT_EQ(layer_->get_input(0).device_type(), DeviceType::kDeviceCUDA);
  EXPECT_EQ(layer_->get_output(0).device_type(), DeviceType::kDeviceCUDA);
}

// 测试 CUDA 配置管理
TEST_F(LayerTest, CudaConfigManagement) {
  auto config1 = std::make_shared<kernel::CudaConfig>();
  auto config2 = std::make_shared<kernel::CudaConfig>();

  // 测试配置设置和获取
  layer_->set_cuda_config(config1);
  EXPECT_EQ(layer_->cuda_config(), config1);

  // 测试配置更新
  layer_->set_cuda_config(config2);
  EXPECT_EQ(layer_->cuda_config(), config2);

  // 测试空指针处理
  layer_->set_cuda_config(nullptr);
  EXPECT_EQ(layer_->cuda_config(), nullptr);
}

// 测试 LayerParam 的量化相关功能
TEST_F(LayerParamTest, QuantizationFeatures) {
  // 测试 scales 管理
  std::shared_ptr<DeviceAllocator> alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor valid_scales(DataType::kDataTypeFp32, 3, 1, true, alloc);
  
  // 正常设置 scales
  layer_param_->set_scales(valid_scales);
  EXPECT_EQ(layer_param_->get_scale_num(), 3);

  // 测试空 scales 设置
  Tensor empty_scales;
  testing::internal::CaptureStderr();
  layer_param_->set_scales(empty_scales);
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.find("scales is empty") != std::string::npos);
}

// 测试权重管理的高级场景
TEST_F(LayerParamTest, AdvancedWeightManagement) {
  layer_param_->reset_weight_size(2);
  std::shared_ptr<DeviceAllocator> alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  
  // 创建类型匹配的 Tensor
  Tensor valid_weight(DataType::kDataTypeInt8, 1, 2, true, alloc);
  
  // 正常设置测试
  testing::internal::CaptureStderr();
  layer_param_->set_weight(0, valid_weight);
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.empty()) << "Unexpected error output: " << output;

  // 测试类型不匹配
  Tensor wrong_type_weight(DataType::kDataTypeFp32, 1, 2, true, alloc);
  testing::internal::CaptureStderr();
  layer_param_->set_weight(1, wrong_type_weight);
  output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.find("weight data type mismatch") != std::string::npos);
}

// 测试边界条件和异常处理
TEST_F(LayerTest, BoundaryConditions) {
  // 测试输入输出重置为0
  layer_->reset_input_size(0);
  layer_->reset_output_size(0);
  EXPECT_EQ(layer_->input_size(), 0);
  EXPECT_EQ(layer_->output_size(), 0);

  // 测试非法索引访问
  testing::internal::CaptureStderr();
  layer_->get_input(0);
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.find("index out of bounds") != std::string::npos);

  // 测试非法尺寸设置
  testing::internal::CaptureStderr();
  layer_->reset_input_size(-1);
  output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.find("size must be >= 0") != std::string::npos);
}

// 测试量化层的 CUDA 转换
TEST_F(LayerParamTest, QuantizedCudaConversion) {
  auto cuda_config = std::make_shared<kernel::CudaConfig>();
  layer_param_->set_cuda_config(cuda_config);

  // 配置权重和 scales
  layer_param_->reset_weight_size(1);
  std::shared_ptr<DeviceAllocator> alloc = DeviceAllocatorSingleton::getInstance(DeviceType::kDeviceCPU);
  Tensor cpu_weight(DataType::kDataTypeInt8, 1, 2, true, alloc);
  Tensor cpu_scales(DataType::kDataTypeFp32, 4, 1, true, alloc);

  layer_param_->set_weight(0, cpu_weight);
  layer_param_->set_scales(cpu_scales);

  // 执行转换并验证
  testing::internal::CaptureStderr();
  layer_param_->to_cuda();
  std::string output = testing::internal::GetCapturedStderr();
  
  EXPECT_TRUE(output.empty()) << "Unexpected error output: " << output;
  EXPECT_EQ(layer_param_->get_weight(0).device_type(), DeviceType::kDeviceCUDA);
  EXPECT_EQ(layer_param_->get_scale_num(), 4);
}


}  // namespace
}  // namespace mybase

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}