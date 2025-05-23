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
    std::unique_ptr<BaseLayer> base_layer_;
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

}  // namespace
}  // namespace mybase

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}