#include <gtest/gtest.h>
#include "layer.h"
#include "base.h"

// 创建具体派生类用于测试基类
class TestBaseLayer : public BaseLayer {
public:
    using BaseLayer::BaseLayer;  // 继承构造函数
    
    // 实现纯虚函数（简单空实现）
    void forward() override {}
};

TEST(BaseLayerTest, ConstructorInitializesMembers) {
    TestBaseLayer layer(
        DeviceType::kDeviceCUDA,
        LayerType::kLayerLinear,
        DataType::kDataTypeFp32,
        "test_layer"
    );
    
    // 验证构造参数初始化
    EXPECT_EQ(layer.device_type(), DeviceType::kDeviceCUDA);
    EXPECT_EQ(layer.layer_type(), LayerType::kLayerLinear);
    EXPECT_EQ(layer.data_type(), DataType::kDataTypeFp32);
    EXPECT_EQ(layer.get_layer_name(), "test_layer");
}

TEST(BaseLayerTest, PropertySettersModifyState) {
    TestBaseLayer layer(
        DeviceType::kDeviceCPU,
        LayerType::kLayerUnknown,
        DataType::kDataTypeUnknown
    );
    
    // 测试设备类型修改
    layer.set_device_type(DeviceType::kDeviceCUDA);
    EXPECT_EQ(layer.device_type(), DeviceType::kDeviceCUDA);
    
    // 测试层名称修改
    layer.set_layer_name("modified_layer");
    EXPECT_EQ(layer.get_layer_name(), "modified_layer");
}

TEST(BaseLayerTest, DefaultNameHandling) {
    TestBaseLayer unnamed_layer(  // 不传递层名
        DeviceType::kDeviceCPU,
        LayerType::kLayerLinear,
        DataType::kDataTypeFp32
    );
    
    EXPECT_TRUE(unnamed_layer.get_layer_name().empty());
}