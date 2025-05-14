# OnnxFlow

## 导出模型：

```bash
optimum-cli export onnx --model [模型路径] [输出目录] [参数]

// 示例
optimum-cli export onnx \
  --model ./models/qwen_model \
  --task text-generation \
  ./qwen_onnx
