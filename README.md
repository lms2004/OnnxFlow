````markdown
# OnnxFlow

## 导出模型：

```bash
optimum-cli export onnx --model [模型路径] [输出目录] [参数]
````

### 示例：

```bash
optimum-cli export onnx \
  --model ./models/qwen_model \
  --task text-generation \
  ./qwen_onnx
```

## 主程序

```bash
cd build
./ONNXFLOW
```

## 测试程序

### 编译程序

```bash
cd build
cmake ..
make
```

### 测试程序

```bash
ctest
```

### 测试结果示例：

```bash
ubun22:~/OnnxFlow/build$ ctest
Test project /home/ubun22/OnnxFlow/build
    Start 1: AllocTest.BasicTest
1/1 Test #1: AllocTest.BasicTest ..............   Passed    0.01 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.02 sec
```

```
```
