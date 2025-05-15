# OnnxFlow

## 1. 导出模型：

```bash
optimum-cli export onnx --model [模型路径] [输出目录] [参数]
```

### 示例：

```bash
optimum-cli export onnx \
  --model ./models/qwen_model \
  --task text-generation \
  ./qwen_onnx
```

## 2. 编译程序

```bash
cd build
cmake .. # 没修改 CMakeLists.txt 可以不用执行
make
```

## 3. 运行程序

### 3.1 主程序

```bash
cd build
./ONNXFLOW
```

### 3.2 测试程序

#### 测试命令：

```bash
ctest
```

#### 测试结果示例：

```bash
Test project /home/ubun22/OnnxFlow/build
    Start 1: AllocTest.BasicTest1
1/2 Test #1: AllocTest.BasicTest1 .............   Passed    0.00 sec
    Start 2: AllocTest.BasicTest2
2/2 Test #2: AllocTest.BasicTest2 .............   Passed    0.00 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.01 sec
```
