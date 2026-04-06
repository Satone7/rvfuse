# ONNX Runtime + YOLO RISC-V 原生编译设计

日期: 2026-04-04
状态: 已批准

## 背景

RVFuse 项目需要通过 QEMU+BBV 插件分析 RISC-V 程序的热点基本块（BB），以发现可融合的指令对。需要一个真实且计算密集的 RISC-V 可执行程序作为分析对象。

ONNX Runtime + YOLO 推理是理想的工作负载：包含大量矩阵运算（Conv、MatMul），且 YOLO11n 模型足够小以保证在 QEMU 模拟下可完成。

之前的交叉编译尝试失败，改用 Docker RISC-V 原生编译以避免交叉编译兼容性问题。

## 目标

1. 在 `docker run --platform riscv64 riscv64/ubuntu:24.04` 中原生编译 ONNX Runtime v1.17
2. 构建一个完全静态链接的 YOLO 推理可执行文件
3. 推理一张测试图片，重复 10 次（1 次 warm-up + 9 次有效推理）
4. 在 QEMU+BBV 环境中运行，收集热点 BB 数据

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| ONNX Runtime | v1.17.x minimal build | 最小化依赖，RISC-V 原生编译可行 |
| YOLO 模型 | YOLO11n (nano) | 最小模型 (~5.4MB ONNX)，推理快 |
| 编译环境 | riscv64/ubuntu:24.04 (Docker) | 原生编译避免交叉编译问题 |
| 链接方式 | 完全静态链接 | QEMU 运行无需 sysroot |
| 图片加载 | stb_image.h | header-only，零额外依赖 |
| ONNX 导出 | x86 宿主机 (Python + ultralytics) | 避免在 RISC-V 容器安装 PyTorch |

## 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│  x86 宿主机                                                    │
│                                                               │
│  ① prepare_model.sh                                           │
│     Python + ultralytics → YOLO11n.onnx                      │
│     下载 COCO 测试图片 → test.jpg                             │
│                                                               │
│  ② docker build --platform riscv64                            │
│     → output/yolo_inference (静态链接 RISC-V ELF)             │
│                                                               │
│  ③ QEMU+BBV 分析                                              │
│     qemu-riscv64 -plugin libbbv.so ./yolo_inference ...       │
│     → tools/analyze_bbv.py → 热点报告                         │
└──────────────────────────────────────────────────────────────┘
```

## Docker 构建设计

### 三阶段 Dockerfile

```
Stage 1: build-env
  FROM --platform=riscv64 riscv64/ubuntu:24.04
  安装: cmake (Kitware >= 3.26), g++, ninja-build, python3, git, ca-certificates
  目的: 提供完整的原生构建环境

Stage 2: onnxrt-build
  FROM --platform=riscv64 riscv64/ubuntu:24.04
  拷贝: Stage 1 的构建工具
  源码: git clone onnxruntime v1.17
  配置:
    --minimal_build
    --disable_rtti
    --cmake_extra_defines
      CMAKE_BUILD_TYPE=Release
      onnxruntime_BUILD_SHARED_LIB=OFF
      onnxruntime_BUILD_UNIT_TESTS=OFF
      onnxruntime_ENABLE_PYTHON=OFF
  输出: libonnxruntime.a + 静态 protobuf

Stage 3: runner-build
  FROM --platform=riscv64 riscv64/ubuntu:24.04
  拷贝: libonnxruntime.a, 头文件
  编译: g++ -static -O2 -g yolo_runner.cpp
  链接: -lonnxruntime -lprotobuf -lpthread -lm -lc
  输出: /out/yolo_inference
```

### BuildKit 缓存加速

```dockerfile
RUN --mount=type=cache,target=/onnxruntime/build \
    cmake --build /onnxruntime/build
```

利用 Docker BuildKit 的缓存挂载，避免重复编译未变化的部分。

## YOLO Runner 程序

### 功能

1. 加载 ONNX 模型文件
2. 使用 stb_image.h 加载 JPEG 图片
3. 预处理: resize 到 640x640, HWC→CHW, 归一化到 [0,1], 构建 NCHW tensor
4. 循环推理 10 次 (第 1 次 warm-up, 第 2-10 次计入 BBV 数据)
5. 后处理: 输出 Top-N 检测框坐标和置信度

### 编译选项

```
-O2 -g -static
```

保留 DWARF 调试信息，用于后续 addr2line 源码定位。

### 命令行接口

```bash
./yolo_inference <model.onnx> <image.jpg> [iterations]
```

默认 10 次推理，可通过第三个参数调整。

## 宿主机端准备

### prepare_model.sh 脚本

1. 检查 Python3、pip、ultralytics、onnx 是否可用，缺失则安装
2. 使用 ultralytics 下载 YOLO11n 预训练权重
3. 导出为 ONNX 格式 (opset 12, 静态 batch size=1)
4. 从 COCO 数据集下载一张测试图片 (bus.jpg)
5. 输出到 `output/` 目录

## BBV 分析流程

### QEMU+BBV 构建 (一次性)

```bash
cd third_party/qemu
git submodule update --init --depth 1
./configure --target-list=riscv64-linux-user --disable-werror --enable-plugins
make -j$(nproc) && make plugins
```

### 运行分析

```bash
qemu-riscv64 \
  -plugin ./third_party/qemu/build/contrib/plugins/libbbv.so,\
interval=10000,outfile=output/yolo.bbv \
  ./output/yolo_inference \
  ./output/yolo11n.onnx \
  ./output/test.jpg
```

### interval 参数选择

| interval | 适用场景 |
|----------|---------|
| 100,000 | 首次测试，快速验证流程 |
| 10,000 | 推荐值，平衡粒度和数据量 |
| 1,000 | 详细分析，数据量大 |

## 文件结构

```
rvfuse/
├── tools/
│   ├── docker-onnxrt/
│   │   ├── Dockerfile          # 三阶段 RISC-V 原生构建
│   │   └── build.sh            # docker build 入口脚本
│   ├── yolo_runner/
│   │   ├── yolo_runner.cpp     # YOLO 推理 runner
│   │   ├── stb_image.h         # header-only 图片加载
│   │   └── CMakeLists.txt      # runner 构建配置
│   └── analyze_bbv.py          # BBV 热点分析工具
├── output/                     # 编译产物和分析结果
│   ├── yolo_inference          # 静态链接 RISC-V ELF
│   ├── yolo11n.onnx            # YOLO 模型文件
│   └── test.jpg                # COCO 测试图片
├── prepare_model.sh            # 导出 ONNX 模型 + 下载测试图片
└── verify_bbv.sh               # QEMU+BBV 构建+验证脚本 (已有)
```

## 一键工作流

```bash
# 步骤 0: 导出 ONNX 模型和测试图片
./prepare_model.sh

# 步骤 1: 构建 QEMU+BBV (首次)
./verify_bbv.sh

# 步骤 2: Docker 原生编译 ONNX Runtime + YOLO runner
./tools/docker-onnxrt/build.sh

# 步骤 3: 运行 BBV 分析
qemu-riscv64 \
  -plugin ./third_party/qemu/build/contrib/plugins/libbbv.so,\
interval=10000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.onnx ./output/test.jpg

# 步骤 4: 生成热点报告
python3 ./tools/analyze_bbv.py \
  --bbv output/yolo.bbv \
  --elf output/yolo_inference
```

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| Docker RISC-V 模拟编译极慢 | ONNX Runtime minimal build, BuildKit 缓存, 预估 2-6 小时 |
| ONNX Runtime v1.17 RISC-V 编译失败 | 降级到更早版本或尝试 latest, minimal build 减少依赖 |
| 静态链接失败 (glibc static 不完整) | 考虑用 musl-libc 或回退到动态链接+sysroot |
| YOLO11n 推理在 QEMU 下超时 | 减少推理次数或使用更小输入分辨率 |
| BBV 输出数据量过大 | 先用大 interval (100000) 测试，确认流程后再缩小 |

## 验证清单

- [ ] prepare_model.sh 成功生成 yolo11n.onnx 和 test.jpg
- [ ] Docker build 成功完成，无错误
- [ ] output/yolo_inference 是有效的 RISC-V ELF (静态链接)
- [ ] QEMU+BBV 构建成功
- [ ] BBV profiling 生成有效的 .bbv 文件
- [ ] analyze_bbv.py 生成热点函数和基本块报告
- [ ] 热点 BB 中包含有意义的计算指令 (非纯框架代码)
