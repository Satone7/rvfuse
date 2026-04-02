# ONNX Runtime + YOLO BBV 热点分析设计

日期: 2026-04-02
状态: 已批准

## 目标

使用 QEMU user mode + BBV 插件，定位 ONNX Runtime 运行 YOLO 目标识别时的热点基本块（Basic Block），为后续 RISC-V 指令融合研究提供数据基础。

## 技术选型

| 组件 | 选择 | 版本/规格 |
|------|------|----------|
| ONNX Runtime | 最小化静态构建 | v1.17 |
| YOLO 模型 | YOLO11n (nano) | 最新版 |
| 构建方式 | Docker 多阶段构建 | 交叉编译 |
| 目标架构 | riscv64-linux-gnu | 静态链接 |
| 分析工具 | QEMU + BBV 插件 + analyze_bbv.py | 已有+增强 |
| 图片加载 | stb_image.h | header-only |
| 输入数据 | 单张测试图片 | COCO 数据集样本 |

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Build Container                  │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  RISC-V  │    │     ONNX     │    │  YOLO Runner  │  │
│  │  sysroot │◄───│  Runtime     │    │  (C++ app)    │  │
│  │ (glibc,  │    │  (minimal,   │    │               │  │
│  │  protobuf│    │   static)    │    │  load model   │  │
│  │  etc.)   │    └──────┬───────┘    │  load image   │  │
│  └──────────┘           │            │  run inference │  │
│                         ▼            └───────┬───────┘  │
│                  静态链接到 runner              │          │
│                                              ▼          │
│                                    yolo_inference (ELF)  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ (copy ELF + model + image out)
┌─────────────────────────────────────────────────────────┐
│                  QEMU + BBV Analysis                     │
│                                                          │
│  qemu-riscv64 -plugin libbbv.so,interval=10000,...      │
│                yolo_inference model.onnx test.jpg        │
│                                                          │
│  输出: yolo.bbv  ──►  analyze_bbv.py  ──►  热点报告     │
│        yolo.disas        (增强版)         (函数级+BB级)  │
└─────────────────────────────────────────────────────────┘
```

## Docker 构建环境

### 多阶段构建

```
阶段 1: RISC-V 交叉编译工具链
  - 安装 gcc-riscv64-linux-gnu, g++-riscv64-linux-gnu
  - 创建 RISC-V sysroot (glibc 头文件和库)

阶段 2: ONNX Runtime 依赖
  - protobuf: 交叉编译为 RISC-V 静态库

阶段 3: ONNX Runtime 最小化构建
  - git clone onnxruntime v1.17
  - CMake 配置:
    --minimal_build=ON
    --disable_rtti
    --target_triple riscv64-linux-gnu
    --cmake_extra_defines CMAKE_SYSROOT=...
  - 输出: libonnxruntime.a

阶段 4: YOLO Runner 编译
  - 链接 libonnxruntime.a (静态)
  - 包含图片预处理 (resize, normalize to NCHW tensor)
  - 输出: yolo_inference (RISC-V ELF, with DWARF debug info)
```

### ONNX Runtime 构建要点

- 仅 CPU Execution Provider
- 静态链接: libonnxruntime.a + libprotobuf.a 打包进 runner
- 保留 DWARF 调试信息: 编译选项 `-O2 -g`

## YOLO Runner 程序

### 程序结构

```cpp
// yolo_runner.cpp
int main(int argc, char* argv[]) {
    // 1. 加载 ONNX 模型
    session = Ort::Session(env, argv[1], session_options);

    // 2. 加载并预处理测试图片
    //    - stb_image.h 读取 JPEG
    //    - Resize 到 640x640
    //    - HWC → CHW, 归一化到 [0,1]
    //    - 构建 NCHW tensor: {1, 3, 640, 640}

    // 3. 执行推理
    session.Run(...)

    // 4. 后处理: NMS + 输出检测框
}
```

### 设计决策

- 图片加载: stb_image.h (header-only), 无额外依赖
- 预处理: 内联双线性插值 resize + 归一化
- 后处理: 基本 NMS 实现
- 命令行: `yolo_inference <model.onnx> <image.jpg>`

## BBV 分析流程

### 运行命令

```bash
qemu-riscv64 \
  -plugin ./libbbv.so,interval=10000,outfile=yolo.bbv \
  -L /path/to/riscv-sysroot \
  ./yolo_inference yolo11n.onnx test_image.jpg
```

### interval 参数

| interval 值 | 适用场景 | 分析粒度 |
|-------------|---------|---------|
| 100,000 | 快速概览 | 粗粒度，适合首次测试 |
| 10,000 | 推荐 | 平衡粒度和分析时间 |
| 1,000 | 详细分析 | 数据量大，适合聚焦分析 |

### 增强版 analyze_bbv.py 功能

1. 解析 .bbv 文件，累加每个 PC 地址的执行次数
2. 排序，取 Top N 热点基本块
3. 对每个热点 BB:
   - riscv64-linux-gnu-objdump -d --source 定位汇编和源码
   - riscv64-linux-gnu-addr2line 获取函数名和源文件位置
4. 按函数聚合并排序

### 输出报告格式

```
══════════════════════════════════════════
ONNX Runtime YOLO 推理热点分析报告
══════════════════════════════════════════

── 热点函数 Top 10 ──
执行次数      函数名                          源文件
12,345,678   onnxruntime::MatMul              /kernels/matmul.cc:142
...

── 热点基本块 Top 20 ──
执行次数      PC地址        函数名              汇编片段
5,432,100   0x4a2c30    MatMul::Compute    vle8.v v8, (a0) ...
...

── 执行阶段分析 ──
阶段         估计执行占比    主要操作
模型加载       ~5%          protobuf 解析
推理执行       ~90%         Conv/MatMul/激活函数
后处理         ~5%          NMS/排序
```

### QEMU 运行注意事项

- sysroot: 需要 -L 指向 RISC-V sysroot
- 执行时间预估: QEMU 模拟 + BBV 开销约 10-50x 原生速度，YOLO11n 单次推理预计 5-30 分钟

## 文件结构

```
rvfuse/
├── tools/
│   ├── analyze_bbv.py              # 增强: 函数聚合+源码定位
│   ├── docker-onnxrt/              # 新增
│   │   ├── Dockerfile              # 多阶段构建
│   │   ├── build.sh                # 构建入口脚本
│   │   ├── toolchain-riscv64.cmake # CMake 工具链文件
│   │   └── README.md               # 构建说明
│   └── yolo_runner/                # 新增
│       ├── yolo_runner.cpp         # YOLO 推理 runner
│       ├── stb_image.h             # header-only 图片加载
│       └── CMakeLists.txt          # runner 构建脚本
├── output/                         # 新增，分析结果输出
│   ├── yolo_inference              # 编译产物
│   ├── yolo11n.onnx               # 模型文件
│   ├── test_image.jpg             # 测试图片
│   └── yolo_hotspot_report.txt    # 分析报告
└── docs/plans/
    └── 2026-04-02-onnxrt-yolo-bbv-design.md  # 本文档
```

## 一键工作流

```bash
# 步骤 1: 构建
./tools/docker-onnxrt/build.sh

# 步骤 2: 运行 BBV 分析
qemu-riscv64 \
  -plugin ./third_party/qemu/build/libbbv.so,interval=10000,outfile=output/yolo.bbv \
  -L ./tools/docker-onnxrt/sysroot \
  ./output/yolo_inference \
  ./output/yolo11n.onnx \
  ./output/test_image.jpg

# 步骤 3: 生成热点报告
python3 ./tools/analyze_bbv.py \
  --bbv output/yolo.bbv \
  --elf output/yolo_inference \
  --objdump riscv64-linux-gnu-objdump \
  --addr2line riscv64-linux-gnu-addr2line \
  --top-funcs 10 \
  --top-blocks 20 \
  --output output/yolo_hotspot_report.txt
```

## 交付物

1. **Docker 构建环境**: 可复现的 ONNX Runtime RISC-V 交叉编译环境
2. **YOLO Runner**: 带调试信息的 RISC-V 推理程序
3. **增强版 analyze_bbv.py**: 支持函数级聚合 + 源码定位的热点分析
4. **热点分析报告**: YOLO 推理的热点函数和基本块列表

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| ONNX Runtime RISC-V 交叉编译失败 | 先用 minimal build，逐步添加依赖 |
| QEMU 模拟运行时间过长 | 用 YOLO11n (最小模型)，必要时缩小输入 |
| BBV 输出数据量过大 | 调整 interval 参数，先用大值测试 |
| DWARF 信息不完整 | 确保编译时 -g 选项，验证 addr2line 输出 |

## 验证清单

- [ ] Docker build completes without errors
- [ ] `output/yolo_inference` is valid RISC-V ELF
- [ ] `output/yolo11n.onnx` is valid ONNX model (~5.4MB)
- [ ] `output/test_image.jpg` is valid image
- [ ] BBV profiling produces valid .bbv file
- [ ] `analyze_bbv.py` generates report with hot functions/blocks
- [ ] addr2line resolves PC addresses to source locations
