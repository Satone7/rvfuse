# RVFuse

RISC-V Instruction Fusion Research Platform

## Overview

RVFuse profiles real workloads (YOLO object detection) on RISC-V via QEMU emulation, collects basic block execution data, and identifies instruction fusion candidates in hot code paths.

The current pipeline:
1. **Build** — Cross-compile ONNX Runtime + YOLO inference runner natively for RISC-V inside Docker
2. **Profile** — Run the RISC-V binary under QEMU with the BBV (Basic Block Vector) plugin to collect execution counts
3. **Analyze** — Map hot addresses back to source code and identify fusion opportunities
4. **Graph** — Generate Data Flow Graphs (DFG) for hot basic blocks to visualize instruction-level dependencies
5. **Discover** — Mine fusible instruction patterns from DFG data and score candidates by hardware feasibility

## 指令分析与调研流程 (Instruction Analysis Workflow)

本流程提供一套规范性的指令分析和向量化优化调研方法：

| 步骤 | 目标 | 输入 | 输出 | Skill |
|------|------|------|------|-------|
| 1 | 交叉编译目标应用 | 源代码 | RISC-V 可执行文件 + sysroot | `cross-compile-app` |
| 2 | 验证应用运行 | RISC-V 二进制 | 运行成功确认 | — |
| 3 | BBV Profiling | RISC-V 二进制 | `.bb` 统计 + `.disas` 反汇编 | `qemu-bbv-usage` |
| 4 | 热点分析 | BBV 数据 | `hotspot.json` | — |
| 5 | RVV 向量化优化 | 热点函数 | 向量化版本 + 正确性验证 | `rvv-op` |
| 6 | 优化效果评估 | 向量化版本 | 新的 BBV 数据 + 对比分析 | — |
| 7 | Patch 文件生成 | 向量化改动 | `.patch` 文件存档 | — |
| 8 | 多架构对比分析 | Patch + 其他架构实现 | 指令分析报告 | `rvv-gap-analysis` |

### 使用 Skills 加速流程

本项目提供 Claude Code Skills 来自动化各步骤。当 LLM 助手执行流程时，应优先调用对应 Skill：

**Step 1 — 交叉编译目标应用**

```
请帮我添加 <app-name> 到 applications/，repo 是 <repo-url>，编译到 RISC-V。
```

LLM 会自动调用 `cross-compile-app` Skill，完成：
- 源码克隆到 `applications/<name>/vendor/`
- 构建系统分析（CMake / Autotools）
- 工具链生成（`riscv64-linux-toolchain.cmake`）
- Sysroot 提取（Docker `riscv64/ubuntu:24.04`）
- 交叉编译 + QEMU 冒烟测试

**Step 3 — BBV Profiling**

调用 `qemu-bbv-usage` Skill 获取 QEMU BBV 插件的正确使用方式和参数配置。

**Step 5 — RVV 向量化优化**

调用 `rvv-op` Skill 生成 RVV vector intrinsic 代码替换热点函数中的 scalar 实现。

详细操作步骤请参考 CLAUDE.md。

## Prerequisites

- Docker with BuildKit enabled
- Python 3.10+ with `pip`
- Git

## Quick Start

### One-command setup

```bash
./setup.sh
```

Options:
- `--shallow` — shallow submodule clone
- `--bbv-interval N` — BBV sampling interval (default: 100000)
- `--top N` — top N blocks for analysis (default: 20)
- `--coverage N` — coverage threshold % (default: 80)

### Manual steps

```bash
# Initialize submodules
git submodule update --init --depth 1

# Prepare YOLO model and test image
./prepare_model.sh

# Build QEMU with BBV plugin
./verify_bbv.sh

# Cross-compile ONNX Runtime + YOLO runner for RISC-V
./applications/yolo/ort/build.sh

# Run BBV profiling
./third_party/qemu/build/qemu-riscv64 \
  -L output/sysroot \
  -plugin ./tools/bbv/libbbv.so,interval=100000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg

# Generate hotspot report
python3 tools/analyze_bbv.py \
  --bbv output/yolo.bbv.0.bb \
  --elf output/yolo_inference \
  --sysroot output/sysroot

# Generate DFG for hot basic blocks
python3 -m tools.dfg \
  --disas output/yolo.bbv.0.disas \
  --report output/hotspot.json \
  --top 20
```

## Project Structure

```
RVFuse/
├── setup.sh               # Full pipeline orchestrator
├── prepare_model.sh       # YOLO model export and test data
├── verify_bbv.sh          # QEMU + BBV plugin build
├── skills/                # Claude Code Skills (cross-compile-app, qemu-bbv-usage, rvv-op, ...)
├── tools/                 # Analysis tools (DFG, BBV, fusion)
├── applications/          # Test applications (YOLO inference)
├── docs/                  # Architecture and design documents
├── third_party/           # Git submodules (QEMU, LLVM)
└── output/                # Build artifacts and profiling data
```

## Dependencies

| Dependency | Purpose |
|------------|---------|
| QEMU | RISC-V emulation + BBV profiling |
| ONNX Runtime v1.24.4 | Neural network inference |
| Eigen 3.4.0 | Linear algebra library |

## License

[License to be determined]