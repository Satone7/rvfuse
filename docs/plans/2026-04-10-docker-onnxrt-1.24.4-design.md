# ONNX Runtime v1.24.4 RISC-V 原生编译设计

日期: 2026-04-10
状态: 已批准

## 背景

现有 `tools/docker-onnxrt/` 使用 ONNX Runtime v1.17.3 + GCC + minimal build 编译 YOLO runner。为了获得更完整的 ONNX Runtime 功能和更好的 RISC-V Vector 扩展支持，需要新建一个独立的编译环境，使用较新的 ORT v1.24.4 + LLVM/Clang + 全量编译。

本设计参考现有 `tools/docker-onnxrt/` 的结构，但仅关注 ONNX Runtime 本身的编译（不包含 YOLO runner 或 sysroot 提取）。

## 目标

1. 在 `tools/docker-onnxrt-1.24.4/` 中新建独立的 Docker 编译环境
2. 使用 LLVM/Clang 编译 ONNX Runtime v1.24.4（全量编译，非 minimal build）
3. 目标架构: riscv64gcv（RISC-V 64-bit + G (IMAFD) + C (Compressed) + V (Vector) 扩展）
4. 产出保留在 Docker image 中（image tag: `rvfuse-onnxrt-1.24.4`）

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| ONNX Runtime | v1.24.4 (最新稳定补丁) | 包含最新 bug 修复和安全改进 |
| 编译器 | LLVM 18 (Ubuntu 24.04 默认 clang) | apt 直接安装，无需从源码构建；支持 riscv64gcv |
| 基础镜像 | riscv64/ubuntu:24.04 | 与现有 docker-onnxrt 一致 |
| 构建类型 | 全量编译 (无 --minimal_build) | 获取完整的 ORT 功能集 |
| 编译模式 | Release + 共享库 | --build_shared_lib，--config Release |
| 构建方法 | 宿主机预克隆源码 + Docker COPY | 与现有 docker-onnxrt 结构一致 |

### LLVM 版本选择说明

LLVM 22 没有 riscv64 官方预构建二进制包，从源码构建在 QEMU 模拟下需要 10-20 小时。Ubuntu 24.04 默认提供 LLVM 18 (clang-18)，支持 RISC-V Vector 扩展，可直接通过 apt 安装。

## 目录结构

```
tools/docker-onnxrt-1.24.4/
├── Dockerfile       # 单阶段 Dockerfile（ORT 编译 + 产出组织）
├── build.sh         # 宿主机预克隆 + Docker 构建入口
└── vendor/          # .gitignored，宿主机预克隆的源代码树
    ├── onnxruntime/ # ORT v1.24.4 + 递归子模块
    └── eigen/       # Eigen（ORT 依赖）
```

## build.sh 设计

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor"

ONNXRUNTIME_REPO="https://github.com/microsoft/onnxruntime.git"
ONNXRUNTIME_VERSION="v1.24.4"
EIGEN_REPO="https://gitlab.com/libeigen/eigen.git"
EIGEN_VERSION="3.4.0"

# clone_if_missing(): 浅克隆 + 跳过子模块（子模块单独获取）
# 带重试的 git submodule update --init --recursive --depth=1

# Docker 构建
DOCKER_BUILDKIT=1 docker build \
    --platform riscv64 \
    --network=host \
    -t rvfuse-onnxrt-1.24.4 \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --progress=plain \
    "${PROJECT_ROOT}"
```

### build.sh 与现有 docker-onnxrt/build.sh 的差异

| 项目 | 现有 docker-onnxrt | docker-onnxrt-1.24.4 |
|------|-------------------|----------------------|
| ORT 版本 | v1.17.3 | v1.24.4 |
| 构建产物提取 | docker cp 到 output/ | 无（保留在 image 中） |
| sysroot 提取 | 有 | 无 |
| 最终清理 | docker rmi | 无（保留 image） |

## Dockerfile 设计

```dockerfile
FROM --platform=riscv64 riscv64/ubuntu:24.04 AS onnxrt-build

# 安装 LLVM 18 + 构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang lld llvm cmake ninja-build python3 git ca-certificates make \
    && rm -rf /var/lib/apt/lists/

ENV CC=clang
ENV CXX=clang++

# 拷贝源代码（由 build.sh 预克隆到宿主机）
COPY tools/docker-onnxrt-1.24.4/vendor/onnxruntime/ /onnxruntime/
COPY tools/docker-onnxrt-1.24.4/vendor/eigen/ /eigen/

# 全量编译 ONNX Runtime v1.24.4
RUN --mount=type=cache,target=/onnxruntime/build \
    cd /onnxruntime \
    && bash build.sh \
        --config Release \
        --build_shared_lib \
        --parallel 8 \
        --allow_running_as_root \
        --skip_tests \
        --cmake_extra_defines \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
            FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen \
            CMAKE_C_FLAGS="-march=riscv64gcv -mtune=riscv64" \
            CMAKE_CXX_FLAGS="-march=riscv64gcv -mtune=riscv64"

# 组织产出
RUN mkdir -p /onnxruntime/build-output \
    && cp /onnxruntime/build/Linux/Release/libonnxruntime.so* /onnxruntime/build-output/ \
    && cp /onnxruntime/build/Linux/Release/onnxruntime_config.h /onnxruntime/build-output/ \
    && cp -r /onnxruntime/include /onnxruntime/build-output/
```

### 关键设计决策

1. **单阶段构建**: 只编译 ORT，不需要 YOLO runner 或最终精简镜像
2. **BuildKit 缓存**: `--mount=type=cache,target=/onnxruntime/build` 加速增量构建
3. **原生构建**: 容器本身运行在 riscv64 上，无需 ORT 的 cross-compilation toolchain
4. **riscv64gcv flags**: 同时通过 CMAKE_C/CXX_FLAGS 和编译器标志启用 RISC-V 扩展

### ORT CMake 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| --config | Release | 优化构建 |
| --build_shared_lib | (flag) | 构建 .so 而非 .a |
| --parallel | 8 | 并行编译 |
| --skip_tests | (flag) | 跳过测试（原生编译下测试需要 QEMU） |
| onnxruntime_BUILD_UNIT_TESTS | OFF | 不构建单元测试 |
| FETCHCONTENT_SOURCE_DIR_EIGEN | /eigen | 使用宿主机预克隆的 Eigen |
| CMAKE_C/CXX_FLAGS | -march=riscv64gcv | 启用 G+C+V 扩展 |

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| ORT v1.24.4 全量编译在 QEMU 模拟下非常慢 | BuildKit 缓存加速增量构建；预计 6-12 小时首次构建 |
| ORT v1.24.4 的 CMake 配置与 v1.17 差异大 | 先用 --minimal_build 验证编译流程，再切换全量编译 |
| LLVM 18 的 riscv64gcv 支持可能有 bug | clang 18 已稳定支持 RISC-V Vector (V) 扩展 |
| Eigen 版本不匹配 ORT v1.24.4 要求 | 检查 ORT v1.24.4 的 deps.txt 确定所需 Eigen 版本 |
| ONNX Runtime 子模块获取失败 | 带重试的 git submodule update（最多 5 次） |

## 验证清单

- [ ] build.sh 成功克隆 ORT v1.24.4 和 Eigen 到 vendor/
- [ ] Docker build 成功完成，无错误
- [ ] 镜像中存在 /onnxruntime/build-output/libonnxruntime.so*
- [ ] 镜像中存在 /onnxruntime/build-output/include/ 头文件
- [ ] `file` 确认 .so 是 RISC-V ELF
- [ ] ORT 版本信息可通过 header 或 cmake 配置确认是 v1.24.4
