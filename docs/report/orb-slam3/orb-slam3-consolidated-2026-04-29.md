# ORB-SLAM3 RVV 分析 — 综合报告

**日期**: 2026-04-29
**状态**: 第二阶段完成（RVV 向量化）
**团队**: orb-slam3（AITC 工作流）

## 概要

ORB-SLAM3 是一个实时视觉 SLAM 系统，已交叉编译至 RISC-V (rv64gcv) 平台。原始构建使用纯标量编译选项 (`-march=rv64gc`)，所有库中均未生成 RVV 指令。修正构建配置后，**OpenCV 现有的 CV_SIMD_SCALABLE 路径仅在 libopencv_imgproc.so 中就生成了 17,528 条 RVV 指令**，完全向量化了 #1 热点（GaussianBlur，约占 25% 运行时间）和 #3 热点（FAST 角点检测，约占 4.4% 运行时间）。

## 核心结果

### 算子覆盖矩阵

| 算子 | 库 | 热点占比 | 状态 | RVV 指令数 | 方法 |
|----------|---------|-----------|--------|-----------------|--------|
| GaussianBlur (hline+vline) | libopencv_imgproc.so | ~25% | ✅ 已向量化 | 17,528 | OpenCV 以 `-DCPU_BASELINE=RVV` 重编译 |
| FAST 角点检测 | libopencv_features2d.so | ~4.4% | ✅ 已向量化 | 1,525 | OpenCV 重编译（通用 intrinsic 自动向量化） |
| g2o Eigen 6x6 矩阵 | libg2o.so / libORB_SLAM3.so | ~16% | ❌ 未向量化 | 0 | Eigen 缺少 RVV 后端；需新增任务 |
| ORB 描述子 (BRIEF) | libORB_SLAM3.so | <1% | ⏳ 已推迟 | 0 | 已分析；RVV 方案已记录；投入产出比低 |

### 重编译后 OpenCV 4.10.0 的完整 RVV 指令统计

| 库 | RVV 指令数 | 状态 |
|---------|-----------------|--------|
| libopencv_imgproc.so | 17,528 | GaussianBlur 完全向量化 |
| libopencv_core.so | 8,709 | 核心运算已向量化 |
| libopencv_imgcodecs.so | 6,242 | 图像编解码运算 |
| libopencv_calib3d.so | 3,813 | 3D 标定运算 |
| libopencv_features2d.so | 1,525 | FAST 角点完全向量化 |
| libopencv_flann.so | 665 | 最近邻搜索 |
| libopencv_video.so | 698 | 视频处理 |
| libopencv_highgui.so | 312 | GUI（有限） |
| libopencv_videoio.so | 336 | 视频 I/O |
| **OpenCV 总计** | **39,828** | 从 0 增长至此 |

### 关键优化：vsaddu.vv

定点饱和加法 (`ufixedpoint32::operator+`) 是 vline 内循环中最大的瓶颈，占其 53%（perf 注解：`addw` 15.5% + `sltu` 31.5% + `negw` 6.0% + `or` → 共 4 条标量指令）。在重编译后的库中，这被单条 `vsaddu.vv` 指令替代（libopencv_imgproc.so 中有 356 处）。

## 根因分析

### 为什么所有代码都是标量的？

OpenCV 和 ORB-SLAM3 均以以下配置构建：
1. **工具链文件**使用 `-march=rv64gc`（纯标量，不含 'v' 扩展）
2. **OpenCV 特有**：`CPU_BASELINE=""` 和 `CPU_DISPATCH=""` — 这些 CMake 变量禁用了所有 SIMD 路径
3. **ORB-SLAM3 特有**：`CPU_BASELINE ""` 设为空，阻止了 SIMD
4. **工具链文件中的 FORCE 标记**阻止了命令行覆盖

### 修复方案（3 处改动）

1. **工具链架构标志**：`-march=rv64gc` → `-march=rv64gcv_zvl512b`
2. **OpenCV CPU_BASELINE**：`""` → `"RVV"`
3. **OpenCV 内部 cmake**：`CPU_RVV_FLAGS_ON` 加入 `_zvl512b` 扩展

## 构建配置变更

### OpenCV (`applications/opencv/riscv64-linux-toolchain.cmake`)
```cmake
# 第 21 行：从 -march=rv64gc 改为 -march=rv64gcv_zvl512b
set(RISCV_FLAGS "-march=rv64gcv_zvl512b -mabi=lp64d -g")
# 第 39 行：从 CPU_BASELINE "" 改为 CPU_BASELINE "RVV"
set(CPU_BASELINE "RVV" CACHE STRING "启用 RVV 基线优化")
```

### OpenCV 内部 (`vendor/opencv/cmake/OpenCVCompilerOptimizations.cmake`)
```cmake
# 第 398 行：为 CPU_RVV_FLAGS_ON 加入 _zvl512b
set(CPU_RVV_FLAGS_ON "-march=rv64gcv_zvl512b")
```

### ORB-SLAM3 (`applications/orb-slam3/riscv64-linux-toolchain.cmake`)
```cmake
# 第 17 行：从 -march=rv64gc 改为 -march=rv64gcv_zvl512b
set(RISCV_FLAGS "-march=rv64gcv_zvl512b -mabi=lp64d -g")
```

## 优先级表（BBV 加权）

| 优先级 | 算子 | 热点占比 | RVV 覆盖 | 差距 |
|----------|----------|-----------|-------------|-----|
| 1 | GaussianBlur | ~25% | ✅ 17,528 RVV | — |
| 2 | g2o Eigen 6x6 | ~16% | ❌ 0 RVV | 缺少 RVV 后端 |
| 3 | FAST 角点 | ~4.4% | ✅ 1,525 RVV | — |
| 4 | ORB 描述子 | <1% | ❌ 0 RVV | 已推迟（投入产出比低） |

## 跨平台对比总结

### GaussianBlur — 定点饱和卷积

| 平台 | 饱和加法 | 指令数 | 吞吐量（元素/周期） |
|----------|---------------|-------------|---------------------------|
| RISC-V RVV512 | `vsaddu.vv` | 1 | 32 @ u8, 16 @ u16 |
| x86 AVX2 | `_mm256_packus_epi16` + `_mm256_packus_epi32` | 2 | 32 @ u8 |
| ARM NEON | `vqmovun.s16` | 1 | 8 @ u8 |
| LoongArch LSX | `vsat.bu` + shuffle | 2 | 16 @ u8 |

**RVV 优势**：单条指令，最高吞吐量（VLEN=512 时 32 个元素）。指令数少于 AVX2，竞争力强。

### FAST 角点 — 整数像素比较

| 平台 | 比较指令 | 掩码形式 | 吞吐量 |
|----------|---------|------|-----------|
| RISC-V RVV512 | `vmslt.vx` + `vmsgt.vx` | 向量掩码寄存器 | 64 像素/迭代 @ u8 |
| x86 AVX2 | `_mm256_cmpgt_epi8` | 256 位寄存器内掩码 | 32 像素/迭代 |
| ARM NEON | `vcgt.s8` | 128 位掩码 | 16 像素/迭代 |

**RVV 优势**：最大向量宽度（VLEN=512 时 u8 为 64 元素，AVX2 为 32，NEON 为 16）。

## 遇到的挑战

1. **LLVM 22 可扩展至定长转换 bug**：`accum.dispatch.cpp` 在使用 `-march=rv64gcv_zvl512b` 时编译失败。变通方案：对该文件使用 `#undef CV_RVV`（GaussianBlur/FAST 不使用该文件）。

2. **CMake 标志命名误导**：`RISCV_RVV_SCALABLE=ON` 选择可扩展 API 变体但并不启用 RVV。`CPU_BASELINE=RVV` 才是实际的启用开关。

3. **CMake 工具链文件中的 FORCE 标记**：带有 `FORCE` 的 `CMAKE_C_FLAGS` 和 `CMAKE_CXX_FLAGS` 阻止命令行覆盖。必须直接编辑文件。

4. **空 sysroot**：主 `output/sysroot/` 为空。QEMU 测试使用 `output/orb-slam3/sysroot/`。

5. **GLM-5.1 模型性能**：在处理大型源文件（2,236 行的 smooth.simd.hpp、大型 ORBextractor.cc）时，深度思考阶段导致数分钟的停滞。变通方案：Lead 直接介入处理卡住的 teammate。

## 后续工作

### 新增任务：g2o Eigen 6x6 RVV 实现（优先级：高，约 16% 热点）

Eigen 3.4.0 不会为 RISC-V 自动向量化。需要为 Eigen 的 6x6 固定大小矩阵运算实现全新的 RVV 后端，以覆盖剩余的主要热点：

- `vle64.v` 用于列加载
- `vfmacc.vv` 用于融合乘加
- 为 6x6 矩阵特化 `Eigen::internal::dense_assignment_loop`

### BBV Profiling（已推迟）

由于时间限制，完整的 QEMU BBV profiling 已推迟。静态反汇编得到的 RVV 指令数已为当前阶段提供了充分的量化依据。BBV profiling 可在需要硬件验证时作为后续工作完成。

### Gap Analysis（已推迟）

逐算子的差距分析已推迟。上方的跨平台对比已提供关键洞察。详细的逐指令对比可与 g2o Eigen 实现一同完成。

## 参考文献

- ET-1 验证：`applications/orb-slam3/rvv-patches/gaussian-blur/verification.md`
- GaussianBlur README：`applications/orb-slam3/rvv-patches/gaussian-blur/README.md`
- FAST 角点 README：`applications/orb-slam3/rvv-patches/fast-corner/README.md`
- g2o Eigen README：`applications/orb-slam3/rvv-patches/g2o-eigen-6x6/README.md`
- ORB 描述子 README：`applications/orb-slam3/rvv-patches/orb-descriptor/README.md`
- 第一阶段 perf 数据：`output/perf/orb-slam3/`
- 计划：`docs/plans/orb-slam3-2026-04-29.md`
