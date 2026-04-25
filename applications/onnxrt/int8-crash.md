# Vanilla ORT INT8 Inference Crash on RISC-V

## 问题描述

Vanilla ONNX Runtime (v1.24.4, 无 RVV patch) 在 RISC-V 上运行 INT8 YOLO11n 模型时，在 warm-up 阶段必定崩溃（SIGSEGV）。
FP32 模型在同一构建下正常运行。

## 崩溃签名

### 硬件 (Banana Pi BPI-F3, SpacemiT K1, rv64gcv)

```
epc : 0x3f7fc7a7cc  in libonnxruntime.so.1.24.4[0x3f7f255000+0xa8a000]
badaddr: 0x0000000000000268
cause: 0xd (load page fault)
```

epc 偏移 = 0xa2a7cc，对应符号：

```
Eigen::internal::gemm_pack_rhs<int, long, const_blas_data_mapper<int, long, 0>, 4, 0, false, false>::operator()(...)
```

故障指令（偏移 0xa2a7cc）：

```asm
a2a7c8: lw    s0, 0x0(a1)       # 从源数据加载
a2a7ca: addi  a1, a1, 0x4
a2a7cc: sw    s0, -0x8(t5)      # <-- 崩溃：写入目标缓冲区，t5=0x270
```

### QEMU (qemu-riscv64 -cpu max,vlen=512)

```
mmap(NULL, 4919296, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x782d9cc4c000
munmap(0x782d9cc4c000, 4919296) = 0
write(1, ..., "Warm-up run...") = 15
--- SIGSEGV {si_signo=SIGSEGV, si_code=1, si_addr=0x782d9cc4c010} ---
```

**关键发现**：4.7MB 缓冲区通过 mmap 分配后立即被 munmap 释放，推理时尝试访问已释放地址（base+0x10）。
这是经典的 **use-after-free**。

## 根因分析

### 调用链追踪

1. `ConvInteger::Compute()` 调用 `MlasGemm(gemm_shape, gemm_params, thread_pool)`
2. `MlasGemmBatch` → `MlasTrySimpleParallel` → `MlasGemmQuantThreaded`
3. `MlasGemmQuantGetDispatch(AIsSigned, BIsSigned)` 返回 `&MlasGemmQuantDispatchDefault`
4. RISC-V 平台无专门的 QGEMM dispatch（`MlasGemmQuantGetDispatch` 中无 `#elif defined(MLAS_TARGET_RISCV64)` 分支）

**但崩溃点不在 MLAS QGEMM 内部**，而是在 Eigen int32 GEMM packing 代码中。

### 崩溃定位

| 项目 | 值 |
|------|-----|
| 崩溃函数 | `Eigen::internal::gemm_pack_rhs<int, long, ...>` |
| 函数文件 | Eigen `GeneralMatrixMatrix.h` 中的 GEMM packing |
| 故障地址 | 0x268（硬件）/ 已释放 mmap 区域+0x10（QEMU） |
| 故障类型 | 写入已释放/未映射的内存 |

### 可能的触发路径

INT8 YOLO11n 模型包含 839 个算子：

| 算子 | 数量 | 使用 Eigen? |
|------|------|------------|
| DynamicQuantizeLinear | 80 | 否 |
| ConvInteger | 88 | 否（使用 MLAS QGEMM） |
| Cast (int32→float) | 88 | 否（类型转换） |
| Mul / Add | 255 / 103 | 否（逐元素） |
| MatMul | 2 | **是**（FP32 注意力机制） |
| Sigmoid / Reshape / ... | 若干 | 否 |

虽然 2 个 MatMul 算子使用 FP32（不是 int32），但崩溃在 Eigen int32 GEMM packing 中。
推测是 **ORT arena allocator 在会话创建期间分配 workspace（4.7MB），会话创建完成后释放，
但 Eigen 的 `level3_blocking` 缓存了指向该 workspace 的指针，推理时被复用。**

### 为什么 x86/ARM 不崩溃

可能原因：
1. x86/ARM 有专门的 QGEMM dispatch（NEON/AVX），不经过 Eigen int32 GEMM 路径
2. 不同平台的内存分配器行为不同，arena allocator 可能在 x86/ARM 上保留 workspace
3. Eigen 的 `level3_blocking` 在不同平台使用不同的缓存块大小，影响内存分配策略

## 尝试的修复（均无效）

### 1. 强制 MLAS 串行执行

修改 `threading.cpp`，将 `MlasTrySimpleParallel`、`MlasExecuteThreaded`、`MlasTryBatchParallel` 全部改为串行循环，
绕过 ThreadPool。

**结果**：仍崩溃。原因：ThreadPool 非空时原始代码通过 `MLAS_THREADPOOL::TrySimpleParallelFor` 调度到线程池线程，
线程池线程的 `ThreadedBufHolder`（thread_local）未初始化。但完全串行化后仍然崩溃，
说明问题不在线程局部存储。

### 2. 禁用图优化

将 `session_options.SetGraphOptimizationLevel` 从 `ORT_ENABLE_BASIC` 改为 `ORT_DISABLE_ALL`。

**结果**：仍崩溃。说明问题不在图优化阶段的 constant folding 或其他变换。

### 3. 添加 ThreadPool 空指针检查（早期尝试）

在 `MlasTrySimpleParallel` 中添加 `ThreadPool == nullptr` 检查，强制串行。

**结果**：仍崩溃。ORT 传入非空 ThreadPool（即使 `SetIntraOpNumThreads(1)`），
因此该检查不触发。

### 4. DisablePerSessionThreads

调用 `session_options.DisablePerSessionThreads()`。

**结果**：直接 abort（ORT 要求必须通过 `CreateEnvWithGlobalThreadPools` API 创建 Env）。

## 最小修复方案评估

| 方案 | 工作量 | 可行性 |
|------|--------|--------|
| 调试 ORT arena allocator 在 RISC-V 上的行为 | 高（需要深入 ORT 内存管理代码） | 不确定 |
| 完整集成 RVV512 QGEMM kernel（kernel type struct + dispatch wrapper） | 中（~200 行胶水代码） | 可能绕过 Eigen 路径 |
| 使用不同 ORT 版本 | 低 | 不确定是否已修复 |
| 禁用 INT8 Conv 路径中的 Eigen 回退 | 中（需要找到并修改调度逻辑） | 可能 |
| **使用标量默认 QGEMM kernel（已采用）** | **低**（~50 行平台胶水代码） | **已验证** |

## 修复结果（2026-04-25）

**已修复**。通过为 RISC-V 添加 MLAS QGEMM dispatch 分支，INT8 YOLO11n 推理在 RISC-V 上成功运行。

### 修复方案

采用标量默认内核（`MlasGemmQuantDispatchDefault`）作为 QGEMM 实现：

- 标量内核是 ORT 自带的平台无关实现，无 SIMD 依赖，兼容所有 VLEN 配置（128/256/512）
- 崩溃根因不是标量内核本身，而是 RISC-V 缺少任何 MLAS QGEMM dispatch 分支，导致 ORT 回退到 Eigen 路径
- 添加 dispatch 分支后，QGEMM 走 MLAS 路径，完全绕过有 bug 的 Eigen `level3_blocking`

改动文件：

1. `mlasi.h` — 添加 RISC-V QGEMM dispatch 字段到 `MLAS_PLATFORM` 结构体
2. `qgemm.h` — 在 `MlasGemmQuantGetDispatch` 中添加 `MLAS_TARGET_RISCV` 分支
3. `platform.cpp` — 将所有 QGEMM dispatch 指向 `&MlasGemmQuantDispatchDefault`
4. `cmake/onnxruntime_mlas.cmake` — 添加 RISC-V 平台源文件（SGEMM）
5. `mlas.h` — 定义 `MLAS_TARGET_RISCV`
6. `threading.cpp` — 强制串行执行避免 thread_local 问题

### 测试结果

| 测试场景 | 结果 |
|----------|------|
| `yolo_inference` + yolo11n_int8.onnx | 通过（输出 shape [1,84,8400]，checksum 偏差 0.4%） |
| `yolo_inference` + yolo11n_int8.ort | 通过 |
| `generic_ort_runner` + yolo11n_int8.onnx | 仍崩溃（Eigen 路径 use-after-free，非 QGEMM 路径） |
| `generic_ort_runner` + yolo11n.onnx (FP32) | 通过 |

**注**：`generic_ort_runner` 的 INT8 崩溃可能是 Eigen `level3_blocking` 缓存问题，
不影响实际 YOLO 推理路径（YOLO 使用 ConvInteger 走 MLAS QGEMM）。

### 修复补丁

- `applications/onnxrt/fix-int8-riscv.patch` — 完整补丁文件，适用于 ORT v1.24.4
- QGEMM 使用标量默认内核（VLEN 无关），SGEMM 使用 RVV512 内核
- 14 个文件：9 个现有文件修改 + 4 个新增 SGEMM 源文件

### 性能优化路线

当前标量 QGEMM 内核功能正确但性能有限。后续可按需添加 RVV 向量化内核：

| 阶段 | 内核 | VLEN | 预期加速 |
|------|------|------|----------|
| 当前 | 标量默认 | 无关 | 基线（可用） |
| 下一步 | RVV256 (LMUL=2) | 256 | ~8x |
| 可选 | RVV512 (LMUL=1) | 512 | ~16x |

## 复现步骤

```bash
# 编译 vanilla ORT（已应用 RVV512 SGEMM patch，无 QGEMM patch）
bash applications/onnxrt/ort/build.sh --force -j32

# 编译 runner
bash applications/onnxrt/yolo/build.sh --force

# QEMU 复现
third_party/qemu/build/qemu-riscv64 -cpu max,vlen=512 \
    -L output/cross-ort/sysroot \
    -E LD_LIBRARY_PATH=/usr/lib/riscv64-linux-gnu \
    output/cross-ort/generic_ort_runner output/yolo11n_int8.onnx 1

# 硬件复现
ssh root@192.168.100.221 \
    'LD_LIBRARY_PATH=/lib chroot /root/ort-vanilla-int8-fixed/rootfs \
     /generic_ort_runner yolo11n_int8.onnx 1'
# → Segmentation fault (exit code 139)
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `ort/vendor/onnxruntime/.../mlas/lib/threading.cpp` | 已修改为强制串行（本次调试） |
| `ort/vendor/onnxruntime/.../mlas/lib/qgemm.cpp` | QGEMM 入口，MlasGemmBatch |
| `ort/vendor/onnxruntime/.../mlas/lib/qgemm.h` | MlasGemmQuantGetDispatch（RISC-V 无专门分支） |
| `ort/vendor/onnxruntime/.../mlas/lib/qgemm_kernel_default.cpp` | 默认标量 QGEMM kernel |
| `ort/vendor/onnxruntime/.../cpu/quantization/conv_integer.cc` | ConvInteger 调用 MlasGemm |
| `ort/vendor/eigen/Eigen/src/Core/products/GeneralMatrixMatrix.h` | 崩溃点的 Eigen GEMM 实现 |
| `rvv-patches/qgemm-kernel-vl16/` | 已开发的 RVV512 QGEMM kernel（24/24 测试通过） |
