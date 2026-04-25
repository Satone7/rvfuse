# SGEMM Kernel (VL=16) 多平台向量实现对比与RVV扩展指令建议 -- WASM SIMD篇

## 概述

**分析目标**: SGEMM (Single-precision General Matrix Multiply) 内核 -- WASM SIMD 实现与 RVV 基准对比
**基准实现**: RVV VL=16 (VLEN=512bit, SEW=32bit, LMUL=1)
**分析平台**: WASM SIMD (128-bit)
**BBV数据**: 未提供，收益为理论估算

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| -- | (无新扩展建议) | WASM SIMD | 参考对比，RVV已具备全面优势 | -- | -- |

**说明**: WASM SIMD 作为 Web 端 128-bit 向量抽象层，其指令集显著弱于 RVV。本报告为参考对比性质，主要展示 RVV 的已有优势，不产生新的扩展指令建议。

**收益计算方式**（基于 BB 内指令计数，归一化到 RVV VLEN=512bit）：
- WASM SIMD 寄存器宽度 128-bit = 4 x float32
- RVV VLEN=512bit = 16 x float32
- 归一化因子 = 512 / 128 = 4x
- 所有比较均归一化到处理 16 个 float32 的等量工作

---

## 基准RVV实现分析

RVV SGEMM 内核 (VL=16) 的 K 循环结构：

- **维度**: 每次调用处理 2 行 x 16 列
- **K 循环展开度**: 2（每次迭代处理 2 个 K 步）
- **单 K-pair 的 1 行指令序列**:
  ```
  flw        fa0, 0(a_ptr)          ; 加载 A[k]         (1)
  flw        fa1, 4(a_ptr)          ; 加载 A[k+1]        (2)
  vle32.v    v_b0, (b_ptr)          ; 加载 B[k][0..15]   (3)
  vle32.v    v_b1, 16(b_ptr)        ; 加载 B[k+1][0..15] (4)
  vfmacc.vf  v_acc0, fa0, v_b0      ; FMA: acc += A[k]*B[k]     (5)
  vfmacc.vf  v_acc0, fa1, v_b1      ; FMA: acc += A[k+1]*B[k+1] (6)
  ```
  **每 K-pair**: 6 条指令, 产生 16 个 FMA (1 行 x 16 列 x 2 K 步)
- **2 行总指令数** (K-pair): 6 x 2 = 12 条指令, 32 个 FMA
- **FMA 效率**: 12 条指令 / 32 FMA = 0.375 条指令/FMA

---

## WASM SIMD 对比分析

### WebAssembly SIMD (128-bit)

**核心特点**:
- 128-bit v128 寄存器 (4 x float32)
- **无 FMA 指令** -- 必须使用 `f32x4.mul` + `f32x4.add` 两条指令实现一次乘累加
- 标量广播需要 `i32x4.splat` 单独指令
- 无掩码、无可变向量长度、无可配置 LMUL

**WASM SIMD 关键指令**:
| 指令 | 功能 | RVV等效 |
|------|------|---------|
| `v128.load` | 加载 128-bit 向量 | `vle32.v` (更宽) |
| `f32.load` | 加载标量 float32 | `flw` |
| `i32x4.splat` | 广播标量到所有 lane | `vfmv.v.f` 或 `vfmacc.vf` 隐含广播 |
| `f32x4.mul` | 逐元素乘法 | `vfmul.vv` |
| `f32x4.add` | 逐元素加法 | `vfadd.vv` |
| -- (无 FMA) | 无融合乘加指令 | `vfmacc.vf` |

**收益分析**:

WASM SIMD 的 GEMM 内循环（处理 1 行 x 4 列 x 1 K 步）:

```wasm
;; WASM SIMD: 1 个 K 步, 1 行, 4 列
(local.set $a_val (f32.load (local.get $a_ptr)))   ;; 加载标量 A[k]
(local.set $a_vec (i32x4.splat (local.get $a_val)));; 广播到 v128
(local.set $b_vec (v128.load (local.get $b_ptr)))  ;; 加载 B[k][0..3]
(local.set $prod  (f32x4.mul (local.get $a_vec) (local.get $b_vec)))  ;; 乘法
(local.set $acc   (f32x4.add (local.get $acc) (local.get $prod)))     ;; 累加
;; 5 条指令, 4 个 FMA (实际为 mul+add, 非融合)
```

**逐项对比** (归一化到 16 个 float32, 即 4 个 WASM SIMD 迭代):

| 指令类别 | WASM SIMD (4 迭代 x 16 列) | RVV (1 迭代, 16 列) | 差异 |
|----------|---------------------------|---------------------|------|
| 标量加载 (A 矩阵) | 4 x `f32.load` = 4 | 2 x `flw` (K-unroll=2) | -- |
| 标量广播 | 4 x `i32x4.splat` = 4 | 0 (vfmacc.vf 隐含) | RVV 省 4 条 |
| 向量加载 (B 矩阵) | 4 x `v128.load` = 4 | 2 x `vle32.v` | -- |
| 乘法 | 4 x `f32x4.mul` = 4 | 0 (含在 FMA 中) | RVV 省 4 条 |
| 加法 | 4 x `f32x4.add` = 4 | 0 (含在 FMA 中) | RVV 省 4 条 |
| FMA | 0 (无 FMA) | 2 x `vfmacc.vf` | -- |
| **合计 (1 K 步, 1 行)** | **20** | **6** | **RVV -70.0%** |
| **合计 (K-pair, 1 行)** | **40** | **6** | **RVV -85.0%** |
| **合计 (K-pair, 2 行)** | **80** | **12** | **RVV -85.0%** |

> 注: WASM SIMD K-pair 按 2 个 K 步直接展开计算 (不利用任何 unroll 优化); RVV K-pair 按实际 unroll=2 计算。

**FMA 效率对比** (归一化到 32 FMA, 即 2 行 x 16 列 x K-pair):

| 平台 | 指令总数 | FMA 数量 | 条指令/FMA | 相对效率 |
|------|---------|---------|-----------|---------|
| WASM SIMD | 80 | 32 | 2.500 | 1.0x (基准) |
| RVV | 12 | 32 | 0.375 | **6.67x** |

RVV 的 FMA 效率是 WASM SIMD 的 **6.67 倍**。

---

## 差异根因分析

### 差异来源拆解

将 RVV 的 85% 指令减少拆解为独立因素：

| 因素 | 说明 | 贡献 (指令减少) |
|------|------|----------------|
| **FMA vs mul+add** | RVV 单条 `vfmacc.vf` = WASM 的 `f32x4.mul` + `f32x4.add` | 每迭代省 2 条 x 4 迭代 = 8 条 |
| **隐含广播** | `vfmacc.vf` 自动广播标量源, 省去 `i32x4.splat` | 每迭代省 1 条 x 4 迭代 = 4 条 |
| **更宽的向量** | VLEN=512 (16 float32) vs 128-bit (4 float32), 减少循环开销 | 循环次数 4x -> 1x |
| **合计** | | 12 条 vs 80 条 = **-85.0%** |

### 核心结论

WASM SIMD 对 RVV 的差距主要来自两个架构层级的限制:

1. **缺少 FMA 指令** -- 这是最关键的差异。WASM SIMD 规范中没有 fused multiply-add, 每次乘累加需要 2 条指令 (`f32x4.mul` + `f32x4.add`)。这不仅增加指令数, 还意味着精度损失 (mul 和 add 之间有一次中间舍入), 性能和数值正确性都受到影响。

2. **缺少隐含广播的 FMA** -- RVV 的 `vfmacc.vf` 将标量广播和乘累加合并在单条指令中完成。WASM SIMD 需要显式 `i32x4.splat` 广播, 再分别执行 mul 和 add, 共 3 条指令完成 RVV 1 条指令的工作。

这些差距在 WASM SIMD 的规范层面存在, 无法通过编译器优化弥补。

---

## RVV扩展指令建议详细说明

### 无新建议

本次分析未发现需要新增的 RVV 扩展指令。WASM SIMD 的所有功能在 RVV 中已有更好或等效的实现:

| WASM SIMD 能力 | RVV 现有覆盖 |
|----------------|-------------|
| `v128.load` | `vle32.v` (更宽, 可配置 VL) |
| `f32.load` + `i32x4.splat` | `vfmv.v.f` / `vfmacc.vf` 隐含广播 |
| `f32x4.mul` + `f32x4.add` | `vfmacc.vf` 单指令融合 |
| 无掩码操作 | `vle32.v` 支持掩码 |
| 固定 4 元素宽度 | 可配置 SEW/LMUL/VL |

RVV 在 GEMM 内核场景下对 WASM SIMD 具有全面的架构优势, 无需额外扩展。

---

## BBV热点加权收益分析

无 BBV profiling 数据，无法计算整体收益。
上表中的"BB内收益"仅反映单个 BB 范围内的指令减少比例。
建议通过 `./tools/profile_to_dfg.sh` 获取 BBV 数据后重新估算整体收益。

---

## 附录

### FMA 指令对比表

| 平台 | FMA 指令 | 操作数宽度 | 广播支持 | 精度 |
|------|---------|-----------|---------|------|
| RVV | `vfmacc.vf` | 16 x float32 (VLEN=512) | 隐含标量广播 | Fused (1 次舍入) |
| WASM SIMD | `f32x4.mul` + `f32x4.add` | 4 x float32 | 需显式 splat | 非融合 (2 次舍入) |

### 向量加载指令对比表

| 平台 | 加载指令 | 宽度 | 掩码 | 可变长度 |
|------|---------|------|------|---------|
| RVV | `vle32.v` | 16 x float32 | 支持 | 支持 (VL) |
| WASM SIMD | `v128.load` | 4 x float32 | 不支持 | 固定 128-bit |

### GEMM 内循环效率对比汇总

| 指标 | WASM SIMD (128-bit) | RVV (VLEN=512) | RVV 优势 |
|------|--------------------|-----------------|---------|
| 每 K-pair 处理元素 (2 行) | 8 x float32 | 32 x float32 | 4x 宽度 |
| 每 K-pair 指令数 (2 行) | 80 | 12 | **-85.0%** |
| 每 K-pair FMA 数 (2 行) | 32 (非融合) | 32 (融合) | 精度更优 |
| 条指令/FMA | 2.500 | 0.375 | 6.67x 效率 |
| 标量广播指令 | 显式 splat (4 条/16 元素) | 隐含 (0 条) | 省去全部广播 |
| FMA 融合 | 否 (mul+add) | 是 (vfmacc) | 精度 + 指令数 |

---

## 结论

本次 WASM SIMD 与 RVV 的 SGEMM 内核对比分析表明:

1. **RVV 对 WASM SIMD 具有压倒性优势**: 在处理等量 GEMM 工作时 (2 行 x 16 列 x K-pair), RVV 仅需 12 条指令 vs WASM SIMD 的 80 条, 指令减少 **85.0%**。

2. **差距核心来自 FMA 缺失**: WASM SIMD 规范没有 fused multiply-add 指令, 每次乘累加需要 2 条分离指令, 且中间结果有一次额外舍入。RVV 的 `vfmacc.vf` 单指令完成融合乘加, 同时隐含标量广播, 架构优势显著。

3. **无新增扩展建议**: WASM SIMD 的所有 GEMM 相关能力在 RVV 中均已具备更优实现, 本次分析不产生新的扩展指令需求。本报告作为参考对比, 展示 RVV 在 Web 端向量计算场景下的已有优势。

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1   |           |        |      |

最终审查结论：待审查
