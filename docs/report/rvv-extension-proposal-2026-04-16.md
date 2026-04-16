# RVV指令扩展建议 — 量化推理与通用矩阵乘实际应用视角

## 概述

本文汇总四份多平台向量实现对比报告（覆盖 llama.cpp 量化推理与 YOLO+ONNX Runtime 推理场景），从实际应用出发提出 RVV 扩展指令建议。

**覆盖算子**:

| 算子 | 应用场景 | 报告来源 |
|------|----------|----------|
| `ggml_quantize_mat_q8_0_4x4` | 权重量化预处理（float32→int8 交织存储） | llama.cpp |
| `ggml_gemm_q4_K_8x4_q8_K` | 4-bit 量化 GEMM（权重×激活矩阵乘） | llama.cpp |
| `ggml_gemv_q4_K_8x8_q8_K` | 4-bit 量化 GEMV（权重×激活向量乘） | llama.cpp |
| MLAS SgemmKernel | 通用 float32 矩阵乘（SGEMM/FGEMM） | YOLO+ONNX Runtime |

**BBV数据状态**: 四份报告均未提供 BBV profiling 数据，下文收益为算子层面的理论预估。

**基准实现**: RVV VLEN=512bit, SEW=32bit

---

## 扩展指令总览

按功能分类汇总所有建议的扩展指令：

| 类别 | 扩展指令 | 功能简述 | 来源平台 | 受益算子 | 算子收益预估 |
|------|----------|----------|----------|----------|-------------|
| 整数点积 | `vdot4a.vv` (signed×signed) | 4个int8×int8→1个int32 点积 | ARM VSDOT | GEMV, GEMM | GEMV算子快3–5倍 |
| 整数点积 | `vdpusd.vx` | int8×int8→int32 逐元素 widening dot | x86 VNNI | GEMM | GEMM算子快60–100% |
| Pairwise乘加 | `vwmaccus.pair.vv` | u8×i8 相邻对乘加→i16 | x86 PMADDUBSW | GEMV | GEMV算子额外快30–50% |
| Pairwise乘加 | `vwmacc.pair.vv` | i16×i16 相邻对乘加→i32 | x86 PMADDWD | GEMV, GEMM | 与vdot4a配合使用 |
| Lane-indexed | `vfmacc.vv_lane` | 从向量选取lane广播后FMA | ARM NEON `fmla .. v.s[lane]` | SGEMM | SGEMM算子K循环快约40% |
| Lane-indexed | `vwmacc.vx_lane` | 从向量选取lane广播后widening MAC | ARM NEON `vdotq_laneq` | GEMM | GEMM算子K循环快约50% |
| 矩阵级操作 | `vmulacc.vv` | 4×4矩阵外积累加 | Power VSX MMA | SGEMM | SGEMM算子快30–100% |
| Pairwise缩减 | `vpairadd.vv` | 相邻元素两两相加 | ARM VPADD | GEMV | GEMV缩减阶段快约66% |
| 数据重排 | `vunzip`/`vzip.vv` | 奇偶元素分离/合并 | AVX vunpck / NEON vtrn | SGEMM | 转置阶段快25–40% |
| 数据重排 | `vshuffle.b.imm` | 立即数控制的字节级shuffle | x86 PSHUFB | GEMV | nibble解压快约50% |
| 数据重排 | `vunpack_epi8_i16.vv` | 单指令nibble提取+符号扩展 | 通用 | GEMM | nibble提取快50% |
| 窄化转换 | `vfncvt_x_f_w_i8` | f32→i8一步窄化（跳过i16中间步骤） | x86 vpmovdb | quantize | quantize算子快约10% |
| 归约融合 | `vfredmax_to_scalar` | 向量最大值归约直接输出标量 | ARM `vmaxvq_f32` | quantize | quantize算子快约10% |

---

## 按应用场景分析

### 场景A: llama.cpp 量化推理

llama.cpp 使用 Q4_K/Q8_K 量化格式进行 LLM 推理。核心计算分为三个层次：

#### A1. GEMV 算子（`ggml_gemv_q4_K_8x8_q8_K`）— 影响最大

**现状**: 当前 RVV 实现因缺乏关键指令，核心 K 循环**退化为纯标量**（correctness-first 策略）。每个 subblock 的 K 循环需要约 290 条标量指令。

**效率差距**:
- vs ARM NEON DOTPROD: 约 **8 倍**（NEON 归一化 ~36 条 vs RVV 标量 ~290 条）
- vs x86 AVX2: 约 **15.5 倍**（AVX2 归一化 ~18 条 vs RVV 标量 ~290 条）

**根本原因**: RVV 缺乏三个关键能力：

1. **signed×signed int8 点积** — ARM 的 `VSDOT` 一条指令完成 4 个 int8×int8→int32，RVV 现有 Zvdot4a8i 扩展仅支持 unsigned 格式，不支持 signed×signed
2. **pairwise 横向乘加** — x86 的 `PMADDUBSW` 可将相邻 u8×i8 乘积对求和，RVV 的 `vwmacc` 只能逐元素操作
3. **pairwise 横向加法** — ARM 的 `VPADD` 将相邻元素两两相加 `[a0+a1, a2+a3, ...]`，RVV 需用 `vrgather` + `vadd` 三条指令模拟

**扩展后预期**:

| 扩展指令组合 | GEMV 算子收益 |
|-------------|--------------|
| 仅 `vdot4a.vv` | K 循环可向量化，算子快 **3–5 倍**（vs 当前标量实现） |
| `vdot4a.vv` + `vwmaccus.pair.vv` | K 循环 + nibble MAC 进一步融合，算子快 **4–6 倍** |
| 全部 P0+P1 指令 | 完整向量化，接近 ARM NEON DOTPROD 效率，算子快 **5–8 倍** |

#### A2. GEMM 算子（`ggml_gemm_q4_K_8x4_q8_K`）— 影响较大

**现状**: RVV 有向量化的 GEMM 实现，使用 `vwmacc_vx` 链（int8×int8→int16），但每个 K 内循环迭代仍需约 30 条指令。

**效率差距**: 核心循环指令数约为 x86 VNNI 的 **5 倍**（30 条 vs 6 条等效）。

**根本原因**:
- RVV 无 int8×int8→int32 单步 widening dot（需两级: int8→int16→int32）
- 每行需单独标量加载 Q8 值，无法像 ARM NEON `vdotq_laneq_s32` 那样从向量中选取 lane

**扩展后预期**:

| 扩展指令组合 | GEMM 算子收益 |
|-------------|--------------|
| `vdpusd.vx`（int8→int32 widening dot） | 消除两级 widening 开销，算子快 **60–100%** |
| `vdpusd.vx` + `vwmacc.vx_lane` | 减少标量加载开销，算子快 **80–130%** |
| 上述 + `vunpack_epi8_i16.vv` | nibble 提取优化，额外快约 **10%** |

#### A3. 量化预处理（`ggml_quantize_mat_q8_0_4x4`）— 影响最小

**现状**: RVV 在此算子上**已是最优**。`vsseg4e32.v` 段存储指令是决定性优势——单条指令完成 ARM NEON 需 256 条逐 lane 操作的交织存储。RVV 总指令数约 41 条/块，ARM NEON 约 416 条，RVV 有 **10 倍**优势。

**扩展后预期**:

| 扩展指令组合 | quantize 算子收益 |
|-------------|-----------------|
| `vfncvt_x_f_w_i8`（f32→i8 一步窄化） | 每块省 4 条指令（41→37），算子快约 **10%** |
| `vfncvt_x_f_w_i8` + `vfredmax_to_scalar` | 共省 8 条指令（41→33），算子快约 **20%** |

**结论**: 此算子的扩展收益有限，RVV 已大幅领先所有其他平台。建议的扩展指令可在其他算子中获得更大收益。

---

### 场景B: YOLO+ONNX Runtime 推理

YOLO 推理中 `MlasSgemmKernel`（SGEMM）是最大热点，约占 **35%** 执行时间。其余为卷积（25%）、BatchNorm（15%）、激活（10%）等。

#### B1. SGEMM 算子（MLAS SgemmKernel）

**现状**: RVV 使用 `vfmacc.vf` 标量广播 FMA 实现。每个 K 步需要 1 条标量 load + 1 条 FMA = 2 条指令。

**效率差距**: ARM NEON 使用 lane-indexed FMA（`fmla v.4s, v.4s, v.s[lane]`），4 个 K 步仅需 1 条 load + 4 条 lane-FMA = 5 条指令，平均 1.25 条/K 步 vs RVV 的 2 条/K 步。

**关键缺失**:
1. **矩阵外积指令** — Power VSX 的 `xvf32gerpp` 一条指令完成 4×4=16 个乘加操作，使用专用 accumulator 寄存器降低寄存器压力
2. **Lane-indexed FMA** — ARM NEON 允许从向量寄存器中选取特定 lane 广播后 FMA，无需单独标量加载

**扩展后预期**:

| 扩展指令组合 | SGEMM 算子收益 | YOLO 整体收益 |
|-------------|--------------|--------------|
| `vmulacc.vv`（矩阵外积） | GEMM 算子快 **30–100%** | YOLO 整体快 **10–35%** |
| `vfmacc.vv_lane`（lane-indexed FMA） | K 循环减少 37.5% 指令 | YOLO 整体快约 **13%** |
| 两者叠加 | GEMM 算子快 **60–150%** | YOLO 整体快 **20–50%** |
| 上述 + `vunzip`/`vzip` | 转置阶段也优化 | 额外快约 **5–10%** |

**注**: `vmulacc.vv` 的收益主要来自专用 accumulator 寄存器降低寄存器压力和更宽的 K 展开，而非原始指令吞吐量差异（归一化后每 K 步指令数相当）。

---

## 各扩展指令简要说明

### 整数点积类

**`vdot4a.vv` (signed×signed int8→int32 4元素点积)**

将向量中每 4 个相邻 signed int8 与对应 int8 相乘求和，结果累加到 int32。等效于 ARM `VSDOT` 指令。

当前 RVV 替代方案: 需 4 次 `vwmul` + 3 次 `vadd` + widening 开销，或退化为标量循环。Zvdot4a8i 扩展类似但不支持 signed×signed，且使用 packed uint32 格式。

算子收益: GEMV 算子快 3–5 倍（使核心 K 循环从纯标量变为向量化）。

**`vdpusd.vx` (int8×int8→int32 逐元素 widening dot)**

将标量 int8 广播后与向量 uint8 逐元素乘加到 int32 累加器，直接跳过 int16 中间步骤。等效于 x86 VNNI `VPDPBUSD` 的逐元素版本。

当前 RVV 替代方案: 需两级 widening MAC 链（`vwmacc_vx` i16 → `vwmacc_vv` i32），约 5 倍指令数。

算子收益: GEMM 算子 K 循环快 60–100%。

### Pairwise 乘加类

**`vwmaccus.pair.vv` (u8×i8 相邻对乘加→i16)**

对向量中相邻的 (unsigned int8 × signed int8) 乘积对求和，结果 widening 累加到 int16。等效于 x86 `PMADDUBSW`。

当前 RVV 替代方案: `vwmaccus.vx` 仅支持标量×向量，无 pairwise（相邻对求和）语义。

算子收益: 配合 `vdot4a.vv`，GEMV 算子额外快 30–50%。

**`vwmacc.pair.vv` (i16×i16 相邻对乘加→i32)**

对向量中相邻的 (int16 × int16) 乘积对求和，结果 widening 累加到 int32。等效于 x86 `PMADDWD`。

算子收益: 在量化推理中将 int16 中间结果进一步缩减到 int32，与 `vwmaccus.pair` 配合使用。

### Lane-indexed 操作类

**`vfmacc.vv_lane` (向量 lane 广播 FMA)**

从 vs1 向量中选取指定 lane（立即数索引），广播后与 vs2 逐元素乘加到 vd。等效于 ARM NEON `fmla v.4s, v.4s, v.s[lane]`。

当前 RVV 替代方案: 逐标量加载 + `vfmacc.vf` 标量广播 FMA，每 K 步需 2 条指令（1 load + 1 FMA）。

改进: 一次向量加载 4 个 A 矩阵元素，逐 lane 广播 FMA，每 K 步平均 1.25 条指令。

算子收益: SGEMM 算子 K 循环减少约 37.5% 指令。

**`vwmacc.vx_lane` (向量 lane 广播 widening MAC)**

从向量中选取指定 lane 的 int8 值，广播后与另一向量逐元素 widening MAC（int8×int8→int16）。等效于 ARM NEON `vdotq_laneq_s32` 的概念。

算子收益: GEMM 算子 K 循环减少约 50% 指令（消除逐行标量加载）。

### 矩阵级操作

**`vmulacc.vv` (4×4 矩阵外积累加)**

一条指令完成 4 元素向量与 4 元素向量的外积累加，产生 4×4=16 个乘加结果。使用专用 accumulator 寄存器（类似 Power VSX MMA）。

当前 RVV 替代方案: 需要 4–16 条 `vfmacc.vf` 完成 4×4 外积，且累加器占用通用向量寄存器。

算子收益: SGEMM 算子快 30–100%（主要来自寄存器压力降低和更宽 K 展开）。

### Pairwise 缩减类

**`vpairadd.vv` (相邻元素两两相加)**

将向量中相邻元素两两相加，输出元素数减半。等效于 ARM `VPADD`。

当前 RVV 替代方案: `vrgather`（偶数索引）+ `vrgather`（奇数索引）+ `vadd` = 3 条指令。

算子收益: GEMV 缩减阶段快约 66%（3 条→1 条）。

### 数据重排类

**`vunzip`/`vzip.vv` (奇偶元素分离/合并)**

`vunzip` 将向量的偶数位和奇数位元素分离到两个目标寄存器；`vzip` 将两个向量的元素交错合并。等效于 AVX `vunpck` / NEON `vtrn`/`vzip`/`vuzp`。

当前 RVV 替代方案: `vrgather` 模拟，需预计算索引向量，约 6–8 条指令。

算子收益: 矩阵转置阶段快 25–40%。

**`vshuffle.b.imm` / `vunpack_epi8_i16.vv`**

字节级 shuffle 和 nibble 提取指令，分别等效于 x86 `PSHUFB` 和通用 nibble unpack。

当前 RVV 替代方案: `vrgather`（需预计算索引）或 `vand` + `vsrl` 两步 nibble 提取。

算子收益: nibble 解压/提取快约 50%。

### 窄化转换类

**`vfncvt_x_f_w_i8` (f32→i8 一步窄化)**

跳过 f32→i16→i8 两步中的 i16 中间步骤，直接 f32→i8。RVV 当前窄化限定每次只能 SEW/2。

当前 RVV 替代方案: `vfncvt.x.f.w`（f32→i16）+ `vnsrl.wi`（i16→i8）= 2 条。

算子收益: quantize 算子快约 10%（省 4 条/块，41→37）。

**`vfredmax_to_scalar` (向量最大值归约直出标量)**

融合 `vfredmax.vs` + `vfmv.f.s` 为一条指令，等效于 ARM `vmaxvq_f32`。

算子收益: quantize 算子快约 10%（省 4 条/块）。

---

## 结论

### 核心发现

1. **量化推理 GEMV 是 RVV 最大的短板**。由于缺乏 signed×signed int8 点积指令，核心 K 循环退化为纯标量，效率差距达 8–15 倍。新增 `vdot4a.vv` 指令可使 GEMV 算子性能提升 3–5 倍，配合 pairwise 操作可达 5–8 倍。

2. **量化推理 GEMM 受限于两级 widening 开销**。RVV 需 int8→int16→int32 两级 widening MAC，而 x86 VNNI 可一步完成 int8→int32。新增 `vdpusd.vx` 可使 GEMM 算子性能提升 60–100%。

3. **通用 SGEMM 的收益来自寄存器管理和加载模式优化**。矩阵外积指令（`vmulacc.vv`）通过专用 accumulator 降低寄存器压力；lane-indexed FMA（`vfmacc.vv_lane`）通过批量加载减少 K 循环中的标量 load 指令。两者叠加可使 SGEMM 快 60–150%，YOLO 整体推理快 20–50%。

4. **量化预处理 RVV 已是最优，扩展收益最小**。`vsseg4e32.v` 段存储是 RVV 的决定性优势，领先 ARM NEON 10 倍。建议的窄化转换改进仅带来约 10–20% 的额外收益。

### 各场景最高价值扩展指令

| 应用场景 | 最高价值指令 | 算子收益 | 收益来源 |
|----------|------------|---------|---------|
| llama.cpp GEMV | `vdot4a.vv` | 快 3–5 倍 | 核心循环从标量→向量化 |
| llama.cpp GEMM | `vdpusd.vx` | 快 60–100% | 消除两级 widening 开销 |
| llama.cpp quantize | `vfncvt_x_f_w_i8` | 快约 10% | 跳过 i16 中间步骤 |
| YOLO SGEMM | `vmulacc.vv` | 快 30–100% | 专用 accumulator + 宽 K 展开 |
| YOLO SGEMM | `vfmacc.vv_lane` | K 循环快约 40% | 批量加载替代逐标量加载 |

---

## 附录：关联报告索引

| 报告 | 路径 |
|------|------|
| quantize_mat_q8_0_4x4 对比分析 | `docs/report/llama.cpp/rvv-gap-analysis-quantize-mat-q8-0-4x4-2026-04-15.md` |
| gemm_q4_K_8x4_q8_K 对比分析 | `docs/report/llama.cpp/rvv-gap-analysis-gemm-q4_K-8x4-q8_K-2026-04-16.md` |
| gemv_q4_K_8x8_q8_K 对比分析 | `docs/report/llama.cpp/rvv-gap-analysis-ggml_gemv_q4_K_8x8_q8_K-2026-04-16.md` |
| YOLO 多平台向量对比分析 | `docs/report/yolo/multi-platform-vector-comparison.md` |
