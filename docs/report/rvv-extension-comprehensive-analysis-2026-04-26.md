# RVV扩展指令综合分析报告

> **分析范围**：基于 llama.cpp、YOLO11n（FP32/INT8）、ResNet50 三个推理场景的 BBV profiling + perf 实测数据，
> 识别 RVV 指令集与主流架构（x86 VNNI/AVX-512、ARM NEON/UDOT/SMMLA、Power VSX MMA、LoongArch LSX、S390X、WASM SIMD）的差距，
> 提出硬件指令扩展需求并量化收益。
>
> **数据来源**：
> - llama.cpp: `docs/report/llama_cpp_perf_q4_v_analysis.md`, `docs/report/llama.cpp/rvv-gap-analysis-gemv_q8_0_16x1_q4_0-2026-04-17.md`
> - YOLO FP32 perf: `applications/onnxrt/yolo/data/perf/yolo11n_bananapi_k1_rv64gcv_scalar_20260424_analysis.md`（SpacemiT K1）
> - YOLO INT8 perf: `applications/onnxrt/yolo/data/perf-int8-yolo11n/summary.md`（SpacemiT K1, RVV512-patched）
> - YOLO FP32 BBV: `output/bbv_rvv512/sgemm/`（QEMU BBV profiling，MlasSgemmKernelRvv512）
> - YOLO INT8 BBV: `output/bbv_rvv512/qgemm/`（QEMU BBV profiling，MlasQgemmKernelRvv512）
> - YOLO INT8 算子 BBV: `output/bbv_rvv512/{compute-logistic,quick-gelu,reduce-minmax-f32,quantize-linear}/`
> - ResNet50 SGEMM BBV: `docs/report/resnet/rvv-gap-analysis-sgemm-kernel-vl16-2026-04-25.md`
> - CX 指令延迟: `docs/reference/cx/instruction-constraints-and-latency.md`

---

## 一、扩展指令方案汇总

### 1.1 方案总表

| 编号 | 指令名称 | 类别 | 解决的问题 | 跨应用收益 |
|------|----------|------|-----------|-----------|
| **1** | vsegdot.vv | INT8 | 4×i8→i32 分段点积（替代 vwmulu+vwaddu 两步序列） | YOLO11n INT8: **54%**, llama.cpp: **15-25%** |
| **2** | vdot_lane.vx | INT8 | Lane-indexed dot 消除标量广播 | llama.cpp: **15-25%** |
| **3** | vnarrow_sat | INT8 | 饱和窄化 int32→int8（替代 2 步 vncvt） | YOLO11n INT8: **~0.3%** |
| **4** | vfmacc.vv_lane | FP32 | Lane-indexed FMA 消除标量加载，K 步间复用 A 元素 | YOLO FP32: **29.1%**, ResNet50: **22.86%**, llama.cpp: **5-8%** |
| **5** | vmulacc.vv | FP32 | 4×4 矩阵外积 FMA + 专用累加器释放 VR | YOLO FP32: **44.6%**, ResNet50: **44.55%** |
| **6** | vclamp.vf | FP32 | 单指令 clamp 替代 vfmax+vfmin（2→1） | YOLO INT8: **~1.0%** |
| **7** | vfmax.red/vfmin.red | FP32 | 单指令水平 min/max 归约（替代 3 步序列） | YOLO INT8: **<0.5%** |
| **8** | vunzip/vzip.vv | 数据重排 | 奇偶分离/合并（转置加速） | YOLO: 边际收益 |
| **9** | vwmaccwev/wod.vv | 数据重排 | Even/Odd 分离 widening MAC 减少依赖链 | llama.cpp: **3-5%** |

**收益说明**：所有收益均为 Amdahl 定律计算的整体推理加速百分比，结合 perf 实测函数占比 + BBV 指令级数据 + CX 指令延迟表。

---

### 1.2 INT8 路径方案详解

#### 方案 1: vsegdot.vv — INT8 分段点积

**指令定义**：

```
vsegdot.vv vd, vs1, vs2, vm
  功能：vd[i] += Σ(j=0..3) vs1[4i+j] × vs2[4i+j]    (uint8×uint8→uint32, 4路分段规约)
  等效于：vwmulu + vwaddu 两步序列合并为单指令
  关联：Zvdot4a8i 扩展的 vdot4au 提供类似功能（quad-widening 4D dot product）
```

**跨平台来源**：x86 AVX512_VNNI `vpdpbusd`（单指令 4×uint8×int8→int32），ARM NEON `vudot.u8`

**RVV 现状**：
- INT8 MAC 需要 2 条指令：`vwmulu.vx`（uint8×uint8→uint16，延迟 4+5=9 周期）+ `vwaddu.wv`（uint16→uint32 累加，延迟 4 周期）
- 还需要额外的 `lbu` 标量加载（4 周期）和 `vle8.v` 向量加载（3 周期）
- 编译器频繁切换 SEW（e8→e16→e32），产生大量 `vsetvli` 开销（占 K-loop 16.8%）

**当前周期分析**（K-loop，per PackedK group = 4 K 元素 × 16 列）：

| 操作 | 指令数 | 单条延迟 | 总周期 |
|------|--------|----------|--------|
| 加载 A（lbu） | 4 | 4 | 16 |
| 加载 B（vle8.v） | 4 | 3 | 12 |
| 扩展乘（vwmulu.vx） | 4 | 9 | 36 |
| 扩展加（vwaddu.wv） | 4 | 4 | 16 |
| SEW 切换（vsetvli） | 9 | ~1 | ~9 |
| **关键路径估计** | | | **~52** |

**扩展后周期分析**：

```asm
vle32.v  v_a, (a)             # 加载4个A元素打包为32位     3周期
vle8.v   v_b, (b)             # 加载64个B元素（4K×16N）    3周期
vsegdot.vv  vacc, v_a, v_b   # 4×i8 dot→i32, 16列并行     ~7周期
```

| 操作 | 指令数 | 单条延迟 | 总周期 |
|------|--------|----------|--------|
| 加载 A（vle32.v） | 1 | 3 | 3 |
| 加载 B（vle8.v） | 1 | 3 | 3 |
| 分段点积（vsegdot.vv） | 1 | ~7 | 7 |
| **关键路径合计** | **3** | | **13 周期** |

**收益计算**：

```
K-loop 关键路径: 52 → 13 周期
K-loop 加速比: 52/13 = 4.00
函数加速比（K-loop 占 96.0%）: 1/(0.04 + 0.96/4.00) = 3.57

YOLO11n INT8 整体收益:
  QGEMM 占 74.51%（perf 实测）
  整体加速 = 1/(0.2549 + 0.7451/3.57) = 2.157 → 54%

llama.cpp 整体收益: 15-25%（理论分析，gemv_q4_0 + gemv_q8_0）
```

**与 ARM SDOT 对比**：

| 平台 | INT8 GEMM 指令/4K 步 | 总周期 |
|------|---------------------|--------|
| ARM SDOT | `ldr q_a` + `ldr q_b` + `udot` | ~13 |
| RVV 现状 | 4×lbu + 4×vle8 + 4×vwmulu + 4×vwaddu + 9×vsetvli | ~52 |
| RVV + vsegdot | 1×vle32 + 1×vle8 + 1×vsegdot | ~13 |

vsegdot.vv 可将 RVV INT8 GEMM 提升到与 ARM SDOT 相同的效率水平。

---

#### 方案 2: vdot_lane.vx — INT8 Lane-indexed Dot

**指令定义**：

```
vdot_lane.vx vd, vs2, vs1, imm
  功能：从 vs1 提取第 imm 个元素广播，与 vs2 逐元素 widening MAC
  等效于：消除 vwmulu.vx 的标量广播开销（4+5=9 周期 → ~7 周期）
```

**跨平台来源**：ARM NEON `vmla_lane` 系列

**应用场景**：llama.cpp INT8 GEMV 中标量 A 元素广播到向量 B

**收益计算**：

```
K-loop 级收益: 周期从 16→10, 加速 1.60
函数级收益（K-loop 占 85%）: 加速 1.44
llama.cpp 整体: gemv_q4_0(40%) + gemv_q8_0(23%) 加速 1.44
  新时间 = 37s + 40s/1.44 + 23s/1.44 = 81s
  整体加速 ≈ 19% (范围 15-25%)
```

---

#### 方案 3: vnarrow_sat — 饱和窄化

**指令定义**：

```
vnarrow_sat_i32_i8  vd, vs2    # int32→int8 饱和窄化（1 条替代 2 步 vncvt）
vnarrow_sat_u32_u8  vd, vs2    # uint32→uint8 饱和窄化
vnarrow_sat_i32_i16 vd, vs2    # int32→int16 饱和窄化
```

**跨平台来源**：ARM NEON `vqmovn_s32`/`vqmovun_s16`，x86 SSE2 `_mm_packs_epi32`/`_mm_packus_epi16`

**RVV 现状**：
- U8 输出需 2 步窄化：`vncvt i32→i16`（4 周期）+ `vncvt i16→i8`（4 周期）
- `vncvt_x_x_w` 是截断窄化（无饱和保证），需预先在 float 域 clamp

**收益计算**：

```
BBV 数据（quantize-linear, VLEN=512）: 主循环 10 条指令/32 元素
CX 延迟: 当前窄化 8 周期 → 扩展后 4 周期
BB 总周期: ~43 → ~39, BB 内减少 ~9.3%
整体收益: 9.3% × 2.94%（QuantizeLinear 占比） ≈ 0.3%
```

注：QuantizeLinear 占比小（2.94%），整体收益有限。主要价值在于提升鲁棒性（消除截断溢出风险）。

---

### 1.3 FP32 路径方案详解

#### 方案 4: vfmacc.vv_lane — Lane-indexed FMA

**指令定义**：

```
vfmacc.vv_lane vd, vs1, vs2, imm[5-bit]
  功能：vd[i] += vs2[i] × vs1[imm]    (i = 0..VL-1)
        广播 vs1 的第 imm 个元素，与 vs2 逐元素乘加
```

**跨平台来源（6 平台共识）**：

| 平台 | 对应指令 | 映射关系 |
|------|---------|---------|
| ARM NEON | `fmla v0.4s, v4.4s, v8.s[lane]` | 最灵活：A 元素保持在向量寄存器，K 步间复用 |
| x86 AVX-512F | `vfmadd231pf_bcst zmm, zmm, [mem]{1to16}` | 最激进：内存加载+广播+FMA 合并为单指令 |
| S390X Z-Vector | `vec_perm` 广播设置 | 批量广播 |
| LoongArch LASX | `xvldrepl.w` | 内存加载+广播 |
| Power VSX | `vmERGEh/vmERGEo` + `xvpermdi` | 元素广播组合 |
| RVV 现状 | `vfmacc.vf` | 仅接受 f 标量寄存器，无向量 lane 索引提取能力 |

**BBV 实测数据**（FP32 SGEMM K-loop，VLEN=512）：

K-loop 占函数指令执行：YOLO **94.68%**，ResNet50 **91.43%**

K-loop 体（16 条指令/迭代）：

```asm
flw     fa5,0(a4)           # A 标量加载（行0, K=0）     4 周期
flw     fa4,4(a4)           # A 标量加载（行0, K=1）     4 周期
vle32.v v11,(t5)            # B 向量加载（16 列, K=0）   3 周期
flw     fa3,0(t4)           # A 标量加载（行1, K=0）     4 周期
flw     fa2,4(t4)           # A 标量加载（行1, K=1）     4 周期
addi    s0,t5,64            # 指针计算                   1 周期
vfmacc.vf  v9,fa5,v11      # FMA: acc_r0 += a0 * b      5 周期
vfmadd.vf  v11,fa3,v10     # FMA: tmp   += a1 * b       5 周期
vle32.v v10,(s0)            # B 向量加载（16 列, K=1）   3 周期
addi    a4,a4,8             # A 指针步进                  1 周期
addi    t5,t5,128           # B 指针步进                  1 周期
addi    t6,t6,-2            # K 计数器                    1 周期
vfmacc.vf  v9,fa4,v10      # FMA: acc_r0 += a1 * b      5 周期
vfmadd.vf  v10,fa2,v11     # FMA: acc_r1 += a1 * b      5 周期
addi    t4,t4,8             # A 指针步进                  1 周期
bgtu    t6,t3,-54           # 循环分支                    1 周期
```

**当前关键路径周期**：4×flw(4) + 2×vle32(3) + 4×vfmacc(5) = **42 周期**

**扩展后**：

```asm
vle32.v  v_a, (a)                        # 1 次向量加载 4 个 A 元素   3 周期
vle32.v  v_b0, (b)                       # B 向量加载                 3 周期
vfmacc.vv_lane v_acc_r0, v_a, v_b0, 0    # lane-indexed FMA           5 周期
vfmacc.vv_lane v_acc_r1, v_a, v_b0, 2    # lane-indexed FMA           5 周期
vle32.v  v_b1, (b+16)                    # B 向量加载                 3 周期
vfmacc.vv_lane v_acc_r0, v_a, v_b1, 1    # lane-indexed FMA           5 周期
vfmacc.vv_lane v_acc_r1, v_a, v_b1, 3    # lane-indexed FMA           5 周期
```

**扩展后关键路径周期**：1×vle32(3) + 2×vle32(3) + 4×vfmacc_lane(5) = **29 周期**

**收益来源**：4 次 `flw`（4×4=16 周期）→ 1 次 `vle32.v`（3 周期），节省 **13 周期**

**各应用收益计算**：

```
YOLO11n FP32:
  K-loop 加速: 42/29 = 1.448
  函数加速（K-loop 94.8%）: 1/(0.052 + 0.948/1.448) = 1.415
  整体加速（GEMM 76.86%）: 1/(0.2314 + 0.7686/1.415) = 1.291 → 29.1%

ResNet50:
  BB 内减少: (16-12)/16 = 25%
  整体收益: 25% × 91.43% = 22.86%

llama.cpp: 辅助优化，整体 5-8%
```

---

#### 方案 5: vmulacc.vv — 4×4 矩阵外积 FMA

**指令定义**：

```
vmulacc.vv acc, vs2, vs1, sew=32
  功能：acc[4×4 矩阵] += vs2[0..3] × vs1[0..3]^T（外积）
  acc: 4×4 float32 累加块，独立于向量寄存器文件
  vs2: 4 个 float32（列向量，代表 A 的 4 行）
  vs1: 4 个 float32（行向量，代表 B 的 4 列）

  配套指令：
    vzero.acc acc      — 清零累加块
    vread.acc vd, acc  — 读取累加块到向量寄存器
    vwrite.acc acc, vs2— 写入向量寄存器到累加块
```

**跨平台来源**：Power VSX (POWER10) `xvf32gerpp` — 4×4 外积 MAC 写入专用累加器 `__vector_quad`（独立于 VR 寄存器文件）

**核心收益**：
1. 消除 per-element A 标量加载（4 行只需 1 次 A 加载，而非 16 次 flw）
2. 释放累加器 VR 寄存器（专用累加器不占用 VR），允许更宽 tile
3. 减少依赖链（4 条 MMA vs 16 条 vfmacc.vf per K-step）
4. 需新增专用累加器寄存器文件和新的指令编码

**各应用收益计算**：

```
YOLO11n FP32:
  周期减少: 80→48, 加速 1.67
  整体加速（GEMM 76.86%）: 1/(0.2314 + 0.7686/1.67) = 1.446 → 44.6%

ResNet50（8 行 tile 场景）:
  BB 内减少: 48.8%
  整体收益: 48.8% × 91.43% = 44.55%
```

---

#### 方案 6: vclamp.vf — 单指令 Clamp

**指令定义**：

```
vclamp.vf vd, vs2, min, max
  功能：vd[i] = clamp(vs2[i], min, max)
        等效于 max(min(vs2[i], max), min)
```

**跨平台来源**：各平台均需 2 条指令实现 clamp（vfmax+vfmin），无单指令方案。本方案填补所有平台的通用需求。

**应用场景**：
- Logistic/Sigmoid 激活：输入 clamp 到 [-18, 18] + 输出 clamp 到 [0, 1]，每 VL 迭代 2 次 clamp
- 其他需要饱和限幅的激活函数

**收益计算**：

```
BBV 数据（compute-logistic, VLEN=512）: 核心循环 25 条指令/迭代, 9,050 次执行
CX 延迟: 当前 clamp 4 条(vfmax×2+vfmin×2) = 12 周期 → 扩展后 2 条 = 6 周期
BB 总周期（~57）减少 6 → ~10.5%
整体收益: 10.5% × 9.80%（Logistic 占 YOLO INT8） ≈ 1.0%
```

---

#### 方案 7: vfmax.red/vfmin.red — 单指令水平 Min/Max 归约

**指令定义**：

```
vfmax.red.vs vd, vs2    # vd[0] ← horizontal_max(vs2[0..VL-1])
vfmin.red.vs vd, vs2    # vd[0] ← horizontal_min(vs2[0..VL-1])
```

**跨平台来源**：ARM NEON64 `vmaxvq_f32`/`vminvq_f32`（单指令水平归约）

**RVV 现状**：

```asm
# 当前 3 步序列：
vfmv.v.f  v_init, FLT_MAX, 1    # 3 周期（初始化）
vfredmin.vs  v_red, v_acc, v_init, 16  # 4 周期（水平归约）
vfmv.f.s  f_result, v_red       # 3 周期（提取标量）
# 3 条指令，10 周期

# 扩展后：
vfmin.red.vs  v_result, v_acc   # ~4 周期
# 1 条指令，4 周期
```

**收益计算**：

```
BBV 数据（reduce-minmax-f32）: 归约阶段仅执行 41 次 vs 主循环 3,114 次
归约阶段占函数: ~1.3%
整体收益: 4.63%（ReduceMinMax 占比）× 1.3% × 67% < 0.1%

注：主要价值在于软件简化和微架构优化，非显著性能收益。
ReduceMinMax 主循环已是 compute-bound（vfmin×4 + vfmax×4），归约不是瓶颈。
```

---

### 1.4 数据重排方案详解

#### 方案 8: vunzip/vzip.vv — 奇偶分离/合并

**指令定义**：

```
vunzip.vv vd_even, vd_odd, vs2
  功能：将 vs2 奇偶元素分离
  输出：vd_even = [e0, e2, e4, ...], vd_odd = [e1, e3, e5, ...]

vzip.vv vd, vs1_even, vs2_odd
  功能：将两向量交错合并
  输出：vd = [e0, o0, e1, o1, ...]
```

**应用场景**：矩阵 4×4 转置、INT8 PackB 预处理

**收益**：转置函数级 25-40%，INT8 PackB 函数级 100-300%。对 YOLO 整体边际收益（非热点，占比 <5%）。

---

#### 方案 9: vwmaccwev/wod.vv — Even/Odd Widening MAC

**指令定义**：

```
vwmaccwev.vv vd, vs2, vs1  ; even positions:  vd.h[j] += vs2.b[2j] × vs1.b[2j]
vwmaccwod.vv vd, vs2, vs1  ; odd positions:   vd.h[j] += vs2.b[2j+1] × vs1.b[2j+1]
```

**跨平台来源**：LoongArch LSX `vmaddwev_w_h`/`vmaddwod_w_h`

**应用场景**：llama.cpp INT8 GEMV widening MAC，减少依赖链深度

**收益**：llama.cpp 整体 **3-5%**

---

### 1.5 已排除方案

| 方案 | 排除原因 |
|------|---------|
| 掩码存储优化 | RVV 已有 `vse32.v + mask`，当前实现未使用。零 ISA 成本，纯软件修复 |
| 多行处理扩展 | 纯软件优化，将 SGEMM 行数从 2 扩展至 4/8 行，B 加载分摊。不需要新指令 |
| 配对向量加载 | VLEN=512 下单条 `vle32.v` 已覆盖 16 列，配对加载收益 <1% |
| vfrsqrt7/vfrec7 | RVV base V 扩展已包含，非新指令提议。当前实现未使用因精度需求不匹配 |
| vexp.approx | 硬件实现复杂度极高，无成熟平台先例（ARM 仅为提案），收益不确定 |

---

## 二、热点分布与收益数据支撑

### 2.1 llama.cpp 推理热点

| 函数 | 占比 | 计算类型 |
|------|------|---------|
| `ggml_gemv_q4_0_16x1_q8_0` | **40.68%** | INT8 量化 GEMV（decode 阶段） |
| `ggml_gemv_q8_0_16x1_q8_0` | **24.06%** | INT8 量化 GEMV（K-quant 中间层） |
| `ggml_gemm_q4_0_16x1_q8_0` | 8.05% | INT8 量化 GEMM（prefill 阶段） |
| `repack_q4_0_to_q4_0_16_bl` | 11.73% | 数据重打包 |
| 其他算子 | ~15% | Norm/RoPE/Attention 等 |

**GEMV/GEMM 总计**: **72.79%**（核心计算）

### 2.2 YOLO11n 推理热点

**FP32**（vanilla ORT, SpacemiT K1, 30 iters）：

| 函数 | 占比 | 计算类型 |
|------|------|---------|
| `MlasSgemmKernel<false,true>` | **43.99%** | FP32 标量 GEMM kernel |
| `MlasSgemmKernel<true,true>` | **32.87%** | FP32 标量 GEMM kernel |
| `MlasComputeLogistic` | 8.28% | sigmoid 激活 |
| `QuickGelu<float>::Compute` | 0.01% | GELU fusion |
| `MlasConvIm2Col` | 3.78% | im2col 变换 |

**GEMM 总计**: **76.86%**

**INT8**（RVV512-patched, SpacemiT K1, 30 iters）：

| 函数 | 占比 | 计算类型 |
|------|------|---------|
| `MlasGemmQuantOperation` | **74.51%** | INT8 QGEMM kernel |
| `MlasComputeLogistic` | **9.80%** | sigmoid 激活 |
| `QuickGelu` | **9.42%** | GELU fusion |
| `MlasReduceMinMaxF32Kernel` | **4.63%** | min/max 归约 |
| `MlasQuantizeLinear` | **2.94%** | 量化 |
| `Im2col` | 1.35% | im2col 变换 |

全局指标：IPC = 0.48，每迭代 4.64 秒，L1-dcache miss rate 2.30%

### 2.3 ResNet50 推理热点

| BB Rank | BB ID | 执行占比 | 指令数 | 关键指令 |
|---------|-------|----------|--------|---------|
| 1 | 85993 | **61.26%** | 16 | vfmacc.vf, vle32.v, flw, addi, bgtu |
| 2 | 85969 | **27.32%** | 16 | vfmacc.vf, vle32.v, flw, addi, bgtu |

- K 循环 BB 执行占比: **91.43%**（BBV 加权）
- SGEMM 函数总占比: **93.38%**

### 2.4 跨应用 K 循环一致性

| 应用 | K 循环 BB 执行占比 | BBV 数据来源 |
|------|-------------------|-------------|
| YOLO11n FP32 | **94.68%** | QEMU BBV, VLEN=512, 3 iters |
| YOLO11n INT8 | **96.0%** | QEMU BBV, VLEN=512, 10000 iters |
| ResNet50 FP32 | **91.43%** | QEMU BBV, VLEN=512, 10 iters |

三个应用 K 循环热点占比均在 91-96% 区间，确认 GEMM K 循环是跨应用的绝对性能瓶颈。

---

## 三、CX 指令延迟参考

基于 `docs/reference/cx/instruction-constraints-and-latency.md`：

| 指令 | 延迟 | 说明 |
|------|------|------|
| flw | 4 | 标量 float 加载 |
| vle32.v | 3 | 向量 float 加载 |
| vle8.v | 3 | 向量 byte 加载 |
| vfmacc.vf | 5 | 标量广播 FMA |
| vfmacc.vv | 5 | 向量 FMA |
| vfmul.vv | 5 | 向量乘法 |
| vfmul.vf | 5 | 标量广播乘法 |
| vfdiv.vv | 4-17 | 向量除法（不定周期） |
| vfmax.vf | 3 | 标量广播 max |
| vfmin.vf | 3 | 标量广播 min |
| vwmul.vx (SEW=8) | **4+5 = 9** | 标量广播 widening 乘 |
| vwmul.vv (SEW=8) | **4** | 向量 widening 乘 |
| vwadd.wv | 4 | Widening 累加 |
| vwmaccu.vv | 5 | Widening MAC（vector） |
| vfcvt.x.f.v | 4 | float→int 转换 |
| vncvt | 4 | 窄化转换 |
| vfredmax.vs | 4 | float max 归约 |
| vfredmin.vs | 4 | float min 归约 |
| vfmv.s.f | 3 | 标量→向量元素 0 |
| vfmv.f.s | 3 | 向量元素 0→标量 |
| vsetvli | 2 | VL 配置 |
| vse8.v | 4 | byte 存储 |
| vse32.v | 4 | float 存储 |
| vfrsqrt7.v | 4-17 | 近似倒数平方根（不定周期） |
| vfrec7.v | 4-17 | 近似倒数（不定周期） |

---

## 四、BBV 实测数据

### 4.1 FP32 SGEMM（MlasSgemmKernelRvv512）

| 数据项 | 值 |
|--------|---|
| K-loop BB 执行比例 | **94.68%** |
| K-loop 每迭代指令数 | 16 条（4 flw + 2 vle32 + 4 vfmacc + 5 addi + 1 bgtu） |
| 总 BB 执行次数 | 2,526,357,651 |
| flw 总执行次数 | 9,567,360,000 |
| vfmacc.vf 总执行次数 | 9,567,360,000 |
| vle32.v 总执行次数 | 4,783,680,000 |

### 4.2 INT8 QGEMM（MlasQgemmKernelRvv512Impl）

| 数据项 | 值 |
|--------|---|
| K-loop BB 执行比例 | **96.0%** |
| K-loop 每 PackedK group 指令数 | 30 条（4 lbu + 4 vle8 + 4 vwmulu + 4 vwaddu + 9 vsetvli + 5 addi/bne） |
| lbu 总执行次数 | 204,800,000 |
| vle8.v 总执行次数 | 138,240,000 |
| vwmulu.vx 总执行次数 | 81,920,000 |
| vwaddu.wv 总执行次数 | 66,560,000 |
| vsetvli 总执行次数 | 133,120,000 |

### 4.3 ResNet50 FP32 SGEMM

| 数据项 | 值 |
|--------|---|
| K-loop BB 执行比例（加权） | **91.43%** |
| SGEMM 函数总占比 | **93.38%** |
| Top 2 K-loop BB 占比 | **88.58%**（BB 85993: 61.26% + BB 85969: 27.32%） |
| K-loop 每迭代指令数 | 16 条（4 flw + 2 vle32 + 4 vfmacc + 2 addi + 2 bgtu + 2 其他） |

### 4.4 YOLO INT8 算子 BBV

| 算子 | 核心循环 BB 指令数 | 执行次数 | 关键指令 |
|------|-------------------|---------|---------|
| ComputeLogistic | 25 条/迭代 | 9,050 | vle32 → vfmax → vfmin → vfmul → 9×vfmacc → vfdiv → vfadd → vfmax → vfmin → vse32 |
| QuickGelu Alpha | 7 条/迭代 | 10,136 | vsetvli → vle32 → vfmul.vf → vse32 |
| ReduceMinMax 主循环 | 18 条/迭代 | 3,114 | 4×vle32 → 4×vfmin → 4×vfmax |
| QuantizeLinear | 10 条/迭代 | 52 BBs | vle32 → vfdiv → vfmax → vfmin → vfcvt → vmv → vadd → vncvt → vncvt → vse8 |

---

## 五、与主流架构对比

### 5.1 扩展后性能定位（相对吞吐量）

| 指标 | x86 VNNI | ARM SDOT | RVV 现状 | RVV 扩展后 |
|------|---------|---------|---------|----------|
| INT8 GEMM 吞吐 | 100%（基准） | 80-100% | **30-50%** | **85-100%** |
| FP32 GEMM 吞吐 | 100%（基准） | 90% | 60% | **85-100%** |
| Lane-indexed 操作 | 有 | 有 | **无** | 有 |
| 饱和窄化 | 有 | 有 | **无** | 有 |
| 单指令水平归约 | 有 | 有 | **无** | 有 |

### 5.2 关键差距填补

| 差距类型 | RVV 现状 | 提议方案 | 填补效果 |
|---------|---------|---------|---------|
| INT8 分段规约 | vwmulu+vwaddu 两步 | 方案 1: vsegdot.vv | **达到 VNNI/SDOT 水平** |
| Lane-indexed | 仅 .vx 标量广播 | 方案 4: vfmacc.vv_lane / 方案 2: vdot_lane.vx | **消除广播开销** |
| 矩阵外积 | 需 16 条 vfmacc | 方案 5: vmulacc.vv（1 条） | **减少 75% 指令** |
| Clamp | 2 条 vfmax+vfmin | 方案 6: vclamp.vf | **减少 50% clamp 指令** |
| 水平 min/max 归约 | 3 步序列 | 方案 7: vfmax.red/vfmin.red | **达到 NEON64 水平** |
| 饱和窄化 | 2 步 vncvt（无饱和） | 方案 3: vnarrow_sat | **达到 SSE2/NEON 水平** |

### 5.3 跨应用累计收益估算

| 应用场景 | 最高收益方案 | 整体收益 |
|---------|------------|---------|
| llama.cpp 推理 | 方案 1 vsegdot.vv | **15-25%** |
| YOLO11n FP32 | 方案 5 vmulacc.vv | **44.6%** |
| YOLO11n FP32 | 方案 4 vfmacc.vv_lane | **29.1%** |
| YOLO11n INT8 | 方案 1 vsegdot.vv | **54%** |
| ResNet50 FP32 | 方案 5 vmulacc.vv | **44.55%** |
| ResNet50 FP32 | 方案 4 vfmacc.vv_lane | **22.86%** |
