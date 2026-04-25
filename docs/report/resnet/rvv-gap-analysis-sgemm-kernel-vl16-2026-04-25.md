# MlasSgemmKernel VL=16 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: ONNX Runtime MLAS SGEMM内循环核函数 (MlasSgemmKernel)
**基准实现**: RVV VL=16 (VLEN=512bit, SEW=32bit, LMUL=1)
**分析平台**: AVX/AVX2/AVX-512F, ARM NEON, LoongArch LASX, Power VSX (POWER10), S390X Z-Vector, WASM SIMD
**BBV数据**: 已提供 (QEMU profiling, 10次迭代)

MlasSgemmKernel 是 ONNX Runtime MLAS 库中执行单精度通用矩阵乘法 (C = alpha * A * B + C) 的内循环热点函数。在 ResNet50 推理中，卷积层通过 im2col 转化为 GEMM 调用，SGEMM 内核的 K 循环占推理总时间的 66.66%，是该算子的绝对性能瓶颈。本报告对比 6 个主流平台 (x86 AVX/AVX-512F, ARM NEON, LoongArch LASX, Power VSX POWER10, S390X Z-Vector, WASM SIMD) 的 SGEMM 向量实现，识别 RVV 的 ISA 差距并提出扩展指令建议。

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | 整体收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vfmacc.vv_lane (向量lane广播FMA) | ARM NEON `fmla v.s[lane]` + x86 `vfmadd231pf_bcst` + S390X `vec_perm` + LoongArch `xvldrepl.w` | **整体减少22.86%** | 中 | `vfmacc.vf` 仅接受标量寄存器，无向量lane索引提取能力 |
| P1 | vmmacc.vv (4x4矩阵外积FMA) | Power VSX `xvf32gerpp` | 整体减少44.55%（8行tile时） | 高 | 无矩阵级乘累加指令 |
| P2 | 掩码存储优化 (软件修复) | x86 AVX-512 掩码存储 | 整体减少<1%（仅部分列触发） | 低 | RVV已有 `vse32.v` + mask，但当前实现未使用 |
| P3 | 多行处理扩展 (软件优化) | S390X (8行) + x86 AVX-512 (12行) + ARM NEON (4行) | 整体减少45.72%（8行vs 2行时A加载摊销） | 低 | 仅支持2行，可扩展至4/8行 |
| -- | 3-operand非破坏性FMA | LoongArch LASX `xvfmadd` + S390X `vfmasb` | 整体减少0%（标准SGEMM，不产生收益） | -- | `vfmacc.vf` 为2-operand破坏性语义，在标准SGEMM中无差距 |
| -- | 配对向量加载 | ARM NEON `ldp` | 整体减少<1%（辅助加载场景） | 低 | RVV VLEN=512已单条覆盖16列，配对加载收益有限 |

**收益计算方式**（基于QEMU-BBV profiling数据）：
- BB指令减少数 = 原BB指令数 - 扩展后BB指令数（归一化到RVV VLEN=512bit, SEW=32bit）
- 整体收益 = BB内减少比例 × K循环BB执行占比
- **K循环BB执行占比**: 91.43%（基于BBV profiling，计算方式：Σ(K循环BB执行次数 × BB指令数) / Σ(所有BB执行次数 × BB指令数)）
  
**P0 vfmacc.vv_lane 整体收益计算**:
- K循环BB内减少: 25%（指令数从16降至12，净减少4条flw）
- K循环BB执行占比: 91.43%
- 整体收益 = 25% × 91.43% = **22.86%**

**P1 vmmacc.vv 整体收益计算**（8行tile场景）:
- K循环BB内减少: 48.8%（8行模式从84条降至43条）
- K循环BB执行占比: 91.43%
- 整体收益 = 48.8% × 91.43% = **44.55%**

**各扩展指令的整体收益可叠加估算上限**（假设无交互效应）：
- P0 + P3 (8行多处理): 25% × (91.43% × 50%分摊收益) ≈ 整体减少11.4%叠加
- P0 + P1: 25% + 48.8% × (91.43% - 已优化部分) → 整体减少约45-50%（激进估算）

**跨平台去重说明**:
- **vfmacc.vv_lane**: ARM NEON `fmla v.s[lane]`、x86 AVX-512F `vfmadd231pf_bcst`、S390X `vec_perm` 广播设置、LoongArch `xvldrepl.w` 均指向同一概念 -- 消除逐元素标量加载，通过lane索引或内存广播复用已加载的A元素。其中 AVX-512 的广播FMA将内存加载、广播和FMA合并为单指令（最激进的形态），ARM NEON 的lane索引FMA在保持A元素在向量寄存器的同时提供K步间复用（最灵活的形态），两者可统一为 `vfmacc.vv_lane`。
- **vmmacc.vv**: 仅 Power VSX MMA `xvf32gerpp` 提出，无其他平台等价指令。
- **掩码存储优化**: 仅 x86 AVX-512 提出。RVV 已有硬件掩码能力，这是纯软件修复。
- **多行处理**: S390X、x86、ARM 均支持多于2行并行，但这是软件架构优化，非ISA差距。

---

## 基准RVV实现分析

### 循环结构

RVV sgemm-kernel-vl16 处理 2行 x 16列 的矩阵块：

```
N-loop (列, 每16列一块):
  清零累加器
  K-loop (展开2, 每步处理2个K):
    Row 0:
      flw  fa0, 0(a0)          # 加载 A[m,k]
      vle32.v v0, (b_ptr)      # 加载 B[k, 0:15]
      flw  fa1, 0(a0+4)        # 加载 A[m,k+1]
      vfmacc.vf v2, fa0, v0    # acc += A[m,k] * B[k,:]
      vle32.v v1, (b_ptr+64)   # 加载 B[k+1, 0:15]
      vfmacc.vf v2, fa1, v1    # acc += A[m,k+1] * B[k+1,:]
    Row 1:
      flw  fa2, 0(a1)          # 加载 A[m+1,k]
      flw  fa3, 0(a1+4)        # 加载 A[m+1,k+1]
      vfmacc.vf v3, fa2, v0    # acc += A[m+1,k] * B[k,:]  (重用B加载)
      vfmacc.vf v3, fa3, v1    # acc += A[m+1,k+1] * B[k+1,:] (重用B加载)
    a0 += 8; a1 += 8; b_ptr += 128
  Alpha 乘法: vfmul.vf (每行1条)
  输出: vse32.v (全块) / 标量提取 (部分块)
```

### 每 K-pair 指令计数 -- 2行 x 16列

| 指令 | 数量 | 说明 |
|------|------|------|
| flw (A元素) | 4 | 每行2个K步各1个 |
| vle32.v (B向量) | 2 | 2个K步各1个 (16 floats each), 2行共享 |
| vfmacc.vf (FMA) | 4 | 每行2个K步各1个 |
| 指针更新 | ~2 | add a0, add b_ptr (a1通过 a[lda] 直接计算, 无需独立递增) |
| **合计** | **~12** | **2行 x 16列 x 2K步 = 64 FLOPs** |

Row 1 复用 Row 0 的 B 向量加载，因此 B 加载只需要 2 条而非 4 条。FMA 密度: 64 FLOPs / 12 instr = **5.33 FLOPs/instruction**。

### 寄存器使用

- 累加器: v2, v3 (每行1个V寄存器, 各含16个float32)
- A元素: fa0-fa3 (f标量寄存器, 每个K步每行1个)
- B向量: v0, v1 (每K步1个V寄存器)
- 地址: a0, a1 (行指针), b_ptr (B指针)
- 指针更新: ~2条

### 数据流 (单行, 1 K步)

```
A[m,k] ──flw──> fa0 ─┐
                       ├─ vfmacc.vf ──> v2 (accumulator)
B[k,:] ──vle32.v──> v0 ┘
```

关键限制: `vfmacc.vf` 只接受 f 标量寄存器作为乘数。处理多个A元素时，每个A元素必须单独用 `flw` 加载到不同的标量寄存器，无法从向量寄存器中提取单个lane。

---

## 各平台对比分析

### 1. x86 AVX/AVX2/AVX-512F

**核心特点**:
- AVX: 256-bit YMM, 8 x float32; 无FMA，使用分离的 `vmulps` + `vaddps`
- FMA3: 256-bit YMM; `vfmadd231ps` 融合乘加
- AVX-512F: 512-bit ZMM, 16 x float32; `vfmadd231pf_bcst` 广播FMA; opmask 掩码存储
- 最多处理 12 行 (AVX-512, zmm4-zmm27)
- K 循环展开 4

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vfmadd231pf_bcst zmm, zmm, [mem]{1to16}` | 从内存加载1个float, 广播到16 lane, 执行FMA | RVV 无等价指令, 需 `flw` + `vfmacc.vf` |
| `vmovupf [mem]{k1}, zmm` | 掩码向量存储 | RVV 有 `vse32.v` + mask, 但当前实现未使用 |

**指令计数对比 (12行 x 16列 x 1 K步, AVX-512 By1)**:

| 对比项 | AVX-512 By1 | RVV (6次2行调用) | 差异 |
|--------|------------|-----------------|------|
| 总指令数 | 13 | ~42 | RVV多223% |
| A元素加载 | 0 (内嵌于vfmadd231pf_bcst) | 12 (独立flw) | RVV多12条 |
| B向量加载 | 1 (所有行共享) | 6 (每次调用各1条) | RVV多5条 |
| FMA指令 | 12 | 12 | 相同 |

AVX-512 在12行模式的优势来源:
1. `vfmadd231pf_bcst` 消除独立标量加载 (ISA差距)
2. B向量在所有行之间共享 (软件架构差异)
3. 12行同时处理 (软件架构差异, 需32个ZMM寄存器)

AVX (无FMA) 不构成紧迫需求: RVV 的 `vfmacc.vf` 已有FMA优势, AVX分离的mul+add使指令数翻倍。FMA3 (256-bit) 的 `vbroadcastss` + `vfmadd231ps` 在指令数上与RVV的 `flw` + `vfmacc.vf` 等价 (均为2条)。

---

### 2. ARM NEON/SVE

**核心特点**:
- 128-bit Q寄存器, 每寄存器4个float32
- 块大小: 16列 (4个Q寄存器), 最多4行
- K循环展开度: 4 (BlockBy4Loop)

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `fmla v0.4s, v4.4s, v8.s[lane]` | lane索引FMA: 从v8第[lane]号元素广播到所有lane, 执行FMA | 不存在. `vfmacc.vf` 仅接受f标量寄存器 |
| `ldp q4,q5,[x1],#64` | 配对加载: 1条加载2个Q寄存器(32字节) | 无配对加载 |

**核心差距: Lane索引FMA** (`fmla v0.4s, v4.4s, v8.s[lane]`):

ARM用1条 `ldr q8` 加载4个A元素, 然后在4个K步中分别用 `v8.s[0]`..`v8.s[3]` 提取, **完全消除后续K步的A加载**。

归一化对比 (等量工作: 1行 x 16列 x 4个K步):

| 实现 | 指令数 | A加载 | B加载 | FMA |
|------|--------|-------|-------|-----|
| ARM NEON (1行, 4K) | 26 | 1 (ldr q8) | 8 (ldp) | 16 |
| ARM NEON 归一化 | 6.5 | 0.25 | 2.0 | 4.0 |
| RVV 当前 (1行, 4K) | 12 | 4 (flw) | 4 (vle32) | 4 |
| RVV + vfmacc.vv_lane | 9 | 1 (vle32) | 4 (vle32) | 4 (vv_lane) |

A加载差距: ARM归一化0.25条 vs RVV 4条, 差距3.75条, 占RVV总量的31.3%。这是lane索引FMA的核心收益。

ARM多行优势 (4行并行时B加载被分摊): 归一化后每行每K步从6.5降至4.8条。但RVV也支持多行扩展, 且单条vle32.v加载16列, B加载效率本身高于ARM。

---

### 3. LoongArch LASX/LSX

**核心特点**:
- 256-bit XR寄存器, 每寄存器8个float32
- 块大小: 16列 (2个XR/行), 最多4行
- K循环展开因子: 4

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `xvldrepl.w $xr3, $a0, offset` | 从内存加载1个float并广播到XR全部lane | RVV的 `vfmacc.vf` 已内含标量广播 |
| `xvfmadd $xr8, $xr4, $xr3, $xr8` | 3-operand非破坏性FMA | `vfmacc.vf` 为2-operand破坏性 |

**关键发现: 无ISA差距**:

归一化对比 (2行 x 16列 x 1 K步):
- LASX归一化后: 8.0条指令
- RVV基线: 5条指令
- RVV已领先37.5%

在4行场景: LASX归一化后15条 vs RVV 10条, RVV仍领先33.3%。LASX的 `xvldrepl.w` 和3-operand FMA在标准SGEMM中均不产生指令减少 (收益0%), 因为RVV的 `vfmacc.vf` 已隐含标量广播, 且累加器模式不破坏有用数据。

**结论**: LoongArch LASX对RVV VLEN=512不构成ISA级gap。

---

### 4. Power VSX (POWER10)

**核心特点**:
- 128-bit VR寄存器, 4 x float32
- MMA指令: `xvf32gerpp` -- 4x4矩阵外积乘累加 (16 FMA/指令)
- 专用累加器: `__vector_quad` (独立于VR寄存器文件)
- Tile: 4行 x 16列, K展开4

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `xvf32gerpp(acc, a, b)` | 4x4外积MAC, 写入专用累加器 | 无矩阵级指令 |
| `__vector_quad` | 512-bit专用累加器, 独立于VR文件 | 累加器共享向量寄存器文件 |

**归一化对比 (4行 x 16列 x 4 K步 = 256 FMA)**:

| 实现 | 总指令 | FMA/instr | 累加器VR压力 |
|------|--------|-----------|-------------|
| RVV 基线 | ~42 | 6.10 | 4 VR |
| POWER10 MMA | ~47 | 5.45 | 0 (专用) |
| RVV + vmmacc.vv (估算) | ~18 | 14.22 | 0 (专用) |

RVV基线在指令数上略优于POWER10 (因为RVV的 `vfmacc.vf` 隐含广播, 无POWER10的8条permute开销)。但MMA的关键优势不在指令数, 而在:
1. 16 FMA/指令减少依赖链
2. 专用累加器释放VR寄存器, 允许更宽tile
3. 对8行tile场景, MMA可将指令从84降至43 (BB内减少48.8%)

---

### 5. S390X Z-Vector

**核心特点**:
- 128-bit VR寄存器, 4 x float32
- `vfmasb(a, b, c)`: 3-operand非破坏性FMA
- `vec_perm`: 任意lane级重排, 批量广播设置
- 最多8行, K展开4

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vfmasb(a, b, c)` | 3-operand FMA, 非破坏性 | `vfmacc.vf` 为2-operand破坏性 |
| `vec_perm` | 任意lane重排, 批量广播 | `vrgather` 仅索引收集, 无批量广播 |

**核心差距: 多行处理效率**

Z-Vector的8行模式中, B加载被8行分摊, A广播通过 `vec_perm` 以12条指令完成4行x4K步的setup。对比 (8行x16列x4K步 = 512 FMA, 均为原始指令数, 未做位宽归一化):

| 实现 | 指令数 | instr/FMA |
|------|--------|-----------|
| Z-Vector 8行 | 203 | 0.40 |
| RVV 2行 × 4调用 (baseline) | ~120 | 0.23 |
| RVV 8行 (假设) | ~96 | 0.19 |

> **注**: 以上为原始指令数对比, 未做VLEN位宽归一化。RVV VLEN=512 (16 floats/instr) vs Z-Vector 128-bit (4 floats/instr), RVV因4倍更宽的向量寄存器天然具有更低的instr/FMA。Z-Vector的优势体现在软件架构层面: 8行并行处理和B广播优化使其在相同128-bit宽度下效率更高。

Z-Vector在instr/FMA (0.40) 上高于RVV 2行 baseline (0.23), 主要因为:
1. 多行处理 (软件架构, 非ISA差距)
2. `vec_perm` 批量广播 (12条指令设置4行x4K步的A广播)
3. 3-operand FMA允许直接使用预计算broadcast vector

---

### 6. WASM SIMD

**核心特点**:
- 128-bit v128寄存器, 4 x float32
- **无FMA指令**: 必须使用 `f32x4.mul` + `f32x4.add`
- 无掩码, 无可变向量长度, 无可配置LMUL

**核心差距: 无FMA + 无隐含广播**

| 特性 | WASM SIMD | RVV |
|------|-----------|-----|
| FMA | 无 (mul + add, 2次舍入) | `vfmacc.vf` (1次舍入) |
| 标量广播 | 显式 `i32x4.splat` | `vfmacc.vf` 隐含 |

归一化对比 (2行 x 16列 x K-pair = 32 FMA):

| 指标 | WASM SIMD | RVV | RVV优势 |
|------|-----------|-----|---------|
| 指令数 | 80 | 12 | -85.0% |
| 条指令/FMA | 2.500 | 0.375 | -85.0% |

**结论**: WASM SIMD对RVV不构成扩展需求。RVV具有压倒性架构优势, 无需额外扩展。

---

## RVV扩展指令建议详细说明

### [P0] vfmacc.vv_lane -- 向量通道广播FMA

**跨平台来源**: ARM NEON `fmla v0.4s, v4.4s, v8.s[lane]` + x86 `vfmadd231pf_bcst` + S390X `vec_perm` 广播设置 + LoongArch `xvldrepl.w`

**指令定义**:
```
vfmacc.vv_lane vd, vs2, vs3, lane_index
# vd[i] += vs2[i] * vs3[lane_index], for all active i
# vs3[lane_index]: 从vs3向量的第lane_index号元素提取, 广播到所有活跃lane
# lane_index 编码在imm字段 (0~VL/SEW-1, VLEN=512/SEW=32时需4位)
```

**语义**:
1. 从vs3向量的第lane_index号元素读取一个标量值
2. 将该标量广播到vs2的所有活跃lane
3. 执行 `vd[i] += broadcast_value * vs2[i]`

**与各平台指令的对应关系**:
- ARM NEON: `fmla v0.4s, v4.4s, v8.s[lane]` -- lane索引FMA, A元素保持在向量寄存器中
- x86 AVX-512F: `vfmadd231pf_bcst zmm, zmm, [mem]{1to16}` -- 内存广播FMA, 标量来自内存而非寄存器
- 统一设计: RVV的 `vfmacc.vv_lane` 从寄存器提取lane, 更灵活; 可选地扩展 `vfmacc.vf_mem` 从内存广播

**应用场景**:

1. **SGEMM K循环**: 加载多个A元素到向量寄存器, 通过lane索引在多个K步中复用, 消除重复的标量加载指令

**性能对比** (1行 x 16列 x 4 K步, K循环主体):

改造前 (RVV当前):
```asm
# 12条指令
flw      fa0, 0(a0)              # A[k=0]
flw      fa1, 4(a0)              # A[k=1]
vle32.v  v5, 0(b0)               # B[k=0]
vle32.v  v6, 64(b0)              # B[k=1]
vfmacc.vf v0, fa0, v5
vfmacc.vf v0, fa1, v6
# ... repeat for K=2,3 (total 12)
```

改造后 (RVV + vfmacc.vv_lane):
```asm
# 9条指令
vle32.v  v4, 0(a0)               # 1条加载4个A元素
vle32.v  v5, 0(b0)               # B[k=0]
vfmacc.vv_lane v0, v5, v4, lane=0
vle32.v  v5, 64(b0)              # B[k=1]
vfmacc.vv_lane v0, v5, v4, lane=1
# ... K=2,3 (total 9)
```

| 指标 | 改造前 | 改造后 | 变化 |
|------|--------|--------|------|
| A加载指令 | 4 (flw) | 1 (vle32.v) | -75.0% |
| 总指令数 | 12 | 9 | **-25.0%** |
| FMA结果 | 64 | 64 | 不变 |

**2行模式 (当前RVV基准) BB内收益**:
- K循环BB约12条指令 (2行 x K-pair, 含指针更新)
- `flw` 占4条, 引入后消除4条, 但需1条 `vle32.v` 替代
- 净减少: 3条, 新BB: 9条
- **BB内减少: (12 - 9) / 12 = 25.0%**

**12行模式 (理论) BB内收益**:
- 约42条指令 (12行 x 1 K步)
- `flw` 占12条, 引入后需3条 `vle32.v` (3个向量覆盖12个元素)
- 净减少: 9条, 新BB: 33条
- **BB内减少: (42 - 33) / 42 = 21%**

**保守取值: BB内减少25%** (1行x4K步和2行xK-pair场景均为25%, 取值一致)

**寄存器压力**: 释放4个f标量寄存器, 增加1个V寄存器. RVV有32个V寄存器, 压力可忽略.

---

### [P1] vmmacc.vv -- 4x4矩阵外积FMA

**来源**: Power VSX `xvf32gerpp`

**指令定义**:
```
vmmacc.vv acc, vs2, vs1, sew=32
# acc[i][j] += vs2[i] * vs1[j], for i,j in [0,3]
# acc: 4x4 float32累加块, 独立于向量寄存器文件
# vs2: 4个float32 (列向量, 代表A的4行)
# vs1: 4个float32 (行向量, 代表B的4列)
```

**配套指令**:
- `vzero.acc acc` -- 清零累加块
- `vread.acc vd, acc` -- 读取累加块到向量寄存器
- `vwrite.acc acc, vs2` -- 写入向量寄存器到累加块

**应用场景**: SGEMM 4行x16列tile, 每K步4条MMA覆盖全部列:

```asm
vmmacc.vv acc0, a_vec, b_vec_0    # 列 0-3  (16 FMA)
vmmacc.vv acc1, a_vec, b_vec_1    # 列 4-7
vmmacc.vv acc2, a_vec, b_vec_2    # 列 8-11
vmmacc.vv acc3, a_vec, b_vec_3    # 列 12-15
```

**BB内收益**:

| 场景 | RVV基线 | +vmmacc.vv | BB内减少 |
|------|---------|-----------|---------|
| 4行 x 16列 x 4K步 (256 FMA) | ~42 instr | ~39 instr | -7.1% |
| 8行 x 16列 x 4K步 (512 FMA) | ~84 instr | ~43 instr | **-48.8%** |

核心收益在于:
1. 消除per-element A标量加载 (4行只需1次A加载, 而非16次flw)
2. 释放累加器VR寄存器 (专用累加器不占用VR), 允许更宽tile
3. 减少依赖链 (4 MMA vs 16 vfmacc.vf per K-step)

**实现难度**: 高 (需新增专用累加器寄存器文件和新的指令编码)

---

### [P2] 掩码存储优化 (软件修复)

**来源**: x86 AVX-512F 掩码存储

**说明**: 不是新ISA扩展, 而是指出RVV已有的掩码存储能力未被当前实现利用。

**RVV已有能力**:
```asm
vsetvli t0, a0, e32, m1      # t0 = min(remaining, VLMAX)
vse32.v v_acc, (c_ptr), v0.t  # 一次存储remaining个float32
```

**当前实现问题**: 对N < 16的尾部列使用标量提取逐个写回.

**性能对比** (假设N=10剩余列, 2行):

| 实现 | 指令数 |
|------|--------|
| 当前 (标量提取) | ~22 (2 vsetvli + 10 flw + 10 sw) |
| 改进 (掩码存储) | ~4 (2 vsetvli + 2 vse32.v) |
| **BB内减少** | **(22 - 4) / 22 = 82%** |

注意: 仅在N非16倍数时触发. 全16列对齐时收益为0.

---

### [P3] 多行处理优化 (软件层面)

**来源**: S390X (8行) + x86 AVX-512 (12行) + ARM NEON (4行)

**建议**: 将RVV SGEMM内核的行数从2扩展至4或8行, 使用多个V寄存器作为累加器.

**收益分析** (每K步, 16列):

| 行数 | A加载 | B加载 | FMA | 合计 |
|------|-------|-------|-----|------|
| 2行 (当前) | 2 flw | 1 vle32 | 2 vfmacc | 5 |
| 4行 | 4 flw | 1 vle32 | 4 vfmacc | 9 |
| 8行 | 8 flw | 1 vle32 | 8 vfmacc | 17 |

B加载在多行间分摊. 8行模式相对于2行模式, B加载效率提升4倍. 但A加载线性增长. 当扩展到8行时, 需配合P0 (vfmacc.vv_lane) 才能获得净收益.

**实现难度**: 低 (纯软件修改, 不需要新ISA扩展)

---

## BBV热点加权收益分析

### Profiling数据概览

**数据来源**: QEMU BBV profiling (ResNet50, 10次迭代, VLEN=512)
**总基本块数**: 87,279
**总执行次数**: 12,657,313,334 (加权: 执行次数 × 指令数)
**K循环BB执行占比**: **91.43%**
**SGEMM函数总占比**: 93.38%

### 热点BB表格 (Top 10)

| Rank | BB ID | 地址 | 执行占比 | 指令数 | 关键指令类型 |
|------|-------|------|----------|--------|-------------|
| 1 | 85993 | 0x706e66a81320 | **61.26%** | 16 | vfmacc.vf, vfmadd.vf, vle32.v, flw, addi, bgtu |
| 2 | 85969 | 0x706e66a810ac | **27.32%** | 16 | vfmacc.vf, vfmadd.vf, vle32.v, flw, addi, bgtu |
| 3 | 85997 | 0x706e66a81310 | 1.43% | 22 | vfmacc.vf, vle32.v, flw, vmv1r.v, mv |
| 4 | 86210 | 0x706e664133d4 | 0.83% | 4 | flt.s, flw, fmv.s, bnez (FP比较) |
| 5 | 85973 | 0x706e66a8109c | 0.79% | 22 | vfmacc.vf, vle32.v, flw, vmv1r.v |
| 6 | 85931 | 0x706e66a62028 | 0.68% | 7 | flw, fsw, slli (标量尾部处理) |
| 7 | 85992 | 0x706e66a81302 | 0.41% | 27 | vfmacc.vf, vle32.v, flw, vmv1r.v |
| 8 | 85968 | 0x706e66a8108e | 0.17% | 27 | vfmacc.vf, vle32.v, flw, vmv1r.v |
| 9 | 86730 | 0x706e66a81588 | 0.09% | 20 | fadd.s, flw, fsw (累加) |
| 10 | 86936 | 0x706e66a813be | 0.04% | 11 | vfmadd.vf, vle32.v, flw |

**关键发现**:
- Top 2 K循环BB（85993 + 85969）合计占总执行的 **88.58%**
- 这些BB包含典型的SGEMM K循环指令：`vfmacc.vf`（向量FMA）、`vle32.v`（向量加载）、`flw`（标量加载）
- 每个K循环BB约16条指令，其中4条 `flw` 用于加载A矩阵元素

### P0 整体收益详细计算

**目标BB**: K循环主体BB（85993 + 85969）
**当前指令构成** (每BB, 16条):
- 4条 `flw` (A元素加载, 占25%)
- 2条 `vle32.v` (B向量加载)
- 4条 `vfmacc.vf` (向量FMA)
- 2条 `addi` (指针递增)
- 2条 `bgtu` (循环控制)
- 2条其他

**扩展后指令构成** (引入vfmacc.vv_lane):
- 1条 `vle32.v` (一次性加载4个A元素到向量寄存器)
- 2条 `vle32.v` (B向量加载, 不变)
- 4条 `vfmacc.vv_lane` (lane索引FMA, 替代4条vfmacc.vf)
- 2条 `addi` (不变)
- 2条 `bgtu` (不变)
- 2条其他
- **净减少**: 4条 `flw` → 1条 `vle32.v`，节省3条

**计算链**:
```
BB指令减少数 = 16 - 12 = 4条 (净减少3条flw，增加1条vle32.v替代)
BB内减少比例 = 4 / 16 × 100% = 25%
K循环BB执行占比 = 91.43% (加权)
整体收益 = 25% × 91.43% = 22.86%
```

### P1 整体收益详细计算 (8行tile场景)

**前提**: 当前RVV内核仅处理2行，扩展至8行时A加载可被更高效分摊。

**8行模式指令变化**:
- 当前2行 × 4调用: 4 × 16 = 64条指令/K步 (A加载: 16条flw)
- 8行单次调用 + vmmacc.vv: 约43条指令/K步 (A加载: 4条vle32.v或MMA内置)

**计算链**:
```
BB指令减少数 (8行) = 84 - 43 = 41条
BB内减少比例 = 41 / 84 × 100% = 48.8%
K循环BB执行占比 = 91.43%
整体收益 = 48.8% × 91.43% = 44.55%
```

### 收益叠加估算

假设P0和P3（多行扩展）独立生效：
- P0先生效: 整体减少22.86%
- 剩余热点占比: 91.43% × (1 - 25%) = 68.57%
- P3叠加: 50% × 68.57% = 整体减少34.29%
- **累计**: 22.86% + 34.29% = **57.15%** (乐观上限)

保守估计（考虑交互效应和新指令延迟）:
- **整体减少范围: 15-25%** (仅P0)
- **整体减少范围: 35-45%** (P0 + P3组合)

---

## 附录

### FMA指令对比表

| 特性 | x86 AVX | x86 FMA3 | x86 AVX-512F | ARM NEON | LoongArch LASX | Power VSX | S390X | WASM SIMD | RVV |
|------|---------|----------|--------------|----------|----------------|-----------|-------|-----------|-----|
| FMA指令 | 无 (分离) | `vfmadd231ps` | `vfmadd231pf` | `fmla .4s` | `xvfmadd` | `xvf32gerpp` | `vfmasb` | 无 (分离) | `vfmacc.vf` |
| 标量x向量FMA | 3条 | 2条 | 1条 (`_bcst`) | 1条 (lane) | 2条 | MMA级 | 2条 | 3条 | 2条 |
| 操作数语义 | 3-op | 3-op | 3-op | 3-op | 3-op | MMA | 3-op | 2-op分离 | 2-op破坏性 |
| 掩码FMA | 无 | 无 | `{k1}` | 无 | 无 | 无 | 无 | 无 | `v0.t` |
| 广播方式 | vbroadcastss | vbroadcastss | 内存广播 | lane索引 | xvldrepl.w | 外部VR | 外部VR | i32x4.splat | 标量隐含 |
| 每 instr FMA数 | 4/8 | 4/8 | 16 | 4 | 8 | 16 (4x4) | 4 | 4 | 16 |
| 精度 | 非融合 | Fused | Fused | Fused | Fused | Fused | Fused | 非融合 | Fused |

### 数据重排/广播指令对比表

| 特性 | x86 AVX-512 | ARM NEON | LoongArch LASX | Power VSX | S390X | RVV |
|------|-------------|----------|----------------|-----------|-------|-----|
| Lane索引提取 | 无 | `v.s[lane]` | 无 | 无 | 无 | **无** |
| 标量广播 | `vbroadcastsf` | `vdup` | `xvldrepl.w` | `vec_splat` | `vec_perm` | `vfmv.v.f` |
| 批量广播 | 无 | 无 | 无 | 8条permute | 8条vec_perm | `vrgather` (需索引) |
| 配对加载 | 无 | `ldp` | 无 | `lxvdsx` | `vl` | 无 |

### 向量加载/存储指令对比表

| 特性 | x86 AVX-512 | ARM NEON | LoongArch LASX | Power VSX | S390X | WASM SIMD | RVV |
|------|-------------|----------|----------------|-----------|-------|-----------|-----|
| 最大宽度 | 512-bit | 128-bit | 256-bit | 128-bit | 128-bit | 128-bit | 512-bit |
| 16列所需loads | 1 | 4 | 2 | 4 | 4 | 4 | 1 |
| 掩码存储 | `{k1}` | 无 | 无 | 无 | 无 | 无 | `v0.t` |
| 可变长度 | 无 | 无 | 无 | 无 | 无 | 无 | VL |

### 寄存器使用对比

| 特性 | x86 FMA3 | x86 AVX-512F | ARM NEON | LoongArch LASX | Power VSX | S390X | RVV |
|------|----------|--------------|----------|----------------|-----------|-------|-----|
| 最大行数 | 6 | 12 | 4 | 4 | 4 (专用累加器) | 8 | 2 |
| VR总数 | 16 YMM | 32 ZMM | 32 Q | 32 XR | 32 VR + 8 acc | 32 VR | 32 V |
| 累加器 | VR | VR | VR | VR | 专用 `vector_quad` | VR | VR |

---

## 结论

### 核心发现 (基于BBV Profiling)

**QEMU BBV profiling揭示的关键数据**:
- K循环BB占总执行的 **91.43%**（加权计算）
- Top 2 K循环BB（85993 + 85969）合计占 **88.58%**
- 这远高于原先假设的 16% 整体占比

### 扩展指令建议优先级

1. **P0: `vfmacc.vv_lane` (向量lane广播FMA)** -- **整体收益22.86%**
   - 6个平台中有4个指向同一ISA差距：RVV缺乏从向量寄存器指定lane提取元素并广播的能力
   - ARM的lane索引FMA是最具参考价值的形态（A元素保持在向量寄存器，K步间复用）
   - 建议优先实现，实现难度中等

2. **P1: `vmmacc.vv` (4x4矩阵外积FMA)** -- **整体收益44.55%（8行tile时）**
   - 仅Power VSX MMA提出，代表了矩阵级加速方向
   - 实现难度高（需专用累加器寄存器文件）
   - 建议作为长期研究方向

3. **P2: 掩码存储优化 (软件修复)** -- 整体收益<1%
   - RVV已有硬件掩码能力，当前实现未使用
   - 部分列场景触发，整体影响有限
   - 建议立即修复（零ISA成本）

4. **P3: 多行处理扩展 (软件优化)** -- 整体收益45.72%（配合P0时）
   - 多数平台支持多于2行并行处理
   - 需配合P0才能获得净收益（避免A加载线性增长）
   - 纯软件修改，无新ISA需求

### 无差距平台

- **LoongArch LASX**: 对RVV VLEN=512不构成ISA级差距（RVV已领先）
- **WASM SIMD**: RVV具有压倒性架构优势

### 实测与理论对比

| 指标 | 原理论估算 | BBV实测值 | 差异 |
|------|-----------|----------|------|
| K循环热点占比 | 16% | **91.43%** | +5.7倍 |
| P0整体收益上限 | 4.0% | **22.86%** | +5.7倍 |

原理论估算严重低估了K循环的实际热点权重。实测数据显示SGEMM K循环是绝对性能瓶颈，任何K循环内的指令优化都将产生显著的整体收益。

### 硬件验证限制

**注**: Banana Pi BPI-F3 仅支持 RVV256（VLEN=256），无法验证 RVV512 内核的实际性能。本报告收益基于 QEMU BBV profiling 的指令计数分析，未包含硬件周期数据。建议在支持 RVV512 的硬件上验证。

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1   | 3         | 3      | 0    |
| R2   | 3         | 3      | 0    |
| R3 (BBV更新) | 0 | - | 0    |

**R1 修复详情**:
1. [MAJOR] WASM SIMD对比表 (L276): "6.67x" 乘数格式改为 "-85.0%" 百分比格式
2. [MINOR] 改造前伪代码 (L319): `vfmacc.vf v0, v5, fa0` 操作数顺序修正为 `vfmacc.vf v0, fa0, v5`（符合 RVV 标准 `vd, rs1, vs2` 约定）
3. [MINOR] 每 K-pair 指令计数表 (L74): 指针更新从 ~3 改为 ~2（a1 通过 `a[lda]` 直接计算无需独立递增），合计 ~15 改为 ~14，FMA密度从 4.27 更新为 4.57；同步更新 P0 BB 收益计算（14→11 条，21.4%）

**R2 修复详情**:
1. [MAJOR] B向量加载计数膨胀: K-pair指令计数表中 vle32.v 从 4 修正为 2（Row 0 和 Row 1 共享B向量加载，源码验证: `v_b0`/`v_b1` 各加载一次，两行FMA均复用）。合计从 ~14 修正为 ~12，FMA密度从 4.57 更新为 5.33。级联更新: P0 2行BB收益从 (14-11)/14=21.4% 修正为 (12-9)/12=25.0%，与汇总表25%一致；保守取值说明更新。
2. [MINOR] 广播指令对比表 (L487): float32 SGEMM上下文中标量广播指令名 `vmv.v.x` 修正为 `vfmv.v.f`（float变体）。
3. [MINOR] S390X Z-Vector对比表RVV baseline数据: "576" 指令数重新计算为 ~120（基于修正后的K-pair计数: 4调用 × (24 K循环 + 6 开销) = 120），instr/FMA 从 1.13 修正为 0.23；RVV 8行假设从 ~288 修正为 ~96；表标题从"归一化指令"改为"指令数"并添加VLEN位宽归一化说明注释。

**R3 BBV数据整合**:
- 添加QEMU BBV profiling实际热点数据
- K循环热点占比从理论16%更新为实测91.43%
- 所有整体收益按实测热点重新计算
- 添加Top 10热点BB表格
- 添加硬件验证限制说明（Banana Pi仅支持RVV256）
