# ggml_gemv_q4_K_8x8_q8_K 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: `ggml_gemv_q4_K_8x8_q8_K` - Q4_K量化权重矩阵与Q8_K量化激活向量的矩阵向量乘法(GEMV)，采用8x8 interleaved tile blocking布局。

**基准实现**: RVV VLEN=512bit, SEW=32bit, LMUL=1

**分析平台**:
- x86 AVX2 (256-bit YMM寄存器)
- ARM NEON + DOTPROD扩展 (128-bit Q寄存器)

**BBV数据**: 未提供，收益为理论估算（基于算法复杂度分析）

**关键发现**: RVV缺乏**pairwise横向乘加指令**和**int8点积指令**，这是量化GEMV向量化实现的核心瓶颈。当前RVV实现退化为标量循环。效率差距分析:
- **vs AVX2向量化**: 约15.5倍 (AVX2归一化~18条指令 vs RVV标量280条)
- **vs RVV理想向量化**: 约3-8倍 (RVV无扩展向量实现需额外指令 vs 有扩展实现)

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | `vwmaccus.pair.vv` | x86 PMADDUBSW | K循环BB内减少约40% | 高 | 无signed/unsigned混合pairwise MAC |
| P0 | `vwmacc.pair.vv` | x86 PMADDWD | K循环BB内减少约20% | 高 | 无pairwise横向MAC |
| P0 | `vdot4a.vv` (signed×signed) | ARM VSDOT | K循环BB内减少约60% | 中 | Zvdot4a8i语义不匹配(无signed×signed) |
| P1 | `vpairadd.vv` | ARM VPADD | 缩减BB内减少约66% | 中 | 无pairwise横向加法 |
| P1 | `vshuffle.b.imm` | x86 PSHUFB | 解压BB内减少约50% | 高 | vrgather需预计算索引 |

**注**: 无BBV profiling数据，上表仅反映单个BB范围内的指令减少比例，无法推算整体收益。
建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算。

---

## 基准RVV实现分析

### 当前实现状态

当前RVV实现(`rvv_gemv_q4_K_8x8_q8_K.inl`)本质上为**标量实现**：

```c
// 核心K循环 - 当前RVV实现
for (int k_local = 0; k_local < 4; k_local++) {
    for (int j = 0; j < ncols_interleaved; j++) {
        int32_t sumi_lo = 0;
        int32_t sumi_hi = 0;
        for (int i = 0; i < blocklen; i++) {  // blocklen=8
            const int v0 = q4_ptr[b].qs[q4_idx] & 0xF;   // scalar nibble extract
            const int v1 = q4_ptr[b].qs[q4_idx] >> 4;    // scalar nibble extract
            sumi_lo += v0 * q8_ptr[b].qs[q8_idx_lo];     // scalar multiply
            sumi_hi += v1 * q8_ptr[b].qs[q8_idx_hi];     // scalar multiply
        }
        acc_f32[j] += (float)(sumi_lo * scales_lo[j] + sumi_hi * scales_hi[j]) * sb_scale[j];
    }
}
```

**指令统计（每K迭代）**:
- nibble提取: 2条标量运算 × 8元素 × 8列 = 128条
- 点积累加: 2条乘法 × 8元素 × 8列 = 128条
- 缩放乘法: 3条浮点运算 × 8列 = 24条
- **总计: 约280条标量指令/K迭代**

### 寄存器宽度归一化

| 平台 | 寄存器宽度 | 等效float32元素数 | 归一化因子(相对RVV 512-bit) |
|------|-----------|-------------------|---------------------------|
| x86 AVX2 | 256-bit | 8 float32 | 512/256 = 2 |
| ARM NEON | 128-bit | 4 float32 | 512/128 = 4 |
| RVV目标 | 512-bit | 16 float32 | 1 |

**注**: 归一化因子表示处理相同数据量所需的指令数倍数，与SEW无关。例如K循环处理int8元素时，AVX2每条指令处理32个int8，需要2条指令才能覆盖RVV 512-bit (64个int8)。

---

## 各平台对比分析

### 1. x86 AVX/AVX2

**核心特点**:
- 256-bit YMM寄存器，每指令处理8个float32或32个int8元素
- 支持signed/unsigned混合乘法指令(PMADDUBSW)
- 强大的字节级shuffle指令(PSHUFB)

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `PMADDUBSW` (`_mm256_maddubs_epi16`) | unsigned byte × signed byte → int16 pair-sum (saturating) | **缺失** - 无signed/unsigned混合pairwise MAC |
| `PMADDWD` (`_mm256_madd_epi16`) | int16 × int16 → int32 pair-sum | **缺失** - 无pairwise横向MAC |
| `PSHUFB` (`_mm256_shuffle_epi8`) | 字节级表查找shuffle | **部分缺失** - vrgather需预计算索引 |
| `VPBROADCASTB` (`_mm256_set1_epi8`) | 字节广播到全向量 | **已有** - vmv_v_x |

**收益分析 - AVX2 K循环**:

归一化到RVV VLEN=512后的等效指令序列（处理16个int8元素）:

```asm
; AVX2归一化K循环 (每16个int8元素)
; 注: AVX2 256-bit指令需2条才能覆盖RVV 512-bit数据
vpbroadcastb  ymm0, [q8_base]      ; 2条归一化: 加载并广播Q8激活值
vmovdqu       ymm1, [q4_base]      ; 2条归一化: 加载Q4权重
vpand         ymm2, ymm1, m4b      ; 2条归一化: 提取低nibble
vpsraw        ymm3, ymm1, 4        ; 2条归一化: 提取高nibble
pmaddubsw     ymm4, ymm2, ymm0     ; 2条归一化: u8×i8→i16 pair-sum
pmaddubsw     ymm5, ymm3, ymm0     ; 2条归一化: u8×i8→i16 pair-sum
pmaddwd       ymm6, ymm4, scales   ; 2条归一化: i16×i16→i32 pair-sum
pmaddwd       ymm7, ymm5, scales   ; 2条归一化: i16×i16→i32 pair-sum
vpaddd        ymm8, ymm6, ymm7     ; 2条归一化: 合并结果
; 总计: 约18条归一化指令/16元素K迭代
```

**vs RVV标量实现**: RVV需约280条标量指令处理相同数据量
**效率差距**: 约**15.5倍** (280/18)

**建议扩展**:

1. **`vwmaccus.pair.vv`** - unsigned byte × signed byte → int16 pair-sum
   - 语义: `vd[i] = sum(vs2[2i]×vs1[2i] + vs2[2i+1]×vs1[2i+1])` (u8×i8)
   - 等效于PMADDUBSW

2. **`vwmacc.pair.vv`** - int16 × int16 → int32 pair-sum
   - 语义: `vd[i] = sum(vs2[2i]×vs1[2i] + vs2[2i+1]×vs1[2i+1])`
   - 等效于PMADDWD

---

### 2. ARM NEON + DOTPROD

**核心特点**:
- 128-bit Q寄存器，每指令处理4个float32或16个int8元素
- DOTPROD扩展提供int8点积指令(VSDOT/UUDOT)
- pairwise操作(VPADD/VPADDL)用于高效缩减

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `VSDOT.S8` (`ggml_vdotq_s32`) | signed int8 × signed int8 → int32 dot product | **部分缺失** - 需Zvdot4a8i扩展 |
| `VPADD.S32` (`vpaddq_s32`) | 相邻元素pairwise加法 | **缺失** - 无pairwise横向加法 |
| `VPADDL.S8` | 相邻元素pairwise加法并扩展 | **缺失** - 无pairwise扩展加法 |
| `VDUP.N` | 标量广播到向量 | **已有** - vmv_v_x |
| `VMLAL.S8` | widening multiply-accumulate | **已有** - vwmacc (但无pairwise) |

**收益分析 - NEON K循环**:

归一化到RVV VLEN=512后的等效指令序列（处理16个int8元素）:

```asm
; NEON归一化K循环 (每16个int8元素，使用DOTPROD)
vdup.8       q0, [q8_base]          ; 4条归一化: 加载并广播Q8激活值
vld1.8       q1, [q4_base]          ; 4条归一化: 加载Q4权重
vand.8       q2, q1, m4b            ; 4条归一化: 提取低nibble
vshr.u8      q3, q1, #4             ; 4条归一化: 提取高nibble
vdot.s8      d4, d2, d0             ; 8条归一化: int8×int8→int32 dot (低nibble)
vdot.s8      d5, d3, d0             ; 8条归一化: int8×int8→int32 dot (高nibble)
vmla.f32     q6, q4, scales         ; 4条归一化: 缩放累加
; 总计: 约36条归一化指令/16元素K迭代 (4+4+4+4+8+8+4=36)
```

**vs RVV标量实现**: RVV需约280条标量指令处理相同数据量
**效率差距**: 约**7.8倍** (280/36)

**vs RVV等效向量实现(无扩展)**:
若RVV使用标准vwmacc+vredsum实现（包含必要开销）:
- vsetvl配置: 1条
- nibble提取(vand+vshr): 2条 × 16元素/VLEN = 2条
- widening mul: vwmul × 2 (低/高nibble) = 2条
- widening add: vwadd × 2 = 2条
- reduction: vredsum × 2 = 2条
- 缩放: vfmul × 1 = 1条
- LMUL管理(vsetvl切换): 2条
- 加载(vle): 2条
- **总计: 约13-15条向量指令/16元素K迭代**（含必要开销）

但需要额外处理scales_per_column的per-column缩放，实际需要更多vrgather操作。

**建议扩展**:

1. **`vdot4a.vv`** - int8 × int8 → int32 dot product with 4-element group
   - 语义: `vd[i] += vs2[4i]×vs1[4i] + vs2[4i+1]×vs1[4i+1] + vs2[4i+2]×vs1[4i+2] + vs2[4i+3]×vs1[4i+3]`
   - 等效于ARM VSDOT，一次处理4个int8→1个int32

2. **`vpairadd.vv`** - 相邻元素pairwise加法
   - 语义: `vd[i] = vs2[2i] + vs2[2i+1]`
   - 等效于ARM VPADD

---

## RVV扩展指令建议详细说明

### [P0] vwmaccus.pair.vv - Unsigned/Signed混合pairwise乘加

**指令定义**:
```
vwmaccus.pair.vv vd, vs2, vs1, vm
// 语义: 对于每个i (0 <= i < VL/2):
//   vd.h[i] += (unsigned)vs2.b[2i] × (signed)vs1.b[2i] +
//              (unsigned)vs2.b[2i+1] × (signed)vs1.b[2i+1]
// 输入: vs2 = unsigned int8向量, vs1 = signed int8向量
// 输出: vd = signed int16向量 (widening)
```

**⚠️ 与PMADDUBSW的语义差异**:

x86 PMADDUBSW执行**saturating add**（饱和加法），即相邻对的乘积结果进行饱和累加，溢出时截断到最大/最小值。本文建议的`vwmaccus.pair.vv`执行**accumulating add**（累积加法），结果累加到目标寄存器。两者语义不同：

| 特性 | PMADDUBSW | vwmaccus.pair.vv (建议) |
|------|-----------|-------------------------|
| 加法类型 | Saturating（饱和） | Accumulating（累积） |
| 溢出处理 | 截断到int16范围 | 依赖vd宽度(widening) |
| 目标寄存器 | 覆盖写 | 累加到现有值 |

对于量化GEMV场景，累积加法更符合需求（多组pairwise结果需累加），但需注意此语义差异。

**编码约束**:
- SEW = 8 (源元素宽度)
- 目标SEW = 16 ( widening)
- LMUL必须满足: LMUL_dest = 2 × LMUL_src

**应用场景**:
Q4_K量化权重中，4-bit nibble提取后为unsigned (0-15)，需与signed Q8_K激活值(-128~127)相乘并pairwise求和，这正是PMADDUBSW的功能。

**命名说明**:
RVV已存在以下widening multiply-accumulate指令:
- `vwmaccu.vx`: unsigned(vs2) × unsigned(vs1) → signed(vd) widening MAC
- `vwmaccus.vx`: unsigned(vs2) × signed(vs1) → signed(vd) widening MAC (**vx形式，无vv形式**)
- `vwmaccsu.vv`: signed(vs2) × unsigned(vs1) → signed(vd) widening MAC

本文建议的`vwmaccus.pair.vv`扩展了`vwmaccus`的vv形式，并添加pairwise横向求和语义。命名遵循RVV惯例，suffix含义:
- `us`: unsigned(第一个操作数) × signed(第二个操作数)
- `.pair`: pairwise横向操作(相邻元素求和)

**性能对比**:
| 实现方式 | 指令数 (归一化VLEN=512) |
|---------|------------------------|
| RVV标量 | 128条 (16次循环 × 8列) |
| RVV+vwmaccus.pair | 4条 (1次向量 × 2对) |
| 减少 | **96.9%** |

---

### [P0] vwmacc.pair.vv - Signed pairwise乘加

**指令定义**:
```
vwmacc.pair.vv vd, vs2, vs1, vm
// 语义: 对于每个i (0 <= i < VL/2):
//   vd.w[i] += vs2.h[2i] × vs1.h[2i] + vs2.h[2i+1] × vs1.h[2i+1]
// 输入: vs2, vs1 = signed int16向量
// 输出: vd = signed int32向量 (widening)
```

**编码约束**:
- SEW = 16 (源元素宽度)
- 目标SEW = 32 (widening)
- LMUL必须满足: LMUL_dest = 2 × LMUL_src

**应用场景**:
int8点积结果(int16)需进一步与scales(int16)相乘并pairwise求和，得到int32累加值。

**性能对比**:
| 实现方式 | 指令数 (归一化VLEN=512) |
|---------|------------------------|
| RVV vwmul + vwadd | 4条 (2次mul + 2次add) |
| RVV+vwmacc.pair | 2条 |
| 减少 | **50%** |

---

### [P0] vdot4a.vv - 四元素组点积

**指令定义**:
```
vdot4a.vv vd, vs2, vs1, vm
// 语义: 对于每个i (0 <= i < VL/4):
//   vd.w[i] += vs2.b[4i]×vs1.b[4i] + vs2.b[4i+1]×vs1.b[4i+1] +
//             vs2.b[4i+2]×vs1.b[4i+2] + vs2.b[4i+3]×vs1.b[4i+3]
// 输入: vs2, vs1 = signed int8向量
// 输出: vd = signed int32向量 (widening + dot)
```

**⚠️ 与Zvdot4a8i的差异说明**:

Zvdot4a8i扩展定义的`vdot4a`指令与本文建议的语义存在以下差异:

| 特性 | Zvdot4a8i | 本文建议 | ARM VSDOT |
|------|-----------|----------|-----------|
| 输入类型 | `vuint32` (packed 4×int8 as uint32) | signed int8向量 | signed int8向量 |
| signed×signed | ❌ 不支持 | ✅ 支持 | ✅ 支持 |
| unsigned×unsigned | ✅ 支持 | ❌ 不需要 | ❌ 不适用 |
| unsigned×signed | ✅ 支持 | ❌ 不需要 | ❌ 不适用 |

**关键差异**:
1. **操作数类型**: Zvdot4a8i使用`vuint32`类型表示packed int8，需要特殊的数据布局。本文建议直接使用int8向量，更符合向量编程习惯。
2. **signed×signed缺失**: Zvdot4a8i提供unsigned×unsigned和unsigned×signed变体，但**不支持signed×signed**。ARM VSDOT使用signed×signed，这是量化推理中最常用的模式。因此，本文建议的`vdot4a.vv`应视为**新指令**，而非Zvdot4a8i的子集。

**编码约束**:
- SEW = 8 (源元素宽度)
- 目标SEW = 32 (widening)
- LMUL必须满足: LMUL_dest = 4 × LMUL_src

**应用场景**:
ARM DOTPROD的核心指令，一次将4个int8×int8乘加结果累加到1个int32，等效于4次vwmul + 1次vwadd + 1次vredsum。

**性能对比**:
| 实现方式 | 指令数 (归一化VLEN=512, 16个int8→4个int32) |
|---------|----------------------------------------|
| RVV vwmul + vwadd + vredsum | 6条 |
| RVV+vdot4a | 1条 |
| 减少 | **83.3%** |

**现有RVV支持**:
Zvdot4a8i扩展已定义类似指令，但为**可选扩展**且语义不完全匹配。建议设计新的signed×signed点积指令并纳入RVV 1.0基础指令集或作为mandatory扩展。

---

### [P1] vpairadd.vv - Pairwise横向加法

**指令定义**:
```
vpairadd.vv vd, vs2, vm
// 语义: 对于每个i (0 <= i < VL/2):
//   vd.h[i] = vs2.h[2i] + vs2.h[2i+1]
// 输入: vs2 = 向量
// 输出: vd = 向量 (元素数减半)
```

**编码约束**:
- 输出LMUL = 输入LMUL / 2 (元素数减半)

**应用场景**:
ARM VPADD用于向量缩减，如16个int32 → 8个int32 → 4个int32 → 2个int32 → 1个int32的递进缩减。

**性能对比**:
| 实现方式 | 指令数 (归一化VLEN=512) |
|---------|------------------------|
| RVV vrgather + vadd | 3条 (vrgather_even + vrgather_odd + vadd) |
| RVV+vpairadd | 1条 |
| 减少 | **66.7%** |

---

### [P1] vshuffle.b.imm - Immediate控制的字节shuffle

**指令定义**:
```
vshuffle.b.imm vd, vs2, imm_ctrl, vm
// 语义: 对于每个字节位置i:
//   若imm_ctrl.byte[i] < 16: vd.b[i] = vs2.b[imm_ctrl.byte[i]]
//   若imm_ctrl.byte[i] >= 16: vd.b[i] = 0
// 输入: vs2 = 向量, imm_ctrl = 立即数控制向量(编码为CSR或指令字段)
```

**编码约束**:
- 需要新的立即数编码空间，或使用CSR存放控制表

**应用场景**:
x86 PSHUFB用于4-bit nibble查找表解压，如将nibble值映射到具体scale值。

**性能对比**:
| 实现方式 | 指令数 |
|---------|--------|
| RVV vrgather (预计算索引) | 2-3条 (索引计算 + vrgather) |
| RVV+vshuffle.b.imm | 1条 |
| 减少 | **约50%** |

---

## 附录

### FMA指令对比表

| 平台 | 指令 | 输入类型 | 输出类型 | Pairwise | RVV等效 |
|------|------|---------|---------|---------|---------|
| x86 AVX2 | PMADDUBSW | u8×i8 | i16 (saturating) | Yes | **缺失** |
| x86 AVX/AVX2 | PMADDWD/VPMADDWD¹ | i16×i16 | i32 | Yes | **缺失** |
| ARM NEON | VSDOT | i8×i8 | i32 | No (4-sum) | Zvdot4a8i |
| ARM NEON | VMLAL | i8×i8 | i16 (MAC) | No | vwmacc |
| RVV | vwmacc | i8×i8 | i16 (MAC) | No | 已有 |
| RVV | vfmacc | f32×f32 | f32 (MAC) | No | 已有 |

¹ PMADDWD (128-bit XMM) 和 VPMADDWD (256-bit YMM) 是同一指令的不同编码宽度版本，语义相同。

### 缩减指令对比表

| 平台 | 指令 | 功能 | RVV等效 |
|------|------|------|---------|
| x86 AVX2 | VPADDD | 向量加法(无pairwise) | vadd |
| ARM NEON | VPADD | pairwise加法 | **缺失** |
| ARM NEON | VPADDL | pairwise加法+widen | **缺失** |
| RVV | vredsum | 全向量缩减 | 已有 |

### Shuffle指令对比表

| 平台 | 指令 | 功能 | RVV等效 |
|------|------|------|---------|
| x86 AVX2 | PSHUFB | 字节级表查找 | **部分缺失** (vrgather) |
| ARM NEON | VTBL | 表查找 | **缺失** |
| RVV | vrgather | 向量索引gather | 已有(但需预计算索引) |

---

## 结论

### 关键差距总结

1. **P0级差距 - Pairwise横向乘加**:
   - x86 PMADDUBSW和PMADDWD/VPMADDWD提供高效的int8/int16 pairwise MAC
   - RVV缺失此类指令，导致量化矩阵乘法效率差距:
     - vs AVX2向量化: 约**15.5倍** (AVX2归一化~18条 vs RVV标量280条)
     - vs RVV理想向量实现: 约3-8倍额外开销
   - 建议: 新增`vwmaccus.pair.vv`和`vwmacc.pair.vv`扩展

2. **P0级差距 - 整数点积**:
   - ARM VSDOT一次完成4个int8×int8→int32点积
   - Zvdot4a8i可选扩展语义不完全匹配(无signed×signed)
   - 建议: 设计新的signed×signed点积指令并纳入mandatory扩展

3. **P1级差距 - Pairwise缩减**:
   - ARM VPADD用于高效向量缩减
   - RVV需vrgather+vadd组合，效率损失约**3倍**
   - 建议: 新增`vpairadd.vv`扩展

### 优先级排序

| 优先级 | 扩展指令 | 预估收益 | 实现复杂度 |
|--------|----------|----------|-----------|
| P0 | vdot4a.vv (signed×signed) | 最高(约60%指令减少) | 中(需新设计，Zvdot4a8i语义不匹配) |
| P0 | vwmaccus.pair.vv | 高(约40%指令减少) | 高(新增signed/unsigned混合pairwise) |
| P0 | vwmacc.pair.vv | 高(约20%指令减少) | 中(pairwise MAC) |
| P1 | vpairadd.vv | 中(约66%缩减指令减少) | 低(pairwise add) |
| P1 | vshuffle.b.imm | 中(约50%shuffle指令减少) | 高(新编码空间) |

### 实现路径建议

1. **短期**: 设计新的signed×signed int8点积指令(区别于Zvdot4a8i)，作为mandatory扩展
2. **中期**: 新增`vwmaccus.pair.vv`和`vwmacc.pair.vv`扩展指令
3. **长期**: 新增`vpairadd.vv`和`vshuffle.b.imm`扩展指令

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1 | 10 | 10 | 0 |
| R2 | 1 | 1 | 0 |

**R1修复详情**:
- Issue 1: Zvdot4a8i operand types - 添加与Zvdot4a8i的差异说明表
- Issue 3: 寄存器宽度归一化表头 - 改为"等效float32元素数"，添加SEW无关说明
- Issue 4: vwmaccus命名混淆 - 添加与现有RVV指令的命名关系说明
- Issue 5: AVX2指令数低估 - 修正为~18条归一化指令，效率差距15.5x
- Issue 6: PMADDUBSW语义说明 - 标注saturating add，添加与vwmaccus.pair语义差异说明
- Issue 7: RVV指令数低估 - 修正为13-15条(含开销)
- Issue 8: vdot4a signed/unsigned变体 - 添加signed×signed缺失说明
- Issue 9: PMADDWD/VPMADDWD合并 - 合并为单行并标注编码宽度变体
- Issue 10: 概述与正文数字冲突 - 概述改为分层说明效率差距

**R2修复详情**:
- Issue R2-1: NEON归一化指令数算术错误 - 修正为36条(4+4+4+4+8+8+4)，效率差距7.8x

**最终审查结论**: 所有R1和R2问题均已修复，报告通过审查。核心差距分析准确，扩展指令建议合理。

最终审查结论: 待R2审查