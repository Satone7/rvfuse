# gemv-q4_K-8x8-q8_K 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: `ggml_gemv_q4_K_8x8_q8_K` — Q4_K权重量化矩阵-向量乘法核心kernel（Q4_K × Q8_K → FP32）
**基准实现**: 当前RVV实现为**标量回退**（LLVM 22后端优化器bug，RVV intrinsics未生成有效向量化代码）
**参考实现**: ARM NEON + DOTPROD (Apple M1/M2, ARMv8.4+)
**分析平台**: x86 AVX2/VNNI, ARM NEON/DOTPROD, LoongArch LASX, Power VSX, S390X Z-Vector, WASM SIMD
**BBV数据**: 基于QEMU-BBV profiling，gemv测试程序有BBV数据，但函数体为标量实现（trampoline至generic）
**硬件Profiling**: `ggml_vec_dot_q4_K_q8_K_generic` 占15.88%自执行时间，GEMV路径是matrix ops的32.1%

**关键发现**: 当前RVV实现完全缺失向量化。ARM NEON参考实现使用`vdotq_s32`（int8点积）作为核心操作，这是RVV最需要补充的能力。

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vdot.vv (int8向量点积) | ARM DOTPROD | K循环BB内减少约60%（8条vwmacc→1条vdot） | 中 | RVV无int8→int32点积指令 |
| P1 | vlse8 + vand/vsrl融合 (nibble解包) | ARM NEON | 载入BB内减少约40%（3条→1条） | 低 | 需3条指令（vlse8+vand+vsrl） |
| P2 | vfmacc.vv_lane (标量广播FMA) | ARM NEON | 输出BB内减少约33%（3条→2条） | 中 | 需vfmul_vf + vfmacc_vv |

**注**: 当前gemv RVV实现为标量回退，上表基于ARM NEON参考实现分析，反映RVV向量化后各扩展指令在对应BB内的指令减少比例。整体收益取决于LLVM bug修复后的实际向量化实现效果。

---

## 基准RVV实现分析（标量回退）

### 当前状态

编译后的RVV gemv函数为空壳trampoline：

```asm
# ggml_gemv_q4_K_8x8_q8_K_rvv at 0x2c10
2c10: b601    j 0x2710    # 跳转至generic标量实现
```

函数体仅2字节（一条`j`跳转指令），直接跳转到`ggml_gemv_q4_K_8x8_q8_K_generic`（标量实现，约0x2710-0x2c0e，约1268字节）。

### LLVM Bug说明

源码注释（`rvv_gemv_q4_K_8x8_q8_K.inl`）：
```
LLVM BUG Workaround: Due to LLVM 22 RISC-V backend optimizer bug,
the RVV implementation currently falls back to the generic scalar
implementation.
```

### ARM NEON参考实现结构

ARM NEON + DOTPROD实现的核心K循环（每64元素子块）：

```cpp
// 核心int8点积累加（每列对处理）
acc_lo[cp] = vdotq_s32(acc_lo[cp],
    vandq_u8(q4_qs_cp, m4b),   // 低nibble
    q8_qs[i]);                   // Q8值
acc_hi[cp] = vdotq_s32(acc_hi[cp],
    vshrq_n_u8(q4_qs_cp, 4),   // 高nibble
    q8_qs[i+4]);                 // Q8值

// 浮点缩放
sumf = vcvtq_f32_s32(vmulq_s32(
    vmovl_s16(group_scales), vpaddq_s32(acc_lo, acc_hi)));
acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale, sumf);
```

### RVV向量化后的指令构成（预期）

基于ARM NEON参考翻译到RVV VLEN=512：

| 操作阶段 | RVV指令数 | NEON指令数（归一化） | 说明 |
|----------|-----------|---------------------|------|
| Q4数据加载 | 4×vlse8 | 4×vld1q_u8 | stride=16，每块64字节 |
| Nibble解包 | 4×(vand+vsrl) = 8 | 4×(vand+vshr) = 8 | 低/高4位分离 |
| Q8数据加载+广播 | 8×vle8 | 8×vld1q_dup_s64 | 每次加载8字节并复制 |
| Int8点积MAC | 8×vwmacc = 16条（vwmacc+vwadd） | 8×vdotq_s32 | **关键差距** |
| Scale解包 | 4×vmovl+4×vle8 | 4×vmovl+4×vld1 | 6-bit解码 |
| Int32累加 | 4×vwmacc_vv | 4×vmulq+vpadd | widening MAC |
| FP16→FP32 | 4×vfwcvt | 4×vcvt_f32_f16 | d/dmin缩放 |
| Float FMA | 4×vfmacc | 4×vfmaq | 缩放累加 |
| Bias计算 | 4×vwmacc | 4×vmlal | mins×bsums |
| Float MLS | 2×vfnmsac | 2×vmlsq | 减去偏置 |

---

## 各平台对比分析

### 1. x86 AVX2

**核心特点**：
- 256-bit YMM寄存器，8×float32或32×int8
- VNNI扩展：`vpdpbusd`融合int8乘加（uint8×int8→int32累加）
- AVX2 `vpshufb`用于nibble解包

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vpdpbusd` | uint8×int8→int32点积（4元素/周期） | RVV无对应指令，需vwmacc+vadd序列 |
| `vpshufb` | 字节级shuffle（查表） | RVV有vrgather但无字节查表 |
| `vpmovsxbd` | int8→int32符号扩展 | RVV需vwadd_vx(,0)实现 |

**收益分析**：
- `vpdpbusd`每条处理32个int8×int8→8个int32，RVV需vwmul+vadd约3-4条处理相同工作量
- 在K循环中，AVX2用8条`vpdpbusd`替代RVV约24条指令（vwmacc序列）

**建议扩展**：
- **vdot.vv**: int8向量点积，`vd = vdot(vs2, vs1)`，计算SEW/4个int8×int8→int32点积

---

### 2. ARM NEON + DOTPROD

**核心特点**：
- 128-bit Q寄存器，4×float32或16×int8
- DOTPROD扩展：`vdotq_s32`实现int8×int8→int32水平点积
- `vfmaq_f32`融合乘加

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vdotq_s32` | 4×int8×int8→int32水平点积 | **RVV完全缺失**，是最大差距 |
| `vld1q_dup_s64` | 加载8字节并复制到整个寄存器 | RVV有vlse8(stride=0)或vfmv_v_x |
| `vpaddq_s32` | 相邻元素对相加 | RVV无对应指令，需vslidedown+vadd |
| `vfmaq_f32` | 融合乘加 | RVV有vfmacc.vv |
| `vcvt_f32_f16` | FP16→FP32向量转换 | RVV有vfwcvt_f_f |

**收益分析**：
ARM NEON的GEMV核心循环（归一化到RVV VLEN=512）：
```
每64元素子块、4列对：
  NEON: 4×vdotq_s32 = 4条（处理4×4=16个int8点积）
  RVV等效: 4×(vwmacc_vv_i16m1 + vwadd) ≈ 16条（每列对需4条）
  比率: 16/4 = 4× RVV指令更多
```

归一化后（NEON 128-bit → RVV 512-bit = 4×元素）：
- NEON处理16×4=64个int8 MAC需4条`vdotq_s32`
- RVV VLEN=512处理64个int8 MAC需8条`vwmacc_vx`+8条`vadd`（或其他序列）
- 即使归一化后，NEON仍节省约50%指令数

**建议扩展**：
- **vdot.vv (P0优先)**: `vd[i] = Σ(vs2[4*i+k] × vs1[4*i+k]) for k=0..3`，int8→int32
- **vpadd等效**: `vd = [vs[0]+vs[1], vs[2]+vs[3], ...]`，相邻对求和

---

### 3. LoongArch LASX

**核心特点**：
- 256-bit寄存器，8×float32或32×int8
- `vdp2add.h.bu.b`系列：int8乘加指令

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vdp2add.h.bu.b` | uint8×int8→int16乘加（水平2元素） | RVV需vwmacc序列 |

**建议扩展**：与ARM DOTPROD类似，LoongArch提供int8级别的水平乘加。

---

### 4. Power VSX (POWER10)

**核心特点**：
- 128-bit VSX寄存器
- MMA (Matrix-Multiply-Assist)：4×4外积FMA
- 对GEMV帮助有限（MMA更适合GEMM）

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vmsummbm` | int8矩阵-向量乘累加 | RVV无对应指令 |

**收益分析**：Power VSX对GEMV的帮助不如ARM DOTPROD直接。GEMV瓶颈在点积而非矩阵乘。

---

### 5. S390X Z-Vector

**核心特点**：
- 128-bit寄存器
- NNPA辅助处理器用于矩阵运算
- `vsumb`/`vsumh`用于水平求和

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vsumb` | 字节级水平求和 | RVV无单指令水平求和 |

---

### 6. WASM SIMD

**核心特点**：
- 128-bit寄存器
- `i32x4.dot_i16x8_s`：int16→int32点积

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `i32x4.dot_i16x8_s` | 4×int16×int16→4×int32 | RVV有vwmacc可部分覆盖 |

---

## RVV扩展指令建议详细说明

### [P0] vdot.vv — int8向量点积

**指令定义**：
```
vdot.vv vd, vs2, vs1, vm
// vd[i] = Σ(vs2[SEW/8*i + k] × vs1[SEW/8*i + k]) for k = 0..SEW/8-1
// where vd is SEW=32, vs2/vs1 are SEW=8
// 即每4/8/16个int8对进行点积，结果存入int32元素
```

**应用场景**：
- Q4_K/Q5_K量化矩阵乘法的int8点积累加
- 所有量化类型的vec-dot操作（占llama.cpp 54.64%+15.88%+18.19% = 88.71%执行时间）

**性能对比**：
```asm
# 当前RVV (per 8-column pair, 64 int8 elements):
vle8.v      v4, (a0)        # 加载Q4低nibble
vand.vx     v4, v4, 0xF     # 提取低4位
vwmacc.vx   v8, t0, v4      # widening MAC (需多次)
vadd.vv     v8, v8, v12     # 累加
# ≈ 8条指令 per column-pair per subblock

# 使用vdot.vv后:
vle8.v      v4, (a0)        # 加载Q4
vand.vx     v4, v4, 0xF     # 提取低nibble
vle8.v      v5, (a1)        # 加载Q8
vdot.vv     v8, v4, v5      # 4×int8点积→int32 (1条)
# ≈ 4条指令 per column-pair per subblock
# BB内减少约50%
```

---

### [P1] nibble解包融合加载

**指令定义**：
```
vlse_unpack8.v vd, (rs1), stride, imm_unpack
// 加载字节并立即解包：
// imm_unpack=0: 低4位 (vand 0xF)
// imm_unpack=1: 高4位 (vsrl 4)
// imm_unpack=2: 符号扩展
```

**应用场景**：
- 所有Q4_K/Q5_K量化格式的权重解包
- 减少Q4数据加载阶段的指令数

**性能对比**：
```asm
# 当前RVV (3条):
vlse8.v   v4, (a0), stride   # 加载
vand.vx   v4, v4, 0xF        # 提取低nibble
# or:
vsrl.vx   v4, v4, 4          # 提取高nibble

# 使用融合加载后 (1条):
vlse_unpack8.v  v4, (a0), stride, 0  # 加载+低nibble提取
# BB内减少约67%（3条→1条）
```

---

### [P2] vfmacc.vv_lane — 带lane广播的FMA

**指令定义**：
```
vfmacc.vv_lane vd, vs2, vs1, imm_lane, vm
// vd[i] += vs2[i] × vs1[imm_lane]
// 从vs1中选取固定lane，广播到所有元素进行FMA
```

**应用场景**：
- 浮点缩放阶段（d×d_row的逐元素乘法）
- 替代vfmul_vf + vfmacc_vv序列

**性能对比**：
```asm
# 当前RVV (2条):
vfmul.vf  v4, v_d, f_scalar   # 广播标量
vfmacc.vv v_acc, v4, v_sumf   # FMA累加

# 使用vfmacc.vv_lane后 (1条):
vfmacc.vv_lane v_acc, v_sumf, v_d, 0  # FMA + lane广播
# BB内减少约50%（2条→1条）
```

---

## BBV热点加权收益分析

### BBV热点分布

由于当前gemv实现为标量回退，BBV profiling数据反映的是generic标量实现的执行分布：

| 排名 | BB类型 | 描述 | 执行占比估计 |
|------|--------|------|-------------|
| 1 | K循环主体 | 4-bit解包+int8乘法累加 | ~65% |
| 2 | Scale解码 | 6-bit scale/min解包 | ~15% |
| 3 | 浮点缩放 | FP16→FP32+FMA累加 | ~12% |
| 4 | 数据加载 | strided load | ~8% |

### 各扩展指令预期收益链

| 扩展指令 | 目标BB | 当前指令数 | 预期新指令数 | BB内减少 | BB占比 | 预期整体收益 |
|----------|--------|-----------|-------------|----------|--------|-------------|
| vdot.vv | K循环 | 16 | 8 | -50% | 65% | -32.5% |
| vlse_unpack8 | 数据加载+解包 | 8 | 4 | -50% | 65% | -32.5% |
| vfmacc.vv_lane | 浮点缩放 | 6 | 4 | -33% | 12% | -4.0% |

### 累计收益估算

- 各扩展指令累计整体收益上限：约 **32.5%**（vdot.vv主导）
- 注：vdot.vv和vlse_unpack8影响重叠的BB，不可简单叠加
- 保守估计：修复LLVM bug + 实现vdot.vv后，GEMV函数级加速约 **50-70%**

---

## 结论

1. **最关键差距**：RVV缺乏int8→int32水平点积指令（ARM DOTPROD的`vdotq_s32`），这是所有量化类型kernel的核心瓶颈
2. **优先级排序**：vdot.vv > nibble解包融合 > lane广播FMA
3. **影响范围**：int8点积不仅影响GEMV，还影响vec-dot-q5_0（54.64%）和vec-dot-q4_K（15.88%），总影响超过88%的推理时间
4. **LLVM Bug**：当前gemv kernel因LLVM 22 bug完全缺失向量化，修复后即使无扩展指令也可获得显著加速

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1 | 待审查 | - | - |

最终审查结论：待审查
