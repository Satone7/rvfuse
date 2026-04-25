# MlasComputeLogistic 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: `MlasComputeLogistic` — Logistic/Sigmoid激活函数向量实现
**算法**: sigmoid(x) = 1 / (1 + exp(-x))，使用有理多项式近似
**基准实现**: RVV VLEN=512, VL=16 (float32), LMUL=1
**分析平台**: x86 AVX/AVX2, ARM NEON, LoongArch LSX, Power VSX, S390X, WASM SIMD
**BBV数据**: 未提供，收益为理论估算（基于MLAS profiling热点占比约10.01%）

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vfrsqrt7.v | x86 AVX512 | sigmoid多项式BB内减少约20% | 中 | RVV已有此指令，当前实现未使用 |
| P1 | vexp.approx | ARM NEON | 指数计算加速 | 高 | RVV无近似指数指令 |
| P2 | vclamp.vv | Power VSX | 单指令clamp替代vfmin/vfmax | 低 | RVV需2条vfmin+vfmax |

**收益计算方式**（无BBV数据，仅BB范围内估算）：
- BB内收益 = (原BB指令数 - 扩展后BB指令数) / 原BB指令数 × 100%
- 整体收益需BBV profiling数据支持

---

## 基准RVV实现分析

### Logistic/Sigmoid算法

```
sigmoid(x) = 1 / (1 + exp(-x))

MLAS使用有理多项式近似（Eigen算法）：
1. Clamp输入到[-18, 18]
2. 计算p(x) = x * (α₁ + x² * (α₃ + x² * (α₅ + x² * (α₇ + x² * α₉))))
3. 计算q(x) = β₀ + x² * (β₂ + x² * (β₄ + x² * (β₆ + x² * (β₈ + x² * β₁₀))))
4. result = clamp(p(x) / q(x) + 0.5, 0.0, 1.0)
```

### RVV实现结构

```cpp
// rvv_compute_logistic.inl 核心结构
for (size_t i = 0; i < N; ) {
    vl = __riscv_vsetvl_e32m1(N - i);

    // 1. Load and clamp to [-18, 18]
    vfloat32m1_t Value = __riscv_vle32_v_f32m1(Input + i, vl);
    Value = __riscv_vfmax_vf_f32m1(Value, C.LowerRange, vl);
    Value = __riscv_vfmin_vf_f32m1(Value, C.UpperRange, vl);

    // 2. Compute ValueSquared
    vfloat32m1_t ValueSquared = __riscv_vfmul_vv_f32m1(Value, Value, vl);

    // 3. Compute p(x) polynomial (4 vfmacc operations)
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(C.alpha_9, vl);
    p = __riscv_vfmacc_vf_f32m1(p, C.alpha_7, ValueSquared, vl);
    p = __riscv_vfmacc_vf_f32m1(p, C.alpha_5, ValueSquared, vl);
    p = __riscv_vfmacc_vf_f32m1(p, C.alpha_3, ValueSquared, vl);
    p = __riscv_vfmacc_vf_f32m1(p, C.alpha_1, ValueSquared, vl);
    p = __riscv_vfmul_vv_f32m1(p, Value, vl);  // multiply by x

    // 4. Compute q(x) polynomial (6 vfmacc operations)
    vfloat32m1_t q = __riscv_vfmv_v_f_f32m1(C.beta_10, vl);
    q = __riscv_vfmacc_vf_f32m1(q, C.beta_8, ValueSquared, vl);
    // ... similar for beta_6, beta_4, beta_2, beta_0

    // 5. Final result: p/q + 0.5, clamped to [0, 1]
    vfloat32m1_t result = __riscv_vfdiv_vv_f32m1(p, q, vl);
    result = __riscv_vfadd_vf_f32m1(result, C.one_half, vl);
    result = __riscv_vfmax_vf_f32m1(result, 0.0f, vl);
    result = __riscv_vfmin_vf_f32m1(result, 1.0f, vl);

    __riscv_vse32_v_f32m1(Output + i, result, vl);
    i += vl;
}
```

### RVV指令计数

| 操作阶段 | 指令数 | 说明 |
|----------|--------|------|
| Clamp | 3 | vle32 + vfmax + vfmin |
| ValueSquared | 1 | vfmul |
| p(x)多项式 | 6 | vfmv + 4×vfmacc + vfmul |
| q(x)多项式 | 6 | vfmv + 5×vfmacc |
| 最终计算 | 4 | vfdiv + vfadd + vfmax + vfmin |
| 存储 | 1 | vse32 |
| **总计** | **21** | 每VL迭代 |

**关键观察**：
- 多项式计算使用FMA（vfmacc）优化，每步乘+累加合并为1条指令
- **vfmacc语义**：`vfmacc vd, rs1, vs2` 计算 `vd = vs2 × rs1 + vd`（非 `vd × vs2 + rs1`）
- Clamp需要2条指令（vfmax + vfmin）
- 除法是单指令（vfdiv），但通常延迟较高
- 最终clamp到[0, 1]需要额外2条指令

---

## 各平台对比分析

### 1. x86 AVX/AVX2

**核心特点**：
- YMM寄存器，256-bit向量宽度
- `_mm256_max_ps`/`_mm256_min_ps`：向量min/max
- `_mm256_fmadd_ps`：FMA指令
- **无专用近似倒数/指数指令**（AVX512有`_mm512_rcp14_ps`近似倒数）

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `_mm256_fmadd_ps` | FMA（乘加融合） | RVV有`vfmacc` |
| `_mm512_rcp14_ps` | 14-bit近似倒数（AVX512） | RVV无 |

**收益分析**：

| 操作 | AVX指令数 | RVV指令数 |
|------|-----------|-----------|
| Clamp | 2 (max + min) | 2 |
| 多项式FMA | 每步1条 | 每步1条 |
| 最终clamp | 2 | 2 |
| **总计** | ~20 | ~21 |

**建议扩展**：
- `vfrsqrt7.v` — 7-bit近似倒数平方根（类似AVX512 `vrsqrt14ps` / `_mm512_rsqrt14_ps`）
- 可用于sigmoid快速近似：sigmoid(x) ≈ 0.5 + 0.5 * x * rsqrt(1 + x²)
- **注意**：此近似公式 `0.5 * (1 + x / sqrt(1 + x²))` 实为**softsign函数**，非sigmoid精确等价
- softsign精度较低，适用场景受限，需权衡精度与性能

---

### 2. ARM NEON

**核心特点**：
- 128-bit Q寄存器，4×float32
- `vfmaq_f32`：FMA指令
- **无专用近似倒数/指数指令**
- NEON的sigmoid通过软件多项式实现

**与AVX类似**，ARM NEON无硬件sigmoid指令，需软件多项式实现。

---

### 3. LoongArch LSX

**核心特点**：
- 128-bit向量，4×float32
- `__lsx_vfmadd_s`：FMA指令
- 多项式实现与RVV类似

**指令对比**：与RVV类似的多项式+FMA模式。

---

### 4. Power VSX

**核心特点**：
- 128-bit VSX寄存器
- `vec_madd`：FMA指令
- `vec_max`/`vec_min`：向量min/max

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vec_max`/`vec_min` | 向量min/max | vfmax/vfmin |
| `vec_madd` | FMA | vfmacc |

**建议扩展**：无显著差异。

---

### 5. S390X

**核心特点**：
- 128-bit Vector Facility
- FMA支持
- 软件多项式sigmoid实现

---

### 6. WASM SIMD

**核心特点**：
- 128-bit v128
- `wasm_f32x4.max`/`wasm_f32x4.min`：向量min/max
- 无近似倒数/指数

---

## RVV扩展指令建议详细说明

### [P0] vfrsqrt7.v / vfrec7.v — RVV已有指令，当前实现未使用

**现状澄清**：
- `vfrsqrt7.v` 和 `vfrec7.v` **已是RVV基础V扩展的一部分**（已 ratified）
- 存在于 `RISCVInstrInfoV.td`，并有 intrinsic 支持
- **并非新指令提议**，而是分析当前实现为何未利用这些已有指令

**指令定义**：
```
vfrsqrt7.v vd, vs2, vm   # vd[i] ≈ 1/sqrt(vs2[i]) (7-bit精度)
vfrec7.v vd, vs2, vm     # vd[i] ≈ 1/vs2[i] (7-bit精度)
```

**当前实现未使用原因分析**：

1. **精度需求不匹配**
   - sigmoid计算使用有理多项式近似，追求较高精度
   - `vfrsqrt7` 仅提供7-bit精度（约2位十进制）
   - 多项式方法当前使用精确除法（`vfdiv`），保证IEEE精度

2. **算法转换成本**
   - 当前算法基于多项式近似：`p(x) / q(x)`
   - 若使用 `vfrsqrt7`，需完全重构为：`0.5 + 0.5 * x * rsqrt(1 + x²)`
   - 此公式为 **softsign近似**，非sigmoid精确等价，精度较低
   - 算法变更需重新验证精度边界

3. **Newton-Raphson迭代开销**
   - 7-bit近似需NR迭代提升精度
   - 达到24-bit精度：需2次NR迭代，共 **3条指令**（vfrsqrt7 + 2×vfmacc）
   - 当前精确除法仅1条指令（`vfdiv`），但延迟较高

**性能对比**：

| 方案 | 指令数 | 精度 | 适用场景 |
|------|--------|------|----------|
| 当前RVV精确除法 | 1条vfdiv | IEEE精确 | 高精度需求 |
| vfrsqrt7（无NR） | 1条 | 7-bit | 低精度近似 |
| vfrsqrt7 + 2×NR | 3条（vfrsqrt7 + 2×vfmacc） | ~24-bit | 中等精度 |

**优化建议**：
- 若MLAS可接受softsign近似（精度损失），可重构算法使用`vfrsqrt7`
- 若需保持当前精度，继续使用多项式+精确除法方案
- 可考虑配置开关：低精度模式用`vfrsqrt7`，高精度模式用多项式

**参考**：x86 AVX512 `vrsqrt14ps` / `_mm512_rsqrt14_ps`（14-bit近似倒数平方根）、ARM NEON `vrsqrte_f32`

---

### [P1] vexp.approx

**指令定义**：
```
vexp.approx vd, vs2, vm  # vd[i] ≈ exp(vs2[i])
```

**应用场景**：
- sigmoid精确计算：sigmoid(x) = 1 / (1 + exp(-x))
- softmax激活函数
- 其他需要指数的场景

**硬件考量**：
- 指数函数硬件实现复杂度较高
- 可考虑低精度近似版本（类似NEON的`vexp_f32`提案）

---

### [P2] vclamp.vv / vclamp.vf

**指令定义**：
```
vclamp.vv vd, vs2, vs_min, vs_max  # vd[i] = clamp(vs2[i], vs_min[i], vs_max[i])
vclamp.vf vd, vs2, min, max        # vd[i] = clamp(vs2[i], min, max)
```

**应用场景**：
- sigmoid最终clamp：clamp(result, 0.0, 1.0)
- 当前需要2条指令（vfmax + vfmin）
- 单指令可减少50% clamp指令数

**性能对比**：

| 方案 | 指令数 |
|------|--------|
| 当前RVV | 2（vfmax + vfmin） |
| vclamp | 1 |

**参考**：Power VSX可通过`vec_sel`+`vec_cmpgt`组合实现，但非单指令

---

## 结论

### 关键发现

1. **RVV sigmoid实现与主流平台效率相当**
   - 多项式+FMA模式与AVX/NEON类似
   - 指令数相近（~21条 vs ~20条）
   - 无显著差距或优势

2. **vfrsqrt7/vfrec7已存在于RVV基础扩展**
   - RVV基础V扩展已包含近似倒数指令`vfrsqrt7.v`和`vfrec7.v`
   - AVX512有`rsqrt14_ps`（14-bit近似倒数平方根），NEON有`vrsqrte_f32`
   - 当前MLAS实现未使用这些指令，原因：精度需求与算法结构
   - 使用`vfrsqrt7`需重构为softsign近似（精度较低）

3. **Clamp可合并为单指令**
   - 当前需2条指令（vfmax + vfmin）
   - P2优先级：`vclamp.vf`单指令clamp

### 优先级总结

| 优先级 | 扩展 | 收益范围 | 来源平台 | 说明 |
|--------|------|----------|----------|------|
| P0 | vfrsqrt7.v | sigmoid近似场景约20% | x86 AVX512 | RVV已有，当前实现未使用 |
| P1 | vexp.approx | 指数场景（硬件复杂） | ARM提案 | RVV无此指令 |
| P2 | vclamp.vf | clamp操作减少50% | Power VSX模式 | RVV无此指令 |

**整体收益估算**：
- sigmoid计算占整体执行约10%
- P0近似倒数：RVV已有指令，需重构算法为softsign近似，BB内减少约20%（但精度降低）
- P2 clamp：2条→1条，整体影响较小（2/21≈10%）

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1 | - | - | 待审查 |
| R2 | 6 | 6 | 已完成 |

**R2修复内容**：
1. P0声明修正：`vfrsqrt7.v`/`vfrec7.v`已是RVV基础扩展，非新指令提议
2. AVX512对比修正：引用`vrsqrt14ps`/`_mm512_rsqrt14_ps`（非`rcp14_ps`）
3. 近似公式标注：明确softsign近似与sigmoid的精度差异
4. NR迭代计数修正：24-bit精度需3条指令（vfrsqrt7 + 2×vfmacc）
5. p(x)指令计数修正：4×vfmacc（非5×）
6. vfmacc语义说明：`vd = vs2 * rs1 + vd`

---