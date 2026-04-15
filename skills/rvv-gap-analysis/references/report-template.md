# Report Template

This template defines the required sections for an RVV gap analysis report.

## Report Structure

```markdown
# [Operator Name] 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: [operator name and purpose]
**基准实现**: RVV VL=<N> (VLEN=<M>bit, SEW=<S>bit, LMUL=<L>)
**分析平台**: AVX/AVX2, ARM NEON/SVE, LoongArch LASX/LSX, Power VSX (POWER10), S390X Z-Vector, WASM SIMD
**BBV数据**: [path or "未提供，收益为理论估算"]

---

## 指令方案汇总

<!-- When BBV data IS available, use "整体收益" column -->
| 优先级 | 扩展指令 | 来源平台 | 整体收益 | 影响BB | BB内减少 | BB占比 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|--------|----------|--------|----------|---------|
| P0     | vmulacc.vv | Power VSX | 整体减少3.2% | K循环BB | -37.5% | 8.5% | 高 | 无矩阵级指令 |
| P1     | vfmacc.vv_lane | ARM NEON | 整体减少2.1% | FMA BB | -41.7% | 5.0% | 中 | 需标量广播 |

<!-- When BBV data is NOT available, use "BB内收益" column instead -->
<!-- | 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 | -->
<!-- | P0     | vmulacc.vv | Power VSX | BB内减少37.5%（K循环BB） | 高 | 无矩阵级指令 | -->

**收益计算方式**（基于QEMU-BBV profiling数据）：
- BB执行权重 = BB执行次数 × BB指令数
- BB执行占比 = BB执行权重 / Σ(所有BB执行权重)
- 整体收益 = BB内减少比例 × BB执行占比
- 各扩展指令的整体收益可叠加估算上限：Σ 整体收益
- 所有计算已归一化到 RVV VLEN=<N>bit, SEW=<S>bit

---

## 基准RVV实现分析

[Detailed breakdown of the RVV baseline: loop structure, instruction count, register usage]

---

## 各平台对比分析

### 1. x86 AVX/AVX2

**核心特点**：
- [register file, key characteristics]

**高价值指令**：
| 指令 | 功能 | RVV现状 |
|------|------|---------|

**收益分析**：
[Code comparison with instruction counts, normalized to RVV VLEN]

**建议扩展**：
- [proposed instructions with semantics]

---

### 2. ARM NEON/SVE

[Same structure as x86]

---

### 3. LoongArch LASX/LSX

[Same structure]

---

### 4. Power VSX (POWER10)

[Same structure]

---

### 5. S390X Z-Vector

[Same structure]

---

### 6. WASM SIMD

[Same structure]

---

## RVV扩展指令建议详细说明

### [P0] [Instruction name]

**指令定义**：
[format, semantics, constraints]

**应用场景**：
[where this instruction helps]

**性能对比**：
[before/after code with instruction counts]

---

### [P1] [Instruction name]

[Same structure]

---

## BBV热点加权收益分析

<!-- When BBV data IS available -->
### BBV热点分布

| 排名 | BB地址 | 所在函数 | 指令数 | 执行次数 | 执行权重 | 执行占比 |
|------|--------|----------|--------|----------|----------|----------|
| 1    | 0x...  | func     | N      | M        | W        | X%       |

### 各扩展指令收益链

| 扩展指令 | 目标BB | 原指令数 | 新指令数 | BB内减少 | BB占比 | 整体收益 |
|----------|--------|----------|----------|----------|--------|----------|
| vmulacc.vv | K循环BB | 12 | 7 | -41.7% | 16.0% | -6.7% |

### 累计收益估算

- 各扩展指令累计整体收益上限：Σ = X%
- 注：假设各扩展指令影响的BB无重叠

<!-- When BBV data is NOT available -->
<!-- 无BBV profiling数据，无法计算整体收益。 -->
<!-- 上表中的"BB内收益"仅反映单个BB范围内的指令减少比例。 -->
<!-- 建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算整体收益。 -->

---

## 附录

### FMA指令对比表
### 数据重排指令对比表
### 加载/存储指令对比表

---

## 结论

[Summary of findings, prioritized recommendations]

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1   |           |        |      |
| R2   |           |        |      |

最终审查结论：[reviewer confirmation]
```
