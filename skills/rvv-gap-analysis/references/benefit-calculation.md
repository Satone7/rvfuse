# Benefit Calculation Methodology

This document defines how to calculate and report performance benefits for proposed RVV extensions.

## Core Principles

1. **Normalize to equivalent workloads**: Compare instruction counts for processing the same
   amount of data, not the same number of instructions per call.

2. **Account for register width differences**: If the reference platform uses 128-bit registers
   and RVV uses 512-bit, the RVV version processes 4× the elements per instruction.

3. **Report both raw and normalized figures**: Show the original platform's numbers and the
   RVV-normalized numbers side by side.

## Calculation Methods

### Method 1: Instruction Count Reduction

```
Instruction reduction = (original_count - proposed_count) / original_count × 100%
```

Example: ARM NEON uses 4 scalar loads + 4 FMA = 8 instructions for 4 K-steps.
With lane-indexed FMA on RVV VLEN=512: 1 vector load + 4 lane-FMA = 5 instructions.
Reduction = (8 - 5) / 8 = 37.5%

### Method 2: Throughput Normalization

When comparing across register widths:

```
Elements per instruction (platform) = register_width / SEW
Elements per instruction (RVV)      = VLEN / SEW

Normalized throughput ratio = elements_RVV / elements_platform

Effective instruction count (platform) = actual_instructions / normalized_throughput_ratio
```

Example: ARM NEON `vmlaq_lane_f32` processes 4 float32 (128-bit).
RVV VLEN=512 processes 16 float32.
Ratio = 16/4 = 4.
One NEON instruction = 1/4 RVV instruction in throughput terms.

### Method 3: Performance Multiplier Estimation

For architectural changes (new hardware units, accumulator registers):

```
Performance multiplier = (improved_throughput × improved_utilization) /
                         (baseline_throughput × baseline_utilization)
```

Factors to consider:
- Register pressure reduction → wider loop unrolling → better pipeline utilization
- Dedicated accumulator registers → free up vector registers → more accumulators
- Memory access pattern improvement → cache efficiency
- Pipeline depth and dependency chain reduction

Document all assumptions explicitly.

### Method 4: BBV-Weighted Overall Benefit (Preferred when BBV data is available)

This is the **primary method** for expressing benefits in the summary table when BBV data exists.
It produces a single "整体减少X%" figure that is directly comparable across all proposed extensions.

#### Data Sources

From `.bbv` file:
- Each BB's execution frequency (执行次数)

From `.disas` file:
- Each BB's instruction count and disassembled content

#### Calculation Chain

```
Step 1 — BB执行权重:
  BB执行权重 = BB执行次数 × BB指令数
  (This weights by both how often a BB runs AND how many instructions it has)

Step 2 — BB执行占比:
  BB执行占比 = BB执行权重 / Σ(所有BB执行权重)
  (What fraction of total dynamic instructions come from this BB)

Step 3 — BB内指令减少数:
  BB内指令减少数 = 原BB指令数 - 扩展后BB指令数
  (Concrete count from comparing the disassembled BB before/after the proposed extension)

Step 4 — BB内减少比例:
  BB内减少比例 = BB内指令减少数 / BB总指令数 × 100%

Step 5 — 整体收益 (the final figure for the summary table):
  整体收益 = BB内减少比例 × BB执行占比
```

#### Worked Example

BBV data for `MlasSgemmKernel`:
- K-loop BB (address 0x40a8c): 执行次数 = 2,400,000, 指令数 = 12
- 该BB执行权重 = 2,400,000 × 12 = 28,800,000
- 程序总执行权重 = 180,000,000
- 该BB执行占比 = 28,800,000 / 180,000,000 = 16.0%

Proposed extension `vfmacc.vv_lane`:
- 原BB指令数 = 12 (4 scalar loads + 4 vfmacc.vf + 4 pointer updates)
- 扩展后BB指令数 = 7 (1 vector load + 4 vfmacc.vv_lane + 2 pointer updates)
- BB内减少比例 = (12 - 7) / 12 = 41.7%
- **整体收益 = 41.7% × 16.0% = 6.7%**

Summary table entry: `整体减少6.7%`

#### Multiple BBs Affected by One Extension

If an extension affects multiple BBs (e.g., a K-loop BB and a transpose BB):

```
整体收益 = Σ(BB_i内减少比例 × BB_i执行占比)
```

#### Cumulative Benefit

Sum all extensions' overall benefits for an upper-bound estimate:

```
累计整体收益上限 = Σ(各扩展指令整体收益)
```

Note: this assumes no interaction between extensions (may overestimate if extensions
affect overlapping BBs). Document any known overlaps.

#### Without BBV Data

When no `.bbv` data is available, the summary table uses BB-scoped figures only:

```
| P1 | vfmacc.vv_lane | ARM NEON | BB内减少41.7%（K循环BB） |
```

Clearly label these as BB-scoped, not overall. Recommend running
`./tools/profile_to_dfg.sh` to obtain BBV data for overall estimation.

## Register Width Normalization Examples

### Example 1: ARM NEON → RVV VLEN=512

ARM NEON: 128-bit Q registers, 4 × float32
RVV: 512-bit V registers, 16 × float32 (SEW=32, LMUL=1)

For a K-loop iteration processing 4 elements:
- NEON: 1 load (4 elements) + 4 lane-FMA = 5 instructions
- RVV equivalent workload (16 elements, 4× more): 4 NEON iterations = 20 instructions
- RVV with lane-indexed FMA (16 elements): 1 load + 16 lane-FMA = 17 instructions
- Normalized reduction: (20 - 17) / 20 = 15%

### Example 2: Power VSX → RVV VLEN=512

Power VSX: 128-bit VSX registers, 4 × float32
POWER10 MMA: processes 4×4 matrix (16 FMA) in 1 instruction

For 2 rows × 16 columns × 4 K-steps:
- RVV baseline: 4 vfmacc × 2 rows × 2 (K-unroll=2) = 16 instructions, 64 FMA
- POWER10 MMA: 2 MMA × 4 (K-steps) = 8 instructions, 128 FMA (2× the work)
- Normalized to same workload (64 FMA): POWER10 needs 4 MMA instructions
- RVV needs 16 vfmacc for 64 FMA
- But MMA requires accumulator disassembly overhead

## Reporting Format

In the report, always show:

1. **Raw comparison**: Original platform numbers vs RVV numbers
2. **Normalization factor**: `VLEN_RVV / VLEN_platform = X`
3. **Normalized comparison**: After scaling
4. **Assumptions**: What was assumed (pipeline behavior, cache effects, etc.)

Example table row:

| 指令 | 平台寄存器宽度 | RVV VLEN | 归一化因子 | 原始指令数 | 归一化指令数 | 减少比例 |
|------|---------------|----------|-----------|-----------|-------------|---------|
| vmlaq_lane | 128-bit | 512-bit | 4× | 5 | 20→17 | 15% |
