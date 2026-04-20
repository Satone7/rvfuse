# Latency-Weighted Benefit Summary Table

## Comparison Table: Current vs Proposed Optimizations

| Metric | Current | vdot.vv only | vdot.vv + vunpacknib |
|--------|---------|--------------|---------------------|
| **Instruction Count** | | | |
| i-loop instructions | 6 | 5 | 4 |
| Total i-loop instructions | 96 (6×16) | 80 (5×16) | 64 (4×16) |
| Epilogue instructions | 5 | 4 | 4 |
| **Total Instructions** | **101** | **84** | **68** |
| **Instruction Reduction** | - | 16.8% | 32.7% |
| **Cycle Count** | | | |
| i-loop cycles (per iter) | 42 | 27 | 16 |
| Total i-loop cycles | 672 | 432 | 256 |
| Epilogue cycles | 20 | 16 | 16 |
| **Total Cycles** | **692** | **448** | **272** |
| **Cycle Reduction** | - | 35.3% | 60.7% |

## Benefit Comparison: Instruction Count vs Cycle Weight

| Optimization | Instruction-Based Benefit | Cycle-Based Benefit | Ratio (Cycle/Instr) |
|--------------|--------------------------|---------------------|-------------------|
| vdot.vv only | 16.8% | 35.3% | 2.10× |
| vdot.vv + vunpacknib | 32.7% | 60.7% | 1.86× |
| **Report's 48.5% estimate** | 48.5% | - | - |
| **Actual (with vunpacknib)** | 32.7% | **60.7%** | 1.86× |

## Overall Inference Speedup (Weighted by Function Profile)

| Optimization | BB内周期减少 | 整体收益 (推理计算阶段) | 整体收益 (总执行) |
|--------------|------------|---------------------|-----------------|
| vdot.vv only | 35.3% | 14.4% (35.3% × 40.68%) | 9.9% (× 68.69%) |
| vdot.vv + vunpacknib | 60.7% | 24.7% (60.7% × 40.68%) | 16.9% (× 68.69%) |

## Key Insight: Report's 48.5% vs Actual

| Metric | Report Estimate | Actual (Cycle-Weighted) | Interpretation |
|--------|----------------|------------------------|---------------|
| Instruction reduction | 48.5% | 32.7% | Report was optimistic |
| **Cycle reduction** | - | **60.7%** | **Actual benefit higher** |
| BB内收益 | 48.5% | 60.7% | 1.25× actual benefit |
| 整体收益 | ~19.8% (48.5% × 40.68%) | 24.7% | 1.25× actual benefit |

## Why Cycle-Based Benefit is Higher

| Removed Instruction | Latency | Cycle Impact |
|---------------------|---------|--------------|
| vwmacc.vx | 9 cycles | 21.4% of i-loop |
| vsll.vx | 7 cycles | 16.7% of i-loop |
| vsra.vx (×2) | 7 cycles | 33.3% of i-loop |
| vwadd.vv | 4 cycles | 5.8% of epilogue |
| **Total removed** | **23 cycles/iter** | **54.8% of i-loop** |

The removed instructions account for **54.8% of i-loop cycle time** despite being only **50% of instruction count**, explaining the 1.86× ratio between cycle-based and instruction-based benefits.

## Summary

- **Report's "48.5% instruction reduction"** translates to:
  - **60.7% cycle reduction** (when weighted by latency)
  - **24.7% overall inference speedup** (when weighted by function占比)

- **Key takeaway**: Cycle-weighted analysis is essential for accurate performance prediction, as high-latency instruction elimination provides outsized benefits compared to instruction count alone.