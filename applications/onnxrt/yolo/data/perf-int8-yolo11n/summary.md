# INT8 YOLO11n Perf Profiling Report

**Date**: 2026-04-25  
**Platform**: Banana Pi K1 (SpacemiT K1, zvl256b)  
**Framework**: ONNX Runtime v1.24.4 (scalar kernels)  
**Model**: yolo11n_int8.ort  

## Global Metrics (perf stat)

| Metric | Value |
|--------|-------|
| Total time | 139.04 seconds (30 iterations) |
| Per-iteration | 4.64 seconds |
| Task-clock | 139,040.67 msec |
| IPC | 0.48 (compute-bound) |
| Instructions | 106.9B |
| Cycles | 221.8B |
| L1-dcache-loads | 22.2B |
| L1-dcache-miss rate | 2.30% |
| Branch-misses | 0.43% |

## Function Hotspots (perf report)

| Function | Self % | Samples | Notes |
|----------|--------|---------|-------|
| **MlasGemmQuantOperation** | 74.51% | 173,011 | INT8 QGEMM - primary target |
| **MlasComputeLogistic** | 9.80% | 22,749 | Sigmoid activation |
| **QuickGelu** | 9.42% | 32 | GELU activation (calls Logistic) |
| **MlasReduceMinMaxF32Kernel** | 4.63% | 10,746 | Min/max reduction |
| **MlasQuantizeLinear** | 2.94% | 6,829 | Quantization |
| **Im2col** | 1.35% | 3,140 | Im2col conversion |
| **BroadcastLooper** | 1.09% | 21 | Broadcasting ops |
| **MlasEltwiseMul** | 0.74% | 1,708 | Element-wise multiply |
| **memset** | 0.52% | 1,203 | libc memset |

## Operators for RVV512 Vectorization (>1%)

| Operator | % Time | Priority | Existing Patch? |
|----------|--------|----------|-----------------|
| QGEMM | 74.51% | P0 | ✅ qgemm-kernel-vl16 |
| Logistic | 9.80% | P1 | ✅ compute-logistic |
| QuickGelu | 9.42% | P1 | ✅ quick-gelu |
| ReduceMinMax | 4.63% | P2 | ❌ needs implementation |
| QuantizeLinear | 2.94% | P3 | ❌ needs implementation |
| Im2col | 1.35% | P4 | Skip (memory-bound) |

## Performance Analysis

### Compute vs Memory
- IPC = 0.48 indicates compute-bound workload
- L1 cache miss rate = 2.30% is low, good cache behavior
- Branch miss rate = 0.43% is excellent

### Optimization Opportunities
1. **QGEMM (74.51%)** - Largest opportunity, RVV512 could provide significant improvement
2. **Activations (19.22% combined)** - Logistic and QuickGelu are vectorizable
3. **Reduction (4.63%)** - Parallel min/max reduction benefits from RVV
4. **Quantization (2.94%)** - Vector scaling and clamping

## Files

| File | Description |
|------|-------------|
| `perf_stat.txt` | Global metrics |
| `perf_report.txt` | Function hotspots |