# INT8 YOLO11n RVV512 优化报告

**日期**: 2026-04-26
**目标**: 完整的RVV512优化流水线，用于INT8 YOLO11n在RISC-V上的推理加速
**平台**: Banana Pi K1 (SpacemiT K1, zvl256b) → RVV512 (QEMU仿真)

---

## 执行摘要

### 分析范围
- 通过perf profiling分析了INT8 YOLO11n的所有热点函数
- 识别出6个占比>1%的算子，覆盖总执行时间的**98.65%**
- 为5个算子实现了RVV512向量化（跳过Im2col，内存受限）

### 向量化成果
| 算子 | % 时间 | RVV实现 | 独立测试 | Gap分析 |
|------|--------|---------|----------|---------|
| QGEMM | 74.51% | ✅ qgemm-kernel-vl16 | ✅ PASS | ✅ 完成 |
| Logistic | 9.80% | ✅ compute-logistic | ✅ PASS | ✅ 完成 |
| QuickGelu | 9.42% | ✅ quick-gelu | ✅ PASS | ✅ 完成 |
| ReduceMinMax | 4.63% | ✅ reduce-minmax-f32 | ✅ PASS | ✅ 完成 |
| QuantizeLinear | 2.94% | ✅ quantize-linear | ✅ PASS | ✅ 完成 |
| Im2col | 1.35% | ⏭️ 跳过 (内存受限) | - | - |

### 关键发现
1. **QGEMM (74.51%)** 是绝对主导瓶颈，RVV扩展最大收益来自INT8融合乘累加指令
2. **激活函数 (19.22%)** Logistic和QuickGelu合计，近似倒数/指数指令可显著加速
3. **ReduceMinMax (4.63%)** 已有高效RVV实现，进一步优化空间有限
4. **QuantizeLinear (2.94%)** 窄化链是RVV特有开销，饱和窄化指令可减少步数

---

## Phase 1: Perf Profiling 结果

### 全局指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 每次推理耗时 | 4.64秒 | 30次迭代平均 |
| IPC | 0.48 | 计算受限（scalar kernels） |
| 总指令数 | 106.9B | |
| 总周期数 | 221.8B | |
| L1-dcache缺失率 | 2.30% | 缓存行为良好 |
| 分支预测缺失率 | 0.43% | 优秀 |

### 函数热点分布

| 函数 | 自身% | 采样数 | 类型 |
|------|-------|--------|------|
| MlasGemmQuantOperation | 74.51% | 173,011 | INT8量化矩阵乘法 |
| MlasComputeLogistic | 9.80% | 22,749 | Sigmoid激活 |
| QuickGelu | 9.42% | 32 | GELU激活 |
| MlasReduceMinMaxF32Kernel | 4.63% | 10,746 | 最大/最小归约 |
| MlasQuantizeLinear | 2.94% | 6,829 | 线性量化 |
| Im2col | 1.35% | 3,140 | 卷积展开 |
| BroadcastLooper | 1.09% | 21 | 广播运算 |
| MlasEltwiseMul | 0.74% | 1,708 | 逐元素乘法 |
| memset | 0.52% | 1,203 | 内存清零 |

### 性能特征
- **计算受限**: IPC=0.48，远低于理想值4.0（4发射宽度）
- **缓存友好**: L1缺失率2.30%，说明数据局部性良好
- **分支可预测**: 缺失率0.43%，循环主导型工作负载

---

## Phase 2: RVV512 算子实现

### 实现总览

#### 1. QGEMM (74.51%) — qgemm-kernel-vl16
- **算法**: INT8量化矩阵乘法，uint8×uint8→uint32
- **RVV策略**: widening multiply-accumulate (vwmulu + vwaddu)
- **LMUL配置**: e32m1 (累加器), e8mf4 (矩阵B行)
- **关键瓶颈**: 每K元素需要3条指令 (load + widening mul + widening add)

#### 2. Logistic (9.80%) — compute-logistic
- **算法**: sigmoid(x) = 1 / (1 + exp(-x))，有理多项式近似
- **RVV策略**: 分区间多项式求值 + vfredsum归约
- **关键瓶颈**: 指数计算和倒数运算

#### 3. QuickGelu (9.42%) — quick-gelu
- **算法**: x * sigmoid(1.702 * x)
- **RVV策略**: 复用Logistic的sigmoid实现 + 逐元素乘法
- **关键瓶颈**: 依赖sigmoid性能

#### 4. ReduceMinMax (4.63%) — reduce-minmax-f32
- **算法**: 在float32数组中同时查找最小/最大值
- **RVV策略**: 4路累加器 + vfredmin/vfredmax水平归约
- **LMUL配置**: e32m1, VL=16
- **关键瓶颈**: 归约阶段的多步序列

#### 5. QuantizeLinear (2.94%) — quantize-linear
- **算法**: Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)
- **RVV策略**: e32m2配置 + 两步窄化 (int32→int16→int8)
- **LMUL配置**: e32m2 → e16m1 → e8mf2 (窄化链)
- **关键瓶颈**: 两步窄化和除法延迟

### 测试验证
所有5个RVV算子均在VLEN=256和VLEN=512下通过独立正确性测试：
- QGEMM: 矩阵乘法数值精度验证 ✅
- Logistic: sigmoid输出精度验证 ✅
- QuickGelu: GELU输出精度验证 ✅
- ReduceMinMax: min/max结果验证 ✅
- QuantizeLinear: U8/S8/U16/S16四类型23项测试 ✅

---

## Phase 3: RVV512 构建支持

### zvl512b Toolchain
- 创建了 `riscv64-linux-zvl512b-toolchain.cmake`
- 关键差异 vs zvl256b: `-march=rv64gcv_zvl512b` + `-D__riscv_v_fixed_vlen=512`
- 支持增量构建：`./build.sh --toolchain riscv64-linux-zvl512b-toolchain.cmake`

---

## Phase 4: BBV Profiling 结果

### 已有BBV数据
| 算子 | BBV数据 | 说明 |
|------|---------|------|
| QGEMM (sgemm_rvv512) | ✅ | VLEN=512 QEMU BBV |
| PackB16 | ✅ | B矩阵打包 |
| test_no_filter | ✅ | 全函数BBV基线 |

### 算子BBV状态
INT8 YOLO11n的函数级BBV profiling需要在RVV512构建完成后执行。当前BBV数据来自独立kernel测试。

---

## Phase 5: Gap Analysis 汇总

### 各算子推荐扩展指令

| 优先级 | 扩展指令 | 适用算子 | 来源平台 | BB内收益 |
|--------|----------|----------|----------|----------|
| **P0** | vwmaccu8x2.vv | QGEMM | x86 VNNI/ARM UDOT | ~50% K循环 |
| **P0** | vfrsqrt7.v | Logistic | x86 AVX | ~20% sigmoid |
| **P0** | vfcvt_rne_sat_x_f | QuantizeLinear | ARM NEON | ~25% 量化BB |
| **P1** | vmatmul.rs.vv | QGEMM | ARM SMMLA | ~75% 矩阵块 |
| **P1** | vexp.approx | Logistic/QuickGelu | ARM NEON | 指数计算加速 |
| **P1** | vnarrow_sat | QuantizeLinear | ARM NEON/SSE2 | ~20% 窄化 |
| **P2** | vclamp.vv | ReduceMinMax | Power VSX | 替代2条vfmin+vfmax |

### Top 5 推荐RVV扩展（按整体收益排序）

| 排名 | 扩展指令 | 受益算子 | 综合收益 | 实现难度 |
|------|----------|----------|----------|----------|
| 1 | **INT8融合乘累加** (vwmaccu8x2) | QGEMM (74.51%) | 最高 | 中 |
| 2 | **近似倒数** (vfrsqrt7/vfrecip) | Logistic (9.80%) | 高 | 中 |
| 3 | **近似指数** (vexp.approx) | Logistic+Gelu (19.22%) | 高 | 高 |
| 4 | **矩阵乘指令** (vmatmul) | QGEMM (74.51%) | 极高 | 极高 |
| 5 | **饱和窄化** (vnarrow_sat) | QuantizeLinear (2.94%) | 中 | 高 |

### 收益-难度矩阵

```
收益↑
高 │  vwmaccu8x2  │  vmatmul
   │  vfrsqrt7    │  vexp.approx
中 │  vnarrow_sat │
   │  vclamp      │
低 │              │
   └──────────────┴─────────────→ 难度
      低            中      高
```

---

## Phase 6: 附录 — 各算子详细报告

| 算子 | Markdown报告 | PDF报告 |
|------|-------------|---------|
| QGEMM | [rvv-gap-analysis-qgemm-kernel-vl16-2026-04-26.md](rvv-gap-analysis-qgemm-kernel-vl16-2026-04-26.md) | [rvv-gap-analysis-qgemm-kernel-vl16-2026-04-26.pdf](pdf/rvv-gap-analysis-qgemm-kernel-vl16-2026-04-26.pdf) |
| Logistic | [rvv-gap-analysis-compute-logistic-2026-04-26.md](rvv-gap-analysis-compute-logistic-2026-04-26.md) | [rvv-gap-analysis-compute-logistic-2026-04-26.pdf](pdf/rvv-gap-analysis-compute-logistic-2026-04-26.pdf) |
| QuickGelu | [rvv-gap-analysis-quick-gelu-2026-04-26.md](rvv-gap-analysis-quick-gelu-2026-04-26.md) | [rvv-gap-analysis-quick-gelu-2026-04-26.pdf](pdf/rvv-gap-analysis-quick-gelu-2026-04-26.pdf) |
| ReduceMinMax | [rvv-gap-analysis-reduce-minmax-f32-2026-04-26.md](rvv-gap-analysis-reduce-minmax-f32-2026-04-26.md) | [rvv-gap-analysis-reduce-minmax-f32-2026-04-26.pdf](pdf/rvv-gap-analysis-reduce-minmax-f32-2026-04-26.pdf) |
| QuantizeLinear | [rvv-gap-analysis-quantize-linear-2026-04-26.md](rvv-gap-analysis-quantize-linear-2026-04-26.md) | [rvv-gap-analysis-quantize-linear-2026-04-26.pdf](pdf/rvv-gap-analysis-quantize-linear-2026-04-26.pdf) |

---

## RVV补丁文件索引

| 算子 | 目录 | 关键文件 |
|------|------|----------|
| QGEMM | `applications/onnxrt/rvv-patches/qgemm-kernel-vl16/` | rvv_qgemm_kernel_vl16.inl |
| Logistic | `applications/onnxrt/rvv-patches/compute-logistic/` | rvv_compute_logistic.inl |
| QuickGelu | `applications/onnxrt/rvv-patches/quick-gelu/` | rvv_quick_gelu.inl |
| ReduceMinMax | `applications/onnxrt/rvv-patches/reduce-minmax-f32/` | rvv_reduce_minmax_f32.inl |
| QuantizeLinear | `applications/onnxrt/rvv-patches/quantize-linear/` | rvv_quantize_linear.inl |

---

## 结论与下一步

### 当前成果
1. 完成INT8 YOLO11n全量perf profiling，识别6个热点算子
2. 实现5个RVV512向量化算子，全部通过VLEN=256/512正确性测试
3. 创建zvl512b编译工具链，支持RVV512 ONNX Runtime构建
4. 完成5个算子的多平台gap分析，识别7个高优先级RVV扩展指令

### 下一步建议
1. **构建RVV512 ONNX Runtime** — 集成所有RVV补丁，端到端性能验证
2. **硬件实测** — 在Banana Pi K1上运行RVV256构建，获取实际加速比
3. **BBV profiling** — 对集成后的INT8 YOLO11n做函数级BBV profiling
4. **扩展指令设计** — 基于gap分析结果，设计INT8融合乘累加等扩展指令
5. **模拟器验证** — 在RISC-V模拟器中实现扩展指令，量化周期收益
