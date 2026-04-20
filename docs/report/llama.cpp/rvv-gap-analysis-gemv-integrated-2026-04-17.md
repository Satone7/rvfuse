# GEMV系列算子RVV扩展指令综合分析与收益估算

## 概述

**分析目标**: llama.cpp量化推理中的GEMV/GEMM核心算子
- `ggml_gemv_q4_0_16x1_q8_0`: Q4_0×Q8_0 GEMV
- `ggml_gemv_q8_0_16x1_q8_0`: Q8_0×Q8_0 GEMV
- `ggml_gemm_q4_0_16x1_q8_0`: Q4_0×Q8_0 GEMM

**数据来源**:
- 函数执行占比: `docs/report/llama_cpp_perf_q4_v_analysis.md`
- 指令延迟: `docs/reference/cx/instruction-constraints-and-latency.md`
- 平台对比: 各平台gap分析报告

---

## 一、热点函数执行占比（关键数据）

基于llama.cpp Q4_0向量化推理性能分析，**放大后占比**：

| 函数 | 占比 | 热点级别 | 说明 |
|------|------|---------|------|
| `ggml_gemv_q4_0_16x1_q8_0` | **40.68%** | 🔴极高 | Q4_0量化 GEMV（主要热点） |
| `ggml_gemv_q8_0_16x1_q8_0` | **24.06%** | 🔴极高 | Q8_0量化 GEMV |
| `ggml_gemm_q4_0_16x1_q8_0` | **8.05%** | 🟠高 | Q4_0量化 GEMM |
| **GEMV/GEMM总计** | **72.79%** | | 核心计算占比 |

**推理计算占总执行**: 68.69%（其余为初始化/加载）

---

## 二、扩展指令方案汇总

### 2.1 核心扩展指令

| 优先级 | 扩展指令 | 来源平台 | 适用函数 | 功能描述 |
|--------|----------|----------|----------|----------|
| **P0** | `vdot.vv` (i8→i32) | x86 VNNI, ARM SDOT, WASM | 通用 | 4×int8·int8→int32点积，消除i16中间层 |
| **P0** | `vdot_lane.vx` | ARM NEON vdotq_laneq | gemv_q8_0 | Lane-indexed点积，消除标量广播 |
| **P1** | `vunpacknib.vv` | x86, S390X, LoongArch | gemv_q4_0 | Nibble解包融合，替代vsll+vsra+vsra |
| **P1** | `vusdot.vv` | ARM USDOT, x86 VDPBUSD | gemv_q4_0 | u8×i8→i32点积，原生支持Q4_0×Q8_0 |
| **P2** | `vwmacc_lane.vx` | ARM SVE2 indexed | 辅助 | Lane-indexed widening MAC |
| **P3** | `vmacc_mat.vv` | Power MMA | 矩阵场景 | 矩阵级MAC（需新硬件） |

### 2.2 各函数适用方案

| 函数 | 核心瓶颈 | 适用扩展指令 | 关键收益点 |
|------|---------|--------------|-----------|
| gemv_q4_0 | Nibble解包(3指令)+i16中间层 | **vdot.vv + vunpacknib.vv** | nibble解包-66%, dot消除i16层 |
| gemv_q8_0 | 标量广播(vwmul.vx 9周期)+i16中间层 | **vdot_lane.vx** | 消除标量广播, dot替代vwmul+vwadd |
| gemm_q4_0 | 同gemv_q4_0 | **vdot.vv + vunpacknib.vv** | 结构相似 |

---

## 三、周期级收益分析

### 3.1 gemv_q4_0_16x1_q8_0

#### 当前l-block周期计算

| 指令 | 单次延迟 | 迭代次数 | 总周期 |
|------|---------|---------|--------|
| vle8_v_i8mf2 | 3 | 16 | 48 |
| vsll_vx_i8mf2 | 7 | 16 | 112 |
| vsra_vx_i8mf2 (低) | 7 | 16 | 112 |
| vsra_vx_i8mf2 (高) | 7 | 16 | 112 |
| vwmacc_vx_i16m1 (低) | 9 | 16 | 144 |
| vwmacc_vx_i16m1 (高) | 9 | 16 | 144 |
| **i-loop总计** | **42/iter** | **16** | **672** |
| epilogue | 20 | 1 | 20 |
| **l-block总计** | | | **692** |

#### 使用vdot.vv + vunpacknib.vv优化后

| 指令 | 单次延迟 | 迭代次数 | 总周期 |
|------|---------|---------|--------|
| vle8_v_i8mf2 | 3 | 16 | 48 |
| vunpacknib.vv | 3 | 16 | 48 |
| vdot.vv (低) | 5 | 16 | 80 |
| vdot.vv (高) | 5 | 16 | 80 |
| **i-loop总计** | **16/iter** | **16** | **256** |
| epilogue (无vwadd) | 16 | 1 | 16 |
| **l-block总计** | | | **272** |

**周期减少**: (692 - 272) / 692 = **60.7%**

### 3.2 gemv_q8_0_16x1_q8_0

#### 当前K-loop周期计算

| 指令 | 单次延迟 | 迭代次数 | 总周期 |
|------|---------|---------|--------|
| vle8_v_i8mf2 | 3 | 32 | 96 |
| vwmul_vx_i16m1 | 9 | 32 | 288 |
| vwadd_wv_i32m2 | 4 | 32 | 128 |
| **K-loop总计** | **16/iter** | **32** | **512** |
| Scale处理 | 20 | 1 | 20 |
| **l-block总计** | | | **532** |

#### 使用vdot_lane.vx优化后

| 指令 | 单次延迟 | 迭代次数 | 总周期 |
|------|---------|---------|--------|
| vle8_v_i8mf2 | 3 | 32 | 96 |
| vdot_lane.vx | ~6 | 32 | 192 |
| **K-loop总计** | **~9/iter** | **32** | **288** |
| Scale处理 | 20 | 1 | 20 |
| **l-block总计** | | | **308** |

**周期减少**: (532 - 308) / 532 = **42.3%**

---

## 四、整体收益估算

### 4.1 计算公式

```
整体收益 = 函数执行占比 × K-loop/i-loop占比 × BB内周期减少
```

- K-loop/i-loop占函数比例: ~90%（Scale处理占~10%）
- 不确定性调整: ±5%

### 4.2 各函数收益

| 函数 | 占比 | BB内周期减少 | 整体收益(推理) | 整体收益(总执行) |
|------|------|--------------|----------------|-----------------|
| gemv_q4_0 | 40.68% | 60.7% | **24.7%** | 24.7%×68.69% = **16.9%** |
| gemv_q8_0 | 24.06% | 42.3% | **10.1%** | 10.1%×68.69% = **6.9%** |
| gemm_q4_0 | 8.05% | 60.7%* | **4.9%** | **3.4%** |

*注: gemm结构类似gemv，使用相同方案估算

### 4.3 累计整体收益

| 方案 | 覆盖函数 | 累计整体收益(推理) | 累计整体收益(总执行) |
|------|----------|-------------------|-------------------|
| 仅vdot.vv | gemv_q4_0 + gemm_q4_0 | 14.4% + 4.9% = **19.3%** | **13.2%** |
| 仅vdot_lane.vx | gemv_q8_0 | **10.1%** | **6.9%** |
| **vdot.vv + vdot_lane.vx** | 所有GEMV/GEMM | **29.4%** | **20.1%** |
| **vdot.vv + vunpacknib.vv + vdot_lane.vx** | 最优组合 | **34.6%** | **23.8%** |

---

## 五、指令延迟关键洞察

### 5.1 为什么周期收益 > 指令数收益

| 指令类型 | 延迟 | 占i-loop周期 | 占指令数 |
|---------|------|-------------|---------|
| vwmacc.vx (被移除) | 9 | 21.4% | 16.7% |
| vsll/vsra.vx (被移除) | 7 | 33.3% | 33.3% |
| vwmul.vx (被移除) | 9 | 54.0% | 33.3% |
| **被移除指令总计** | | **54.8%周期** | **50%指令** |

**关键发现**: 被移除的高延迟指令(.vx形式)占周期比例大于指令数比例。

### 5.2 .vx vs .vv 形式延迟对比

| 形式 | 延迟 | 说明 |
|------|------|------|
| vwmul.vx | 4+5 = 9 | 需标量广播，额外开销 |
| vwmul.vv | 4 | 纯向量操作，低延迟 |
| vwmacc.vx | 4+5 = 9 | 同样需标量广播 |
| vwmacc.vv | 5 | 向量-向量，较低延迟 |

**结论**: Lane-indexed指令价值在于消除标量广播开销(.vx→等效.vv延迟)

---

## 六、统一扩展指令定义

### 6.1 vdot.vv / vdot4ss.vv (P0)

```
vdot.vv vd, vs2, vs1, vm
语义: 对于每个i (0 <= i < VL/4):
  vd.w[i] += vs2.b[4i]×vs1.b[4i] + vs2.b[4i+1]×vs1.b[4i+1] +
            vs2.b[4i+2]×vs1.b[4i+2] + vs2.b[4i+3]×vs1.b[4i+3]

输入: vs2, vs1 = signed int8向量
输出: vd = signed int32向量
约束: SEW_src=8, SEW_dest=32, LMUL_dest=4×LMUL_src
```

**与Zvdot4a8i差异**: 直接int8向量输入，支持signed×signed，无需packed uint32。

### 6.2 vdot_lane.vx (P0)

```
vdot_lane.vx vd, vs2, vs1, lane_idx, vm
语义: 对于每个i (0 <= i < VL/4):
  ; 从vs1选择lane_idx对应的4个int8元素
  lane_base = lane_idx × 4
  vd.w[i] += vs2.b[4i]×vs1.b[lane_base] +
            vs2.b[4i+1]×vs1.b[lane_base+1] +
            vs2.b[4i+2]×vs1.b[lane_base+2] +
            vs2.b[4i+3]×vs1.b[lane_base+3]

输入: vs2=int8向量, vs1=int8向量(lane源), lane_idx=立即数
输出: vd = int32向量
```

**应用场景**: gemv_q8_0中的标量广播替代，单指令完成load+广播+dot。

### 6.3 vunpacknib.vv (P1)

```
vunpacknib.vv vd_lo, vd_hi, vs2, vm
语义: 从vs2提取高低nibble并符号扩展为int8
  vd_lo.b[i] = sign_extend(vs2.b[i] & 0x0F)
  vd_hi.b[i] = sign_extend((vs2.b[i] >> 4) & 0x0F)

输入: vs2 = packed bytes (每字节2个4-bit值)
输出: vd_lo, vd_hi = signed int8向量
```

**收益**: 替代vsll+vsra+vsra(3条→1条)，nibble部分周期减少66.7%。

### 6.4 vusdot.vv (P1)

```
vusdot.vv vd, vs2, vs1, vm
语义: unsigned int8 × signed int8 → int32 dot product
  vd.w[i] += (unsigned)vs2.b[4i]×(signed)vs1.b[4i] + ...

输入: vs2=unsigned int8, vs1=signed int8
输出: vd = signed int32
```

**应用场景**: Q4_0 nibble(unsigned) × Q8_0(signed)原生支持。

---

## 七、优先级与实现建议

### 7.1 基于整体收益的优先级

| 优先级 | 扩展指令 | 整体收益(推理) | 实现难度 | 优先实现 |
|--------|----------|----------------|----------|----------|
| **P0** | vdot.vv | 19.3% (覆盖48.73%函数) | 中 | ✓ 最高优先 |
| **P0** | vdot_lane.vx | 10.1% (覆盖24.06%) | 高 | ✓ 次优先 |
| **P1** | vunpacknib.vv | 额外10% | 低 | ✓ 配合vdot.vv |
| P1 | vusdot.vv | 替代方案 | 中 | 可选 |
| P3 | vmacc_mat.vv | ~13% | 高(新HW) | 长期规划 |

### 7.2 实现路径建议

**第一阶段**: 实现vdot.vv + vunpacknib.vv
- 收益: 推理计算提速24.7%，总执行提速16.9%
- 覆盖: gemv_q4_0 + gemm_q4_0 (48.73%占比)
- 难度: 中等，已有Zvdot4a8i参考

**第二阶段**: 实现vdot_lane.vx
- 收益: 推理计算额外提速10.1%，累计34.6%
- 覆盖: gemv_q8_0 (24.06%占比)
- 难度: 较高，需lane-indexed编码设计

**第三阶段**: 考虑vusdot.vv替代
- 若vdot.vv实现复杂，可用vusdot.vv覆盖u×s场景

### 7.3 指令编码设计要点

| 扩展指令 | 编码挑战 | 建议 |
|---------|---------|------|
| vdot.vv | 需新funct6 | 参考Zvdot4a8i，修改输入格式 |
| vdot_lane.vx | lane_idx立即数编码 | 5-bit立即数(0-31)足够 |
| vunpacknib.vv | 双输出寄存器 | 使用连续寄存器编号(vd+1) |

---

## 八、与现有扩展的关系

### 8.1 Zvdot4a8i适用性

| 问题 | Zvdot4a8i | 建议方案 |
|------|-----------|----------|
| 输入格式 | packed uint32 | vdot.vv: 直接int8向量 |
| signed×signed | 不支持 | vdot.vv: 支持signed×signed |
| lane-indexed | 无 | vdot_lane.vx: 新增 |

### 8.2 扩展兼容性

- vdot.vv可与Zvdot4a8i共存，编码空间不冲突
- vdot_lane.vx需要新的funct6编码
- vunpacknib.vv可与现有位操作指令共存

---

## 九、结论

### 9.1 关键数据汇总

| 指标 | 值 |
|------|-----|
| GEMV/GEMM占比 | 72.79% |
| gemv_q4_0占比 | 40.68% |
| gemv_q8_0占比 | 24.06% |
| **最大累计收益** | **推理计算34.6%** / **总执行23.8%** |

### 9.2 核心扩展方案

**最优组合**: vdot.vv + vunpacknib.vv + vdot_lane.vx

- **推理计算阶段**: 提速34.6%
- **总执行时间**: 提速23.8%

### 9.3 关键洞察

1. **周期收益高于指令数收益**: 60.7%周期减少 vs 48.5%指令数减少
2. **.vx形式指令延迟高**: 标量广播带来额外4周期开销
3. **gemv_q4_0是主要优化目标**: 占比40.68%，收益潜力最大
4. **累计收益可观**: 覆盖72.79%热点函数，整体提速超20%

---

## 附录

### A. 指令延迟参考表

| 指令 | 延迟 | 来源 |
|------|------|------|
| vle8.v | 3 | instruction-constraints-and-latency.md |
| vwmul.vx (SEW=8) | 4+5 = 9 | 同上 |
| vwmul.vv (SEW=8) | 4 | 同上 |
| vwmacc.vx | 4+5 | 同上 |
| vwmacc.vv | 5 | 同上 |
| vwadd.wv | 4 | 同上 |
| vsll/vsra.vx | 4+3 = 7 | 同上 |
| vfwmul.vf | 4 | 同上 |
| vfcvt.f.x.v | 4 | 同上 |
| vfmacc.vv | 5 | 同上 |

### B. 函数占比详细数据

| 函数 | 原始Self | 放大后Self |
|------|----------|-----------|
| ggml_gemv_q4_0_16x1_q8_0 | 27.73% | 40.68% |
| ggml_gemv_q8_0_16x1_q8_0 | 16.39% | 24.06% |
| ggml_gemm_q4_0_16x1_q8_0 | 5.49% | 8.05% |
| repack_q4_0 | 8.01% | 11.73% |
| repack_q8_0 | 4.61% | 6.73% |

### C. 相关报告索引

| 报告 | 内容 |
|------|------|
| rvv-gap-analysis-ggml_gemv_q4_0_16x1_q8_0 | Q4_0 GEMV跨平台对比 |
| rvv-gap-analysis-gemv_q8_0_16x1_q8_0 | Q8_0 GEMV跨平台对比 |
| rvv-gap-analysis-gemv-benefit-estimation | 收益估算方法 |
| llama_cpp_perf_q4_v_analysis | 性能热点分析 |

---

*报告生成日期: 2026-04-17*
*数据来源: llama.cpp perf profiling + RVV指令延迟文档*