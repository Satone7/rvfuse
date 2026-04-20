# gemv_q8_0_16x1_q8_0 RVV扩展指令收益综合分析

## 数据来源

| 数据项 | 来源 |
|-------|------|
| 函数执行占比 | `docs/report/llama_cpp_perf_q4_v_analysis.md` |
| 指令延迟 | `docs/reference/cx/instruction-constraints-and-latency.md` |
| 扩展指令方案 | `docs/report/llama.cpp/rvv-gap-analysis-gemv_q8_0_16x1_q8_0-2026-04-17.md` |

---

## 一、热点函数执行占比（关键数据）

基于 llama.cpp Q4_0 向量化推理性能分析，**放大后占比**：

| 函数 | 占比 | 说明 |
|------|------|------|
| `ggml_gemv_q4_0_16x1_q8_0` | **40.68%** | Q4_0量化 GEMV（主要热点） |
| `ggml_gemv_q8_0_16x1_q8_0` | **24.06%** | Q8_0量化 GEMV（本次分析目标） |
| `ggml_gemm_q4_0_16x1_q8_0` | 8.05% | Q4_0量化 GEMM |

**GEMV/GEMM 总计**: **72.79%**

---

## 二、K-loop 周期级分析（基于指令延迟）

### 当前 RVV 实现每迭代周期数

| 操作 | RVV指令 | 延迟(cycles) | 说明 |
|------|---------|-------------|------|
| 加载 b向量 | `vle8_v_i8mf2` | **3** | 加载16个int8 |
| Scalar×Vector widening乘法 | `vwmul_vx_i16m1` | **4+5 = 9** | SEW=8, `.vx`形式需标量广播 |
| Widening累加 | `vwadd_wv_i32m2` | **4** | int16→int32 |
| **每迭代总计** | | **16 cycles** | |

**K-loop 32迭代**: 32 × 16 = **512 cycles**
**Scale处理(4条)**: ~20 cycles (vle16:3 + vfwmul:4 + vfcvt:4 + vfmacc:5 + 依赖开销)
**每 l-block 总计**: ~**532 cycles**

### 使用 vdot_lane.vx 的优化版本

假设新指令 `vdot_lane.vx` 执行延迟为 **5-6 cycles**（参考 `vwmacc.vx` 的 4+5 和 `vdot4a` 的预估延迟）：

| 操作 | 指令 | 延迟(cycles) | 说明 |
|------|------|-------------|------|
| 加载 b向量 | `vle8_v_i8mf2` | 3 | 不变 |
| Lane-indexed dot | `vdot_lane.vx` | **~6** | 替代 vwmul.vx + vwadd.wv |
| **每迭代总计** | | **~9 cycles** | |

**K-loop 32迭代**: 32 × 9 = **288 cycles**
**每 l-block 总计**: ~**308 cycles**

### 周期级收益计算

```
K-loop BB内周期减少 = (512 - 288) / 512 = 43.75%
```

**注**: 周期级收益高于指令数收益(33%)，因为 `vwmul.vx` 的 9 cycles 被 `vdot_lane.vx` 的 6 cycles 替代。

---

## 三、整体收益估算（基于热点占比）

### 方法：热点权重 × BB内收益

```
整体收益 = 函数执行占比 × K-loop BB内周期减少比例
```

#### 针对 gemv_q8_0_16x1_q8_0

| 指标 | 值 |
|------|------|
| 函数执行占比 | **24.06%** |
| K-loop周期减少 | **43.75%** |
| **整体收益** | **24.06% × 43.75% = 10.53%** |

#### 针对 gemv_q4_0_16x1_q8_0（结构相似）

`gemv_q4_0_16x1_q8_0` 有相似的 K-loop 结构（反量化 + vec_dot），若应用相同优化：

| 指标 | 值 |
|------|------|
| 函数执行占比 | **40.68%** |
| K-loop周期减少 | **43.75%**（假设相似） |
| **整体收益** | **40.68% × 43.75% = 17.80%** |

### 累计整体收益

| 扩展指令应用范围 | 整体收益 |
|-----------------|----------|
| 仅 gemv_q8_0 | **10.53%** |
| 仅 gemv_q4_0 | **17.80%** |
| **两者合计** | **28.33%** |

---

## 四、各方案收益对比表

| 优先级 | 扩展指令 | 函数应用 | BB内周期减少 | 函数占比 | 整体收益 |
|--------|----------|----------|-------------|----------|----------|
| **P0** | `vdot_lane.vx` | gemv_q8_0 | 43.75% | 24.06% | **10.53%** |
| **P0** | `vdot_lane.vx` | gemv_q4_0 | 43.75%* | 40.68% | **17.80%** |
| **P0** | `vdot4ss.vv` | gemv_系列 | ~50-60%* | 64.74% | **32-39%** |
| **P1** | `vwmacc_lane.vx` | 辅助优化 | ~15% | 64.74% | **~9.7%** |
| **P2** | `vwmaccwev/wod.vv` | 减少依赖链 | ~10% | 64.74% | **~6.5%** |

*注: gemv_q4_0 和 vdot4ss.vv 的周期减少为估算值，需进一步反汇编分析验证。

---

## 五、指令延迟对比分析

### 当前 vs 优化后关键指令延迟

| 操作类型 | 当前方案 | 延迟 | 优化方案 | 延延 | 减少 |
|---------|---------|------|---------|------|------|
| int8×int8 wid. MAC | vwmul.vx + vwadd.wv | 9+4=13 | vdot_lane.vx | ~6 | **-54%** |
| int8×int8 wid. MAC (vv) | vwmul.vv + vwadd.wv | 4+4=8 | vdot4ss.vv | ~5 | **-38%** |
| int16×int16 MAC | vwmacc.vv | 5 | vwmaccwev/wod | ~4+4 | 取决于并行度 |

### 关键观察

1. **`.vx` 形式延迟显著高于 `.vv` 形式**:
   - `vwmul.vx`: 4+5 = 9 cycles（需标量广播）
   - `vwmul.vv`: 4 cycles（SEW=8）
   
   这验证了 lane-indexed 指令的价值——消除标量广播开销。

2. **widening 操作链延迟**:
   - `vwmul.vx` (9) + `vwadd.wv` (4) = 13 cycles
   - 若有 `vwmacc.vx` 直接累加，可减少至 ~9 cycles
   - `vdot_lane.vx` 进一步优化至 ~6 cycles

---

## 六、实际整体收益估算

### 考虑实际情况的调整因素

| 因素 | 影响 | 说明 |
|------|------|------|
| K-loop占函数比例 | ~90% | Scale处理占~10%，需按比例调整 |
| gemv调用频率波动 | ±5% | 不同模型/token长度下占比变化 |
| 指令融合实现延迟 | 待测量 | 实际硬件实现可能有差异 |

### 调整后的整体收益估算

```
实际整体收益 = 函数占比 × K-loop占比 × K-loop周期减少 × (1 - 不确定性)

gemv_q8_0: 24.06% × 90% × 43.75% = 9.47% (±1.5%)
gemv_q4_0: 40.68% × 90% × 43.75% = 16.07% (±2.5%)

累计整体收益: ~25.5% (±4%)
```

---

## 七、结论

### 整体收益汇总

| 扩展指令 | 应用范围 | 整体收益 | 实现难度 |
|----------|----------|----------|----------|
| **vdot_lane.vx** | gemv_q8_0 + gemv_q4_0 | **~25.5%** | 高（需lane-indexed设计） |
| **vdot4ss.vv** | gemv系列(需数据重排) | **~32-39%** | 中 |
| vwmacc_lane.vx | 辅助优化 | ~9.7% | 中 |
| vwmaccwev/wod.vv | 减少依赖链 | ~6.5% | 中 |

### 优先级建议（基于整体收益）

1. **最高优先级**: `vdot_lane.vx` — 整体收益 ~25.5%，直接应用于现有代码结构
2. **次高优先级**: `vdot4ss.vv` — 整体收益 ~32-39%，但需数据布局调整
3. **辅助优先级**: `vwmacc_lane.vx` / `vwmaccwev/wod.vv` — 收益较小

### 关键发现

- **周期级收益高于指令数收益**: 43.75% vs 33%，因为 `.vx` 形式指令延迟显著高于 `.vv` 形式
- **gemv_q4_0 收益更高**: 占比40.68%，是优化重点
- **累计整体收益可达 25-30%**: 若同时优化 gemv_q4_0 和 gemv_q8_0

---

## 附录：延迟数据参考

### K-loop 核心指令延迟

| 指令 | 延延 | 来源 |
|------|------|------|
| vle8.v | 3 | instruction-constraints-and-latency.md |
| vwmul.vx (SEW=8) | 4+5 | instruction-constraints-and-latency.md |
| vwmul.vv (SEW=8) | 4 | instruction-constraints-and-latency.md |
| vwadd.wv | 4 | instruction-constraints-and-latency.md |
| vwmacc.vx | 4+5 | instruction-constraints-and-latency.md |
| vwmacc.vv | 5 | instruction-constraints-and-latency.md |

### Scale 处理指令延迟

| 指令 | 延延 |
|------|------|
| vle16.v | 3 |
| vfwmul.vf | 4 |
| vfcvt.f.x.v | 4 |
| vfmacc.vv | 5 |