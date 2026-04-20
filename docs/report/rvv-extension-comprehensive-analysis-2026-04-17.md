# RVV扩展指令综合分析报告（llama.cpp + YOLO）

> **数据来源**：
> - llama.cpp: `docs/report/llama_cpp_perf_q4_v_analysis.md`, `docs/report/llama.cpp/rvv-gap-analysis-gemv_q8_0_16x1_q8_0-2026-04-17.md`
> - YOLO: `docs/report/yolo/rvv-extension-analysis-report.md`, `docs/report/yolo/multi-platform-vector-comparison.md`
> - 指令延迟: `docs/reference/cx/instruction-constraints-and-latency.md`

---

## 一、热点分布对比

### 1.1 llama.cpp 推理热点

| 函数 | 占比 | 计算类型 |
|------|------|---------|
| `ggml_gemv_q4_0_16x1_q8_0` | **40.68%** | INT8量化 GEMV（decode阶段） |
| `ggml_gemv_q8_0_16x1_q8_0` | **24.06%** | INT8量化 GEMV（K-quant中间层） |
| `ggml_gemm_q4_0_16x1_q8_0` | 8.05% | INT8量化 GEMM（prefill阶段） |
| `repack_q4_0_to_q4_0_16_bl` | 11.73% | 数据重打包 |
| 其他算子 | ~15% | Norm/RoPE/Attention等 |

**GEMV/GEMM 总计**: **72.79%**（核心计算）

### 1.2 YOLO 推理热点

| 算子 | 占比 | 计算类型 |
|------|------|---------|
| Conv/GEMM | **35%** | INT8 MAC密集（量化推理） |
| Conv/GEMM | 35% | FP32 MAC密集（部分层） |
| BatchNorm | 15% | 元素级运算 |
| Activation | 10% | SiLU/ReLU等 |
| 其他 | 15% | 数据搬运/Concat |

**Conv/GEMM 总计**: **70%**（核心计算）

---

## 二、扩展指令方案汇总（区分来源应用）

### 2.1 方案总表

| 指令名称 | 来源应用 | 计算类型 | 解决的问题 | 整体收益 |
|----------|----------|----------|-----------|----------|
| **vsegdot.vv** | YOLO | INT8 GEMM | 分段规约缺失（4×i8→i32） | **YOLO: 1.5-1.7×** |
| **vdot_lane.vx** | llama.cpp | INT8 GEMV | Lane-indexed dot消除标量广播 | **llama.cpp: 25.5%** |
| **vmulacc.vv** | YOLO | FP32 GEMM | 矩阵外积（4×4 MAC） | **YOLO: 1.1-1.4×** |
| **vfmacc.vv_lane** | 通用 | FP32 MAC | Lane-indexed FMA | **YOLO: 28%, llama.cpp: 辅助** |
| **vdot4ss.vv** | llama.cpp | INT8 GEMV | Signed×Signed直接点积 | **llama.cpp: 32-39%** |
| **vunzip/vzip.vv** | YOLO | 数据重排 | 奇偶分离/合并（转置） | **YOLO: 边际收益** |
| **vwmaccwev/wod.vv** | llama.cpp | Widening MAC | Even/Odd分离减少依赖链 | **llama.cpp: ~6.5%** |

### 2.2 跨应用通用方案

以下指令对两个应用均有收益：

| 指令 | llama.cpp收益 | YOLO收益 | 说明 |
|------|--------------|---------|------|
| `vsegdot.vv` / `vdot4ss.vv` | **32-39%**（gemv系列） | **1.5-1.7×**（INT8 Conv） | 功能等效，命名差异 |
| `vfmacc.vv_lane` / `vdot_lane.vx` | **25.5%**（gemv） | **28%**（FP32 GEMM K-loop） | Lane-indexed思想通用 |

---

## 三、INT8 路径方案详解

### 3.1 核心瓶颈：分段规约缺失

**问题本质**（来自YOLO分析）：

```
x86 VNNI vpdpbusd（512-bit）：
  输入：64个int8 → 分成16组，每组4个
  输出：16个独立int32结果（并行产出）

ARM SDOT（128-bit）：
  输入：16个int8 → 分成4组，每组4个
  输出：4个独立int32结果

RVV vredsum：
  输入：VL个int8
  输出：仅1个int32标量（串行瓶颈）
```

**对GEMM数据流的影响**：

| 架构 | 每次加载产出的C元素数 | Store效率 |
|------|---------------------|----------|
| x86 VNNI | 8个int32/指令 | SIMD批量写回 |
| ARM SDOT | 4个int32/指令 | SIMD批量写回 |
| RVV（无分段规约） | 1个int32 | 串行scalar写回 |

### 3.2 方案A：vsegdot.vv（YOLO来源）

**指令定义**：

```
vsegdot.vv vd, vs1, vs2, seg=4
  输入：
    vs1 = vint8m1_t [a0 a1 a2 a3 | a4 a5 a6 a7 | ...]（直接int8向量）
    vs2 = vint8m1_t [b0 b1 b2 b3 | b4 b5 b6 b7 | ...]
  输出：
    vd = vint32mf2_t（元素数为VL/4）
    vd[0] += a0*b0 + a1*b1 + a2*b2 + a3*b3
    vd[1] += a4*b4 + a5*b5 + a6*b6 + b7*b7
```

**YOLO收益量化**：

| 操作 | RVV外积法（现状） | vsegdot方案 | x86 VNNI |
|------|------------------|-------------|---------|
| 加载A | 32次broadcast | 1次vle8 | 1次broadcast |
| 加载B | 32次vle8 | 8次vle8 | 1次vmovdqu |
| 计算 | 32次vwmacc | 8次vsegdot | 1次vpdpbusd |
| 产出C元素 | 逐行产出 | 8个int32/指令 | 8个int32/指令 |
| **总周期** | ~256 | ~72 | ~8（归一化相当） |

**YOLO整体推理加速**：

```
YOLO推理热点分布：
  Conv/GEMM（INT8）：35% 热点

整体推理加速计算：
  若INT8 GEMM提升3.5×：
    整体加速 = 1 + 0.35 × (3.5 - 1) ≈ 1.87x

考虑混合精度开销（约20%）：
  实际整体加速 ≈ 1.5-1.7×
```

### 3.3 方案B：vdot_lane.vx + vdot4ss.vv（llama.cpp来源）

**vdot_lane.vx 定义**（来自ARM NEON `vdotq_laneq_s32`）：

```
vdot_lane.vx vd, vs2, vs1, lane_idx
  语义：对于每个i (0 <= i < VL/4)：
    ; 从vs1选择lane_idx对应的4个int8元素
    vd.w[i] += vs2.b[4i]×vs1.b[lane_base] + ... + vs2.b[4i+3]×vs1.b[lane_base+3]
```

**llama.cpp收益量化**（基于指令延迟表）：

| 操作 | 当前RVV | 延迟 | 优化方案 | 延迟 | 减少 |
|------|---------|------|---------|------|------|
| 加载b向量 | vle8.v | 3 | vle8.v | 3 | 不变 |
| Scalar×Vector wid.乘 | vwmul.vx | **9** | — | — | — |
| Widening累加 | vwadd.wv | **4** | vdot_lane.vx | **~6** | **13→6** |
| **每迭代总计** | — | **16** | — | **~9** | **43.75%** |

**llama.cpp整体收益**：

```
整体收益 = 函数占比 × K-loop占比 × K-loop周期减少

gemv_q8_0: 24.06% × 90% × 43.75% = 9.47%
gemv_q4_0: 40.68% × 90% × 43.75% = 16.07%

llama.cpp累计整体收益: ~25.5%
```

### 3.4 Zvdot4a8i 扩展现状分析

**已存在的分段点积指令**（需`zvdot4a8i`扩展）：

```c
vint32m1_t __riscv_vdot4a_vv_i32m1(vint32m1_t vd, vuint32m1_t vs2, vuint32m1_t vs1, size_t vl);
```

**与理想方案的差距**：

| 对比项 | Zvdot4a8i（现状） | vsegdot/vdot4ss（提议） |
|--------|-------------------|------------------------|
| 输入格式 | `vuint32`（打包32位） | 直接`vint8`向量 |
| Signed×Signed | 仅 unsigned variants | 支持 signed×signed |
| 数据预处理 | 需4个int8打包成1个32位 | 无需预处理 |
| 与现有框架兼容 | 需重写CopyPackB逻辑 | 可直接复用ARM模式 |

**结论**：Zvdot4a8i 功能接近但格式不兼容，仍需设计直接int8输入的新指令。

---

## 四、FP32 路径方案详解

### 4.1 方案A：vmulacc.vv（YOLO来源）

**指令定义**：

```
vmulacc.vv vd, vs1, vs2
  功能：vd[4×4矩阵] += vs1[0..3] × vs2[0..3]^T（外积）
  
  等效操作：16个乘累加
    vd[0,0] += vs1[0] × vs2[0]
    vd[0,1] += vs1[0] × vs2[1]
    ...
    vd[3,3] += vs1[3] × vs2[3]
```

**YOLO收益量化**（FP32 SGEMM，处理2行×16列，K=4步）：

| 实现方式 | 指令数 | 周期数 |
|---------|--------|--------|
| RVV vfmacc×16 | 16条 | 16×5 = 80周期 |
| vmulacc×4 | 4条 | 4×12(预估) = 48周期 |

**收益来源**：
- 指令数减少：75%
- 寄存器压力：专用accumulator释放向量寄存器
- B加载共享：4行A共享同一次B加载

**YOLO整体收益**：
- FP32 GEMM提升：**1.3-2.0×**
- 整体推理加速：**1.1-1.5×**

### 4.2 方案B：vfmacc.vv_lane（通用）

**指令定义**：

```
vfmacc.vv_lane vd, vs1, vs2, imm[5-bit]
  功能：vd[i] += vs2[i] × vs1[imm]  (i = 0..VL-1)
        广播vs1的第imm个元素，与vs2逐元素乘加
```

**YOLO收益**（FP32 SGEMM K循环）：

| 实现方式 | 指令序列 | 指令数 |
|---------|---------|--------|
| RVV现状 | flw×4 + vfmacc.vf×4 | 8条 |
| Lane-indexed | vle32×1 + vfmacc_lane×4 | 5条 |

- 指令数减少：**37.5%**
- 周期数减少：**~28%**

**llama.cpp收益**（作为辅助优化）：
- gemv Scale处理中可替代 `vfwmul.vf` + `vfmacc.vv` 序列
- 整体收益约 **9.7%**

---

## 五、数据重排方案

### 5.1 vunzip/vzip.vv（YOLO来源）

**指令定义**：

```
vunzip.vv vd_even, vd_odd, vs2
  功能：将vs2奇偶元素分离
  输出：vd_even = [e0, e2, e4, ...], vd_odd = [e1, e3, e5, ...]

vzip.vv vd, vs1_even, vs2_odd
  功能：将两向量交错合并
  输出：vd = [e0, o0, e1, o1, ...]
```

**收益量化**（矩阵4×4转置）：

| 实现方式 | 指令数 |
|---------|--------|
| RVV vrgather | 6-8条（含索引向量准备） |
| vunzip/vzip | 4条 |

**应用场景**：
- 矩阵转置：**25-40%加速**
- INT8 PackB预处理：**2-4×加速**
- 对YOLO整体：边际收益（非热点）

### 5.2 vwmaccwev/wod.vv（llama.cpp来源）

**指令定义**：

```
vwmaccwev.vv vd, vs2, vs1  ; even positions
  vd.h[j] += vs2.b[2j] × vs1.b[2j]

vwmaccwod.vv vd, vs2, vs1  ; odd positions
  vd.h[j] += vs2.b[2j+1] × vs1.b[2j+1]
```

**收益**：
- 减少依赖链深度
- llama.cpp整体收益：**~6.5%**

---

## 六、整体收益汇总

### 6.1 按应用分列

| 应用 | 扩展组合 | 整体收益 |
|------|---------|----------|
| **llama.cpp** | vdot_lane.vx（gemv_q4+q8） | **25.5%** |
| **llama.cpp** | vdot4ss.vv（gemv系列） | **32-39%** |
| **llama.cpp** | vwmacc_lane.vx + wev/wod | **~16%**（辅助） |
| **YOLO** | vsegdot.vv（INT8路径） | **1.5-1.7×** |
| **YOLO** | vmulacc + lane-FMA（FP32路径） | **1.1-1.4×** |
| **YOLO** | 全套（含vzip） | **1.9-2.5×** |

### 6.2 跨应用累计收益估算

若同时应用于两个应用（假设相同硬件平台）：

| 场景 | 收益计算 |
|------|---------|
| llama.cpp推理优化 | **25-40%**加速 |
| YOLO推理优化 | **1.5-2.5×**加速 |
| **综合提升** | 两个应用均受益于INT8分段点积指令 |

---

## 七、与主流架构对比

### 7.1 扩展后性能定位

| 指标 | x86 VNNI | ARM SDOT | RVV现状 | RVV扩展后 |
|------|---------|---------|---------|----------|
| INT8 GEMM吞吐 | 1.0× | 0.8-1.0× | **0.3-0.5×** | **0.85-1.0×** |
| FP32 GEMM吞吐 | 1.0× | 0.9× | 0.6× | **0.85-1.0×** |
| 矩阵转置效率 | 高 | 高 | 低 | 中 |
| Lane-indexed操作 | 有 | 有 | **无** | 有 |

### 7.2 关键差距填补

| 差距类型 | RVV现状 | 提议方案 | 填补效果 |
|---------|---------|---------|---------|
| 分段规约（INT8） | 仅vredsum（单输出） | vsegdot/vdot4ss | **达到VNNI水平** |
| Lane-indexed | 仅.vx标量广播 | vfmacc_lane/vdot_lane | **消除广播开销** |
| 矩阵外积 | 需16条vfmacc | vmulacc（1条） | **减少75%指令** |

---

## 八、指令延迟对比（关键数据）

基于 `docs/reference/cx/instruction-constraints-and-latency.md`：

### 8.1 当前RVV关键指令延迟

| 指令 | 延迟 | 说明 |
|------|------|------|
| vle8.v | 3 | 加载 |
| vwmul.vx (SEW=8) | **4+5 = 9** | 标量广播开销 |
| vwmul.vv (SEW=8) | **4** | 向量乘法 |
| vwadd.wv | 4 | Widening累加 |
| vwmacc.vx | 4+5 | Widening MAC（scalar） |
| vwmacc.vv | 5 | Widening MAC（vector） |
| vfmacc.vv | 5 | FP32 FMA |
| vredsum.vs | 4 | 规约（单输出） |

### 8.2 提议指令预估延迟

| 指令 | 预估延迟 | 依据 |
|------|---------|------|
| vsegdot.vv | ~6 | 类似vwmacc.vv + 规约逻辑 |
| vdot_lane.vx | ~6 | 类似vwmacc.vx（消除广播） |
| vmulacc.vv | ~12 | 16个FMA并行（硬件复杂度高） |
| vfmacc.vv_lane | ~5 | 类似vfmacc.vf |

**关键发现**：`.vx`形式延迟(9 cycles)显著高于`.vv`形式(4 cycles)，验证了lane-indexed指令的价值。

---

## 附录：方案命名对照表

| 功能描述 | llama.cpp命名 | YOLO命名 | 实质等效 |
|---------|--------------|---------|---------|
| INT8分段点积（直接输入） | vdot4ss.vv | vsegdot.vv | ✓ 命名差异 |
| Lane-indexed dot/MAC | vdot_lane.vx | vfmacc.vv_lane | ✓ 类型差异(INT8/FP32) |
| 矩阵外积 | — | vmulacc.vv | YOLO特有 |
| 奇偶分离 | vwmaccwev/wod | vunzip/vzip | 功能不同 |