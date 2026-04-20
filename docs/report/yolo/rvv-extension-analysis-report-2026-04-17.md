# RVV指令扩展方案综合分析与收益评估报告

> **数据来源**：
> - docs/report/yolo/multi-platform-vector-comparison.md（FP32 GEMM跨平台对比）
> - docs/report/yolo/onnxruntime_quantization_support_report.md（INT8量化推理分析）
> - docs/reference/cx/instruction-constraints-and-latency.md（指令延迟参考）
> - third_party/riscv-rvv-intrinsic-doc（RVV intrinsic手册验证）

---

## 一、核心发现：RVV的架构瓶颈

### 1.1 分段规约缺失——INT8推理的根本障碍

**问题本质**：x86 VNNI / ARM SDOT 的核心能力是将向量寄存器中的多个int8分组求和产出多个独立int32结果：

```
x86 VNNI vpdpbusd（512-bit寄存器）：
  输入：64个int8 → 分成16组，每组4个
  输出：16个独立int32结果（并行产出）

ARM SDOT（128-bit寄存器）：
  输入：16个int8 → 分成4组，每组4个
  输出：4个独立int32结果（并行产出）

RVV vredsum：
  输入：VL个int8
  输出：仅1个int32标量（串行瓶颈）
```

**对GEMM数据流的影响**：

| 架构 | 每次加载产出的C元素数 | Store效率 |
|------|---------------------|----------|
| x86 VNNI (256-bit) | 8个int32 | SIMD批量写回 |
| ARM SDOT (128-bit) | 4个int32 | SIMD批量写回 |
| RVV (无分段规约) | 1个int32 | 串行scalar写回 |

这意味着RVV需要比VNNI多**8倍迭代次数**才能算出相同数量的C元素。

### 1.2 验证发现的Zvdot4a8i扩展

**已存在的分段点积指令**（需要`zvdot4a8i`扩展）：

```c
vint32m1_t __riscv_vdot4a_vv_i32m1(vint32m1_t vd, vuint32m1_t vs2, vuint32m1_t vs1, size_t vl);
```

**语义**：将32位元素视为4个打包int8，执行分段点积。

**局限性分析**：

| 对比项 | x86 VNNI / ARM SDOT | Zvdot4a8i | 影响 |
|--------|---------------------|-----------|------|
| 输入格式 | 直接int8向量加载 | 需预打包成32位 | 额外预处理开销 |
| ISA状态 | 核心ISA | 非标准扩展 | 编译器/生态支持有限 |
| ONNX Runtime | 原生兼容 | 需重写数据打包逻辑 | 迁移成本高 |

**结论**：Zvdot4a8i功能上接近分段点积，但数据格式差异使其难以直接应用于现有MLAS kernel。更理想的方案是设计直接接受int8输入的`vsegdot`。

---

## 二、扩展指令方案设计

### 2.1 按优先级排序的完整扩展清单

| 优先级 | 指令名称 | 类型 | 功能描述 | 解决的问题 |
|--------|---------|------|---------|-----------|
| **P0** | `vsegdot.vv` | INT8 | 直接int8输入的分段点积（4×i8→1×i32） | INT8 GEMM核心瓶颈 |
| **P0** | `vmulacc.vv` | FP32 | 4×4矩阵外积累加 | FP32 GEMM寄存器压力 |
| **P1** | `vfmacc.vv_lane` | FP32 | Lane-indexed FMA | 减少A加载指令数 |
| **P2** | `vunzip/vzip.vv` | 通用 | 奇偶元素分离/合并 | 矩阵转置、PackB效率 |

---

## 三、指令详细设计与收益量化

### 3.1 P0-INT8：`vsegdot.vv` 分段点积指令

#### 指令定义

```
vsegdot.vv vd, vs1, vs2, seg=4
  输入：
    vs1 = vint8m1_t [a0 a1 a2 a3 | a4 a5 a6 a7 | ...]（直接int8向量）
    vs2 = vint8m1_t [b0 b1 b2 b3 | b4 b5 b6 b7 | ...]
  输出：
    vd = vint32mf2_t（元素数为VL/4）
    vd[0] += a0*b0 + a1*b1 + a2*b2 + a3*b3
    vd[1] += a4*b4 + a5*b5 + a6*b6 + a7*b7
    ...
  一条指令产出 VL/4 个独立int32结果
```

#### 与Zvdot4a8i对比

| 特性 | Zvdot4a8i（已存在） | vsegdot（提议） |
|------|---------------------|-----------------|
| 输入类型 | `vuint32m1_t`（打包32位） | `vint8m1_t`（直接int8） |
| 数据预处理 | 需要4个int8打包成1个32位 | 无需预处理 |
| 与MLAS兼容 | 需重写CopyPackB逻辑 | 可直接复用ARM SDOT模式 |
| 编译器支持 | 需要`-march=...+zvdot4a8i` | 提议加入核心ISA |

#### 收益量化（基于指令延迟表）

**INT8 GEMM内循环对比**（VLEN=256，SEW=8，VL=32）：

| 操作 | RVV外积法（现状） | vsegdot方案 | x86 VNNI |
|------|------------------|-------------|---------|
| 加载A | 32次broadcast（3周期×32） | 1次vle8（3周期） | 1次broadcast（3周期） |
| 加载B | 32次vle8（3周期×32） | 8次vle8（3周期×8） | 1次vmovdqu（3周期） |
| 计算 | 32次vwmacc（4周期×32） | 8次vsegdot（预估6周期×8） | 1次vpdpbusd（内置规约） |
| 产出C元素 | 逐行产出（效率低） | 8个int32/指令（并行） | 8个int32/指令 |
| **总周期（K=1步）** | ~256周期 | ~72周期 | ~8周期（归一化后相当） |

**性能提升估算**：
- 外积法 → vsegdot：**约3.5× INT8 GEMM吞吐提升**
- vsegdot → ARM SDOT水平：**85-100%**

#### 对YOLO推理的整体影响

```
YOLO推理热点分布：
  Conv/GEMM（INT8）：35% 热点
  
整体推理加速计算：
  若INT8 GEMM提升3.5×：
    整体加速 = 1 + 0.35 × (3.5 - 1) = 1 + 0.35 × 2.5 ≈ 1.87x
  
  考虑混合精度开销（SiLU等DQ/Q转换约20%）：
    实际整体加速 ≈ 1.5-1.7x
```

---

### 3.2 P0-FP32：`vmulacc.vv` 矩阵外积指令

#### 指令定义

```
vmulacc.vv vd, vs1, vs2
  功能：vd[4×4矩阵] += vs1[0..3] × vs2[0..3]^T（外积）
  要求：SEW=32, LMUL≥1，vd为4×4矩阵accumulator
  
  等效操作：
    vd[0,0] += vs1[0] × vs2[0]
    vd[0,1] += vs1[0] × vs2[1]
    vd[0,2] += vs1[0] × vs2[2]
    vd[0,3] += vs1[0] × vs2[3]
    vd[1,0] += vs1[1] × vs2[0]
    ... (共16个乘累加)
```

#### 收益量化

**FP32 SGEMM内循环对比**（处理2行×16列，K=4步）：

| 实现方式 | 指令数 | 周期数（参考延迟表） |
|---------|--------|---------------------|
| RVV vfmacc×16 | 16条 | 16×5 = 80周期 |
| vmulacc×4 | 4条 | 4×12(预估) = 48周期 |

**收益来源分析**：

| 收益维度 | 具体效果 |
|---------|---------|
| 指令数减少 | 16条→4条，减少75% |
| 寄存器压力 | 专用accumulator，释放向量寄存器 |
| K展开宽度 | 天然处理4个K元素的外积 |
| B加载共享 | 4行A共享同一次B加载 |

**性能提升估算**：
- FP32 GEMM提升：**1.3-2.0×**
- 对YOLO推理整体加速：**1.1-1.5×**

---

### 3.3 P1：`vfmacc.vv_lane` Lane-indexed FMA

#### 指令定义

```
vfmacc.vv_lane vd, vs1, vs2, imm[5-bit]
  功能：vd[i] += vs2[i] × vs1[imm]  (i = 0..VL-1)
        广播vs1的第imm个元素，与vs2逐元素乘加
  
  应用：批量加载A向量后，逐lane广播FMA
```

#### 收益量化

**FP32 SGEMM K循环对比**（K=4步）：

| 实现方式 | 指令序列 | 指令数 |
|---------|---------|--------|
| RVV现状 | flw×4 + vfmacc.vf×4 | 8条 |
| Lane-indexed | vle32×1 + vfmacc_lane×4 | 5条 |

**收益**：
- 指令数减少：**37.5%**
- 周期数减少：约28%（基于延迟表计算）

---

### 3.4 P2：`vunzip/vzip.vv` 奇偶分离/合并

#### 指令定义

```
vunzip.vv vd_even, vd_odd, vs2
  功能：将vs2奇偶元素分离
  输出：vd_even = [e0, e2, e4, ...], vd_odd = [e1, e3, e5, ...]

vzip.vv vd, vs1_even, vs2_odd
  功能：将两向量交错合并
  输出：vd = [e0, o0, e1, o1, ...]
```

#### 收益量化

**矩阵4×4转置对比**：

| 实现方式 | 指令数 |
|---------|--------|
| RVV vrgather | 6-8条（含索引向量准备） |
| vunzip/vzip | 4条 |

**应用场景收益**：

| 场景 | 收益 |
|------|------|
| 矩阵转置（FP32） | 25-40%加速 |
| INT8 PackB预处理 | 2-4×加速 |

---

## 四、综合收益评估

### 4.1 YOLO推理热点与扩展匹配

| YOLO热点算子 | 占比 | 计算类型 | 受益扩展 | 预期提升 |
|-------------|------|---------|---------|---------|
| Conv/GEMM | 35% | INT8 MAC密集 | vsegdot | **3.5×局部** |
| Conv/GEMM | 35% | FP32 MAC密集 | vmulacc + lane-FMA | **1.5-2×局部** |
| BatchNorm | 15% | 元素级（可融合） | — | 无额外收益 |
| Activation | 10% | Memory-bound | — | 无显著收益 |
| 其他 | 15% | 数据搬运 | vzip/vunzip | 边际收益 |

### 4.2 整体推理加速汇总

| 扩展组合 | INT8路径收益 | FP32路径收益 | 整体推理加速 |
|---------|-------------|-------------|-------------|
| 仅`vsegdot` | 3.5× GEMM | — | **1.5-1.7×** |
| 仅`vmulacc` + `lane-FMA` | — | 1.5-2× GEMM | **1.1-1.4×** |
| `vsegdot` + `vmulacc` | 3.5× | 1.5-2× | **1.8-2.2×** |
| 全套（含vzip） | 3.5-4× | 1.5-2× | **1.9-2.5×** |

### 4.3 与主流架构对比（扩展后）

| 指标 | x86 VNNI | ARM SDOT | RVV现状 | RVV扩展后 |
|------|---------|---------|---------|----------|
| INT8 GEMM吞吐 | 基准1.0× | 0.8-1.0× | **0.3-0.5×** | **0.85-1.0×** |
| FP32 GEMM吞吐 | 1.0× | 0.9× | 0.6× | **0.85-1.0×** |
| 矩阵转置效率 | 高（vunpck） | 高（vtrn） | 低（vrgather） | 中（vzip） |

---

## 五、实现路线图

### 5.1 阶段划分

| 阶段 | 目标 | 关键任务 | 优先级 |
|------|------|---------|--------|
| **Phase 1** | INT8基础能力 | 利用现有Zvdot4a8i实现INT8 kernel | P0 |
| **Phase 2** | FP32优化 | vmulacc + lane-FMA验证 | P0-P1 |
| **Phase 3** | 完善INT8 | 设计直接int8输入的vsegdot | P0 |
| **Phase 4** | 数据重排优化 | vzip/vunzip扩展 | P2 |

### 5.2 Phase 1：利用Zvdot4a8i快速验证

**现状**：Zvdot4a8i已存在，可立即用于INT8 GEMM验证。

**实施方案**：

```c
// 利用Zvdot4a8i的INT8 GEMM（需数据打包）
// 步骤1：将4个int8打包成1个uint32
vuint32m1_t packed_a = pack_int8_to_uint32(a_vec);  // 预处理
vuint32m1_t packed_b = pack_int8_to_uint32(b_vec);

// 步骤2：分段点积
vint32m1_t acc = __riscv_vdot4a_vv_i32m1(acc, packed_b, packed_a, vl);
```

**预期收益**：
- 相比外积法：**约2×提升**
- 相比理想vsegdot：略低（因打包开销）

### 5.3 Phase 2：FP32扩展验证

**任务清单**：

1. QEMU添加`vmulacc`和`vfmacc_lane`模拟
2. 编写测试kernel验证语义
3. 对比现有`vfmacc.vf`性能
4. 评估寄存器压力改善

### 5.4 Phase 3：理想vsegdot设计

**设计要点**：

```
编码方案建议：
  - 复用RVV OPIVV格式（funct6 + vs1 + vs2 + vd）
  - 新增funct6编码点（需申请ISA扩展空间）
  
硬件实现建议：
  - 可复用现有FMA单元并行化
  - 分段求和逻辑：4个8-bit乘法器 → 1个32-bit加法器
  
编译器支持：
  - LLVM后端添加vsegdot intrinsic
  - 自动向量化识别分段点积模式
```

---

## 六、风险与限制

### 6.1 技术风险

| 风险项 | 影响 | 缓解措施 |
|--------|------|---------|
| ISA扩展审批 | 新指令需RISC-V国际批准 | 先用Zvdot4a8i验证，积累数据 |
| 编译器生态 | LLVM/GCC支持周期 | 从intrinsic开始，逐步完善 |
| 硬件实现成本 | 分段规约需新硬件单元 | 复用现有FMA阵列并行化 |

### 6.2 收益限制

| 限制因素 | 说明 |
|---------|------|
| Memory-bound算子 | SiLU/Concat等无收益（占总推理15-25%） |
| 混合精度开销 | SiLU需DQ→FP32→Q转换，约20%额外开销 |
| 现有Zvdot4a8i格式 | 打包开销可能抵消部分收益 |

---

## 七、结论

### 7.1 核心结论

1. **RVV分段规约缺失是INT8推理的根本瓶颈**，导致GEMM性能仅为ARM SDOT的30-50%。

2. **Zvdot4a8i扩展已存在但受限**：提供分段点积功能，但数据格式（打包32位）与主流框架不兼容，仍需设计直接int8输入的vsegdot。

3. **FP32扩展全部缺失**：矩阵外积、Lane-indexed FMA均不存在，建议优先添加。

4. **整体收益显著**：全套扩展可使YOLO推理加速**1.9-2.5×**，使RVV达到x86/ARM主流水平。

### 7.2 推荐优先级

```
优先实现顺序：
  1. 利用Zvdot4a8i验证INT8可行性（Phase 1，立即可行）
  2. vmulacc + vfmacc_lane（Phase 2，FP32优化）
  3. 设计理想vsegdot（Phase 3，长期目标）
  4. vzip/vunzip（Phase 4，完善生态）
```

### 7.3 下一步行动

| 行动项 | 负责 | 时间线 |
|--------|------|--------|
| Zvdot4a8i INT8 kernel原型 | RVFuse项目 | 2周 |
| QEMU vmulacc模拟 | 工具链团队 | 4周 |
| vsegdot ISA提案草案 | RISC-V贡献 | 8周 |
| LLVM intrinsic扩展 | 编译器团队 | 12周 |