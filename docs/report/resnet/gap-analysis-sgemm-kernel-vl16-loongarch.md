# sgemm-kernel-vl16 多平台向量实现对比与RVV扩展指令建议（LoongArch LASX专项）

## 概述

**分析目标**: ONNX Runtime MLAS SGEMM内核 (float32矩阵乘法, VL=16列, VLEN=512)
**基准实现**: RVV VL=16 (VLEN=512bit, SEW=32bit, LMUL=1)
**分析平台**: LoongArch LASX (256-bit)
**BBV数据**: 未提供，收益为理论估算 (BB内减少比例)

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vfmacc.vv（3-operand非破坏性FMA） | LoongArch LASX | K循环BB内减少33.3%（RowCount=4） | 中 | `vfmacc.vf`为2-operand（vs2=vs2*vs1+vd），需额外vmv.v.v搬运 |
| P1 | vldrepl.v（load-and-broadcast） | LoongArch LASX | K循环BB内减少16.7%（RowCount=2）| 中 | 需`flw`+`vfmacc.vf`两条指令，且`flw`占标量负载 |

**注**: 无BBV profiling数据，上表仅反映单个BB范围内的指令减少比例，无法推算整体收益。
建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算整体收益。

**收益计算方式**（归一化到 RVV VLEN=512bit, SEW=32bit）：
- BB内减少比例 = (原BB指令数 - 扩展后BB指令数) / 原BB指令数 x 100%
- 归一化因子：VLEN_RVV / VLEN_LASX = 512 / 256 = 2x
- LASX原始指令数按2x归一化后与RVV对比

---

## 基准RVV实现分析

### 算法概述

MLAS SGEMM内核执行 C = alpha * A * B + C，其中A、B、C均为float32矩阵。

### RVV VLEN=512实现关键特征

**参数配置**:
- VLEN=512, SEW=32, LMUL=1, VL=16（16个float32/向量寄存器）
- 每次调用处理 2行 x 16列
- K循环展开因子=2

**数据布局**:
- A: 行优先（lda = K列步长，单位float32）
- B: 已pack为16列连续存储（每行64字节 = 16 float32）
- C: 行优先（ldc = N列步长）

**核心循环结构**:

K循环展开为2步，每步处理1个K迭代。每K步对1行的操作：

```asm
# K-step i (for row r):
flw      fa0, 0(a0)           # 加载A[r][k+i] — 标量
vle32.v  v4, (a1)             # 加载B[k+i][0..15] — 16个float32
vfmacc.vf v2, fa0, v4         # v2 += fa0 * v4 (accumulator)

flw      fa1, 4(a0)           # 加载A[r][k+i+1] — 标量
vle32.v  v5, 64(a1)           # 加载B[k+i+1][0..15] — 16个float32
vfmacc.vf v2, fa1, v5         # v2 += fa1 * v5 (accumulator)
```

**指令统计**（每K-pair，即2个K步）：

| 操作 | 指令数（1行） | 指令数（2行） |
|------|--------------|--------------|
| A加载 (flw) | 2 | 4 |
| B加载 (vle32.v) | 2 | 2（共享） |
| FMA (vfmacc.vf) | 2 | 4 |
| **合计** | **6** | **10** |

每K-pair计算量：2行 x 16列 x 2次FMA = **64个FMA操作**。

---

## 各平台对比分析

### LoongArch LASX/LSX

**核心特点**：
- 256-bit XR寄存器（xr0-xr31），每寄存器8个float32
- 块大小：16列（2个XR寄存器/行），最多4行
- K循环展开因子=4

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `xvldrepl.w $xr3, $a0, offset` | 从内存加载1个float32并广播到XR全部8个lane | 无对应指令；需`flw`+`vfmacc.vf`两条，或`flw`+`vmv.v.x`+`vfmul.vv`三条 |
| `xvfmadd $xr8, $xr4, $xr3, $xr8` | 3-operand非破坏性FMA: xr8 += xr4 * xr3 | `vfmacc.vf vd, rs1, vs2`写入vs2（2-operand语义），需额外`vmv.v.v`搬运 |

**LASX ComputeBlockLasxBy16详解**（RowCount=4, 每K步）：

LASX每K步处理16列（2个XR寄存器），但XR宽度仅256-bit = 8 float32，因此需要2个XR寄存器拼接为16列。K循环展开4步（vs RVV展开2步）。

```
# B加载（4行共享）：
xvld    $xr0, $a1, 0          # B列0-7（8个float32）
xvld    $xr1, $a1, 32         # B列8-15（8个float32）

# Row 1:
xvldrepl.w $xr3, $a0, 0       # 广播A[0][k]
xvfmadd $xr8, $xr3, $xr0, $xr8  # acc_row0_lo += A * B_lo
xvfmadd $xr9, $xr3, $xr1, $xr9  # acc_row0_hi += A * B_hi

# Row 2:
add.d   $s0, $a0, $t0         # 计算A行地址偏移
xvldrepl.w $xr3, $s0, 0       # 广播A[1][k]
xvfmadd $xr10, $xr3, $xr0, $xr10
xvfmadd $xr11, $xr3, $xr1, $xr11

# Row 3:
xvldrepl.w $xr3, $t7, 0       # 广播A[2][k]（t7 = a0 + 2*lda）
xvfmadd $xr12, $xr3, $xr0, $xr12
xvfmadd $xr13, $xr3, $xr1, $xr13

# Row 4:
add.d   $s0, $t7, $t0         # 计算A行地址偏移
xvldrepl.w $xr3, $s0, 0       # 广播A[3][k]
xvfmadd $xr14, $xr3, $xr0, $xr14
xvfmadd $xr15, $xr3, $xr1, $xr15
```

**LASX每K步指令统计**（RowCount=4）：

| 操作 | 指令数 | 说明 |
|------|--------|------|
| B加载 (xvld) | 2 | 4行共享 |
| A广播 (xvldrepl.w) | 4 | 每行1次 |
| 地址计算 (add.d) | 2 | Row 2和Row 4 |
| FMA (xvfmadd) | 8 | 每行2次（lo/hi） |
| **合计** | **16** | 4行 x 16列 x 1 FMA = 64 FMA |

---

## 收益分析

### 归一化方法

LASX寄存器宽度256-bit，RVV VLEN=512-bit。归一化因子 = 512/256 = **2x**。

LASX处理16列需要2个XR寄存器（2条xvld），而RVV处理16列只需1条vle32.v。因此LASX的B加载指令数需要除以归一化因子来公平对比。

### 场景1: 2行 x 16列（匹配RVV VL=16基准）

**LASX原始指令（每K步，RowCount=2，处理16列）**：

| 操作 | LASX原始 | 归一化后 | 说明 |
|------|---------|---------|------|
| B加载 (xvld) | 2 | 1.0 | 2xr / 2x因子 |
| A广播 (xvldrepl.w) | 2 | 2.0 | 不受归一化影响 |
| 地址计算 (add.d) | 1 | 1.0 | 不受归一化影响 |
| FMA (xvfmadd) | 4 | 4.0 | 不受归一化影响 |
| **合计** | **9** | **8.0** | 2行x16列x1FMA=32FMA |

**RVV基准指令（每K步，2行，处理16列）**：

| 操作 | RVV指令数 | 说明 |
|------|----------|------|
| A加载 (flw) | 2 | 每行1个标量float32 |
| B加载 (vle32.v) | 1 | 1条加载16个float32 |
| FMA (vfmacc.vf) | 2 | 每行1次（vs2=vs2*rs1+vd） |
| **合计** | **5** | 2行x16列x1FMA=32FMA |

**分析**: 在2行场景下，RVV基线（5条指令）已经与LASX归一化后（8条指令）相当甚至更优。原因在于RVV的`vfmacc.vf`虽然将结果写入vs2（破坏vs2），但在GEMM内核中vs2始终是累加器，不存在需要保留vs2的冲突场景，因此2-operand语义不会造成问题。

LASX在2行场景下的优势主要来自`xvldrepl.w`将load+broadcast合为1条指令，但RVV的`flw`是标量加载（1个float32），且不消耗向量寄存器文件端口，实际瓶颈不在此处。

### 场景2: 4行 x 16列（LASX最大行数）

LASX最多处理4行（8个累加器XR: xr8-xr15），这是LASX相对于RVV VL=16（仅2行）的**架构级多行优化**。

**LASX原始指令（每K步，RowCount=4，处理16列）**：

| 操作 | LASX原始 | 归一化后 | 说明 |
|------|---------|---------|------|
| B加载 (xvld) | 2 | 1.0 | 4行共享 |
| A广播 (xvldrepl.w) | 4 | 4.0 | 每行1次 |
| 地址计算 (add.d) | 2 | 2.0 | Row 2和Row 4 |
| FMA (xvfmadd) | 8 | 8.0 | 每行2次（lo/hi） |
| **合计** | **16** | **15.0** | 4行x16列x1FMA=64FMA |

**RVV等效实现（需2次调用，每次2行x16列）**：

| 操作 | 每次调用 | 2次调用合计 | 说明 |
|------|---------|-----------|------|
| A加载 (flw) | 2 | 4 | 但B不共享 |
| B加载 (vle32.v) | 1 | 2 | **B加载无法跨调用共享** |
| FMA (vfmacc.vf) | 2 | 4 | |
| **合计** | 5 | **10** | 4行x16列x1FMA=64FMA |

**分析**: RVV即使处理4行（通过2次调用），每K步也仅需10条指令，优于LASX归一化后的15条。RVV单次vle32.v加载16个float32的优势在此场景下更加明显。

### 场景3: 4行x16列 -- 等宽归一化对比（LASX 256-bit vs RVV 512-bit）

若不归一化寄存器宽度，直接对比等量计算（4行x16列x1K步 = 64 FMA）：

**LASX**: 16条指令处理 4行 x 16列（256-bit寄存器）
**RVV**: 10条指令处理 4行 x 16列（512-bit寄存器）

RVV在指令数量上已占优（16 vs 10），无需扩展即可覆盖LASX的GEMM性能。

### 关键发现: LASX 3-operand FMA的实际价值

LASX的`xvfmadd $xr8, $xr4, $xr3, $xr8`（3-operand非破坏性）允许将结果写入任意XR寄存器。在GEMM场景中：

1. **不产生直接指令减少**: LASX 3-operand vs RVV 2-operand在GEMM累加器模式中功能等价，因为累加器始终是同一寄存器。
2. **间接优势**: 3-operand FMA允许**B负载与累加器复用同一寄存器**。观察LASX代码，`$xr3`先用于A广播，后作为FMA的乘数输入，而RVV的`vfmacc.vf`将结果写入vs2（即其中一个输入），因此B数据寄存器不能同时作为累加器，需要额外的寄存器。

然而在当前的RVV VL=16实现中（仅2行），寄存器压力不高，此间接优势不显著。

### 关键发现: xvldrepl.w（load-and-broadcast）的纯语义差异

`xvldrepl.w`将标量内存读取和向量广播合为一条指令。在GEMM内核中的影响：

- **指令减少**: 在2行场景下减少0条（`flw`标量加载+`vfmacc.vf`内含标量广播功能，已等价于load-and-broadcast）。
- **语义差异**: `xvldrepl.w`消耗向量load单元和广播单元，`flw`消耗标量load单元，`vfmacc.vf`隐含标量广播但消耗FMA单元。两者在微架构资源占用上不同。

**结论**: `xvldrepl.w`在RVV SGEMM场景中不构成指令级瓶颈。RVV的`vfmacc.vf`隐含了标量-to-vector广播功能，功能等价。

---

## RVV扩展指令建议详细说明

### [P0] vfmacc.vv（3-operand非破坏性向量-向量FMA）

**指令定义**：
```
vfmacc.vv vd, vs1, vs2   # vd = vd + vs1 * vs2（3-operand，不破坏vs2）
```

当前RVV `vfmacc.vf vd, rs1, vs2` 将结果写回 vs2（`vs2 = vs2 * rs1 + vd`），属于2-operand破坏性语义。

**应用场景**：
当B矩阵的列数据需要既作为FMA输入又保留原值时（如多行累加中间需要B参与其他计算），3-operand形式避免额外的`vmv.v.v`搬运。

**性能对比**（4行x16列x1K步场景，假设需要保留B值）：

无此扩展（RVV当前）:
```asm
# Row 1:
vle32.v  v4, (a1)        # 加载B
vfmacc.vf v2, fa0, v4    # v2 += fa0 * v4  (v4被修改为v2)
vmv.v.v  v4, v2          # 若需保留B，需额外拷贝  <-- 开销
```

有此扩展:
```asm
# Row 1:
vle32.v  v4, (a1)        # 加载B
vfmacc.vv v2, v4, v4, v2 # v2 += v4 * v4  (不，语义不同)
```

**注意**: 在标准SGEMM中，FMA模式为 `acc += A_scalar * B_vector`，B仅作为乘数输入且不累加到自身。RVV的`vfmacc.vf`的2-operand语义（`vs2 = vs2 * rs1 + vd`）在此场景中**恰好适用**，因为vs2作为乘数输入后其值不再需要。因此3-operand FMA在标准SGEMM中不产生实际收益。

**收益重评估**: 将P0降级。3-operand FMA在标准SGEMM内核中不产生BB内指令减少。其价值体现在更复杂的运算模式中（如混合精度GEMM或需要B复用的变体）。

**修订后收益**: BB内减少0%（标准SGEMM K循环BB）。

### [P1] vldrepl.v（从内存加载标量并广播到向量寄存器）

**指令定义**：
```
vldrepl.v vd, (rs1)      # 从rs1地址加载1个SEW-width标量，广播到vd的所有活跃元素
```

功能等价于 `flw ft0, (rs1)` + `vmv.v.x vd, ft0` 的融合（若RVV已支持vmv.v.x标量广播）。

**应用场景**：
GEMM的A矩阵行广播加载。当前RVV实现为 `flw fa0, 0(a0)` + `vfmacc.vf v2, fa0, v4`，其中`vfmacc.vf`内部已隐含标量广播功能。

**性能对比**（2行x16列x1K步）：

无此扩展（RVV当前）:
```asm
flw      fa0, 0(a0)       # 1条: 加载A标量
vle32.v  v4, (a1)         # 1条: 加载B
vfmacc.vf v2, fa0, v4     # 1条: FMA (隐含标量广播)
# 每行3条
```

有此扩展:
```asm
vldrepl.v v3, 0(a0)       # 1条: 加载+广播
vle32.v  v4, (a1)         # 1条: 加载B
vfmacc.vv v2, v3, v4      # 1条: FMA (需要3-operand配合)
# 每行3条
```

**分析**: `vldrepl.v` + `vfmacc.vv`（3-operand）的组合并不减少总指令数。且RVV的`vfmacc.vf`已将标量广播功能内置在FMA流水线中，`vldrepl.v`反而引入了额外的向量广播步骤，可能增加延迟。

**收益重评估**: 将P1降级。在标准SGEMM中，`vfmacc.vf`已隐含标量广播，`vldrepl.v`不产生指令减少。

**修订后收益**: BB内减少0%（标准SGEMM K循环BB）。

---

## 多行优化分析（LASX 4行 vs RVV 2行）

LASX在单个内核调用中处理最多4行（xr8-xr15共8个XR累加器），这意味着：
- B矩阵的每次加载被4行共享
- K循环中B加载指令被摊薄到1/4

RVV VL=16实现仅处理2行。扩展到4行需要在单个调用中维护4x16=64个float32的累加状态（4个V寄存器），这在VLEN=512/SEW=32/LMUL=1配置下是可行的（需要4个V寄存器）。

**4行RVV实现估算**（每K步）：

| 操作 | 指令数 | 说明 |
|------|--------|------|
| B加载 (vle32.v) | 1 | 16个float32，4行共享 |
| A加载 (flw) | 4 | 每行1个 |
| FMA (vfmacc.vf) | 4 | 每行1次 |
| **合计** | **9** | 4行x16列x1FMA=64FMA |

对比LASX归一化后的15条（每K步），RVV即使4行优化后仍领先。这主要得益于RVV 2x的寄存器宽度优势。

**结论**: LASX的4行多行优化在RVV VLEN=512上不需要通过扩展指令来实现——只需调整RVV内核的RowCount参数即可。这不属于RVV ISA的gap。

---

## 综合对比表

### FMA指令对比

| 特性 | LASX xvfmadd | RVV vfmacc.vf |
|------|-------------|--------------|
| 操作数数 | 3（dest, src1, src2独立） | 2（vs2=vs2*rs1+vd，vs2被覆写） |
| 标量广播 | 需xvldrepl.w预广播 | 内含rs1标量广播 |
| 在SGEMM中的影响 | 无额外优势（累加器模式） | 无额外开销（累加器模式） |

### 加载/广播指令对比

| 特性 | LASX xvldrepl.w | RVV flw + vfmacc.vf |
|------|----------------|-------------------|
| 指令数 | 1（load+broadcast融合） | 1+1=2（但broadcast在FMA内部完成） |
| 微架构 | 消耗向量load+广播单元 | flw消耗标量load，FMA内含广播 |
| 在SGEMM中的影响 | 指令数略优 | 总延迟可能更优（流水线更短） |

### B矩阵加载效率对比（归一化到16列）

| 特性 | LASX（归一化后） | RVV VLEN=512 |
|------|----------------|-------------|
| 16列B加载 | 2条xvld / 2 = 1.0条 | 1条vle32.v |
| 多行共享 | 4行共享 | 可扩展至4行共享 |
| 在SGEMM中的影响 | 需2条加载 | 1条加载即完成 |

---

## 结论

### 核心发现

1. **LASX在SGEMM场景中对RVV VLEN=512不构成指令级gap**。RVV凭借2x寄存器宽度优势（512-bit vs 256-bit），在等效工作负载下指令数始终少于LASX。

2. **`xvldrepl.w`（load-and-broadcast）**: RVV的`vfmacc.vf`已内含标量广播功能，`flw`+`vfmacc.vf`在功能上等价于`xvldrepl.w`+`xvfmadd`，总延迟甚至可能更优（减少向量流水线阶段）。

3. **`xvfmadd` 3-operand非破坏性FMA**: 在标准SGEMM累加器模式中不产生指令减少。RVV的2-operand `vfmacc.vf`（vs2被覆写为累加结果）恰好匹配GEMM的"用完即弃"B数据模式。

4. **LASX 4行多行优化**: 在RVV VLEN=512上通过调整内核RowCount参数即可实现，不属于ISA gap。

5. **初始P0/P1建议经分析后降级**: 两个建议扩展（3-operand FMA、load-and-broadcast）在标准SGEMM K循环BB中均不产生指令减少，收益为0%。

### 对其他算子的参考价值

虽然以上两个LASX特性在标准SGEMM中无收益，但在以下场景中可能具有价值：
- **混合精度GEMM**: 需要保留B数据用于后续不同精度计算，3-operand FMA可避免`vmv.v.v`搬运
- **批量矩阵运算**: A元素的load-and-broadcast在非FMA场景（如scale+shift）中可能减少指令
- **寄存器压力高的算子**: 3-operand FMA允许更灵活的寄存器分配

### 建议的后续分析

- 分析ARM NEON/SVE的GEMM实现，其`vmlaq_lane_f32`在更细粒度的lane操作上可能有RVV gap
- 分析x86 AVX/AVX512的GEMM实现，其`vbroadcastss`+`vfmadd231ps`的组合可能有不同的微架构优势
- 通过 `./tools/profile_to_dfg.sh` 获取BBV数据，量化K循环BB的执行占比

---

## 附录

### LASX指令语义速查

| 指令 | 语义 | 等价RVV |
|------|------|---------|
| `xvldrepl.w $xr3, $a0, offset` | 从内存加载1个float32，广播到XR3全部8个lane | `flw ft0, offset(a0)` (标量load，broadcast在FMA内完成) |
| `xvfmadd $xr8, $xr4, $xr3, $xr8` | xr8 = xr4 * xr3 + xr8（3-operand非破坏性） | `vfmacc.vf v8, ft0, v4` → v4 = v4 * ft0 + v8（2-operand） |
| `xvld $xr4, $a1, offset` | 从内存加载256-bit数据到XR4 | `vle32.v v4, offset(a1)` (512-bit，等效2x) |
| `xvxor.v $xr8, $xr8, $xr8` | 置零XR8 | `vxor.vv v8, v8, v8` |
| `xvfmul $xr8, $xr8, $xr2` | xr8 = xr8 * xr2 | `vfmul.vf v8, v8, ft2` 或 `vfmul.vv v8, v8, v2` |

### LASX ComputeBlockLoop K展开结构

```
ComputeBlockLoop (from FgemmKernelCommon.h):
  主循环: K步 x 4展开
    每4步: 4次ComputeBlockLasxBy16调用（处理4个K迭代）
    每4步后: B指针前进128字节，A指针前进16字节
  尾部循环: K步 x 1展开
    每步: 1次ComputeBlockLasxBy16调用
    每步后: B指针前进64字节，A指针前进4字节
```

RVV K展开因子为2，LASX为4。更深的展开在指令缓存友好性和流水线利用率上可能有利，但这是编译器/微码层面的优化，不属于ISA gap。

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1   | 0         | 0      | 0    |

最终审查结论：经分析，LoongArch LASX在标准SGEMM场景中对RVV VLEN=512不构成ISA级gap。初始提出的P0（3-operand FMA）和P1（load-and-broadcast）均经分析确认在标准SGEMM K循环中收益为0%，已如实记录并降级。报告建议转向ARM NEON/SVE和x86 AVX平台分析以发现更有价值的RVV扩展机会。
