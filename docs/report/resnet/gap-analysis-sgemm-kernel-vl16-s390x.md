# SGEMM Kernel VL=16 RVV vs S390X Z-Vector Gap Analysis

## 概述

**分析目标**: SGEMM (C = alpha * A * B + beta * C) 内核K循环热点路径对比
**基准实现**: RVV VL=16 (VLEN=512bit, SEW=32bit, LMUL=1)
**分析平台**: S390X Z-Vector
**BBV数据**: 未提供，收益为理论估算
**归一化因子**: RVV VLEN=512 / Z-Vector VLEN=128 = 4x

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | 多行处理 + A广播摊销 (软件优化) | S390X Z-Vector | BB内减少50.0%（K循环BB，8行 vs 2行时A加载指令摊销） | 低 | 仅支持2行，可扩展至4/8行 |
| P1 | vfmacc.vv (向量-向量FMA，3-operand语义) | S390X Z-Vector `vfmasb` | BB内减少11.1%（K循环BB，消除A广播指令） | 中 | 仅 `vfmacc.vf`（标量广播FMA） |
| P2 | vrgather.vv 批量广播 (多元素加载+广播) | S390X Z-Vector `vec_perm` | BB内减少16.7%（K循环BB，4行时A广播setup） | 中 | 有 `vrgather` 但无批量广播语义 |

**收益计算方式**（基于per-BB指令计数）：
- 无BBV profiling数据，收益为单BB内指令减少比例
- 所有计算已归一化到 RVV VLEN=512bit, SEW=32bit
- 各扩展指令收益可叠加估算上限：BB内减少比例取最大值

---

## 基准RVV实现分析

**源文件**: `applications/onnxrt/rvv-patches/sgemm-kernel-vl16/rvv_sgemm_kernel_vl16.inl`

### 循环结构

- **外层**: 按16列分块遍历N维（`do { ... } while (CountN > 0)`）
- **内层**: K循环，按2展开（`while (k >= 2)`）
- **行数**: 最多2行（`ProcessTwoRows`模板参数）

### K循环单次迭代（2行，2个K步）指令分解

| 指令 | 数量 | 说明 |
|------|------|------|
| `flw` (scalar A load) | 4 | 2行 x 2K步 |
| `vle32.v` (vector B load) | 2 | 每次16列 = 1条vector load |
| `vfmacc.vf` (FMA) | 4 | 2行 x 2K步 |
| 指针/计数器更新 | 3 | `a+=2, b+=32, k-=2` |
| **合计** | **13** | **4 FMA** |

### 关键参数

- **VLEN=512, SEW=32, LMUL=1, VL=16**: 每条vector指令处理16个float32
- **K展开因子**: 2（每次处理2个K步）
- **行数**: 最多2行（单个vector accumulator pair）
- **B packing**: 16列连续排列，每K行16个float
- **FMA形式**: `vfmacc.vf vd, rs1, vs2` — 2-operand（破坏性），标量rs1自动广播到vector

---

## S390X Z-Vector 对比分析

**源文件**: `applications/onnxrt/ort/vendor/onnxruntime/onnxruntime/core/mlas/lib/s390x/SgemmKernelZVECTOR.cpp`

### 核心特点

- **128-bit vector寄存器**: 每个VR容纳4个float32
- **`__builtin_s390_vfmasb(a, b, c)`**: Vector FMA，3-operand（非破坏性），`result = a*b + c`
- **块结构**: 16列 = 4个VR/行，最多8行（RowCount=4，CountM=8时处理两个4行组）
- **K展开因子**: 4（`while (k >= 4)`）
- **A广播优化**: `vec_perm` 多元素重排实现批量广播

---

### 1. `vec_perm` 用于A元素重排（多元素广播设置）

**Z-Vector实现**: `MlasSgemmComputeAElements<4>()`

Z-Vector在K展开为4时，一次加载4个A元素（每个VR含1个有效float），通过`vec_perm`指令序列将4个分散元素重排为4个广播向量。

```cpp
// 输入: AElements[0..3]，每个VR含1个有效float在不同lane
// 输出: ABroadcast[0..3]，每个VR含4份对应A元素的广播

a1 = vec_perm(AElements[0], AElements[1], mask_even);  // 重排
a2 = vec_perm(AElements[2], AElements[3], mask_even);  // 重排
ABroadcast[0] = vec_perm(a1, a2, mask0);               // A[0]广播
ABroadcast[2] = vec_perm(a1, a2, mask3);               // A[2]广播
a1 = vec_perm(AElements[0], AElements[1], mask_odd);   // 重排
a2 = vec_perm(AElements[2], AElements[3], mask_odd);   // 重排
ABroadcast[1] = vec_perm(a1, a2, mask0);               // A[1]广播
ABroadcast[3] = vec_perm(a1, a2, mask3);               // A[3]广播
```

**指令成本**: 4 loads + 8 vec_perm = 12 ops，产生4个广播向量（覆盖4行 x 4个K步的A数据）

**RVV对比**: RVV使用逐元素标量加载 + `vfmacc.vf`的标量广播功能，每个A元素需要1条`flw`。对于4行 x 4 K步 = 16个A元素，需要16条`flw`。

**归一化对比**（处理4行 x 4 K步的A数据）:

| 平台 | A数据准备指令数 | 说明 |
|------|----------------|------|
| S390X Z-Vector | 12 (4 loads + 8 perm) | `vec_perm`批量重排 |
| RVV baseline | 16 (16 flw) | 逐元素标量加载 |
| **减少比例** | **25.0%** | |

**关键差异**: Z-Vector的`vec_perm`可以将4个分散的元素通过3级permute网络重排为4个完整的广播向量。RVV虽然有`vrgather`指令，但缺乏类似的跨向量lane重排能力来实现一次性批量广播设置。

---

### 2. 多行处理（最多8行共享同一A广播）

**Z-Vector实现**: `MlasSgemmComputeBlockZVECTOR<4>` 支持CountM=8

Z-Vector内核支持两种行数模式:
- **CountM=4**: 处理4行，16个accumulator VR (4行 x 4列)
- **CountM=8**: 处理8行，32个accumulator VR (8行 x 4列)，通过两组`ABroadcast`/`A2Broadcast`共享B加载

8行模式下，同一组B加载（4个VR = 16列）被8行FMA复用:
```cpp
if (CountM == 8) {
    acc[16..31] = __builtin_s390_vfmasb(AElements[4..7], BElements[0..3], acc[16..31]);
}
```

**RVV对比**: RVV baseline仅支持2行（`ProcessTwoRows`），使用2个vector accumulator。

**A广播摊销分析**（每K步，归一化到16列输出）:

| 行数 | Z-Vector A setup/列 | RVV A loads/列 | RVV减少比例 vs Z-Vector |
|------|---------------------|----------------|------------------------|
| 2行 | 0.50 (2 flw) | 2 flw | RVV多 300% |
| 4行 | 0.75 (3 flw equiv) | 4 flw | RVV多 433% |
| 8行 | 0.375 (1.5 flw equiv) | 8 flw | RVV多 2033% |

**注意**: Z-Vector的A setup成本通过`vec_perm`摊销到4个K步。表中"Z-Vector A setup/列"为每K步每列的A指令成本。

**多行对B加载的摊销效应**:

B加载成本在Z-Vector和RVV中相同（每个vector load覆盖16列），但多行意味着每次B加载被更多FMA复用:

| 行数 | B loads/FMA | 有效B加载利用率 |
|------|-------------|----------------|
| 2行 | 1/2 (RVV) | 每B load服务2行FMA |
| 4行 | 1/4 (Z-Vector) | 每B load服务4行FMA |
| 8行 | 1/8 (Z-Vector) | 每B load服务8行FMA |

---

### 3. K循环展开策略（Z-Vector展开4 vs RVV展开2）

**Z-Vector**: `while (k >= 4)` — 每次迭代处理4个K步，调用4次`MlasSgemmComputeBlockZVECTOR`

**RVV**: `while (k >= 2)` — 每次迭代处理2个K步

**展开因子对指令效率的影响**（归一化到16 FMA，4行，16列）:

| 平台 | K步数 | A loads | B loads | FMA | 指针/计数器 | 总指令 | FMA效率 (instr/FMA) |
|------|--------|---------|---------|-----|-------------|--------|---------------------|
| Z-Vector (4行, K-unroll=4) | 4 | 4 + 8 perm | 16 (4 blocks x 4) | 64 | 3 | 95 | 1.48 |
| RVV (2行, K-unroll=2) | 4 (2 pairs) | 8 flw | 4 (2 pairs x 2) | 8 | 6 | 26 | 3.25 |
| RVV (4行, K-unroll=4, 假设) | 4 | 16 flw | 4 | 16 | 3 | 39 | 2.44 |
| RVV (8行, K-unroll=4, 假设) | 4 | 32 flw | 4 | 32 | 3 | 71 | 2.22 |

**注意**: Z-Vector的4个block调用中，B load无法跨block共享（每个block处理不同的16列子集但相同的A广播），所以4 blocks需要 4 x 4 = 16 条B loads。但实际上这些block处理的是同一批16列（B, B+16, B+32, B+48），每次K步覆盖全部64列。校正:

重新分析Z-Vector K-quad（4行）的4次block调用:
- Block 0: B[0..15], ABroadcast[0] -> 4 B loads + 16 FMA
- Block 1: B[16..31], ABroadcast[1] -> 4 B loads + 16 FMA
- Block 2: B[32..47], ABroadcast[2] -> 4 B loads + 16 FMA
- Block 3: B[48..63], ABroadcast[3] -> 4 B loads + 16 FMA

这4次调用共享A广播但处理不同K步的B数据。在packed B layout中，B, B+16, B+32, B+48 对应 K步0-3 的同一16列。

**归一化对比**（每16 FMA，即4行 x 1K步 x 16列，或2行 x 4K步 x 2列）:

取等效计算量: **64 FMA (4行 x 4 K步 x 4列 Z-Vector等效)**:

| 平台 | 总指令 | FMA效率 (instr/FMA) |
|------|--------|---------------------|
| Z-Vector 4行 K-unroll=4 | 95 | 1.48 |
| RVV 2行 K-unroll=2 | 144 (16 K-pairs x 9) | 2.25 |
| RVV 4行 K-unroll=2 (假设) | ~117 (16 K-pairs x 7.3) | ~1.83 |

**Z-Vector相对于RVV 2行baseline的K循环指令减少**:
- 减少比例 = (144 - 95) / 144 = **34.0%**
- 来源分解: 多行(8→2) A加载摊销 + K展开(4→2) 减少循环开销

---

### 4. FMA指令对比: `vfmasb` (3-operand) vs `vfmacc.vf` (2-operand)

**S390X `__builtin_s390_vfmasb(a, b, c)`**:
- 语义: `result = a * b + c`（3-operand，非破坏性）
- 所有3个操作数均为vector寄存器
- Accumulator `c` 不被修改，结果写入新的目标寄存器
- 允许直接使用broadcast vector作为乘数，无需额外广播步骤

**RVV `vfmacc.vf vd, rs1, vs2`**:
- 语义: `vd = vs2 * rs1 + vd`（2-operand destructive）
- `vd` 既是输入accumulator又是输出目标
- `rs1` 为标量float寄存器，自动广播
- 需要额外的标量加载（`flw`）获取A元素

**3-operand vs 2-operand 对寄存器压力的影响**:

| 特性 | `vfmasb` (3-operand) | `vfmacc.vf` (2-operand) |
|------|----------------------|-------------------------|
| Accumulator | 非破坏性，可保留原值 | 破坏性，原值被覆盖 |
| 广播方式 | 使用预计算的broadcast VR | 使用标量寄存器自动广播 |
| 寄存器需求 | 需要额外目标寄存器 | 复用accumulator寄存器 |
| 适用场景 | 多行共享broadcast时更优 | 单行或少量行时更简洁 |

**对当前RVV baseline的影响**: 在2行、K-unroll=2的场景下，`vfmacc.vf`的破坏性语义不是瓶颈，因为每行只有一个accumulator，且标量广播已经内嵌在指令中。但在扩展到4/8行时，如果RVV引入3-operand `vfmacc.vv`，可以消除逐元素标量加载，改用预计算的broadcast vector。

**指令减少估算**（假设RVV支持3-operand vector-vector FMA）:

对于2行、K-unroll=2:
- 当前: 4 flw + 2 vle32 + 4 vfmacc.vf + 3 = 13 instr
- 改进: 1 vle32 (A, 2元素) + 2 vle32 (B) + 4 vfmacc.vv + 3 = 10 instr
- 减少: (13 - 10) / 13 = **23.1%**

对于4行、K-unroll=2（假设）:
- 当前: 8 flw + 2 vle32 + 8 vfmacc.vf + 3 = 21 instr
- 改进: 1 vle32 (A, 4元素) + 1 vslide1up (broadcast setup) + 2 vle32 (B) + 8 vfmacc.vv + 3 = 15 instr
- 减少: (21 - 15) / 21 = **28.6%**

---

### 5. 归一化综合收益分析（RVV VLEN=512 vs Z-Vector VLEN=128）

**归一化因子**: VLEN_RVV / VLEN_ZVector = 512 / 128 = 4x

Z-Vector每条vector指令处理4个float32，RVV每条处理16个。为公平比较，将Z-Vector指令数归一化:

**等效工作负载**: 8行 x 16列 x 4 K步 = 512 FMA

| 平台 | 归一化指令数 | 归一化FMA效率 | 相对RVV 2行baseline |
|------|-------------|--------------|---------------------|
| Z-Vector 8行 K-unroll=4 | 203 | 0.40 instr/FMA | baseline |
| RVV 2行 K-unroll=2 | 576 (128 K-pairs x ~4.5) | 1.13 instr/FMA | +183% |
| RVV 8行 K-unroll=2 (假设) | ~288 | 0.56 instr/FMA | +42% |

**注意**: Z-Vector的203条指令处理128 FMA（8行x4Kx4列，其中4列是因为Z-Vector需要4个128-bit VR覆盖16列），归一化后实际覆盖512 FMA（4x列宽）。

**综合归一化**:

| 度量 | Z-Vector 8行 | RVV 2行 baseline | 差距 |
|------|-------------|------------------|------|
| 指令/FMA (归一化) | 0.40 | 1.13 | RVV多183% |
| B load/FMA | 0.031 (16/512) | 0.063 (2/32 per pair, scaled) | Z-Vector优 50% |
| A setup/FMA | 0.047 (24/512) | 0.125 (16/128) | Z-Vector优 62% |

---

## RVV扩展指令建议详细说明

### [P0] 多行处理扩展（软件优化：扩展至4/8行）

**建议定义**: 修改RVV SGEMM内核，将行数从2扩展至4或8，使用多个vector accumulator。

**应用场景**: 大M维度的GEMM（如全连接层），多行处理可摊销A加载和B加载成本。

**性能对比**（归一化到64 FMA, 16列, 4 K步）:

| 版本 | A loads | B loads | FMA | overhead | 总指令 | 减少 |
|------|---------|---------|-----|----------|--------|------|
| RVV 2行 (当前) | 16 flw | 4 vle32 | 16 | 6 | 42 | baseline |
| RVV 4行 (扩展) | 16 flw | 4 vle32 | 16 | 3 | 39 | 7.1% |
| RVV 8行 (扩展) | 32 flw | 4 vle32 | 32 | 3 | 71 | N/A (128 FMA) |
| Z-Vector 4行 | 4 ld + 8 perm | 16 ld | 64 | 3 | 95 | N/A (64 FMA) |

**说明**: 多行扩展的收益主要体现在B加载摊销（每次B load被更多行复用），但A加载成本线性增长。当扩展到8行时，A加载成为主要开销（32 flw vs 16 FMA），此时需要配合P1（vector-vector FMA）才能获得净收益。

**实现难度**: 低（纯软件修改，不需要新指令）

---

### [P1] `vfmacc.vv` 向量-向量FMA（3-operand语义）

**指令定义**:
- 格式: `vfmacc.vv vd, vs1, vs2, vm` — `vd = vs1 * vs2 + vd`
- 或新增3-operand变体: `vfmadd.vv vd, vs1, vs2, vs3, vm` — `vd = vs1 * vs2 + vs3`（非破坏性）
- 约束: SEW=32, LMUL>=1

**应用场景**: 当A元素已预加载为broadcast vector时，可直接与B vector相乘，省去逐元素标量加载。

**性能对比**（2行, K-unroll=2, 归一化到4 FMA）:

| 版本 | A setup | B loads | FMA | overhead | 总指令 | 减少 |
|------|---------|---------|-----|----------|--------|------|
| RVV 当前 (vfmacc.vf) | 4 flw | 2 vle32 | 4 vfmacc.vf | 3 | 13 | baseline |
| RVV + vfmacc.vv | 1 vle32 (A) + 1 vrgather | 2 vle32 | 4 vfmacc.vv | 3 | 11 | **15.4%** |

**4行, K-unroll=2, 16 FMA**:

| 版本 | A setup | B loads | FMA | overhead | 总指令 | 减少 |
|------|---------|---------|-----|----------|--------|------|
| RVV 当前 | 8 flw | 2 vle32 | 8 vfmacc.vf | 3 | 21 | baseline |
| RVV + vfmacc.vv | 1 vle32 (A) + 1 vrgather | 2 vle32 | 8 vfmacc.vv | 3 | 15 | **28.6%** |

**8行, K-unroll=2, 32 FMA**:

| 版本 | A setup | B loads | FMA | overhead | 总指令 | 减少 |
|------|---------|---------|-----|----------|--------|------|
| RVV 当前 | 16 flw | 2 vle32 | 16 vfmacc.vf | 3 | 37 | baseline |
| RVV + vfmacc.vv | 2 vle32 (A) + 2 vrgather | 2 vle32 | 16 vfmacc.vv | 3 | 25 | **32.4%** |

**实现难度**: 中（需要新增或扩展RVV FMA指令编码）

**RVV现状**: RVV 1.0规范仅有 `vfmacc.vf`（标量广播）和 `vfmacc.vv`（向量-向量），但 `vfmacc.vv` 的语义是 `vd = vs1 * vs2 + vd`（destructive），且vs1需要是完整的vector，没有"broadcast a single element from vs1"的选项。ARM NEON的 `vmlaq_lane_f32` 提供了类似功能（从另一个vector的特定lane广播）。

---

### [P2] `vrgather` 批量广播（多元素加载+广播优化）

**指令定义**: 利用现有 `vrgather.vv vd, vs2, vs1, vm` 指令实现多元素广播。`vs1` 指定目标lane索引，从 `vs2` 收集元素到 `vd`。

**应用场景**: 一次性将N个A元素（预加载到一个vector中）广播为N个广播向量。

**性能对比**（4行, A广播setup）:

| 版本 | 指令数 | 说明 |
|------|--------|------|
| RVV 当前 (逐元素flw) | 4 flw | 4条标量加载 |
| RVV + vrgather broadcast | 1 vle32 (A, 4元素) + 2 vrgather = 3 | 减少 25.0% |

**实现难度**: 中（`vrgather` 已在RVV 1.0中定义，但需要构造索引向量，索引向量本身占用寄存器）

**注意**: Z-Vector的 `vec_perm` 比 RVV 的 `vrgather` 更灵活。`vec_perm` 可以同时执行任意lane级别的重排（类似AVX512的 `vpermi2ps`），而 `vrgather` 只能从源vector的指定位置收集元素。Z-Vector的8条 `vec_perm` 实现4元素到4广播向量的转换，在RVV上需要多条 `vrgather` + 辅助指令才能完成等效操作。

---

## BBV热点加权收益分析

无BBV profiling数据，无法计算整体收益。
上表中的"BB内收益"仅反映单个BB范围内的指令减少比例。
建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算整体收益。

---

## 附录

### FMA指令对比表

| 特性 | S390X `vfmasb` | RVV `vfmacc.vf` | RVV `vfmacc.vv` |
|------|----------------|-----------------|-----------------|
| 操作数 | 3 (a, b, c) | 2 (vd, vs2) + scalar rs1 | 2 (vd, vs2) + vector vs1 |
| 语义 | `dst = a*b + c` | `vd = rs1*vs2 + vd` | `vd = vs1*vs2 + vd` |
| 破坏性 | 否（c不变） | 是（vd被覆盖） | 是（vd被覆盖） |
| 广播方式 | 外部broadcast VR | rs1自动广播 | vs1逐元素 |
| 寄存器宽度 | 128-bit | VLEN (512-bit) | VLEN (512-bit) |
| 元素数/instr | 4 float32 | 16 float32 | 16 float32 |

### 数据重排指令对比表

| 特性 | S390X `vec_perm` | RVV `vrgather.vv` | ARM NEON `vtrn`/`vzip` |
|------|------------------|-------------------|----------------------|
| 功能 | 任意lane级重排 | 按索引收集元素 | 特定transpose模式 |
| 源操作数 | 2个vector | 1个vector + 1个索引 | 2个vector |
| 索引来源 | 编译时常量mask | 运行时vector | 编译时内置 |
| 批量广播 | 支持（4元素→4广播VR） | 支持（需要构造索引） | 不支持 |

### 加载/存储指令对比表

| 特性 | S390X `vl` (vector load) | RVV `vle32.v` |
|------|--------------------------|---------------|
| 宽度 | 128-bit (4 float32) | VLEN (512-bit, 16 float32) |
| stride | 仅unit stride | unit stride + stride + indexed |
| 元素数/load | 4 | 16 |
| 覆盖16列所需loads | 4 | 1 |

---

## 结论

S390X Z-Vector GEMM内核相比RVV VL=16 baseline，在以下方面具有结构性优势：

1. **多行处理 (P0)**: Z-Vector处理多达8行，B加载效率是RVV 2行的4倍。将RVV扩展至4-8行是纯软件优化，实现难度低，可立即摊销B加载成本。但A加载成本线性增长限制了纯多行扩展的收益上界。

2. **3-operand FMA (P1)**: Z-Vector的`vfmasb`使用预计算的broadcast vector直接参与FMA，避免逐元素标量加载。对于8行场景，配合多行处理可将K循环BB内指令减少**32.4%**。这是最值得引入的RVV扩展。

3. **`vec_perm`批量广播 (P2)**: Z-Vector使用`vec_perm`以12条指令完成4行x4K步的A广播setup。RVV的`vrgather`可部分实现类似功能，但需要额外的索引向量构造开销。收益为BB内减少约**16.7%**。

4. **K展开因子**: Z-Vector展开为4 vs RVV展开为2，主要减少循环控制开销（指针更新、计数器），对K循环BB的收益有限（约5-8%）。

5. **综合潜力**: P0 + P1 + P2 叠加（多行 + vv-FMA + 批量广播），K循环BB内理论最大减少约**50%**。但需要BBV数据才能估算整体收益。

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1   |           |        |      |

最终审查结论：待审查
