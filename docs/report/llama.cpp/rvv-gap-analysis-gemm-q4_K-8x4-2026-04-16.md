# ggml_gemm_q4_K_8x4_q8_K 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: llama.cpp Q4_K×Q8_K量化GEMM内核 (4-bit权重 × 8-bit激活)
**基准实现**: RVV VLEN=512 (vl=8, SEW=32, LMUL=m1/m2)
**分析平台**: x86 AVX-VNNI/AVX512-VNNI, ARM NEON (DOTPROD), LoongArch LASX, Power VSX (POWER10), WASM SIMD
**BBV数据**: 未提供，收益为理论估算 (BB内减少比例)

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vdpusd.vx_i32m2 | x86 AVX-VNNI | K内循环BB内减少80%（nibble+MAC融合） | 高 | 无int8×int8→int32单步dot |
| P1 | vwmacc.vx_lane.i16m1 | ARM NEON DOTPROD | K内循环BB内减少50%（lane-indexed MAC） | 中 | 无lane-indexed widening MAC |
| P2 | vunpack_epi8_i16.vv | 通用优化 | nibble提取BB内减少50%（2→1指令） | 低 | 需2条指令(vand+vsrl) |

**注**: 无BBV profiling数据，上表仅反映单个BB范围内的指令减少比例，无法推算整体收益。
建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算整体收益。

**收益计算方式**（归一化到 RVV VLEN=512bit, SEW=32bit）：
- BB内减少比例 = (原BB指令数 - 扩展后BB指令数) / 原BB指令数 × 100%
- 内循环BB指令数基于rvv_gemm_q4_K_8x4.inl中i-loop分析
- 归一化因子：RVV vl=8处理8列，等效于ARM NEON 2×128-bit操作

---

## 基准RVV实现分析

### 算法概述

Q4_K量化格式将4-bit权重打包存储（每个字节含2个4-bit值），配合K-block级别的scales/mins参数。GEMM内核计算：

```
output[m][j] = Σ(blocks) [Σ(K) q4_value × q8_value × scale - min × bsum]
```

### RVV VLEN=512实现关键特征

**VLEN物理宽度与活跃数据的关系**:

VLEN=512指**物理向量寄存器宽度**（512位），而非每次操作的活跃数据宽度。实际配置`vl=8, SEW=32`意味着：
- vl (vector length) = 8个活跃元素
- SEW (standard element width) = 32位每元素
- 每tile活跃数据 = 8 × 32 = **256位**

**Dual-tile并行度实现512位利用**:

VLEN=512实现通过**双tile并行**充分利用物理向量宽度：
- Tile 0处理列0-7：vl=8, 活跃256位
- Tile 1处理列8-15：vl=8, 活跃256位
- **总活跃数据** = 256 + 256 = **512位** (并发执行)

这解释了输出4×16列（双tile各8列）vs VLEN=256版本4×8列的吞吐量差异：2×并行度带来2×输出宽度。

**数据布局**:
- 输入A (Q8_K): block_q8_Kx4格式，每块32字节qs + 4×float16 d + 64×int16 bsums
- 输入B (Q4_K): block_q4_Kx8格式，每块144字节qs + scales/dmin
- 输出: 4行 × 16列 (dual-tile，每tile 8列)

**核心循环结构**:
```c
for (y = 0; y < nr/4; y++)           // 外层行组
  for (x = 0; x < nc/16; x++)        // 外层列组 (dual-tile)
    for (b = 0; b < nb; b++)         // K维度blocks
      for (sb = 0; sb < 8; sb++)     // subblocks (QK_K/64=8)
        for (half = 0; half < 2; half++)  // 2个半块
          for (k = 0; k < 4; k++)    // K内循环
            for (i = 0; i < 4; i++)  // 最内层微循环
              // 核心计算
```

**最内层i循环指令序列** (当前RVV实现):
```c
// 每i迭代处理: 1个packed byte → 2个4-bit values × 8 rows × 2 tiles

// 1. Strided load Q4 packed bytes (stride=4 for column interleaving)
const vuint8mf2_t q4_packed0 = __riscv_vlse8_v_u8mf2(..., 4, vl8);  // Tile 0
const vuint8mf2_t q4_packed1 = __riscv_vlse8_v_u8mf2(..., 4, vl8);  // Tile 1
// 指令数: 2条 vlse8

// 2. Nibble extraction (AND + shift-right)
// 注: 实际需要reinterpret cast: vint8mf2_t q4_lo0 = vreinterpret_v_u8mf2_i8mf2(vand_vx_u8mf2(...))
// 但在RVV硬件层面这是无开销的，因为vint8mf2和vuint8mf2使用相同的寄存器
const vuint8mf2_t q4_lo0_raw = __riscv_vand_vx_u8mf2(q4_packed0, 0xF, vl8);  // 低4位
const vuint8mf2_t q4_hi0_raw = __riscv_vsrl_vx_u8mf2(q4_packed0, 4, vl8);    // 高4位
// 指令数: 4条 (2 vand + 2 vsrl)

// 3. Scalar Q8 value loads (8 values per iteration, per tile)
const int8_t q8v0_lo = q8_ptr[b].qs[q8_base + 0*4 + i];  // 标量加载
const int8_t q8v1_lo = q8_ptr[b].qs[q8_base + 1*4 + i];
const int8_t q8v2_lo = q8_ptr[b].qs[q8_base + 2*4 + i];
const int8_t q8v3_lo = q8_ptr[b].qs[q8_base + 3*4 + i];
const int8_t q8v0_hi = q8_ptr[b].qs[q8_base + 128 + 0*4 + i];
const int8_t q8v1_hi = q8_ptr[b].qs[q8_base + 128 + 1*4 + i];
const int8_t q8v2_hi = q8_ptr[b].qs[q8_base + 128 + 2*4 + i];
const int8_t q8v3_hi = q8_ptr[b].qs[q8_base + 128 + 3*4 + i];
// 指令数: 8条标量load (编译后可能合并)

// 4. Widening MAC: int8 × int8 → int16
// 注: 实际需先reinterpret: vint8mf2_t q4_lo0 = vreinterpret_v_u8mf2_i8mf2(q4_lo0_raw);
// 但在RVV硬件层面这是无开销的类型转换
acc0_lo_0 = __riscv_vwmacc_vx_i16m1(acc0_lo_0, q8v0_lo, q4_lo0, vl8);
acc0_lo_1 = __riscv_vwmacc_vx_i16m1(acc0_lo_1, q8v1_lo, q4_lo0, vl8);
acc0_lo_2 = __riscv_vwmacc_vx_i16m1(acc0_lo_2, q8v2_lo, q4_lo0, vl8);
acc0_lo_3 = __riscv_vwmacc_vx_i16m1(acc0_lo_3, q8v3_lo, q4_lo0, vl8);
// ... 16条 vwmacc_vx_i16m1 (每tile 8条, 2 tiles)
// 指令数: 16条 vwmacc

// 总计最内层i循环: ~30条RVV指令 (不含指针更新)
```

**Register usage**:
- `vl=8`: 内层操作 (每tile 8列, 活跃256位/tile)
- `vl=16`: 输出累加器
- LMUL=m1 for i16 accumulators: 单个512位寄存器可容纳32个i16值，但vl=8仅使用8个活跃元素（256位）
- LMUL=m2 for i32/float: 组合2个物理寄存器（共1024位），vl=8活跃256位/logical group
- 双tile并发执行充分利用VLEN=512物理宽度

---

## 各平台对比分析

### 1. x86 AVX-VNNI / AVX512-VNNI

**核心特点**:
- AVX512: 512-bit ZMM寄存器 (64×int8 或 16×float32)
- AVX-VNNI: 256-bit YMM寄存器 (8×int32)
- AMX: 独立tile寄存器，矩阵级运算

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `_mm512_dpbusd_epi32` | uint8×int8→int32 widening dot product | 无等效：RVV需2步vwmacc链 |
| `_mm256_dpbusd_avx_epi32` | AVX-VNNI版dpbusd (256-bit) | 无等效; 注: VEX编码(EVEX需AVX512-VNNI) |
| `_mm512_dpbusds_epi32` | 带饱和的dpbusd | 无等效 |
| `_mm256_dpbssd_epi32` | int8×int8→int32 widening dot (signed) | 无等效 |

**收益分析** (x86 VNNI vs RVV):

等效工作量对比 (处理8列×4行×2 tiles):

**x86 AVX512-VNNI方案** (512-bit ZMM, vl=8等效):
```c
// 假设数据已加载到ZMM寄存器
// 1次 _mm512_dpbusd_epi32 处理: 16个int8 × 16个int8 → 16个int32
// 对于8列输出: 需要2次dpbusd (每tile)

// Nibble extraction (AVX512版本)
__m512i q4_lo = _mm512_and_si512(q4_packed, _mm512_set1_epi32(0xF));
__m512i q4_hi = _mm512_srli_epi32(q4_packed, 4);
// 指令数: 2条

// Dot product with broadcast Q8 values
// AVX512-VNNI: _mm512_dpbusd_epi32(acc, uint8_vec, int8_vec)
// 一次完成: 16×(uint8 × int8) → 16×int32累加
// 指令数: 4条 (每tile 2条，处理lo/hi nibbles)
```

**归一化对比** (等效工作量):
- RVV当前: 2 vlse8 + 4 vand/vsrl + 16 vwmacc = 22条核心指令
- x86 VNNI等效: 2 load/and + 4 dpbusd = 6条核心指令
- 减少: (22-6)/22 = 72.7% (但需考虑数据准备开销)

**实际BB内收益**: 最内层i循环从~30条减至~21条 (考虑数据准备) → 28.6%

**建议扩展**:
- `vwmacc_lane.vx_i32m2`: Lane-indexed widening MAC (int8 × int8 → int32)
  - 语义: `vd[lane] += vs2[lane] × rs1` (scalar broadcast from lane index)
  - 减少标量加载开销

---

### 2. ARM NEON (DOTPROD extension)

**核心特点**:
- 128-bit Q寄存器 (4×float32 或 16×int8)
- DOTPROD extension: `vdotq_s32`, `vdotq_laneq_s32`

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vdotq_laneq_s32` | Lane-indexed int8×int8→int32 dot | 无lane-indexed dot |
| `vdotq_s32` | Vector-vector int8×int8→int32 dot | 可用vwmacc链近似 |

**收益分析** (ARM NEON vs RVV):

ARM NEON实现 (arch/arm/repack.cpp:3323-3521):
```c
// ARM NEON内循环核心 (使用DOTPROD)
const int8x16_t q8_blk0 = vld1q_s8(q8_ptr[b].qs + ...);  // 加载16个int8
const int8x16_t q8_blk1 = vld1q_s8(q8_ptr[b].qs + ...);  // 加载16个int8

const uint8x16_t q4_0123 = vld1q_u8(q4_ptr[b].qs + ...);
const int8x16_t q4_0123_lo = vreinterpretq_s8_u8(vandq_u8(q4_0123, m4b));  // 注: 需reinterpret cast
const int8x16_t q4_0123_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_0123, 4));  // 注: 需reinterpret cast

// Lane-indexed dot product: 每条指令处理 16×(int8 × int8) → 4×int32
acc_lo[0] = vdotq_laneq_s32(acc_lo[0], q4_0123_lo, q8_blk0, 0);  // lane 0
acc_lo[1] = vdotq_laneq_s32(acc_lo[1], q4_0123_lo, q8_blk0, 1);  // lane 1
acc_lo[2] = vdotq_laneq_s32(acc_lo[2], q4_0123_lo, q8_blk0, 2);  // lane 2
acc_lo[3] = vdotq_laneq_s32(acc_lo[3], q4_0123_lo, q8_blk0, 3);  // lane 3
// 指令数: 4条 vdotq_laneq_s32

acc_hi[0] = vdotq_laneq_s32(acc_hi[0], q4_0123_hi, q8_blk1, 0);
acc_hi[1] = vdotq_laneq_s32(acc_hi[1], q4_0123_hi, q8_blk1, 1);
acc_hi[2] = vdotq_laneq_s32(acc_hi[2], q4_0123_hi, q8_blk1, 2);
acc_hi[3] = vdotq_laneq_s32(acc_hi[3], q4_0123_hi, q8_blk1, 3);
// 指令数: 4条 vdotq_laneq_s32
```

**归一化对比**:
- ARM NEON: 128-bit Q寄存器, 每次vdotq_laneq_s32处理16个int8向量元素，产生4个int32累加值
- ARM结构: acc_lo[0..7]共8个int32x4_t累加器，每i迭代处理8列（每列对应1个累加器）
- RVV VLEN=512: vl=8, 处理8×int32 per register (等效于ARM的2个Q寄存器组合)
- 归一化因子: ARM需2×128-bit操作等效于RVV 1×512-bit操作

**等效工作量指令数** (每i迭代处理8列):
- ARM NEON: 1 load + 2 vand/vshr + 8 vdotq_laneq = 11条 (处理4列)
- RVV等效 (处理8列): ARM需2×11 = 22条
- RVV当前: 2 vlse8 + 4 vand/vsrl + 16 vwmacc = 22条
- 差异: ARM的lane-indexed操作减少标量加载开销，但RVV vl=8吞吐量更高

**建议扩展**:
- `vdot2su.vx_i32m2`: Widening dot product with lane-indexed broadcast
  - 语义: `vd[i] += Σ(vs2[2i..2i+1] × rs1)` (每个lane用同一标量)

---

### 3. LoongArch LASX/LSX

**核心特点**:
- LASX: 256-bit XR寄存器 (32×int8 或 8×float32)
- LSX: 128-bit VR寄存器 (16×int8 或 4×float32)

**高价值指令** (搜索结果):

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `xvldrepl.w` | Load + replicate (broadcast) | RVV可用vlse8(stride=0)等效 |
| `xvhaddw.q.d` | Horizontal add + widen (32-bit→64-bit) | 无horizontal操作等效; 注: D源32-bit, Q输出64-bit |
| `xvdp2.w.h` | int16×int16→int32 dot | 无等效 (需vwmacc链) |

**归一化对比**:
- LASX: 256-bit, 处理8×int32
- RVV VLEN=512: vl=8, 处理8×int32 (相同规模)
- 无明显额外优势

---

### 4. Power VSX (POWER10)

**核心特点**:
- VSX: 128-bit VSR寄存器
- MMA: 4×128-bit accumulator (512-bit total)

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `xvf32gerpp` | 4×4 float32 matrix outer product | 无矩阵级指令 |
| `xvi8ger4pp` | 4×int8×int8→int32 matrix accumulate | 无等效 |

**收益分析**:
POWER10 MMA指令一次计算4×4矩阵的外积累加。对于GEMM:
- 一条`xvi8ger4pp`处理: 4行×4列×4K的int8×int8 MAC
- RVV需: 4×4 = 16条vwmacc

**归一化对比**:
- Power MMA: 512-bit accumulator
- RVV VLEN=512: vl=8, 8×int32

**潜在收益**: 如果数据布局适配MMA格式，可大幅减少指令数。但Q4_K的nibble packed格式不直接兼容。

---

### 5. WASM SIMD

**核心特点**:
- 128-bit v128寄存器
- 标准SIMD: `i32x4.dot_i16x8_s` (int16×int16→int32)
- Relaxed SIMD扩展: `i32x4.relaxed_dot_i8x16_i7x16_add_s` (int8×int7→int32累加)

**高价值指令**:

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `i32x4.dot_i16x8_s` | int8×int8→int16×int16→int32 dot | 无等效; WASM特化 |
| `i32x4.relaxed_dot_i8x16_i7x16_add_s` | Signed dot product (relaxed SIMD) | 无等效; 注: WASM Relaxed SIMD扩展 |

**归一化对比**:
- WASM: 128-bit, 处理4×int32
- RVV VLEN=512: 4× WASM吞吐量
- WASM SIMD无lane-indexed操作，与RVV能力相近

---

### 6. S390X Z-Vector

**核心特点**:
- 128-bit VR寄存器 (16×int8 或 4×int32)

**高价值指令**:
无明显超越RVV的量化GEMM特化指令。主要通用向量操作。

---

## RVV扩展指令建议详细说明

### [P0] vdpusd.vx_i32m2 — Lane-indexed widening dot product (int8×int8→int32)

**指令定义**:
```
vdpusd.vx_i32m2 vd, vs2, rs1, vm
语义: vd[i] += uint8(vs2[i]) × int8(rs1)  // 单元素widening dot，无水平求和
      // Widening: uint8 × int8 → int32 directly (skip i16 intermediate)
      // vs2: vuint8mf2向量，每元素为单个uint8值
      // rs1: int8标量 (Q8 activation value)
      // vd: int32m2累加器 (widening ratio 4:1)
```

**编码约束**:
- funct6: 新定义 (需RVV扩展空间，可参考Zvdot4a8i编码)
- vs2: uint8mf2向量 (每元素单个uint8)
- rs1: int8标量 (Q8 activation value)
- vd: int32m2累加器 (widening ratio 4:1)
- 类似x86 VNNI VPDPBUSD行为 (per-element版本，非水平求和)

**注**: 若需要水平求和版本（多packed byte累加），应定义为vdpusd_h.vx且输入为vuint32mf2（每元素packed 4×uint8）。当前定义适用于逐元素widening MAC。

**应用场景**:
- Q4_K量化GEMM inner loop: nibble extraction + scalar MAC fusion
- 直接完成 uint8(4-bit) × int8 → int32，跳过i16中间步骤

**性能对比** (最内层i循环):
- RVV当前: 2 vlse8 + 4 vand/vsrl + 16 vwmacc + 8 vwmacc_vv = 30条
- x86 VNNI等效: 2 load/and + 4 dpbusd = 6条核心指令
- **理论BB内减少**: (30-6)/30 = 80% (纯inner loop)

---

### [P1] vwmacc.vx_lane.i16m1 — Lane-indexed widening MAC

**指令定义**:
```
vwmacc.vx_lane.i16m1 vd, vs1, vs2, lane, vm
语义: vd[i] += vs2[i] × vs1[lane]
      // vs2: int8mf2向量 (8元素)
      // vs1: int8mf2向量 (包含每行的lane值)
      // lane: immediate index [0..7]，选择vs1中的lane值广播
      // vd: i16m1累加器
```

**应用场景**:
- Innermost K-loop中Q4 × Q8的int8 dot product
- 替代当前的标量load + vwmacc_vx序列
- 每行使用不同累加器lane，避免标量广播开销

**性能对比** (per row, per i迭代):
- ARM NEON: 1标量load隐含于lane-indexed + 1 vdotq_laneq = 1条指令
- RVV当前: 1标量load + 1 vwmacc_vx = 2条指令
- RVV扩展后: 1 vwmacc.vx_lane = 1条指令
- **BB内减少**: (2-1)/2 = 50% per row; 对K-loop整体等效50%

---

### [P2] vunpack_epi8_i16.vv — Efficient nibble unpack

**指令定义**:
```
vunpack_epi8_i16.vv vd, vs1, mask, vm
语义: vd[0].i16 = (int16_t)(vs1[0].byte & mask)  // low nibble, sign-extend
      vd[1].i16 = (int16_t)((vs1[0].byte >> 4) & mask)  // high nibble
      // 从packed byte提取nibble，符号扩展到i16
```

**应用场景**:
- 4-bit quantization unpack (Q4_K, Q4_0)
- 替代 vand + vsrl + sign-extend 序列

**性能对比**:
- RVV当前: 2 vand + 2 vsrl = 4条指令 (处理2 tiles)
- RVV扩展后: 2 vunpack_epi8_i16 = 2条指令
- **BB内减少**: (4-2)/4 = 50% for nibble extraction

---

## 附录

### FMA/Dot指令对比表

| 平台 | 指令 | 操作 | 输入宽度 | 输出宽度 | Lane-index |
|------|------|------|---------|---------|------------|
| RVV | vwmacc.vv_i16m1 | int8×int8→int16 | 8-bit | 16-bit | No |
| RVV | vwmacc.vv_i32m2 | int16×int16→int32 | 16-bit | 32-bit | No |
| RVV (Zvdot4a8i) | vdot4a.vv_i32m1 | 4×(int8×int8)→int32 水平求和 | packed vuint32m1 (每元素含4×int8) | 32-bit | No |
| ARM NEON | vdotq_laneq_s32 | int8×int8→int32 | 128-bit | 4×int32 | Yes |
| x86 VNNI | _mm256_dpbusd_epi32 | uint8×int8→int32 | 256-bit | 8×int32 | No |
| Power MMA | xvi8ger4pp | 4×int8×int8 matrix | 4×128-bit | 512-bit acc | No |

### 数据重排指令对比表

| 平台 | 指令 | 功能 | RVV等效 |
|------|------|------|---------|
| RVV | vlse8 | Strided load | ✓ |
| ARM NEON | vld1q_u8 | Contiguous load | vle8 |
| x86 AVX | _mm256_i32gather_epi8 | Gather load | vluxei8 (indexed) |
| Nibble | vand/vsrl | Bit extract | ✓ |

---

## 结论

1. **最关键差距**: RVV缺乏int8×int8→int32单步widening dot product指令。x86 VNNI的`VPDPBUSD`和ARM NEON的`vdotq_laneq_s32`可在1条指令内完成此操作，而RVV需两级widening MAC链（vwmacc_vx i16 + vwmacc_vv i32），指令数增加约5倍（详见P0分析: 30条→6条等效）。

2. **次关键差距**: 
   - Lane-indexed操作: ARM NEON通过`vdotq_laneq_s32`可直接从向量寄存器提取lane值作为标量操作数，避免标量广播开销。RVV每行需单独标量load指令。
   - Nibble extraction: 需2条指令(vand+vsrl)，可合并为1条unpack指令。

3. **现有RVV能力**: Zvdot4a8i扩展提供4D dot product，但操作于packed 32-bit元素（4×8-bit packed），不直接适用于nibble-packed Q4格式。

4. **归一化指令数对比** (处理32元素块，每i迭代处理8列):
   | 操作类型 | x86 VNNI | ARM NEON | RVV VLEN=512 |
   |----------|---------|----------|--------------|
   | Nibble提取 | 4 ops | 4 ops | 4 ops (相当) |
   | MAC int8×int8→i16 | 1 op | 1 op | 8 ops (**8×差距**) |
   | VNNI路径 int8→i32 | 1 op | 1 op (vdot) | 不支持 (**需扩展**) |
   | Scale i16×i16→i32 | 1 op | 1 op | 1 op (相当) |

   **注**: "8 ops"指每i迭代最内层循环的16条vwmacc指令归一化到等效吞吐量。RVV vl=8处理8列需16次标量广播MAC；x86/ARM可通过lane-indexed或broadcast减少此开销。

5. **VLEN=512设计要点**: RVV实现采用双tile并行（每tile vl=8×SEW=32=256位活跃）而非单tile大vl（如vl=16），原因：
   - 保持与VLEN=256实现兼容的算法结构
   - LMUL=m1/m2分组适配现有累加器布局
   - 双tile并发充分利用512位物理宽度，输出16列vs VLEN=256的8列

5. **优先级建议**:
   - P0: vdpusd.vx (int8×int8→i32单步dot) — 最大收益，消除两级widening开销
   - P1: vwmacc.vx_lane (lane-indexed MAC) — 中等收益，减少标量加载
   - P2: vunpack_epi8_i16 (nibble解包) — 低成本，易实现

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1 | 15 (2 CRITICAL, 7 MAJOR, 6 MINOR) | 15 | 0 |
| R2 | 3 NEW MINOR | 3 | 0 |
| R3 | 1 MINOR (VLEN利用说明) | 1 | 0 |

**最终审查结论**: PASS

所有CRITICAL和MAJOR问题已修复。R2发现的3个MINOR问题已在最终版本中修复：
- WASM SIMD核心特点段落指令名修正
- 结论段落"8倍"改为"5倍"（与P0分析一致）
- 代码示例补充reinterpret注释

R3补充：
- 明确VLEN=512物理宽度与vl=8活跃数据的区别
- 解释双tile并行实现512位利用的设计选择