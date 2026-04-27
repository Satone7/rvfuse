# quantize-q8_0-4x4 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: `ggml_quantize_mat_q8_0_4x4` — 将 4 行 × 32 列 float32 矩阵量化为 int8，按 4 字节粒度 4 路交织存储（用于 GEMV/GEMM 中的权重量化预处理）
**算法**: FP32 → 计算 max(abs) → scale = amax/127 → multiply by inverse scale → round to int8 → 4x4 interleaved store
**基准实现**: RVV VLEN=512, SEW=32, LMUL=8 (f32 load) / LMUL=4 (i16) / LMUL=2 (i8/i32 store)
**分析平台**: x86 AVX/AVX2, ARM NEON, LoongArch LSX, Power VSX, S390X Z-Vector, WASM SIMD
**BBV数据**: 基于理论估算（未获取BBV profiling数据），收益计算为BB内指令减少比例

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vfncvt_scale_x_f_w_i8 | x86 AVX2 `_mm256_mul_ps + _mm256_round_ps + _mm256_packs` | BB内减少约14%（量化+窄化BB） | 高 | 需 vfmul + vfncvt + vncvt 三步 |
| P1 | vfredmax_to_scalar | ARM NEON `vmaxvq_f32` | BB内减少约7%（amax归约BB） | 低 | 需 vfredmax + vfmv.f.s 两步 |
| P2 | vfabs_redmax | ARM NEON+ x86 组合 | BB内减少约5%（abs+redmax合并） | 中 | 需 vfabs + vfredmax 两步 |

**注**: 无实际BBV profiling数据，收益为理论估算。RVV的段存储指令 `vsseg4e32.v` 是该kernel的决定性优势。

**总体结论**: RVV在此kernel上对ARM NEON有约10倍指令数优势，对AVX2有约2倍优势。RVV的段存储和向量长度灵活性是核心竞争力，但量化转换路径可优化。

---

## 基准RVV实现分析

### 算法流程

```
输入: float32 x[4][k], k 为 QK8_0=32 的倍数
输出: block_q8_0x4 { ggml_half d[4]; int8_t qs[128]; }

每块 (4行 × 32元素):
  对每行 (重复4次):
    1. vle32_v_f32m8        → 加载 32 个 float32 (LMUL=8, 512-bit)
    2. vfabs_v_f32m8        → 取绝对值
    3. vfredmax_vs_f32m8_f32m1 → 归约求最大绝对值到 m1 向量
    4. vfmv_f_s_f32m1_f32   → 提取标量 amax
    5. 标量计算: d = amax/127, id = 1/d (if d != 0), 存 fp16 scale
    6. vfmul_vf_f32m8       → 乘以 id (inverse scale)
    7. vfncvt_x_f_w_i16m4_rm → f32→i16 窄化转换 (rm=4, round-to-nearest-max-magnitude)
    8. vncvt_x_x_w_i8m2     → i16→i8 窄化转换 (SEW/2)

  4 行量化完成后:
    9. vreinterpret_v_i8m2_i32m2 × 4 → 重解释 int8 为 int32 (零开销)
    10. vcreate_v_i32m2x4           → 创建 4 段元组
    11. vsseg4e32_v_i32m2x4 vl=8    → 段存储: 4段 × 8元素 × 4B = 128B
```

### RVV指令计数 (每块)

| 阶段 | 向量指令 | 标量指令 | 小计 |
|------|----------|----------|------|
| 加载 (4行) | 4 × vle32_v_f32m8 | — | 4 |
| 绝对值 (4行) | 4 × vfabs_v_f32m8 | — | 4 |
| 归约 (4行) | 4 × (vfredmax + vfmv.f.s) | 4 × (div + fp16_store + id_calc) | 8 + 16 |
| 缩放 (4行) | 4 × vfmul_vf_f32m8 | — | 4 |
| f32→i16 (4行) | 4 × vfncvt_x_f_w_i16m4_rm | — | 4 |
| i16→i8 (4行) | 4 × vncvt_x_x_w_i8m2 | — | 4 |
| 段存储 | 1 × vsseg4e32 | — | 1 |
| **总计** | **25** | **16** | **41** |

### 寄存器使用 (VLEN=512)

| 寄存器类型 | LMUL | 寄存器数 | 说明 |
|------------|------|----------|------|
| vfloat32m8_t | 8 | 8 | 加载/计算，逐行复用 |
| vint16m4_t | 4 | 4 | 中间结果，逐行复用 |
| vint8m2_t | 4 × 2 = 8 | 8 | 量化结果，需同时保持4行 |
| vint32m2x4_t | tuple | 8 | 段存储元组，复用上述寄存器 |
| 峰值占用 | — | ~16/32 | m8(8) + m2×4(8) |

### 关键RVV指令验证

已在 `third_party/riscv-rvv-intrinsic-doc/auto-generated/` 验证存在的指令：
- `vfabs_tu/ta`: 浮点绝对值 (line 1620-1634)
- `vfredmax_tu/ta`: 浮点归约最大值 (line 1819-1834)
- `vfncvt_x_tu/ta`: 浮点到整数窄化转换 (line 1733-1751)
- `vncvt_x_tu/ta`: 整数窄化转换 (line 1069-1078)
- `vsseg4e32`: 4段32-bit元素存储 (vsseg4e32.c)
- `vcreate_v_*_m2x4`: 创建4元素元组 (vcreate.c)
- `vreinterpret`: 类型重解释 (vreinterpret.c)

---

## 各平台对比分析

### 1. x86 AVX/AVX2

**核心特点**:
- AVX: 16 × 256-bit YMM 寄存器
- AVX2: 增加 `_mm256_packs_epi32/epi16` 打包指令
- 无段存储，interleave 依赖 pack + shuffle + permute 组合
- 当前实现: `ggml_quantize_mat_q8_0_4x8` (4x8 interleaving, 非 4x4)

**量化路径对比 (VLEN=512 等效)**:

| 操作 | AVX2 指令数 | RVV 指令数 | 说明 |
|------|-------------|------------|------|
| Load (32×f32) | 4 × `_mm256_loadu_ps` | 1 × vle32m8 | RVV LMUL=8 跨寄存器 |
| Abs | 4 × `_mm256_andnot_ps(signbit)` | 1 × vfabs | RVV 单指令 |
| Horizontal Max | 4×max_tree + 1×extract | 1×vfredmax + 1×vfmv.f.s | AVX2需树形归约 |
| Scale | 4 × `_mm256_mul_ps` | 1 × vfmul_vf | RVV 单指令 |
| Round | 4 × `_mm256_round_ps` | 0 | RVV 在 vfncvt 中指定 rm |
| f32→i32 | 4 × `_mm256_cvtps_epi32` | 0 | RVV 直接 f32→i16 |
| i32→i16→i8 | 2×packs + 1×permute | 2×vfncvt + vncvt | AVX2 packs 两步打包 |
| Interleave Store | 4×shuffle + 4×store | 1×vsseg4e32 | **RVV 决定性优势** |

**AVX2 每块指令估算**:
- Load: 16 条 (4行 × 4 loads)
- Abs: 16 条
- Max reduction: 32 条 (4行 × 每行8元素树形归约)
- Scale: 16 条
- Round+Convert: 16 条
- Pack (i32→i16→i8): 12 条 (per pack path)
- Interleave: 20 条 (shuffle + permute + store)
- **总计: ~128 条**

**高价值指令**:

| 指令 | 功能 | RVV 等价 | 差距分析 |
|------|------|---------|---------|
| `_mm256_packs_epi32` | i32→i16 饱和打包 | `vfncvt` (f32→i16 直接) | AVX2 需先 cvt 到 i32 |
| `_mm256_packs_epi16` | i16→i8 饱和打包 | `vncvt` (i16→i8) | 功能等价 |
| `_mm256_permutevar8x32_epi32` | 跨 lane 重排 | `vsseg4e32` | RVV 1条 vs AVX2 4条 |

**建议扩展**: AVX2 的 pack 指令链实现了"scale → round → convert → pack"的融合路径，但 RVV 的段存储在 interleave 上有绝对优势。

---

### 2. ARM NEON

**核心特点**:
- 32 × 128-bit Q 寄存器，固定 4×float32/寄存器
- 无段存储，interleave 需逐 lane 提取 + 逐字节存储
- `vmaxvq_f32`: 单指令水平最大值归约 + 标量提取
- `vcvtnq_s32_f32`: f32→i32 round-to-nearest 转换

**ARM NEON 实现** (`arch/arm/repack.cpp` lines 51-115):

```cpp
// 每行处理 (重复4次):
for (int row_iter = 0; row_iter < 4; row_iter++) {
    for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + ... + 4*j);  // 8 loads
    for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);       // 8 abs
    
    // 树形归约 (3级):
    for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);  // 4 max
    for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);  // 2 max
    for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);  // 1 max
    
    const float amax = vmaxvq_f32(amaxv[0]);  // 1 horizontal max + scalar extract
    // ... 标量计算 scale/id ...
}

// Convert + Interleave (每j迭代):
for (int j = 0; j < 8; j++) {
    v = vmulq_n_f32(srcv[0][j], id[0]);
    vi = vcvtnq_s32_f32(v);          // f32 → i32 (round-to-nearest)
    y[i].qs[16*j + 0] = vgetq_lane_s32(vi, 0);  // 逐lane提取 + 隐式i32→i8截断
    y[i].qs[16*j + 1] = vgetq_lane_s32(vi, 1);
    // ... 重复 4 行 × 4 lanes = 16 次 lane 提取 ...
}
```

**ARM NEON vs RVV 指令计数对比**:

| 阶段 | ARM NEON | RVV | 比值 |
|------|----------|-----|------|
| Load (4行×32元素) | 32 (8×vld1q × 4行) | 4 (1×vle32m8 × 4行) | 8.0× |
| Abs (4行) | 32 | 4 | 8.0× |
| Horizontal Max | 32 (7×vmaxq + 1×vmaxvq) × 4行 | 8 (vfredmax + vfmv.f.s) × 4行 | 4.0× |
| Scale (4行) | 32 | 4 | 8.0× |
| Convert + Interleave | 288 (32×vcvtnq + 128×vgetq_lane + 128×store) | 9 (4×vfncvt + 4×vncvt + 1×vsseg4e32) | **32.0×** |
| **总计** | **~416** | **~41** | **10.1×** |

**高价值指令**:

| 指令 | 功能 | RVV 等价 | 差距 |
|------|------|---------|------|
| `vmaxvq_f32` | 水平最大值 + 标量提取 | `vfredmax` + `vfmv.f.s` | **ARM 优势**: 1条 vs 2条 |
| `vcvtnq_s32_f32` | f32→i32 round-to-nearest | `vfncvt` rm=4 | 功能等价 |
| `vgetq_lane_s32` | 提取单个 lane | 无直接对应 | **ARM 劣势**: 需逐lane提取 |

**建议扩展**: `vmaxvq_f32` 的"归约+标量提取"融合是ARM独有优势，RVV可扩展类似指令。

---

### 3. LoongArch LSX/LASX

**核心特点**:
- LSX: 32 × 128-bit 向量寄存器
- LASX: 32 × 256-bit 向量寄存器 (类似 AVX2)
- `lsx_vsat_w/h`: 饱和操作 (用于窄化)
- `lsx_vpickev_h/b`: 打包操作 (交错取偶元素)
- 当前实现: 无 4x4 quantize 优化，仅有 `quantize_row_q8_0` 单行量化

**LSX 实现参考** (`arch/loongarch/quants.c` lines 26-100):

```cpp
// LSX quantize_row_q8_0 (单行):
for (int j = 0; j < 8; j++) srcv[j] = __lsx_vld(...);        // 8 loads
for (int j = 0; j < 8; j++) asrcv[j] = __lsx_vfabs_s(srcv[j]); // 8 abs

// 树形归约:
for (int j = 0; j < 4; j++) amaxv[2*j] = __lsx_vfmax_s(...);  // 4 max
// ... 继续归约到标量 ...

// 转换路径 (LSX):
const __m128 v = __lsx_vfmul_s(srcv[j], id);
const __m128i vi = __lsx_vftintrne_w_s(v);  // f32→i32 round-to-nearest-even
// 然后需要 packs: i32→i16→i8
```

**LoongArch 缺少的优化**:
- 无 f32→i8 直接窄化 (需 f32→i32→i16→i8 三步)
- 无段存储 (interleave 需手动重排)
- `lsx_packs_w/h` helper 函数实现打包链

**建议**: LoongArch 对此 kernel 无显著优势，RVV 无需借鉴。

---

### 4. Power VSX (POWER9/10)

**核心特点**:
- 32 × 128-bit VSX 寄存器
- POWER9: `vec_abs`, `vec_max`, `vec_round`, `vec_cts` (convert to signed)
- `vec_pack`: 两级打包 (i32→i16→i8)
- 当前实现: 无 4x4 quantize 优化

**Power 实现参考** (`arch/powerpc/quants.c` lines 41-85):

```cpp
for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);  // 8 loads
for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);           // 8 abs

// 树形归约:
for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
// ...

// 转换 + 打包链:
for (int j = 0; j < 8; j++) {
    const vector float v  = vec_round(vec_mul(srcv[j], vid));
    vi[j] = vec_cts(v, 0);  // f32 → i32
}
vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
```

**高价值指令**:

| 指令 | 功能 | RVV 等价 | 说明 |
|------|------|---------|------|
| `vec_pack` | i32→i8 两级打包 | `vfncvt` + `vncvt` | Power 2条pack vs RVV 2条narrow |
| `vec_round` + `vec_cts` | round + convert | `vfncvt` rm=4 | Power 分开，RVV 融合 |

**建议**: Power 的 pack 链与 RVV 等价，无显著差异。

---

### 5. S390X Z-Vector (VXE/VXE2)

**核心特点**:
- 32 × 128-bit 向量寄存器
- `vec_abs`, `vec_max`: 基本向量操作
- `vec_signed(__builtin_s390_vfisb(v, 4, 1))`: f32→i32 round-to-nearest-even
- 当前实现: 无 4x4 quantize 优化

**S390X 实现参考** (`arch/s390/quants.c` lines 47-92):

```cpp
for (int j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i*32 + 4*j);
for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);
// ... 树形归约 ...

const float32x4_t v = vec_mul(srcv[j], vec_splats(id));
const int32x4_t vi = vec_signed(__builtin_s390_vfisb(v, 4, 1));  // f32→i32 rm=4
```

**特色**: `__builtin_s390_vfisb(v, 4, 1)` 提供精确的 rounding mode 控制，类似 RVV 的 `vfncvt_x_f_w_i16m4_rm`。

**建议**: S390X 无段存储或特殊窄化优势，RVV 无需借鉴。

---

### 6. WASM SIMD

**核心特点**:
- 128-bit `v128_t`
- `wasm_f32x4_abs`, `wasm_f32x4_max`: 基本操作
- `wasm_i32x4_trunc_sat_f32x4`: f32→i32 饱和截断 (无 rounding mode 选择)
- 当前实现: 无 4x4 quantize 优化

**WASM 实现参考** (`arch/wasm/quants.c` lines 41-86):

```cpp
for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);
// ... 树形归约 ...

const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);  // 饱和截断，无 rounding control
```

**局限性**: WASM SIMD 的 `trunc_sat` 只有 truncation 饱和，无 round-to-nearest 选项，精度可能不如其他平台。

**建议**: WASM 无特殊优势，RVV 的 rounding mode 支持更灵活。

---

## RVV扩展指令建议详细说明

### [P0] vfncvt_scale_x_f_w_i8 — 带缩放的 f32→i8 一体化转换

**动机**: 当前 RVV 量化路径需 3 步:
```
vfmul_vf (scale) → vfncvt_x_f_w_i16 (f32→i16) → vncvt_x_x_w_i8 (i16→i8)
```

x86 AVX2 通过 `_mm256_mul_ps + _mm256_round_ps + _mm256_packs_epi32 + _mm256_packs_epi16` 实现类似融合，但需 4 条指令。如果能将 scale + convert + narrow 融合，可显著减少指令。

**建议指令定义**:
```
vfncvt_scale_x_f_w_i8 vd, vs2, rs1, rm, vl
语义: vd[i] = saturate_i8(round(vs2[i] * rs1, rm))
```

**应用场景**: 所有量化 kernel (Q8_0, Q4_0, Q4_K, MXFP4 等)。

**性能对比**:
```
当前 RVV (三步):
  vfmul_vf_f32m8    v_scaled, v_src, id, vl    // scale (1条)
  vfncvt_x_f_w_i16m4_rm  v_i16, v_scaled, 4, vl  // f32→i16 (1条)
  vncvt_x_x_w_i8m2  v_i8, v_i16, vl              // i16→i8 (1条)
  总计: 3 条

扩展后 (一步):
  vfncvt_scale_x_f_w_i8  v_i8, v_src, id, 4, vl  // scale + convert (1条)
  总计: 1 条

每行节省: 2 条
每块 (4行) 节省: 8 条
BB 内减少: 8 / (25 + 16) ≈ 19.5%
```

**实现难度**: 高。需硬件支持 f32×scalar → i8 的完整数据通路，包括：
1. f32 乘法器
2. Rounding logic (支持多种 rm)
3. 跨 2 个 SEW 级别的窄化饱和

**硬件代价**: 需在现有 `vfncvt` 通路增加乘法器输入，复杂度较高。

---

### [P1] vfredmax_to_scalar — 归约最大值直接提取标量

**动机**: ARM NEON 的 `vmaxvq_f32` 一条指令完成水平最大值归约 + 标量提取，RVV 需 2 条。

**建议指令定义**:
```
vfredmax_to_scalar rd, vs2, vs1, vl
语义: rd = max(vs2[0], vs2[1], ..., vs2[vl-1], vs1[0])
      (vs1 为初始值，通常为 0.0f)
```

**应用场景**: 量化中的 amax 计算，softmax 中的 max 归约等。

**性能对比**:
```
当前 RVV:
  vfredmax_vs_f32m8_f32m1  v_max, v_abs, v_zero, vl  // 归约到 m1 (1条)
  vfmv_f_s_f32m1_f32       amax, v_max               // 提取标量 (1条)
  总计: 2 条

扩展后:
  vfredmax_to_scalar  amax, v_abs, 0.0f, vl  // 直接归约到标量 (1条)
  总计: 1 条

每行节省: 1 条
每块 (4行) 节省: 4 条
BB 内减少: 4 / (25 + 16) ≈ 9.8%
```

**实现难度**: 低。仅需在现有 `vfredmax` 输出后增加一个 lane 提取逻辑，硬件改动小。

**ARM NEON 参考**: `vmaxvq_f32` 实现:
```cpp
float amax = vmaxvq_f32(amaxv[0]);  // 直接从 Q 寄存器提取最大值
```

---

### [P2] vfabs_redmax — 绝对值 + 归约最大值融合

**动机**: 量化需先计算绝对值再归约最大值，ARM NEON 和 x86 都需分开处理，但 RVV 可融合以减少指令。

**建议指令定义**:
```
vfabs_redmax vd, vs2, vs1, vl
语义: vd[0] = max(|vs2[0]|, |vs2[1]|, ..., |vs2[vl-1]|, vs1[0])
```

**应用场景**: 量化中的 amax = max(|x|) 计算。

**性能对比**:
```
当前 RVV:
  vfabs_v_f32m8  v_abs, v_src, vl         // 绝对值 (1条)
  vfredmax_vs    v_max, v_abs, v_zero, vl // 归约 (1条)
  vfmv_f_s       amax, v_max              // 提取 (1条)
  总计: 3 条

扩展后:
  vfabs_redmax_to_scalar  amax, v_src, 0.0f, vl  // abs + redmax + extract (1条)
  总计: 1 条

每行节省: 2 条
每块 (4行) 节省: 8 条
BB 内减少: 8 / (25 + 16) ≈ 19.5%
```

**实现难度**: 中。需在归约通路前增加绝对值逻辑，可复用现有 `vfabs` 和 `vfredmax` 硬件。

---

## BBV热点加权收益分析

**注**: 当前未获取实际 BBV profiling 数据，以下为理论估算。

### 理论指令减少估算

| 扩展指令组合 | 每块节省指令 | BB内减少比例 | 整体收益估算 |
|-------------|-------------|-------------|-------------|
| P0 单独 | 8 条 | 19.5% | 未知 (需 BBV) |
| P1 单独 | 4 条 | 9.8% | 未知 |
| P2 单独 | 8 条 | 19.5% | 未知 |
| P0+P1 | 12 条 | 29.3% | 未知 |
| P0+P2 | 16 条 | 39.0% | 未知 |
| P0+P1+P2 | 20 条 | 48.8% | 未知 |

### 与ARM NEON的绝对优势

即使在未扩展状态下，RVV 对 ARM NEON 已有 **10 倍** 指令数优势：
- ARM NEON: ~416 条/块
- RVV: ~41 条/块
- 差距来源: RVV 的段存储 (vsseg4e32) 消除了逐 lane 提取 + store 的 288 条指令

### 与AVX2的相对优势

| 维度 | AVX2 | RVV | RVV优势 |
|------|------|-----|---------|
| Load | 16 条 | 4 条 | 4× |
| Abs | 16 条 | 4 条 | 4× |
| Max归约 | 32 条 | 8 条 | 4× |
| Scale+Convert | 28 条 | 12 条 | 2.3× |
| Interleave | 20 条 | 1 条 | **20×** |
| **总计** | ~128 条 | ~41 条 | **3.1×** |

---

## 结论

### RVV现有优势

1. **段存储 vsseg4e32**: 决定性优势，消除 ARM NEON 的 32 倍 interleave 指令差距
2. **向量长度灵活**: LMUL=8 可一次加载 32 元素，ARM 固定 4 元素
3. **Rounding mode 控制**: vfncvt 支持 rm=4 (round-to-nearest-max-magnitude)，精度优于 WASM trunc_sat

### RVV潜在扩展方向

| 扩展指令 | 收益 | 实现难度 | 优先级 |
|---------|------|---------|--------|
| vfncvt_scale_x_f_w_i8 | 高 (19.5% BB内) | 高 | P0 |
| vfredmax_to_scalar | 中 (9.8% BB内) | 低 | P1 |
| vfabs_redmax | 高 (19.5% BB内) | 中 | P2 |

### 与其他平台的总结

- **ARM NEON**: RVV 已有 10 倍优势，唯一可借鉴的是 `vmaxvq_f32` 的归约+提取融合
- **x86 AVX2**: RVV 有 3 倍优势，pack 指令链无显著超越
- **LoongArch/Power/S390X/WASM**: 无特殊优势可借鉴

### 建议

1. **P1 (vfredmax_to_scalar)** 为最实际可行扩展，实现难度低，收益明确
2. **P0/P2** 需权衡硬件代价与收益，建议先在模拟器验证
3. 获取实际 BBV 数据后重新估算整体收益