# gemm-q4_K-8x4-q8_K 多平台向量实现对比与RVV扩展指令建议

## 概述

### 分析目标
本文档对 llama.cpp 中 `ggml_gemm_q4_K_8x4_q8_K` 函数进行跨平台向量实现对比分析，识别各平台向量指令集相对于 RVV 的优势和差距，提出 RVV 扩展指令建议。

### 基准实现
- **文件位置**: `/home/pren/wsp/rvfuse/applications/llama.cpp/rvv-patches/gemm-q4_K-8x4-q8_K/rvv_gemm_q4_K_8x4.inl`
- **VLEN**: 512 (双tile处理), 256 (单tile处理)
- **SEW**: 32 (float32累加器)
- **量化类型**: Q4_K (4-bit packed in uint8) × Q8_K (8-bit activations)
- **Block布局**: `block_q4_Kx8` (8列交错) × `block_q8_Kx4` (4行交错)
- **处理单元**: 8×4 block, K-loop处理 32×4 = 128 nibbles per subblock

### 对比平台
| 平台 | 文件位置 | 主要向量宽度 |
|------|----------|--------------|
| x86 AVX2/AVX-512 | `arch/x86/repack.cpp` | 256-bit / 512-bit |
| ARM NEON/SVE | `arch/arm/repack.cpp` | 128-bit / 可变长度 |
| LoongArch LASX | `arch/loongarch/quants.c` | 256-bit |
| Power VSX (POWER9) | `arch/powerpc/quants.c` | 128-bit |
| S390X Z-Vector (VXE/VXE2) | `arch/s390/quants.c` | 128-bit |
| WASM SIMD | `arch/wasm/quants.c` | 128-bit |

### BBV数据状态
基于 RISC-V 板卡硬件 perf 数据（Q4_K量化）：
- `ggml_compute_forward_mul_mat`: 96.84% (children) / 0.11% (self)
- `ggml_vec_dot_q4_K_q8_K_generic`: 15.88% self time
- GEMM kernel 对应矩阵乘法主路径，是推理性能瓶颈

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | RVV替代操作数 | BB内收益 | 整体收益估计 |
|--------|----------|----------|---------------|----------|--------------|
| P0 | `vusdot.vv` (uint8×int8→int32 dot) | ARM NEON `vdotq_s32` | 2 (vwmacc_vx + add) | ~15% | ~3-5% |
| P0 | `vusdot_lane.vv` (lane-indexed dot) | ARM NEON `vdotq_laneq_s32` | 4 (load+dup+vwmacc+add) | ~20% | ~4-6% |
| P1 | `vfmuladd_vf_f32` (scalar FMA) | x86 `_mm256_fmadd_ps` | 2 (vfmul+vfadd) | ~10% | ~2-3% |
| P1 | `vfwmuladd_vf_f32` (widening FMA) | x86/ARM float16 FMA | 3 (vfwcvt+vfmul+vfadd) | ~12% | ~3-4% |
| P2 | `vlseg2e8` (segment load 2) | x86 `_mm256_unpacklo_epi8` | 2 (load+shuffle) | ~8% | ~1-2% |
| P2 | `vnibunpack.vv` (nibble unpack) | x86 `_mm256_and+srli` pattern | 2 (vand+vsrl) | ~10% | ~2-3% |
| P3 | `vgetextract_i32m1` (fast LMUL trunc) | All platforms register access | 1 (vget) | ~5% | ~1-2% |

---

## 基准RVV实现分析

### 关键操作分解

#### 1. Strided Load (字节级 stride=4)
```c
// 代码位置: 第160-165行
vuint8mf2_t q4_packed0 = __riscv_vlse8_v_u8mf2(
    (const uint8_t *)&q4_ptr0[b].qs[...], (ptrdiff_t)4, vl8);
```
**用途**: 从 interleaved block_q4_Kx8 结构中按 stride=4 加载，每字节对应一个列组的 packed nibbles。

**RVV指令**: `vlse8.v` (Unit-stride with byte stride)
**替代方案**: ARM NEON `vld1q_u8` + shuffle, x86 `_mm256_loadu_si256` + `_mm256_permute`

#### 2. Bitwise Unpack (nibble extraction)
```c
// 代码位置: 第167-174行
vint8mf2_t q4_lo0 = __riscv_vreinterpret_v_u8mf2_i8mf2(
    __riscv_vand_vx_u8mf2(q4_packed0, 0xF, vl8));
vint8mf2_t q4_hi0 = __riscv_vreinterpret_v_u8mf2_i8mf2(
    __riscv_vsrl_vx_u8mf2(q4_packed0, 4, vl8));
```
**用途**: 从 packed uint8 中提取低/高 nibble (4-bit values)。

**RVV指令**: `vand.vx` + `vsrl.vx` + `vreinterpret`
**替代方案**: x86 `_mm256_and_si256` + `_mm256_srli_epi16`, ARM `vandq_u8` + `vshrq_n_u8`

#### 3. Widening Multiply-Accumulate (int8×int8→int16)
```c
// 代码位置: 第187-204行
acc0_lo_0 = __riscv_vwmacc_vx_i16m1(acc0_lo_0, q8v0_lo, q4_lo0, vl8);
```
**用途**: 将 q8 scalar (int8) 与 q4 vector (uint8→int8) 进行 widening MAC 到 int16。

**RVV指令**: `vwmacc.vx` (Widening multiply-accumulate)
**替代方案**: ARM `vdotq_s32` (直接 dot 到 int32), x86 `_mm256_maddubs_epi16` (uint8×int8→int16)

#### 4. Widening Multiply-Accumulate (int16×int16→int32)
```c
// 代码位置: 第209-225行
sumi0_0 = __riscv_vwmacc_vv_i32m2(sumi0_0, v_sc0_lo, acc0_lo_0, vl8);
```
**用途**: 将 int16 scale 与 int16 accumulator 进行 widening MAC 到 int32。

**RVV指令**: `vwmacc.vv` (Widening multiply-accumulate)
**替代方案**: x86 `_mm256_madd_epi16` (int16×int16→int32), ARM `vmlal_lane_s16`

#### 5. Float16 → Float32 Widening Conversion
```c
// 代码位置: 第88-95行
vfloat32m1_t q4_d0 = __riscv_vfwcvt_f_f_v_f32m1(
    __riscv_vle16_v_f16mf2((const _Float16 *)q4_ptr0[b].d, vl8), vl8);
```
**用途**: 将 Q4_K block 的 float16 scale/dmin 转换为 float32 进行后续计算。

**RVV指令**: `vfwcvt.f.f.v` ( Widening floating-point convert)
**替代方案**: ARM `vcvt_f32_f16`, x86 无直接指令需软件实现

#### 6. Float FMA with Accumulator
```c
// 代码位置: 第304-305行
cur = __riscv_vfmacc_vv_f32m1(cur, sumf, sbd_scale, vl8);
cur = __riscv_vfnmsac_vv_f32m1(cur, biasf, sbd_min, vl8);
```
**用途**: Float32 FMA/NMSAC 用于最终累加。

**RVV指令**: `vfmacc.vv`, `vfnmsac.vv`
**替代方案**: x86 `_mm256_fmadd_ps`, ARM `vfmaq_f32`

### 指令统计

基于 VLEN=512 版本 (`ggml_gemm_q4_K_8x4_q8_K_rvv_512`)，统计关键循环内的 RVV 指令：

| 操作类别 | 指令数 (per K-loop iteration) | 占比 |
|----------|-------------------------------|------|
| Strided load (`vlse8`) | 2 | ~2% |
| Bitwise unpack (`vand`, `vsrl`) | 4 | ~4% |
| Widening MAC int8→int16 (`vwmacc.vx`) | 16 | ~15% |
| Widening MAC int16→int32 (`vwmacc.vv`) | 16 | ~15% |
| Float16 widen (`vfwcvt`) | 4 | ~4% |
| Float MAC (`vfmacc`, `vfnmsac`) | 8 | ~8% |
| Vector load/store (`vle`, `vse`) | 12 | ~12% |
| Vector init/copy (`vmv`, `vmv_v_x`) | 32 | ~30% |
| Type conversion (`vfcvt`) | 8 | ~8% |
| 其他 (`vget`, `vfmul`) | 6 | ~6% |

**总计**: ~105 RVV instructions per function (估算)

---

## 各平台对比分析

### 1. x86 AVX2/AVX-512

#### 关键指令映射

| x86指令 | 功能 | RVV等价操作 | 差距分析 |
|---------|------|-------------|----------|
| `_mm256_maddubs_epi16` | uint8×int8→int16 (并行8组) | `vwmaccsu.vx` | RVV有等价，但需逐元素MAC而非并行dot |
| `_mm256_madd_epi16` | int16×int16→int32 (并行8组) | `vwmacc.vv` | RVV有等价 |
| `_mm256_fmadd_ps` | float32 FMA | `vfmacc.vv` | RVV有等价 |
| `_mm256_permute2f128_si256` | 128-bit lane复制 | `vmv.v.v` + shuffle | RVV无lane复制指令 |
| `_mm256_shuffle_epi32` | 32-bit元素shuffle | 无直接等价 | RVV缺乏fixed-width shuffle |
| `_mm256_cvtepu8_epi16` | uint8→int16 widening | `vwaddu.vx` (需zero扩展) | RVV需2指令 (load+extend) |
| `_mm256_hadd_epi16` | pairwise add | `vredsum.vs` (reduction) | RVV reduction需额外setup |

#### 独特优势
1. **`_mm256_maddubs_epi16`**: 单指令完成 uint8×int8→int16 的 8-way parallel multiply-add，RVV 需要 `vwmaccsu.vx` 逐元素MAC
2. **Fixed-width shuffle**: `_mm256_shuffle_epi32` 可以精确控制每个32-bit lane的位置，RVV 只能通过 `vrgather` 或软件循环实现

#### RVV缺失指令
- **Lane replication**: 类似 `_mm256_permute2f128_si256` 的 128-bit lane复制/广播指令
- **Fixed-width shuffle**: 类似 `_mm256_shuffle_epi32` 的 element-position shuffle

### 2. ARM NEON/SVE

#### 关键指令映射 (来自 repack.cpp 内联汇编)

| ARM指令 | 功能 | RVV等价操作 | 差距分析 |
|---------|------|-------------|----------|
| `vdotq_s32` | int8×int8→int32 (4-way dot) | 无直接等价 | **RVV缺失** - 需要vwmacc+vadd |
| `vdotq_laneq_s32` | lane-indexed dot product | 无直接等价 | **RVV缺失** - 需要load+dup+vwmacc+add |
| `sdot v20.4s, v12.16b, v24.4b[0]` | byte dot with lane index | 无直接等价 | **RVV缺失** |
| `vmlal_lane_s16` | widening MAC with lane index | `vwmacc.vx` (无lane index) | RVV无lane-indexed MAC |
| `vfmaq_f32` | float32 FMA | `vfmacc.vv` | RVV有等价 |
| `vcvt_f32_f16` | float16→float32 widen | `vfwcvt.f.f.v` | RVV有等价 |
| `scvtf v20.4s, v20.4s` | int32→float32 convert | `vfcvt.f.x.v` | RVV有等价 |
| `fcvtl v20.4s, v20.4h` | float16→float32 widen (pair) | `vfwcvt.f.f.v` | RVV有等价 |
| `fmul v9.4s, v17.4s, v29.s[0]` | scalar lane multiply | `vfmul.vf` | RVV有等价 (broadcast) |
| `ldr q24, [x21, #0x0]` | 128-bit load with offset | `vle8.v` | RVV有等价 |

#### 独特优势
1. **`vdotq_s32`**: **关键差距** - 单指令完成 16×int8 × 16×int8 → 4×int32 dot product，RVV 需要 `vwmacc.vx` (16次) + `vadd` (多次)
2. **`vdotq_laneq_s32` / `sdot ... v24.4b[0]`**: Lane-indexed dot product，从向量中提取特定lane作为dot operand，RVV 需要额外的 `vlse` + `vmv` + `vwmacc`

#### 代码示例分析 (ARM NEON内联汇编)
```asm
// 第2030-2033行: Lane-indexed dot product
".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
```
这条指令实现了：
- 取 `v12` (16×int8) 与 `v24` 的 lane 0/1/2/3 (4×int8) 进行 dot product
- 结果累加到 `v20/v10/v26/v2` 的 int32 accumulator

**RVV实现等价功能需要**:
```c
// RVV等价 (需要4+条指令)
int8_t lane0 = ...; // 提取lane元素 (需要vget或scalar load)
vint8mf2_t v_lane = __riscv_vmv_v_x_i8mf2(lane0, vl); // 广播
acc = __riscv_vwmacc_vx_i16m1(acc, lane0, vec, vl);  // MAC
// 还需要多次累加实现dot效果
```

### 3. LoongArch LASX

#### 关键指令映射 (来自 quants.c)

| LASX指令 | 功能 | RVV等价操作 | 差距分析 |
|----------|------|-------------|----------|
| `lasx_madd_h_b` | int8×int8→int16 MAC | `vwmacc.vx` | RVV有等价 |
| `lasx_madd_h` | int16×int16→int32 MAC | `vwmacc.vv` | RVV有等价 |
| `__lasx_xvfmadd_s` | float32 FMA | `vfmacc.vv` | RVV有等价 |
| `__lasx_xvffint_s_w` | int32→float32 convert | `vfcvt.f.x.v` | RVV有等价 |
| `__lasx_xvandi_b` | byte AND | `vand.vx` | RVV有等价 |
| `__lasx_xvsrli_b` | byte shift right | `vsrl.vx` | RVV有等价 |
| `lsx_hadd_h` | horizontal add (int16) | `vredsum.vs` | RVV有等价 |
| `lsx_madd_h` | int16×int16→int32 MAC | `vwmacc.vv` | RVV有等价 |

#### 代码示例分析
```c
// 第1235-1236行: LASX dot-like operation
__m256i p16l = lasx_madd_h_b(q4l, q8l);  // int8×int8→int16
p16l = lasx_madd_h(scale_l, p16l);        // int16×int16→int32
```
LASX使用 `madd_h_b` 模拟 dot product，但仍是逐元素MAC而非并行dot。

#### RVV对比
LoongArch LASX 的指令设计更接近 RVV，两者差距较小。主要差距在于：
- LASX 有 `lsx_hadd_h` (horizontal add) 做reduction，RVV 用 `vredsum.vs`
- LASX 的 `xvrepl128vei_h` 可以从128-bit lane复制元素，RVV 需要 `vrgather` + `vmv`

### 4. Power VSX (POWER9)

#### 关键指令映射 (来自 quants.c)

| VSX指令 | 功能 | RVV等价操作 | 差距分析 |
|---------|------|-------------|----------|
| `vec_msum` | int8×int8→int32 (partial sum) | `vwmacc.vv` + add | RVV需2指令 |
| `vec_madd` | int32 FMA | `vfmacc.vv` | RVV有等价 |
| `vec_mul` | int32 multiply | `vmul.vv` | RVV有等价 |
| `vec_ctf` | int→float convert | `vfcvt.f.x.v` | RVV有等价 |
| `vec_unpackh/l` | widen int8→int16 | `vwaddu.vx` | RVV有等价 |
| `vec_mergeh/l` | merge vector halves | `vrgather` | RVV无直接merge |
| `vec_splat` | broadcast element | `vmv.v.x` | RVV有等价 |
| `vec_nmsub` | float NMSUB | `vfnmsac.vv` | RVV有等价 |

#### 代码示例分析
```c
// 第1009-1016行: Power VSX multiply-sum
vector signed int qv00 = vec_msum(q8y00, q4x00, v0);  // int8×int8→int32
```
Power VSX 的 `vec_msum` 是 128-bit 内的 partial multiply-sum，每指令完成 16×int8×int8 → 4×int32 的累加。

#### RVV对比
Power VSX 的 `vec_msum` 接近 dot product，但范围仅限于 128-bit 内。RVV 的 `vwmacc` 是逐元素MAC，需要多次累加才能达到同样效果。

### 5. S390X Z-Vector (VXE/VXE2)

#### 关键指令映射 (来自 quants.c)

| Z-Vector指令 | 功能 | RVV等价操作 | 差距分析 |
|--------------|------|-------------|----------|
| `vec_mulo` | multiply odd elements | 无直接等价 | **RVV缺失** |
| `vec_meadd` | multiply-even add | 无直接等价 | **RVV缺失** |
| `vec_xl` | load with length | `vle.v` | RVV有等价 |
| `vec_add` | vector add | `vadd.vv` | RVV有等价 |
| `vec_sl` | shift left | `vsll.vx` | RVV有等价 |
| `vec_sr` | shift right | `vsrl.vx` | RVV有等价 |
| `vec_and` | bitwise AND | `vand.vv` | RVV有等价 |
| `vec_or` | bitwise OR | `vor.vv` | RVV有等价 |

#### 代码示例分析
```c
// 第1005-1006行: S390X odd/even multiply
const int32x4_t v_minsho = vec_mulo(v_ysums, v_minsh);
const int32x4_t v_mins = vec_meadd(v_ysums, v_minsh, v_minsho);
```
S390X 有独特的 `vec_mulo` / `vec_meadd` 指令，可以选择 odd/even 元素进行 multiply-add。

#### RVV对比
S390X 的 odd/even multiply 是 RVV 完全缺失的功能。RVV 需要通过 mask + `vmul` + `vadd` 实现，需要 3+ 条指令。

### 6. WASM SIMD

#### 关键指令映射 (来自 quants.c)

| WASM指令 | 功能 | RVV等价操作 | 差距分析 |
|----------|------|-------------|----------|
| `wasm_i32x4_dot_i16x8` | int16×int16→int32 dot | 无直接等价 | **RVV缺失** |
| `wasm_i16x8_extend_low/high_i8x16` | int8→int16 widen | `vwadd.vx` | RVV有等价 |
| `wasm_i8x16_splat` | broadcast | `vmv.v.x` | RVV有等价 |
| `wasm_v128_and` | AND | `vand.vv` | RVV有等价 |
| `wasm_u8x16_shr` | shift right | `vsrl.vx` | RVV有等价 |
| `wasm_i32x4_add` | add | `vadd.vv` | RVV有等价 |
| `wasm_i32x4_extract_lane` | extract lane | `vget` | RVV有等价 |

#### 代码示例分析
```c
// 第917-924行: WASM dot product
v128_t vacc1 = wasm_i32x4_dot_i16x8(
    wasm_i16x8_extend_low_i8x16(q4l0),
    wasm_i16x8_extend_low_i8x16(q8x0)
);
```
WASM SIMD 提供了 `wasm_i32x4_dot_i16x8`，这是 RVV 缺失的 dot product 指令。

#### RVV对比
WASM 的 `dot_i16x8` 是 128-bit 内的 8×int16 × 8×int16 → 4×int32 dot product。RVV 需要 `vwmacc.vv` + `vadd` 实现，需要多条指令。

---

## RVV扩展指令建议详细说明

### E1: `vusdot.vv` — Unsigned-Signed Dot Product (int32 output)

#### 功能描述
```
vusdot.vv vd, vs2, vs1, vl
vd[i] = sum_{j=0}^{3} (unsigned(vs2[4i+j]) * signed(vs1[4i+j])) + vd[i]
```
每条指令完成 4×uint8 × 4×int8 → int32 的 dot product 并累加。

#### 来源平台
- ARM NEON: `vdotq_s32`, `sdot v20.4s, v12.16b, v24.4b[0]`
- WASM SIMD: `wasm_i32x4_dot_i16x8` (after widen)
- x86 AVX2: `_mm256_maddubs_epi16` + `_mm256_madd_epi16` (组合实现)

#### RVV替代操作
```c
// 当前RVV实现 (需要 ~6 条指令)
vint16m1_t acc = __riscv_vmv_v_x_i16m1(0, vl);
for (int i = 0; i < 4; i++) {
    acc = __riscv_vwmacc_vx_i16m1(acc, q8_scalar[i], q4_vec, vl);
}
vint32m2_t sumi = __riscv_vwmacc_vv_i32m2(sumi, scale, acc, vl);
```

#### 收益分析
- **指令减少**: 6 → 1 (~83% BB内减少)
- **实际收益**: 考虑循环内使用频率 (16次 per subblock)，整体 BB 内指令数从 ~105 → ~89 (~15% 减少)
- **整体收益**: ~3-5% (基于 GEMM kernel 占推理时间 96.84% 的 15-20%)

### E2: `vusdot_lane.vv` — Lane-Indexed Dot Product

#### 功能描述
```
vusdot_lane.vv vd, vs2, vs1, lane_idx, vl
vd[i] = sum_{j=0}^{3} (unsigned(vs2[4i+j]) * signed(vs1[lane_idx*4+j])) + vd[i]
```
从 vs1 的特定 lane 组提取 4×int8 进行 dot product。

#### 来源平台
- ARM NEON: `vdotq_laneq_s32`, `sdot v20.4s, v12.16b, v24.4b[0]`
- Power VSX: `vec_msum` with element selection

#### RVV替代操作
```c
// 当前RVV实现 (需要 ~8 条指令)
int8_t lane_values[4];
for (int j = 0; j < 4; j++) {
    lane_values[j] = __riscv_vget_lane_i8mf2(q8_vec, lane_idx*4+j);
}
for (int j = 0; j < 4; j++) {
    acc = __riscv_vwmacc_vx_i16m1(acc, lane_values[j], q4_vec, vl);
}
```

#### 收益分析
- **指令减少**: 8 → 1 (~88% BB内减少)
- **实际收益**: 考虑循环内使用频率 (8次 per subblock for 4 lanes)，整体 BB 内指令数从 ~89 → ~81 (~20% BB内减少)
- **整体收益**: ~4-6%

### E3: `vfmuladd_vf_f32` — Scalar Float Multiply-Add

#### 功能描述
```
vfmuladd_vf_f32 vd, vs2, rs1, vl
vd[i] = vs2[i] * rs1 + vd[i]
```
单指令完成 scalar broadcast multiply + vector add。

#### 来源平台
- x86 AVX2: `_mm256_fmadd_ps` (with scalar broadcast via shuffle)
- ARM NEON: `vfmaq_f32` (with scalar lane)
- LoongArch: `__lasx_xvfmadd_s`

#### RVV替代操作
```c
// 当前RVV实现 (需要 2 条指令)
vfloat32m1_t tmp = __riscv_vfmul_vf_f32m1(vec, scalar, vl);
vd = __riscv_vfadd_vv_f32m1(vd, tmp, vl);
```

#### 收益分析
- **指令减少**: 2 → 1 (~50% BB内减少)
- **实际收益**: 考虑循环内使用频率 (8次 per block for 4 rows)，整体收益 ~10% BB内减少
- **整体收益**: ~2-3%

### E4: `vfwmuladd_vf_f32` — Widening Float Multiply-Add

#### 功能描述
```
vfwmuladd_vf_f32 vd, vs2, rs1, vl
vd[i] = (float32)(vs2[i] (float16) * rs1 (float16)) + vd[i]
```
单指令完成 float16 widening multiply + float32 add。

#### 来源平台
- x86 AVX-512: `_mm512_fmadd_ph` (FP16 FMA)
- ARM NEON: `vfmaq_f32` + `vcvt_f32_f16` (组合)

#### RVV替代操作
```c
// 当前RVV实现 (需要 3 条指令)
vfloat32m1_t f32_vec = __riscv_vfwcvt_f_f_v_f32m1(f16_vec, vl);
float f32_scalar = (float)f16_scalar;
vd = __riscv_vfmacc_vf_f32m1(vd, f32_scalar, f32_vec, vl);
```

#### 收益分析
- **指令减少**: 3 → 1 (~67% BB内减少)
- **实际收益**: 考虑 d/dmin 加载频率 (4次 per block)，整体收益 ~12% BB内减少
- **整体收益**: ~3-4%

### E5: `vlseg2e8` — Segment Load (2 elements)

#### 功能描述
```
vlseg2e8 vd0, vd1, rs1, vl
vd0[i] = Memory[rs1 + 2*i]
vd1[i] = Memory[rs1 + 2*i + 1]
```
单指令加载 interleaved 数据到两个向量。

#### 来源平台
- x86 AVX2: `_mm256_unpacklo_epi8` + load
- ARM NEON: `vld2q_u8` (interleaved load)

#### RVV替代操作
```c
// 当前RVV实现 (需要 2 条指令)
vuint8mf2_t v0 = __riscv_vle8_v_u8mf2(ptr, vl);
vuint8mf2_t v1 = __riscv_vle8_v_u8mf2(ptr + 1, vl);  // 或 stride load
```

#### 收益分析
- **指令减少**: 2 → 1 (~50% BB内减少)
- **实际收益**: 考虑 q4 加载频率，整体收益 ~8% BB内减少
- **整体收益**: ~1-2%

### E6: `vnibunpack.vv` — Nibble Unpack

#### 功能描述
```
vnibunpack.vv vd_lo, vd_hi, vs2, vl
vd_lo[i] = vs2[i] & 0xF
vd_hi[i] = vs2[i] >> 4
```
单指令从 packed uint8 提取两个 nibble。

#### 来源平台
- x86 AVX2: `_mm256_and_si256` + `_mm256_srli_epi16` (pattern)
- ARM NEON: `vandq_u8` + `vshrq_n_u8` (pattern)
- LoongArch: `__lasx_xvandi_b` + `__lasx_xvsrli_b` (pattern)

#### RVV替代操作
```c
// 当前RVV实现 (需要 2 条指令)
vuint8mf2_t lo = __riscv_vand_vx_u8mf2(vec, 0xF, vl);
vuint8mf2_t hi = __riscv_vsrl_vx_u8mf2(vec, 4, vl);
```

#### 收益分析
- **指令减少**: 2 → 1 (~50% BB内减少)
- **实际收益**: 考虑 unpack 频率 (4次 per k-loop), 整体收益 ~10% BB内减少
- **整体收益**: ~2-3%

---

## BBV热点加权收益分析

### 热点数据
基于 RISC-V 板卡硬件 perf 数据：
- `ggml_compute_forward_mul_mat`: 96.84% 总时间占比 (包含子函数)
- `ggml_vec_dot_q4_K_q8_K_generic`: 15.88% self time
- GEMM kernel 是推理性能瓶颈

### 收益计算方法
由于缺少精确的 BBV 数据，使用理论估算：

| 扩展指令 | BB内指令减少 | 函数级指令减少 | 整体推理收益 |
|----------|--------------|----------------|--------------|
| `vusdot.vv` | ~15% | ~12% | ~3-5% (96.84% × 12% × 0.3) |
| `vusdot_lane.vv` | ~20% | ~15% | ~4-6% |
| `vfmuladd_vf_f32` | ~10% | ~8% | ~2-3% |
| `vfwmuladd_vf_f32` | ~12% | ~10% | ~3-4% |
| `vlseg2e8` | ~8% | ~5% | ~1-2% |
| `vnibunpack.vv` | ~10% | ~6% | ~2-3% |

**综合收益**: 如果全部实现，预计整体推理性能提升 **15-25%**。

### 已有RVV扩展 (Zvdot4a8i)

RVV 已有 `Zvdot4a8i` 扩展，提供 `vdot4a` 指令：
```c
vint32m1_t __riscv_vdot4a_vv_i32m1(vint32m1_t vd, vuint32m1_t vs2, vuint32m1_t vs1, size_t vl);
```

这是 **quad widening 4D dot product**，从 uint32 packed (4×uint8) 中提取并进行 4-way dot。

**与本文建议的区别**:
- `Zvdot4a8i` 的输入是 32-bit packed (需要预处理)
- 本建议的 `vusdot.vv` 输入是 8-bit vector (直接可用)
- `Zvdot4a8i` 更适合 block layout，而 `vusdot.vv` 更适合 element-wise processing

---

## 结论

### 主要发现

1. **ARM NEON 的 lane-indexed dot product 是 RVV 最大的差距** (`vdotq_laneq_s32`, `sdot ... v24.4b[0]`)
   - ARM 用单指令完成 16×int8 × 4×int8(lane) → 4×int32 dot
   - RVV 需要 8+ 条指令实现等价功能
   - 预估收益: 4-6%

2. **WASM/S390X 的 odd/even multiply 也是 RVV 缺失的功能**
   - WASM `wasm_i32x4_dot_i16x8` 提供 8×int16 dot
   - S390X `vec_mulo/vec_meadd` 提供 odd/even 选择
   - 预估收益: 2-4%

3. **x86 AVX2 的 fixed-width shuffle RVV 也缺失**
   - `_mm256_shuffle_epi32` 可以精确控制元素位置
   - RVV 只能通过 `vrgather` 或软件循环实现
   - 预估收益: 1-2%

4. **LoongArch LASX 和 Power VSX 与 RVV 差距最小**
   - LASX 的指令设计接近 RVV
   - VSX 的 `vec_msum` 接近 dot 但范围有限

### 建议优先级

| 优先级 | 扩展指令 | 理由 |
|--------|----------|------|
| P0 | `vusdot.vv`, `vusdot_lane.vv` | 最大收益，ARM NEON 竞争差距 |
| P1 | `vfmuladd_vf_f32`, `vfwmuladd_vf_f32` | Float FMA 是 GEMM 核心操作 |
| P2 | `vlseg2e8`, `vnibunpack.vv` | 量化数据处理优化 |
| P3 | `vgetextract_i32m1` | LMUL truncation 效率 |

### 后续工作

1. **BBV 精确数据采集**: 使用 QEMU BBV plugin 对 llama.cpp inference 进行精确热点分析
2. **扩展指令模拟**: 在 QEMU 中模拟建议的扩展指令，验证收益
3. **提案文档**: 将分析结果转化为 RISC-V 扩展指令提案文档

---

## 附录: RVV现有指令验证

经查证 `/home/pren/wsp/rvfuse/third_party/riscv-rvv-intrinsic-doc/auto-generated/intrinsic_funcs.adoc`:

- `vwmacc.vv/vx`: **存在** (Widening integer multiply-accumulate)
- `vwmaccsu.vv/vx`: **存在** (Widening unsigned-signed multiply-accumulate)
- `vfwcvt.f.f.v`: **存在** (Widening floating-point convert)
- `vfmacc.vv`: **存在** (Floating-point multiply-add)
- `vfnmsac.vv`: **存在** (Floating-point negative multiply-subtract)
- `vlse8.v`: **存在** (Unit-stride load with byte stride)
- `vand.vx`: **存在** (Bitwise AND with scalar)
- `vsrl.vx`: **存在** (Shift right logical with scalar)
- `vdot4a`: **存在** (Zvdot4a8i extension)

**缺失指令**:
- Lane-indexed dot product (`vdot_lane`)
- Odd/even multiply (`vmul_odd`, `vmul_even`)
- Fixed-width element shuffle (`vshuffle_epi32`)
- Segment load without explicit segment intrinsics (`vlseg2e8` 现有，但需要更灵活的版本)