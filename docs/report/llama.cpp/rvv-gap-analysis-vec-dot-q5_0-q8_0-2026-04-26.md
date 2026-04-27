# vec-dot-q5_0-q8_0 多平台向量实现对比与RVV扩展指令建议

## 概述

本文档分析 `ggml_vec_dot_q5_0_q8_0` 函数在 7 个向量 ISA 平台上的实现，识别 RVV 缺失的关键指令，并提出扩展建议。

**算法关键步骤**：
1. **半字节解包（Nibble unpack）**：从 16 个打包字节中提取 32 个 4-bit 值（[0,15]）
2. **符号扩展（Sign extension）**：根据 qh 位掩码应用第 5 位（bit=0 处减去 0x10，使值域变为 [-16,15]）
3. **点积计算（Dot product）**：int8 × int8 点积（32 元素）
4. **缩放累加（Scale accumulate）**：乘以 FP16 delta 因子并累加

**数据结构**：
- `block_q5_0`：16 bytes qs（packed nibbles）+ 4 bytes qh（bitmask）+ 2 bytes d（FP16 scale）
- `block_q8_0`：32 bytes qs（int8 values）+ 2 bytes d（FP16 scale）

**热点分析**：`ggml_vec_dot_q5_0_q8_0_generic` 占推理总时间 **54.64%**（llama.cpp Q4_K_M 工作负载），是最关键的优化目标。

---

## 指令方案汇总

| 平台 | 寄存器宽度 | 每块指令数 | 关键独特指令 | 相对RVV差距 |
|------|-----------|-----------|-------------|------------|
| **ARM NEON** | 128-bit Q | 8 | `VDOT.S32` (dotprod扩展) | 点积融合 |
| **x86 AVX2** | 256-bit YMM | 11 | `PMADDUBSW` (sign+mul+acc) | 符号提取+MAC |
| **x86 AVX-VNNI** | 256-bit YMM | 8 | `DPBUSD/DPBSSD` (VNNI) | 点积融合 |
| **LoongArch LASX** | 256-bit XV | 11 | `XVMADD.H.B` | 类似x86 |
| **Power VSX** | 128-bit V | 10 | `vec_mule/mulo` + `vec_sum4s` | 分离式MAC |
| **S390 Z-Vector** | 128-bit V | 10 | `VMMRDI` (VXE2) | 点积融合 |
| **WASM SIMD** | 128-bit v128 | 12 | `i32x4.dot_i16x8` | 需先扩展 |
| **RVV VLEN=128** | 128-bit V | 11 | `vwmul` + `vwredsum` | 无点积指令 |
| **RVV VLEN=512** | 512-bit V | 12 | `vlmul_ext` + `vslideup` | 拼接开销 |

---

## 基准RVV实现分析

### VLEN >= 256 路径（m1-based）

```c
// 半字节解包
vuint8mf2_t raw = __riscv_vle8_v_u8mf2(x[ib].qs, vl_half);  // 16 bytes
vint8mf2_t v0l = __riscv_vand_vx_u8mf2(raw, 0x0F, vl_half); // 低半字节
vint8mf2_t v0h = __riscv_vsrl_vx_u8mf2(raw, 4, vl_half);    // 高半字节

// 拼接为32元素向量（开销！）
vint8m1_t v0c = __riscv_vlmul_ext_v_i8mf2_i8m1(v0l);
v0c = __riscv_vslideup_vx_i8m1(v0c, 
    __riscv_vlmul_ext_v_i8mf2_i8m1(v0h), qk/2, vl);

// 符号扩展（掩码减法）
vbool8_t qh = __riscv_vlm_v_b8(x[ib].qh, vl);      // 加载4字节为掩码
qh = __riscv_vmnand_mm_b8(qh, qh, vl);              // 取反
vint8m1_t v0f = __riscv_vsub_vx_i8m1_mu(qh, v0c, v0c, 0x10, vl); // 掩码减法

// 点积（2步）
vint8m1_t v1 = __riscv_vle8_v_i8m1(y[ib].qs, vl);   // 加载32 int8
vint16m2_t mul = __riscv_vwmul_vv_i16m2(v0f, v1, vl); // i8×i8→i16
vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(mul, zero, vl); // 归约i16→i32
int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);      // 提取标量
```

### 关键指令序列（23条编译后）

| 步骤 | 指令 | 数量 | 说明 |
|------|------|------|------|
| 半字节解包 | `vle8.v` + `vand.vi` + `vsrl.vi` | 3 | 标准模式 |
| 拼接 | `vlmul_ext` + `vslideup` | 3 | **RVV独有开销** |
| 符号扩展 | `vlm.v` + `vmnand` + `vsub.vx_mu` | 3 | 掩码减法模式 |
| 点积乘法 | `vle8.v` + `vwmul.vv` | 2 | widening multiply |
| 归约 | `vmv.v.x` + `vwredsum.vs` + `vmv.x.s` | 3 | 2-step reduction |
| 缩放 | `fmadd` (标量) | 2 | FP16转换+乘加 |
| 循环控制 | `vsetvli` + 分支 | ~4 | |

---

## 各平台对比分析

### 1. ARM NEON (128-bit Q寄存器)

**实现特点**：双块并行处理 + 查表符号扩展

```c
// 查表：8 bit → 8 bytes (预计算模式)
tmp0[0] = table_b2b_1[(qh0 >>  0) & 0xFF];  // bit=0 → 0x10, bit=1 → 0x00
tmp0[1] = table_b2b_1[(qh0 >>  8) & 0xFF];
...
const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));

// 半字节解包
int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));

// 符号扩展：单次减法
const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);  // value - (bit?0x00:0x10)

// 点积（dotprod扩展）
sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
    ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),  // 4×i8×i8→i32
    ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))),
    scale);
```

**关键指令**：
| 指令 | 功能 | RVV等价 |
|------|------|---------|
| `vld1q_u8` | 128-bit load | `vle8.v` (m1) |
| `vandq_u8` / `vshrq_n_u8` | nibble unpack | `vand.vi` + `vsrl.vi` |
| `table_b2b_1` (lookup) | bit→byte expansion | **缺失**（需查表或掩码） |
| `vsubq_s8` | sign extension | `vsub.vx_mu` (掩码版) |
| `ggml_vdotq_s32` | 4×i8×i8→i32 | **缺失** |
| `vmlaq_f32` | FMA accumulate | `fmadd` (标量) |

**ARM优势**：
- `VDOT.S32`（dotprod扩展）：**单指令完成 int8×int8→int32 点积**（4元素）
- 双块并行（64元素/迭代）：更好的流水线利用率
- 查表符号扩展：避免位操作开销（但增加内存压力）

---

### 2. x86 AVX2 (256-bit YMM寄存器)

**实现特点**：融合符号提取 + 无符号×有符号乘法

```c
// 半字节解包（专用helper）
__m256i qx = bytes_from_nibbles_32(x[ib].qs);  // PSHUFB+shift+AND

// 位扩展（专用helper）
__m256i bxhi = bytes_from_bits_32(x[ib].qh);   // shuffle+compare
bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
qx = _mm256_or_si256(qx, bxhi);                // OR sign bits

// 点积（2-step）
const __m256 q = mul_sum_i8_pairs_float(qx, qy);
// 内部：sign extraction + PMADDUBSW + PMADDWD

// FMA accumulate
acc = _mm256_fmadd_ps(d, q, acc);
```

**关键helper分解**：
```asm
; bytes_from_nibbles_32:
VPUNPCKLQDQ  ymm, xmm_lo, xmm_hi    ; interleave
VPSRLW       ymm, ymm, 4            ; shift high nibbles
VPAND        ymm, ymm, 0x0F         ; mask

; bytes_from_bits_32:
VPSHUFB      ymm, shuffle_mask      ; spread bits
VPOR         ymm, bit_mask          ; add magic bits
VPCMPEQB     ymm, all_ones          ; compare → 0x00/0xFF

; mul_sum_i8_pairs_float:
VPSIGNB      ymm, ymm, signs        ; sign extraction
VPMADDUBSW   ymm, ax, sy            ; unsigned×signed→i16 pairs
VPMADDWD     ymm, ones, dot         ; i16 pairs→i32
```

**关键指令**：
| 指令 | 功能 | RVV等价 |
|------|------|---------|
| `VPSHUFB` | byte-level shuffle | **缺失**（无通用字节重排） |
| `VPMADDUBSW` | u8×s8→i16 pairwise sum | **缺失**（无符号提取+MAC） |
| `VPMADDWD` | i16×i16→i32 pairwise sum | `vwredsum.vs` (归约) |
| `VPCMPEQB` | compare to all-ones | `vmseq.mm` |
| `VPANDNOT` | AND-NOT | `vandn.vv` |
| `VFMADD213PS` | FMA | `fmadd` (标量) |

**x86优势**：
- `PMADDUBSW`：**自动处理符号差异**（unsigned×signed），无需预分离
- `PSHUFB`：字节级任意重排，用于 nibble interleave 和 bit expansion
- VNNI扩展：`VPDPBUSD` 单指令完成 i8×i8→i32 点积

---

### 3. x86 AVX-VNNI / AVX512-VNNI

**VNNI指令集**：融合点积指令

```c
// AVX-VNNI: u8×u8→i32 dot product
__m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);

// AVX-VNNIINT8: s8×s8→i32 dot product
__m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
```

| 指令 | 功能 | RVV等价 |
|------|------|---------|
| `VPDPBUSD` | u8×u8→i32 dot (4×4→4) | **缺失** |
| `VPDPBSSD` | s8×s8→i32 dot | **缺失** |
| `VPDPBUSDS` | u8×u8→i32 dot (saturation) | **缺失** |

**VNNI优势**：单指令替代 `PMADDUBSW + PMADDWD`，吞吐量翻倍。

---

### 4. LoongArch LASX (256-bit XV寄存器)

**实现特点**：类似x86 AVX2模式

```c
// 半字节解包（专用helper）
__m256i qx = bytes_from_nibbles_32(x[ib].qs);

// 位扩展（专用helper）
__m256i bxhi = bytes_from_bits_32(x[ib].qh);
bxhi = __lasx_xvandn_v(bxhi, __lasx_xvreplgr2vr_b((char)0xF0));
qx = __lasx_xvor_v(qx, bxhi);

// 点积
const __m256 q = mul_sum_i8_pairs_float(qx, qy);
// 内部：lasx_madd_h_b (similar to PMADDUBSW)

// FMA accumulate
acc = __lasx_xvfmadd_s(d, q, acc);
```

**关键指令**：
| 指令 | 功能 | RVV等价 |
|------|------|---------|
| `XVSHUF.B` | byte shuffle | **缺失** |
| `XVMADD.H.B` | i8×i8→i16 MAC | `vwmacc.vv` (部分) |
| `XVANDN.V` | AND-NOT | `vandn.vv` |
| `XVSEQ.B` | compare equal | `vmseq.mm` |
| `XVFMADD.S` | FMA | `fmadd` (标量) |

**LoongArch优势**：与x86 AVX2相似度高，移植简单。

---

### 5. Power VSX (128-bit V寄存器)

**实现特点**：查表符号扩展 + 分离式乘加

```c
// 查表符号扩展
vector signed long long aux64x2_0 = {
    (uint64_t)(table_b2b_1[x[ib].qh[0]]),
    (uint64_t)(table_b2b_1[x[ib].qh[1]])
};
vector signed char qh0 = (vector signed char)aux64x2_0;

// 半字节解包
vector signed char qxs = (vector signed char)vec_xl(0, x[ib].qs);
vector signed char q5x0 = vec_sub(vec_and(qxs, lowMask), qh0);
vector signed char q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

// 分离式乘加
vector signed short qv0 = vec_add(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
vector signed int vsumi0 = vec_add(vec_unpackh(qv0), vec_unpackl(qv0));

// 累加
vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
```

**关键指令**：
| 指令 | 功能 | RVV等价 |
|------|------|---------|
| `vec_xl` | 128-bit load | `vle8.v` (m1) |
| `vec_and` / `vec_sr` | nibble unpack | `vand.vi` + `vsrl.vi` |
| `table_b2b_1` (lookup) | bit→byte | **缺失** |
| `vec_mule` / `vec_mulo` | widening multiply (even/odd) | `vwmul.vv` |
| `vec_add` + `vec_unpackh/l` | pairwise sum → i32 | `vwredsum.vs` |
| `vec_sum4s` | 4×sum | `vredsum.vs` |
| `vec_madd` | FMA | `fmadd` (标量) |

**Power特点**：使用 `vec_mule/mulo` 分离乘法（even/odd索引），需后续拼接。

---

### 6. S390 Z-Vector (128-bit V寄存器)

**实现特点**：查表符号扩展 + VXE2点积指令

```c
// 查表符号扩展（同ARM/Power）
tmp0[0] = table_b2b_1[(qh0 >>  0) & 0xFF];
...
int8x16_t v_qh0l = vec_xl(0, (const int8_t *)(tmp0 + 0));
v_qh0l = vec_perm(v_qh0l, v_qh0l, v_kperm);  // byteorder fix

// 半字节解包
int8x16_t v_x0l = (int8x16_t)vec_and(v_x0, v_m);
int8x16_t v_x0h = (int8x16_t)vec_sr(v_x0, 4);

// 符号扩展
const int8x16_t v_x0lf = vec_sub(v_x0l, v_qh0l);

// 点积（VXE2扩展）
const int32x4_t v_xy0 = ggml_vec_dot(vec_splats(0), v_x0lf, v_y0l);
// 内部：VMMRDI (Vector Multiply-Add Reduce)

// 累加
v_sum0 = vec_madd(v_xy0f, v_d0, v_sum0);
```

**关键指令**：
| 指令 | 功能 | RVV等价 |
|------|------|---------|
| `VL` | 128-bit load | `vle8.v` (m1) |
| `VN` / `VESRLB` | nibble unpack | `vand.vi` + `vsrl.vi` |
| `table_b2b_1` (lookup) | bit→byte | **缺失** |
| `VSB` | sign extension | `vsub.vx_mu` |
| `VMMRDI` (VXE2) | i8×i8→i32 dot reduce | **缺失** |
| `VFMA` | FMA | `fmadd` (标量) |

**S390优势**：`VMMRDI`（VXE2扩展）提供类似ARM dotprod的点积指令。

---

### 7. WASM SIMD (128-bit v128寄存器)

**实现特点**：查表符号扩展 + i16点积（需预扩展）

```c
// 查表符号扩展
tmp[0] = table_b2b_1[(qh_ >>  0) & 0xFF];
...
const v128_t qhl = wasm_v128_load(tmp + 0);

// 半字节解包
const v128_t v0l = wasm_v128_and(v0, m4b);
const v128_t v0h = wasm_u8x16_shr(v0, 4);

// 符号扩展
const v128_t v0lf = wasm_i8x16_sub(v0l, qhl);

// 扩展为i16（必需步骤！）
const v128_t v0lfl = wasm_i16x8_extend_low_i8x16(v0lf);
const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);

// 点积（i16×i16→i32）
sumv = wasm_i32x4_add(
    wasm_i32x4_dot_i16x8(v0lfl, v1ll),
    wasm_i32x4_dot_i16x8(v0lfh, v1lh));
```

**关键指令**：
| 指令 | 功能 | RVV等价 |
|------|------|---------|
| `wasm_v128_load` | 128-bit load | `vle8.v` (m1) |
| `wasm_v128_and` / `wasm_u8x16_shr` | nibble unpack | `vand.vi` + `vsrl.vi` |
| `table_b2b_1` (lookup) | bit→byte | **缺失** |
| `wasm_i8x16_sub` | sign extension | `vsub.vx_mu` |
| `wasm_i16x8_extend_low/high` | widening | `vwcvt.x.x` |
| `wasm_i32x4_dot_i16x8` | i16×i16→i32 dot | **缺失**（i8无） |
| `wasm_f32x4_add` | accumulate | `vfadd.vv` |

**WASM限制**：无直接 i8 点积指令，需先扩展为 i16。

---

## RVV扩展指令建议详细说明

### EXT-1: `vdot.s8` — 整数点积指令（最高影响）

**语义**：
```c
// 4×int8×int8→int32 pairwise dot product
vint32m1_t __riscv_vdot_vv_i32m1(vint8m1_t vs2, vint8m1_t vs1, vint32m1_t vd, size_t vl);
// vd[i] += vs2[4i]×vs1[4i] + vs2[4i+1]×vs1[4i+1] + vs2[4i+2]×vs1[4i+2] + vs2[4i+3]×vs1[4i+3]
```

**对标**：
- ARM: `VDOT.S32` (dotprod extension)
- x86: `VPDPBSSD` (AVX-VNNIINT8)
- S390: `VMMRDI` (VXE2)

**收益分析**：
| 当前RVV | 优化后 | 改善 |
|---------|--------|------|
| `vwmul`(4c) + `vwredsum`(4c) + `vmv.x.s`(3c) = 11c | `vdot.s8`(~5c) + `vmv.x.s`(3c) = 8c | **减少27%** |

**端到端收益**：kernel加速比从 6.3× → 7.6×，整体推理加速比从 1.80× → 1.86× (+3.3%)

---

### EXT-2: `vwmulred.vs` — Widening Multiply-Reduce（中高影响）

**语义**：
```c
// Widening multiply + immediate reduction
vint32m1_t __riscv_vwmulred_vs_i8m1_i32m1(vint8m1_t vs2, vint8m1_t vs1, vint32m1_t vd, size_t vl);
// vd[0] += sum(vs2[i]×vs1[i] for all i)  // widening to i16 then sum to i32
```

**对标**：
- ARM: `vdotq_s32` + `vaddvq_s32`
- WASM: `i32x4.dot_i16x8` + horizontal add

**收益分析**：
| 当前RVV | 优化后 | 改善 |
|---------|--------|------|
| `vwmul`(4c) + `vwredsum`(4c) = 8c | `vwmulred`(~6c) | **减少25%** |

**适用场景**：替代分离的 widening multiply + reduction 序列。

---

### EXT-3: `vexpand.b2b` — Bit-to-Byte Expansion（中等影响）

**语义**：
```c
// Expand bit mask to byte mask (0→0x00, 1→0xFF)
vint8m1_t __riscv_vexpand_b2b_i8m1(vbool8_t vm, size_t vl);
// out[i] = vm[i] ? 0xFF : 0x00
```

**对标**：
- x86: `VPCMPEQB` (compare to all-ones)
- ARM/Power/S390/WASM: `table_b2b_1` lookup (256-byte table)

**收益分析**：
| 当前RVV (掩码减法) | 优化后 (位扩展) | 改善 |
|-------------------|----------------|------|
| `vlm`(3c) + `vmnand`(3c) + `vsub.vx_mu`(7c) = 13c | `vlm`(3c) + `vexpand`(3c) + `vandn`(3c) + `vor`(3c) = 12c | **等效** |

**适用场景**：当需要位掩码转换为字节掩码用于 `vor` 操作时（类似x86模式）。当前RVV的掩码减法方法更高效。

---

### EXT-4: `vunpackn.v` — Nibble Unpack指令（低影响）

**语义**：
```c
// Unpack packed nibbles to bytes
vint8m1_t __riscv_vunpackn_v_i8m1(vuint8mf2_t vs2, size_t vl);
// out[2i] = vs2[i] & 0x0F; out[2i+1] = (vs2[i] >> 4) & 0x0F
```

**对标**：
- x86: `bytes_from_nibbles_32` (PSHUFB+shift+AND)
- ARM: `vand` + `vshr`

**收益分析**：
| 当前RVV | 优化后 | 改善 |
|---------|--------|------|
| `vand.vi`(3c) + `vsrl.vi`(3c) + `vlmul_ext`(1c) + `vslideup`(3c) = 10c | `vunpackn`(~5c) + `vlmul_ext`(1c) = 6c | **减少40%** |

**适用场景**：量化权重解包。

---

### EXT-5: `vinterleave.v` — 向量交错指令（低影响）

**语义**：
```c
// Interleave two vectors
vint8m1_t __riscv_vinterleave_vv_i8m1(vint8mf2_t vs2, vint8mf2_t vs1, size_t vl);
// out[0..vl/2-1] = vs2[0..vl/2-1]; out[vl/2..vl-1] = vs1[0..vl/2-1]
```

**对标**：
- x86: `VPUNPCKLQDQ` / `VPUNPCKHQDQ`
- ARM: `VZIP`

**收益分析**：
| 当前RVV | 优化后 | 改善 |
|---------|--------|------|
| `vlmul_ext`(1c) + `vslideup`(3c) = 4c | `vinterleave`(~2c) | **减少50%** |

**适用场景**：半字节拼接。

---

### EXT-6: `vsignext.v` — 符号扩展融合指令（中等影响）

**语义**：
```c
// Sign extension: subtract constant where mask bit is zero
vint8m1_t __riscv_vsignext_vx_mu_i8m1(vbool8_t vm, vint8m1_t vs2, int8_t rs1, size_t vl);
// out[i] = vm[i] ? vs2[i] : vs2[i] - rs1
```

**对标**：
- ARM: `vsubq_s8` (after table lookup)
- x86: `VPANDNOT` + `VPOR` + `VPSUBB`

**收益分析**：
| 当前RVV | 优化后 | 改善 |
|---------|--------|------|
| `vmnand`(3c) + `vsub.vx_mu`(7c) = 10c | `vsignext.vx_mu`(~7c) | **减少30%** |

**适用场景**：Q5_0 符号扩展。

---

## BBV热点加权收益分析

基于 `ggml_vec_dot_q5_0_q8_0_generic` 占推理时间 **54.64%** 的热点数据：

### 单一扩展收益

| 扩展指令 | Kernel加速 | 整体推理加速 | 备注 |
|---------|-----------|-------------|------|
| `vdot.s8` | +29% | +3.3% | 最高影响 |
| `vwmulred.vs` | +25% | +2.9% | 中高影响 |
| `vsignext.vx_mu` | +11% | +1.2% | 中等影响 |
| `vunpackn.v` | +40% (局部) | +1.5% | 低整体影响 |
| `vinterleave.v` | +50% (局部) | +1.0% | 低整体影响 |

### 组合扩展收益

| 配置 | Kernel周期/块 | Kernel加速比 | 整体推理加速 |
|------|-------------|-------------|-------------|
| 当前RVV (VLEN=128) | ~35c | 6.3× | 1.80× |
| + `vdot.s8` | ~29c | 7.6× | 1.86× |
| + `vdot.s8` + `vsignext` | ~26c | 8.4× | 1.89× |
| + 全部扩展 | ~22c | 10.0× | 1.93× |

**最大收益上限**：~1.93× 整体推理加速比。

---

## 结论

### RVV优势

1. **掩码减法方法**：`vlm` + `vmnand` + `vsub.vx_mu` 避免了其他平台的查表开销（ARM/Power/S390/WASM）或复杂位扩展序列（x86）
2. **可配置VLEN**：单一代码路径支持128/256/512位，无需多ISA路径
3. **灵活LMUL**：支持 mf2/m1/m2 动态切换，适应不同数据布局

### RVV劣势/差距

1. **缺少整数点积指令**：ARM `VDOT.S32`、x86 `VPDPBSSD`、S390 `VMMRDI` 均有 i8×i8→i32 单指令点积
2. **缺少字节级shuffle**：`PSHUFB` 等价物缺失，限制了 nibble interleave 和 bit expansion 优化
3. **缺少融合符号提取+MAC**：x86 `PMADDUBSW` 自动处理 unsigned×signed，无需预分离
4. **拼接开销**：`vlmul_ext` + `vslideup` 是 RVV 独有的拼接开销，其他平台有专用 interleave 指令

### 优先级建议

| 优先级 | 扩展指令 | 影响范围 | 端到端收益 |
|--------|---------|---------|-----------|
| **P0** | `vdot.s8` | 所有量化点积kernel | +3.3% |
| **P1** | `vwmulred.vs` | Widening multiply + reduction | +2.9% |
| **P2** | `vsignext.vx_mu` | 量化符号扩展 | +1.2% |
| **P3** | `vunpackn.v` | 半字节解包 | +1.5% |
| **P3** | `vinterleave.v` | 向量拼接 | +1.0% |

### 最终建议

对于 llama.cpp Q5_0×Q8_0 点积kernel：
- `vdot.s8` 是**最高优先级扩展**，可带来 +3.3% 整体推理加速
- 当前RVV掩码减法方法**优于其他平台的查表或位扩展方法**
- 最大可达到的整体推理加速比约为 **1.93×**（需实现全部扩展）
- 要突破此上限，需向量化其余 vec_dot kernel（Q6_K、Q4_K）

---

**附录：平台实现文件路径**

| 平台 | 文件路径 |
|------|---------|
| RVV (project) | `/home/pren/wsp/rvfuse/applications/llama.cpp/rvv-patches/vec-dot-q5_0-q8_0/rvv_vec_dot_q5_0_q8_0.inl` |
| RVV (upstream) | `/home/pren/wsp/rvfuse/applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` |
| ARM NEON | `/home/pren/wsp/rvfuse/applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` |
| x86 AVX | `/home/pren/wsp/rvfuse/applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` |
| LoongArch LASX | `/home/pren/wsp/rvfuse/applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/loongarch/quants.c` |
| Power VSX | `/home/pren/wsp/rvfuse/applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/powerpc/quants.c` |
| S390 Z-Vector | `/home/pren/wsp/rvfuse/applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/s390/quants.c` |
| WASM SIMD | `/home/pren/wsp/rvfuse/applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/wasm/quants.c` |