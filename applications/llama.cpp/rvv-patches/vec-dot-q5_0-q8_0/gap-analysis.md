# RVV Gap Analysis: vec_dot_q5_0_q8_0

## Comparison: x86 AVX2 vs ARM NEON vs RISC-V RVV

### Algorithm Overview

Q5_0 dot product requires four steps:
1. **Nibble unpack**: Extract 32 4-bit values from 16 packed bytes
2. **Sign extension**: Apply 5th-bit from qh bitmask (subtract 0x10 where bit=0)
3. **Multiply-accumulate**: Compute int8 x int8 dot product
4. **Scale**: Multiply by combined delta factors

### Instruction-Level Comparison

| Step | x86 AVX2 | ARM NEON | RISC-V RVV |
|------|----------|----------|------------|
| **Nibble unpack** | `PSHUFB` + `VPSRLDQ` + `VPAND` | `VAND` + `VSHR` | `VAND.VI` + `VSRL.VI` |
| **qh sign ext** | `VPSHUFB` + `VPCMPEQB` + `VPANDNOT` + `VPOR` | LUT (`table_b2b_1`) + `VSUB` | `VLM` + `VMNAND` + `VSUB.MU` |
| **Multiply** | `VPMADDUBSW` + `VPMADDWD` | `VDOT.S32` (dotprod ext) | `VWMUL.VV` |
| **Reduction** | Horizontal sum (`VPHADDD`) | `VADDV` | `VWREDSUM.VS` |
| **FMA** | `VFMADD` (fused) | `VMLA` (scalar) | `FMADD` (scalar) |

### Register Pressure and Throughput

| Metric | x86 AVX2 | ARM NEON | RISC-V RVV (VLEN=512) | RISC-V RVV (VLEN=128) |
|--------|----------|----------|----------------------|----------------------|
| Register width | 256-bit YMM | 128-bit Q | 512-bit V | 128-bit V |
| Elements per iter | 32 | 64 (2 blocks) | 32 (1 block) | 32 (1 block) |
| LMUL used | N/A | N/A | m1 (mf2 load) | m2 |
| Mask registers | N/A (implicit) | N/A (implicit) | v0 (b8) | v0 (b4) |
| Accumulators | 1 (scalar) | 2 (parallel vec) | 1 (scalar) | 1 (scalar) |

### Critical Gap: qh Sign Extension

The qh sign extension step reveals the most significant architectural differences:

**x86 AVX2** (3 instructions):
```asm
VPSHUFB  xmm_qh, shuf_mask     ; Broadcast bits to bytes via shuffle
VPCMPEQB bytes, all_ones         ; Compare to get 0x00/0xFF mask
VPANDNOT qh_mask, 0xF0           ; Invert and mask
VPOR     result, nibbles         ; OR with nibble values
```
- `VPSHUFB` can arbitrarily permute bytes using a 256-bit shuffle mask — extremely flexible
- The qh→byte expansion is a single-instruction operation with a precomputed mask

**ARM NEON** (table lookup + subtract):
```c
// Table lookup: each qh byte index → 8-byte precomputed pattern
tmp[i] = table_b2b_1[(qh >> (i*8)) & 0xFF];  // 0 → 0x00, 1 → 0x10
qhl = vld1q_s8((const int8_t *)(tmp + 0));
result = vsubq_s8(nibbles, qhl);               // Single subtract
```
- Uses a **256-byte lookup table** (2^8 entries) — requires memory but very fast
- The subtract-based approach is elegant: `value - 0x10` when bit=1, `value - 0x00` when bit=0
- However, the table load adds memory pressure

**RISC-V RVV** (masked subtract — closest to ARM approach):
```c
vbool8_t qh = __riscv_vlm_v_b8(x->qh, vl);     // Load 32 bits as mask
qh = __riscv_vmnand_mm_b8(qh, qh, vl);          // Invert all bits
vint8m1_t v0f = __riscv_vsub_vx_i8m1_mu(qh, v0c, v0c, 0x10, vl);  // Masked subtract
```
- Uses **vector load-as-mask** (`VLM`) — converts byte-loaded qh directly to mask register
- **Masked subtract** (`VSUB.MU`) — only subtracts where mask is active (inverted)
- No lookup table needed — fully register-based
- This is actually **simpler** than both x86 and ARM approaches

### Gap Assessment

#### RVV Strengths (vs x86 AVX2 / ARM NEON)

1. **Configurable VLEN**: Single code path handles 128/256/512-bit — x86 needs separate SSE/AVX2 paths; ARM needs feature detection
2. **Mask-based sign extension**: `VLM` + `VMNAND` + `VSUB.MU` is 3 instructions vs 4 for AVX2. No table memory overhead vs ARM.
3. **Scalable element count**: VLEN=512 processes 64 elements with m2 — potentially 2x throughput over AVX2's 32
4. **Natural nibble unpack**: `VAND.VI` + `VSRL.VI` is identical to ARM NEON's approach, equally efficient

#### RVV Weaknesses / Gaps

1. **No fused multiply-subtract with sign extraction**: x86 `VPMADDUBSW` combines sign extraction + unsigned×signed multiply + accumulate in one instruction. RVV needs separate sign extension + `VWMUL.VV` + `VWREDSUM.VS` (3 instructions vs 1).

2. **No integer dot product instruction**: ARM `VDOT.S32` computes `sum(dp[i]*dq[i])` for 4×int8→int32 in a single instruction. RVV `VWMUL.VV` produces int16 intermediates, requiring a separate reduction. This costs 2 instructions vs 1.

3. **Reduction bottleneck**: `VWREDSUM.VS` requires a scalar accumulator, preventing parallel accumulation. ARM NEON uses 2 vector accumulators (sumv0, sumv1) for better pipeline utilization. RVV could improve this with `vslideup` + `vadd` to maintain vector accumulators across iterations.

4. **No byte shuffle (PSHUFB equivalent)**: RVV lacks a general-purpose byte-level shuffle. This limits some optimizations possible on x86. The `vrgather` instruction is the closest but requires index computation.

5. **Mask register type mismatch across VLEN**: VLEN=128 requires `b4` mask type; VLEN>=256 requires `b8` mask type. This necessitates runtime VLEN detection and dual code paths — an avoidable complexity.

### Proposed RVV Extensions for Q5_0 Dot Product

| Gap | Proposed Extension | Benefit |
|-----|-------------------|---------|
| No int8 dot product | `vdot.s8` (4×i8→i32) | Replace VWMUL+VWREDSUM with single instruction; match ARM VDOT |
| Masked multiply-subtract | `vmsub.vx` (masked scalar subtract) | Combine VMNAND+VSUB.MU into single sign-extension instruction |
| Configurable mask type | Auto-promote b4↔b8 based on destination LMUL | Eliminate runtime VLEN branching for mask operations |
| Vector accumulate across iterations | `vredsum.vs` with vector (not scalar) source | Enable multi-iteration vector accumulation like ARM NEON's dual accumulators |

### Instruction Count Comparison (per Q5_0 block)

| Step | x86 AVX2 | ARM NEON | RVV VLEN=512 | RVV VLEN=128 |
|------|----------|----------|-------------|-------------|
| Nibble unpack | 3 | 2 | 4 | 3 |
| qh sign ext | 4 | 2 (LUT) | 3 | 3 |
| Multiply | 1 (PMADDUBSW) | 1 (VDOT) | 1 (VWMUL) | 1 (VWMUL) |
| Reduction | 1 (PMADDWD) | 1 (VADDV) | 2 (VWREDSUM+MV) | 2 (VWREDSUM+MV) |
| Scale | 2 (FP16+FMUL) | 2 (FP16+VMLA) | 2 (FP16+FMUL) | 2 (FP16+FMUL) |
| **Total** | **11** | **8** | **12** | **11** |

### Throughput Comparison (elements per instruction)

| Platform | Elements/iter | Instructions/iter | Throughput (elem/instr) |
|----------|--------------|-------------------|------------------------|
| x86 AVX2 | 32 | 11 | 2.91 |
| ARM NEON | 64 | 8 | 8.00 |
| RVV VLEN=512 | 32 | 12 | 2.67 |
| RVV VLEN=128 | 32 | 11 | 2.91 |

ARM NEON leads due to `VDOT.S32` (1 instr for multiply+reduce) and dual-block processing (64 elem/iter). RVV could match or exceed this with a `vdot.s8` extension and multi-block iteration.

### Summary

The RVV implementation of `vec_dot_q5_0_q8_0` is functionally correct and achieves comparable instruction count to x86 AVX2. The mask-based sign extension approach (`VLM`+`VMNAND`+`VSUB.MU`) is elegant and avoids the ARM NEON's lookup table memory overhead. The primary gap is the lack of an integer dot product instruction (`vdot.s8`), which would replace the 2-instruction `VWMUL`+`VWREDSUM` sequence with a single operation — this is the single highest-impact extension for Q5_0 dot product performance.
