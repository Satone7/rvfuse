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

---

## Performance Benefit Estimation

> **Data source**: Perf hotspot analysis on Spacemit K1-X (rv64imafdcv), Qwen2.5-0.5B Q4_K_M, 128-token generation.
> **Instruction latencies**: K1-X RVV pipeline (see `docs/reference/cx/instruction-constraints-and-latency.md`).

### Methodology

The estimation follows a three-step approach:

1. **Cycle-level kernel analysis**: Count critical-path cycles for scalar generic vs RVV implementation, using measured instruction latencies from the K1-X pipeline.
2. **Kernel speedup derivation**: Ratio of scalar-to-RVV cycles per Q5_0 block (32 elements).
3. **End-to-end projection**: Apply Amdahl's law using the perf-measured Q5_0 time fraction (55.45% of total inference).

All cycle counts are **compute-bound** estimates (assume data is in L1/L2 cache). Memory-bound analysis is provided separately where relevant.

### K1-X Instruction Latencies Used

| Instruction | Latency (cycles) | Notes |
|-------------|-----------------|-------|
| `vsetvli` | 2 | Loop setup (amortized) |
| `vle8.v` (vector load) | 3 | 16 bytes (m1) or 32 bytes (m2) |
| `vlm.v` (mask load) | 3 | 4 bytes → mask register |
| `vand.vi` | 3 | Immediate operand |
| `vsrl.vi` | 3 | Immediate operand |
| `vmnand.mm` | 3 | Mask operation |
| `vsub.vx` (masked) | 4+3 = 7 | Scalar broadcast penalty |
| `vwmul.vv` (SEW=8) | 4 | int8×int8 → int16 |
| `vwredsum.vs` | 4 | int16 → int32 reduction |
| `vmv.x.s` | 3 | Vector → scalar move |
| `lb` (scalar byte load) | 3 | Scalar baseline |
| `mul` (scalar multiply) | 5 | RVG baseline |

### Scalar Generic: Cycle Analysis (per Q5_0 block, 32 elements)

The generic implementation iterates 16 times (qk/2), processing 2 elements per iteration:

```c
for (int j = 0; j < 16; ++j) {
    xh_0 = ((qh >> j) & 1) << 4;        // bit extract: ~2 cycles
    xh_1 = ((qh >> (j+16)) & 1);        // bit extract: ~2 cycles
    x0 = (qs[j] & 0x0F) | xh_0) - 16;   // load+mask+or+sub: ~7 cycles
    x1 = ((qs[j] >> 4) | xh_1) - 16;    // shift+or+sub: ~4 cycles
    sumi0 += x0 * y[j];                  // load+mul+add: ~9 cycles
    sumi1 += x1 * y[j+16];               // load+mul+add: ~9 cycles
}
```

**Critical path per iteration** (in-order execution, conservative):
- Data ready for multiply: ~7 cycles (load + nibble ops)
- `mul` (5 cycles) — dominates, is loop-carried through `add`
- Accumulate: 1 cycle
- **Per iteration: ~13 cycles**

**Total per block**: 16 × 13 + memcpy(3) + FP scale(8) = **~219 cycles**

The scalar multiply (`mul`) at 5 cycles is the single biggest bottleneck — it appears twice per iteration and cannot be pipelined on in-order cores.

### RVV Implementation: Cycle Analysis (per Q5_0 block, VLEN=128)

Dependency graph with K1-X latencies:

```
t0:  vle8.qs(3)          ─┬─→ vand(3)=6  ─┐
                           └─→ vsrl(3)=6  ─┤
t1:  vlm.qh(3)           ──→ vmnand(3)=6  ┤
t2:  vle8.y(3)           (independent)    │
t3:  vmv.zero(3)          (independent)   │
                                             ↓
t4:  max(6,6)+vcreate(~3)=9  ──────────→ vsub_mu(7)=16
t5:  max(16,3)+vwmul(4)=20  ←─────────── (y ready at 3)
t6:  max(20,3)+vwredsum(4)=24
t7:  t6+vmv_x_s(3)=27
```

**RVV critical path per block: 27 cycles** (+ ~8 for FP16 scale + loop overhead = **~35 cycles total**)

### Kernel Speedup Estimate

| Implementation | Cycles/block (32 elem) | Speedup vs Scalar |
|---------------|----------------------|-------------------|
| Scalar generic | ~219 | 1.0× (baseline) |
| RVV (VLEN=128) | ~35 | **6.3×** |
| RVV (VLEN=256) | ~33 | **6.6×** |
| RVV (VLEN=512) | ~31 | **7.1×** |

The primary speedup driver is replacing 16 iterations of scalar `mul`(5 cycles) with a single `vwmul.vv`(4 cycles) that processes all 32 elements in parallel. The scalar version spends 16×2×5 = 160 cycles on multiplies alone; RVV spends 4 cycles.

### End-to-End Inference Impact (Amdahl's Law)

From the perf hotspot analysis, `ggml_vec_dot_q5_0_q8_0_generic` accounts for **55.45%** of total inference time on K1-X (Qwen2.5-0.5B Q4_K_M, 128 tokens).

Amdahl's law: `Speedup_total = 1 / [(1-f) + f/s]` where f=0.5545, s=kernel speedup.

| Scenario | Kernel Speedup | Q5_0 Time Fraction | Overall Inference Speedup |
|----------|---------------|-------------------|--------------------------|
| Scalar baseline | 1.0× | 55.45% | 1.00× |
| RVV VLEN=128 (realistic) | 5.0× | 11.09% | **1.80×** |
| RVV VLEN=128 (optimistic) | 6.3× | 8.80% | 1.91× |
| RVV VLEN=256 | 5.5× | 10.08% | 1.86× |
| RVV VLEN=512 | 6.0× | 9.24% | 1.89× |

**Realistic estimate: 1.8× overall inference speedup** for Q5_0 RVV optimization alone. This assumes compute-bound operation (data fits in cache), which is consistent with the long scalar execution times observed in the perf profile.

### Extension Benefit Estimates

#### 1. `vdot.s8` — Integer Dot Product (Highest Impact)

Replaces `vwmul.vv`(4) + `vwredsum.vs`(4) + `vmv.x.s`(3) = **11 cycles** with a single instruction.

Estimated `vdot.s8` latency: **5 cycles** (comparable to `vwmacc.vv` = 5 cycles, as both perform multiply + accumulate).

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Multiply-reduce phase | 11 cycles | 5 cycles | **54% reduction** |
| Total RVV critical path | 27 cycles | 21 cycles | **22% faster** |
| Kernel speedup vs scalar | 6.3× | 8.1× | **+29%** |
| Overall inference (VLEN=128) | 1.80× | 1.89× | **+5%** |

The relatively modest overall impact (+5%) is because the multiply-reduce phase (11 cycles) is only part of the total pipeline (27 cycles), and the Q5_0 kernel itself is only 55% of total inference time.

#### 2. Masked Multiply-Subtract `vmulsub.vx` (Medium Impact)

Replaces `vmnand.mm`(3) + `vsub.vx_mu`(7) = **10 cycles** with a single instruction.

Estimated latency: **7 cycles** (same as `vsub.vx` masked variant, since the inversion is absorbed).

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Sign extension phase | 10 cycles | 7 cycles | **30% reduction** |
| Total RVV critical path | 27 cycles | 24 cycles | **11% faster** |
| Kernel speedup vs scalar | 6.3× | 7.0× | **+11%** |
| Overall inference (VLEN=128) | 1.80× | 1.84× | **+2%** |

#### 3. Vector Accumulator Across Iterations (Pipeline Improvement)

Current implementation uses scalar accumulator (`vwredsum.vs` → `vmv.x.s`), which serializes iterations. Adding a vector accumulator (like ARM NEON's dual `sumv0`/`sumv1` pattern) would overlap iteration pipelines.

Estimated benefit: **10-15% throughput improvement** by hiding load and setup latency across iterations.

#### 4. Configurable Mask Type (No Compute Benefit)

Eliminates the VLEN-dependent code path branching (b4 vs b8). Benefits code maintainability and reduces binary size, but **no cycle savings** for a given VLEN configuration.

### Combined Extension Scenario

If all proposed extensions are implemented:

| Configuration | Cycles/block | Kernel Speedup | Overall Inference |
|--------------|-------------|----------------|-------------------|
| RVV only (VLEN=128) | ~35 total | 6.3× | 1.80× |
| RVV + `vdot.s8` | ~29 total | 7.6× | 1.86× |
| RVV + `vdot.s8` + `vmulsub` | ~26 total | 8.4× | 1.89× |
| All extensions + multi-block | ~22 total | 10.0× | 1.93× |

**Maximum achievable overall inference speedup from Q5_0 kernel optimization alone: ~1.9×**

To go beyond this, the remaining vec_dot kernels must also be vectorized:
- Q6_K × Q8_K (18.22%): would add another ~0.3× overall speedup
- Q4_K × Q8_K (15.92%): would add another ~0.2× overall speedup
- Full vectorization of all 4 kernels: **estimated ~2.5-3.0× overall inference speedup**

### Memory-Bound Ceiling

Each Q5_0 block touches 56 bytes (16B qs + 4B qh + 2B d + 32B y + 2B d). With L2 bandwidth of ~20 GB/s on K1-X, the theoretical minimum time per block is 56/20 ≈ 2.8 ns. At 1.8 GHz, this is ~5 cycles per block — well below the compute-bound estimate of 27 cycles. **The kernel is firmly compute-bound**, confirming that instruction-level optimizations (extensions) provide real speedup.

### Summary

The RVV implementation of `vec_dot_q5_0_q8_0` is functionally correct and achieves comparable instruction count to x86 AVX2. The mask-based sign extension approach (`VLM`+`VMNAND`+`VSUB.MU`) is elegant and avoids the ARM NEON's lookup table memory overhead. The primary gap is the lack of an integer dot product instruction (`vdot.s8`), which would replace the 2-instruction `VWMUL`+`VWREDSUM` sequence with a single operation — this is the single highest-impact extension for Q5_0 dot product performance.

Based on K1-X instruction latency analysis, the RVV implementation achieves an estimated **6.3× kernel speedup** over the scalar generic baseline, translating to **~1.8× overall inference speedup** for the Q5_0-heavy Q4_K_M workload. The proposed `vdot.s8` extension would improve this to **~1.9×** overall — a meaningful but incremental gain. The largest remaining optimization opportunity lies in vectorizing the other three vec_dot kernels (Q6_K, Q4_K, Q8_0) to reach **~2.5-3.0×** total inference speedup.
