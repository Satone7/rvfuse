# RVV Extension Proposal Verification Report

> **Source**: Cross-reference of the 18 proposals from `docs/report/rvv-extension-comprehensive-analysis-2026-04-26.md`
> against official RVV intrinsic documentation in `third_party/riscv-rvv-intrinsic-doc/auto-generated/intrinsic_funcs.adoc`.
>
> **Date**: 2026-04-26
>
> **Key reference files**:
> - `auto-generated/intrinsic_funcs.adoc` (53957 lines) -- complete intrinsic list
> - `auto-generated/intrinsic_funcs/05_vector_reduction_operations.adoc` -- reduction intrinsics
> - `auto-generated/intrinsic_funcs/09_zvdot4a8i_-_vector_quad_widening_4d_dot_product.adoc` -- Zvdot4a8i extension

---

## Key Findings Summary

- **1 EXISTS**: 方案 10 `vfwmacc.vf` -- already in RVV specification
- **7 PARTIAL**: 方案 1, 3, 7, 11, 12, 13, 17 -- similar operations exist but don't fully match the proposal (narrower width ratios, missing fusion, or different API)
- **10 MISSING**: 方案 2, 4, 5, 6, 8, 9, 14, 15, 16, 18 -- no equivalent in current RVV

**Critical finding**: 方案 10 `vfwmacc.vf` is already implemented as `__riscv_vfwmacc_vf_f32m1(vd, rs1_f16, vs2_f16)`, making it a false-positive proposal. Its inclusion in the gap analysis report is an error.

---

## Individual Verification

### 方案 1: vsegdot.vv -- PARTIAL

- **RVV equivalent**: `vdot4au_vv` / `vdot4a_vv` (Zvdot4a8i extension)
- **Status**: PARTIAL
- **Evidence**: `auto-generated/intrinsic_funcs/09_zvdot4a8i_-_vector_quad_widening_4d_dot_product.adoc:59-78`
  ```
  vuint32mf2_t __riscv_vdot4au_vv_u32mf2(vuint32mf2_t vd, vuint32mf2_t vs2,
                                         vuint32mf2_t vs1, size_t vl);
  ```
- **Analysis**: Zvdot4a8i already provides quad-widening 4D dot product (4xuint8 dot -> uint32 accumulation). However, it requires data packed as uint32 words (4 bytes per word), while `vsegdot.vv` expects unpacked byte vectors. The operations are functionally equivalent but differ in data layout requirements, and Zvdot4a8i requires the input bytes to be assembled into 32-bit words first. The proposal's claim of "replacing vwmulu+vwaddu" is valid -- both could do the same mathematical operation, but with different in-register data arrangements.

### 方案 2: vdot_lane.vx -- MISSING

- **RVV equivalent**: NONE (closest: `vdot4a_vx`, but semantics differ)
- **Status**: MISSING
- **Evidence**: `auto-generated/intrinsic_funcs/09_zvdot4a8i_-_vector_quad_widening_4d_dot_product.adoc:11-12`
  ```
  vint32mf2_t __riscv_vdot4a_vx_i32mf2(vint32mf2_t vd, vuint32mf2_t vs2,
                                       uint32_t rs1, size_t vl);
  ```
- **Analysis**: `vdot4a_vx` takes a scalar `uint32_t rs1` and broadcasts ALL 4 bytes to every lane. It does NOT extract a specific lane from a vector register. `vdot_lane.vx` extracts a specific byte lane (e.g., byte imm from a vector) and broadcasts it -- this is equivalent to ARM NEON's `vmla_lane` pattern. RVV has no lane-extract+broadcast+dot instruction.

### 方案 3: vnarrow_sat -- PARTIAL

- **RVV equivalent**: `vnclip` / `vnclipu` (narrowing clip with saturation)
- **Status**: PARTIAL
- **Evidence**: `auto-generated/intrinsic_funcs.adoc` (vnclip entries)
  ```
  vint8mf8_t __riscv_vnclip_wv_i8mf8(vint16mf4_t vs2, vuint8mf8_t vs1, ...);
  ```
- **Analysis**: `vnclip` already provides saturating narrowing (shift right + round + saturate). With shift=0, it does simple saturating narrow. However, it only supports 2:1 SEW ratio (e.g., i16->i8). The proposal wants 4:1 ratio (i32->i8 in one step). RVV currently requires two `vnclip` operations for i32->i8, just like it requires two `vncvt` operations. The saturation semantics exist; the width ratio is the gap.

### 方案 4: vfmacc.vv_lane -- MISSING

- **RVV equivalent**: NONE (closest: `vfmacc.vv` + `vfmacc.vf`, but no lane extraction)
- **Status**: MISSING
- **Evidence**: `auto-generated/intrinsic_funcs.adoc` has `vfmacc.vv` (vector-vector FMA) and `vfmacc.vf` (scalar broadcast FMA), but no lane-indexed FMA that extracts a specific element from a vector register.
- **Analysis**: RVV has `vfmacc.vf` which broadcasts a scalar floating-point register to all lanes, and `vfmacc.vv` which does element-wise FMA. But no instruction combines lane-extract from a vector + broadcast + FMA. This is a genuine gap. ARM NEON has `fmla v0.4s, v4.4s, v8.s[lane]` for this exact purpose.

### 方案 5: vmulacc.vv -- MISSING

- **RVV equivalent**: NONE
- **Status**: MISSING
- **Evidence**: grep for `matrix.*multiply\|matmul\|mma\|outer.*product\|accumulator.*block` returned no results in the intrinsic documentation.
- **Analysis**: No matrix multiply, outer product, or MMA (matrix multiply-accumulate) instructions exist in the standard RVV specification. The proposal's 4x4 outer product FMA with dedicated accumulator registers is conceptually similar to POWER10's `xvf32gerpp` (Matrix Math Assist). This is a major architectural gap requiring a new register file and instruction encoding -- the highest implementation cost among all proposals.

### 方案 6: vclamp.vf -- MISSING

- **RVV equivalent**: NONE (emulated via `vfmax.vf` + `vfmin.vf`)
- **Status**: MISSING
- **Evidence**: RVV has `vfmax` (element-wise max) and `vfmin` (element-wise min), requiring 2 instructions for clamp semantics. No single clamp instruction exists.
- **Analysis**: No ISA has a dedicated clamp instruction -- the proposal notes this explicitly ("各平台均需 2 条指令实现 clamp"). It is a convenience instruction that would reduce 2 operations to 1 across all platforms, not just RVV.

### 方案 7: vfmax.red / vfmin.red -- PARTIAL

- **RVV equivalent**: `vfredmax.vs` / `vfredmin.vs` (float reduction max/min)
- **Status**: PARTIAL
- **Evidence**: `auto-generated/intrinsic_funcs/05_vector_reduction_operations.adoc:1261-1299`
  ```
  vfloat32m1_t __riscv_vfredmax_vs_f32m1_f32m1(vfloat32m1_t vs2, vfloat32m1_t vs1, size_t vl);
  vfloat32m1_t __riscv_vfredmin_vs_f32m1_f32m1(vfloat32m1_t vs2, vfloat32m1_t vs1, size_t vl);
  ```
- **Analysis**: `vfredmax.vs` and `vfredmin.vs` already provide single-instruction horizontal max/min reduction. Semantics: `vd[0] = max(vs2[0..VL-1], vs1[0])` / `vd[0] = min(vs2[0..VL-1], vs1[0])`. The key difference: RVV requires (a) an accumulator init value in `vs1[0]` (via `vfmv.v.f`), and (b) post-extraction of the result from element 0 (via `vfmv.f.s`). This makes the current path 3 instructions vs the proposed 1. The reduction operation IS a single instruction; the init+extract are the overhead.

### 方案 8: vunzip / vzip.vv -- MISSING

- **RVV equivalent**: NONE (emulated via `vrgather` with precomputed index vectors)
- **Status**: MISSING
- **Evidence**: RVV has `vrgather.vv` / `vrgather.vx` for arbitrary permutations, and `vslideup`/`vslidedown` for element shifts. No dedicated interleave/deinterleave (zip/unzip) instruction exists.
- **Analysis**: A dedicated zip/unzip could be faster than `vrgather` (which requires an index vector load and potentially O(VLEN) cross-lane communication). However, since the proposal acknowledges "marginal benefit" for YOLO, the hardware cost may not be justified given `vrgather` already provides a general emulation path.

### 方案 9: vwmaccwev/wod.vv -- MISSING

- **RVV equivalent**: NONE (closest: `vwmacc.vv`, but no even/odd selection)
- **Status**: MISSING
- **Evidence**: `auto-generated/intrinsic_funcs.adoc` has `vwmacc.vv` (widening multiply-accumulate, e.g., i8xi8 -> i16 accumulation) but no even/odd element selection variants. Search for `vwmacc.*[we]v|vwmacc.*[wo]d` returned empty.
- **Analysis**: LoongArch LSX has `vmaddwev_w_h` / `vmaddwod_w_h` (even/odd widening MAC). RVV's `vwmacc.vv` processes all elements sequentially, creating longer dependency chains when even-index and odd-index computations are independent. This is a genuine gap for dependency chain reduction.

### 方案 10: vfwmacc.vf -- EXISTS

- **RVV equivalent**: `__riscv_vfwmacc_vf_f32m1`
- **Status**: EXISTS
- **Evidence**: `auto-generated/intrinsic_funcs.adoc` (various vfwmacc entries)
  ```
  vfloat32mf2_t __riscv_vfwmacc_vf_f32mf2(vfloat32mf2_t vd, _Float16 vs1,
                                         vfloat16mf4_t vs2, size_t vl);
  vfloat32m1_t __riscv_vfwmacc_vf_f32m1(vfloat32m1_t vd, _Float16 vs1,
                                        vfloat16mf2_t vs2, size_t vl);
  ```
- **Analysis**: **This instruction already exists in RVV.** The operation `vd(f32) += f32(rs1_f16) * f32(vs2[i]_f16)` is exactly what the existing `vfwmacc.vf` does. The "widening" in the name refers to widening both FP16 operands to FP32 before multiply-accumulate. The proposal's described `vfwcvt + vfmul + vfadd` sequence is semantically identical to what `vfwmacc.vf` already fuses. **This proposal should be removed from the gap analysis report.**

### 方案 11: vfncvt_scale_x_f_w_i8 -- MISSING

- **RVV equivalent**: NONE (fused from `vfmul + vfncvt_f_f_w + vfncvt_rtz_x_f_w`)
- **Status**: MISSING
- **Evidence**: Individual components exist:
  - `vfncvt_f_f_w`: f32 -> f16 narrowing float (`__riscv_vfncvt_f_f_w_f16mf4`)
  - `vfncvt_rtz_x_f_w`: f16 -> i8 narrowing float-to-int (`__riscv_vfncvt_rtz_x_f_w_i8mf8`)
  - `vfmul.vf`: scalar multiply
  But no single instruction fuses f32*scale -> i8 quantization.
- **Analysis**: The existing path requires 3 instructions because RVV's narrowing operations only halve the width. `vfncvt_f_f_w` narrows f32->f16, then `vfncvt_rtz_x_f_w` narrows f16->i8. No instruction skips both steps. A fused `f32 * scale -> i8` would be genuinely new.

### 方案 12: vwmulred.vs -- MISSING

- **RVV equivalent**: NONE (fused from `vwmul.vv + vwredsum.vs`)
- **Status**: MISSING
- **Evidence**: Individual components exist:
  - `vwmul.vv`: widening multiply (e.g., i8*i8 -> i16)
  - `vwredsum.vs`: widening reduction sum (e.g., sum i16 elements -> i32 in element 0)
  But no fused multiply-reduce instruction.
- **Analysis**: Both components exist individually, but the fusion eliminates intermediate register write/read, reducing register pressure and latency. The proposal positions this as a lighter-weight alternative to `vsegdot.vv` (方案 1), since it only changes the SEW once (e8->e32) rather than requiring multiple vsetvli switches.

### 方案 13: vfabs_redmax -- MISSING

- **RVV equivalent**: NONE (fused from `vfabs + vfredmax + vfmv.f.s`)
- **Status**: MISSING
- **Evidence**: Individual components exist:
  - `vfabs.vv`: vector absolute value
  - `vfredmax.vs`: float reduction max
  - `vfmv.f.s`: extract element 0 to scalar
  But no fused abs+redmax instruction.
- **Analysis**: All three component operations exist. The fusion saves 2 instructions per amax computation. Note that the proposal describes this as exceeding ARM NEON `vmaxvq_f32` (which only does reduction, no ABS).

### 方案 14: prefetch.v -- MISSING

- **RVV equivalent**: NONE
- **Status**: MISSING
- **Evidence**: grep for `prefetch|prefetch\.v` returned no results in the intrinsic documentation.
- **Analysis**: RVV has no software data prefetch instruction. x86 has `prefetcht0`/`prefetcht1`/`prefetcht2`/`prefetchnta` and ARM has `PRFM`. This is a genuine gap in the memory system interface. The proposal references x86 SGEMM's standard practice of inserting `prefetcht0 [rdx+256]` in K-loop bodies.

### 方案 15: vsignext.vx_mu -- MISSING

- **RVV equivalent**: NONE (emulated via `vmnand + vsub.vx_mu`)
- **Status**: MISSING
- **Evidence**: Individual components exist:
  - `vmnand.mm`: mask NAND
  - `vsub.vx_mu`: masked scalar subtraction
  But no fused sign extension instruction.
- **Analysis**: The existing 2-instruction sequence (`vmnand + vsub.vx_mu`) is used for Q5_0 sign extension (convert sign bit to +/-1). This is a narrow but common pattern in 5-bit quantization formats. The proposal has the lowest expected overall gain (~1.2%) but also low implementation complexity.

### 方案 16: vnibunpack.vv -- MISSING

- **RVV equivalent**: NONE (emulated via `vand.vx + vsrl.vx`)
- **Status**: MISSING
- **Evidence**: Individual components exist (`vand.vx`, `vsrl.vx`) but no dedicated nibble unpack instruction. No `vnsrl` narrowing shift would help since the operation keeps the same element width.
- **Analysis**: Two-step sequence exists (`vand.vx` for low nibble, `vsrl.vx` for high nibble). The fusion eliminates one instruction and one intermediate register. This is a low-complexity, broadly useful instruction for all 4-bit quantization formats (Q4_K, Q5_0, Q5_K).

### 方案 17: vfadd.red.vs -- PARTIAL

- **RVV equivalent**: `vfredusum.vs` / `vfredosum.vs` (float reduction sum)
- **Status**: PARTIAL
- **Evidence**: `auto-generated/intrinsic_funcs/05_vector_reduction_operations.adoc:1243-1252`
  ```
  vfloat32m1_t __riscv_vfredusum_vs_f32m1_f32m1(vfloat32m1_t vs2, vfloat32m1_t vs1, size_t vl);
  vfloat32m1_t __riscv_vfredusum_vs_f32m2_f32m1(vfloat32m2_t vs2, vfloat32m1_t vs1, size_t vl);
  vfloat32m1_t __riscv_vfredusum_vs_f32m4_f32m1(vfloat32m4_t vs2, vfloat32m1_t vs1, size_t vl);
  vfloat32m1_t __riscv_vfredusum_vs_f32m8_f32m1(vfloat32m8_t vs2, vfloat32m1_t vs1, size_t vl);
  ```
- **Analysis**: This is the most important verification question. `vfredusum.vs` (unordered) and `vfredosum.vs` (ordered) already provide **single-instruction horizontal float sum reduction**. Semantics: `vd[0] = sum(vs2[0..VL-1]) + vs1[0]`. The reduction itself IS a single instruction. What the proposal `vfadd.red.vs` adds is:
  1. **Implicit zero init**: no need for `vfmv.v.f v_init, 0.0f` (3-cycle setup)
  2. **Scalar output**: result goes directly to a scalar FP register, eliminating `vfmv.f.s` (3-cycle extraction)
  
  This makes the full sequence 1 instruction instead of 3. Compared to ARM NEON `vaddvq_f32` (single instruction, scalar output), RVV's approach is 3x the instruction count for the same end result. The operation exists; the API overhead is the gap.

### 方案 18: vfexp.v -- MISSING

- **RVV equivalent**: NONE
- **Status**: MISSING
- **Evidence**: RVV has `vfsqrt.v` (sqrt), `vfrec7.v` (7-bit approximate reciprocal), and `vfrsqrt7.v` (7-bit approximate reciprocal sqrt), but no vector exp instruction.
- **Analysis**: No hardware vector exponential function exists in RVV. The current software implementation requires ~28 instructions per VL elements using Horner-form polynomial evaluation. This is the highest-potential single proposal for Softmax-heavy workloads (transformer inference), but also the highest hardware implementation cost due to the area required for floating-point exp evaluation.

---

## Verification Summary Table

| 方案 | Instruction | Status | Existing RVV Equivalent | Gap Description |
|------|-------------|--------|------------------------|-----------------|
| 1 | vsegdot.vv | **PARTIAL** | vdot4au_vv (Zvdot4a8i) | Data layout: Zvdot4a8i expects packed uint32 words, proposal expects unpacked bytes |
| 2 | vdot_lane.vx | **MISSING** | NONE (vdot4a_vx broadcasts scalar, not lane) | Lane extraction from vector + broadcast + dot product |
| 3 | vnarrow_sat | **PARTIAL** | vnclip (2:1 saturating narrow) | Width ratio: vnclip halves SEW, proposal quarters it (i32->i8 in one step) |
| 4 | vfmacc.vv_lane | **MISSING** | NONE (only vfmacc.vf scalar broadcast) | Lane-indexed FMA from vector register |
| 5 | vmulacc.vv | **MISSING** | NONE | 4x4 matrix outer product MMA with dedicated accumulators |
| 6 | vclamp.vf | **MISSING** | NONE (vfmax+vfmin 2-instruction emulation) | Single-instruction clamp(min,max) |
| 7 | vfmax.red / vfmin.red | **PARTIAL** | vfredmax.vs / vfredmin.vs | The reduction instruction exists but needs init+extract overhead (3 insns vs. 1) |
| 8 | vunzip / vzip.vv | **MISSING** | NONE (vrgather-based emulation) | Dedicated interleave/deinterleave |
| 9 | vwmaccwev/wod.vv | **MISSING** | NONE (vwmacc.vv without even/odd selection) | Even/odd element widening MAC |
| 10 | vfwmacc.vf | **EXISTS** | `__riscv_vfwmacc_vf_f32m1` | **Already in RVV spec -- should be removed from proposals** |
| 11 | vfncvt_scale_x_f_w_i8 | **MISSING** | NONE (fused from vfmul+vfncvt_f_f_w+vfncvt_rtz_x_f_w) | Fused f32*scale->i8 quantization |
| 12 | vwmulred.vs | **MISSING** | NONE (fused from vwmul.vv+vwredsum.vs) | Fused widening multiply-reduce |
| 13 | vfabs_redmax | **MISSING** | NONE (fused from vfabs+vfredmax+vfmv.f.s) | Fused abs+reduce-max |
| 14 | prefetch.v | **MISSING** | NONE | Software data prefetch |
| 15 | vsignext.vx_mu | **MISSING** | NONE (fused from vmnand+vsub.vx_mu) | Fused mask sign extension |
| 16 | vnibunpack.vv | **MISSING** | NONE (fused from vand.vx+vsrl.vx) | Fused nibble unpack |
| 17 | vfadd.red.vs | **PARTIAL** | vfredusum.vs / vfredosum.vs | Reduction exists but needs init+extract (3 insns vs. 1) |
| 18 | vfexp.v | **MISSING** | NONE (vfsqrt/vfrec7/vfrsqrt7 exist but no exp) | Hardware vector exponential |

---

## Detailed Analysis: 方案 17 vfadd.red.vs vs. Existing RVV Reductions

This proposal warrants special attention because `vfredusum.vs` / `vfredosum.vs` already exist in RVV and perform the exact same mathematical operation (horizontal float sum reduction). The question is what makes the proposal different.

### Current RVV Reduction API

```c
// Existing RVV -- horizontal float sum reduction
vfloat32m1_t vfredusum_vs_f32m1_f32m1(
    vfloat32m1_t vs2,    // source vector to reduce
    vfloat32m1_t vs1,    // accumulator (result placed in element 0)
    size_t vl);
```

This does: `vd[0] = vs1[0] + sum(vs2[0..vl-1])` where vd is the return value (a vector register with the scalar result in element 0).

**Required usage pattern**:
```asm
vfmv.v.f    v_init, 0.0f          # Step 1: zero-init accumulator (3 cycles)
vfredusum.vs v_result, v_data, v_init  # Step 2: horizontal sum (4 cycles)
vfmv.f.s    f_result, v_result    # Step 3: extract scalar from element 0 (3 cycles)
# Total: 10 cycles, 3 instructions
```

### Proposed vfadd.red.vs API

```
vfadd.red.vs vd, vs2, vm
  功能：vd[0] ← Σ(vs2[0..VL-1])
```

This would:
1. Implicitly zero-initialize the accumulator (no separate init instruction)
2. Perform the horizontal sum reduction
3. Place result directly in a scalar FP register (no separate extract instruction)

### ARM NEON Equivalent

ARM NEON's `vaddvq_f32` is a single instruction that horizontally sums all 4 float32 lanes into a scalar.

### Verdict

**The operation is NOT new.** RVV already has single-instruction horizontal float sum reduction. The proposal adds:
- Implicit zero initialization of the accumulator
- Direct scalar register output (eliminating the `vfmv.f.s` extraction)

These are API/usability improvements, not a new operation. The functional gap is in instruction count (3 vs 1) for what ARM NEON achieves in a single instruction. The proposal is best understood as an "API simplification" rather than a new arithmetic capability.

### Comparison with ARM NEON vaddvq_f32

| Feature | ARM vaddvq_f32 | RVV vfredusum.vs | Proposed vfadd.red.vs |
|---------|---------------|------------------|-----------------------|
| Single instruction | Yes | Yes (reduction only) | Yes (incl. init+extract) |
| Implicit zero init | Yes | No (needs vfmv.v.f) | Yes |
| Scalar output | Yes (scalar register) | No (needs vfmv.f.s) | Yes |
| VLEN configurable | No (fixed 128-bit) | Yes | Yes |
| Ordered variant | N/A | vfredosum.vs | Not specified |

---

## Detailed Analysis: 方案 7 vfmax.red/vfmin.red vs. Existing RVV Reductions

The same pattern as 方案 17 applies here: `vfredmax.vs` and `vfredmin.vs` already exist.

```c
vfloat32m1_t vfredmax_vs_f32m1_f32m1(vfloat32m1_t vs2, vfloat32m1_t vs1, size_t vl);
// Semantics: vd[0] = max(vs1[0], max(vs2[0..vl-1]))
```

Current usage:
```asm
vfmv.v.f    v_init, -Infinity      # init with -Inf for max (or +Inf for min)
vfredmax.vs v_result, v_data, v_init  # horizontal max
vfmv.f.s    f_result, v_result     # extract scalar
```

The reduction IS a single instruction. The proposal only eliminates init+extract overhead.

---

## Actionable Recommendations

1. **Remove 方案 10 (vfwmacc.vf)**: Already exists as `__riscv_vfwmacc_vf_f32m1` in the RVV specification. The gap analysis report should be corrected.

2. **Reclassify 方案 7 and 方案 17**: These are "API simplifications" rather than new operations. The reduction IS single-instruction; the gap is init+extract overhead. Marketing them as "single-instruction horizontal sum" in the report while `vfredusum` already does that is misleading -- the accurate framing is ""single-instruction init+reduce+extract fusion".

3. **Clarify 方案 1 vs Zvdot4a8i**: The report should acknowledge that Zvdot4a8i already provides quad-widening 4D dot product, and clarify whether `vsegdot.vv` is a byte-layout variant (unpacked vs. packed uint32) or genuinely adds new functionality.

4. **Correct 方案 3 framing**: `vnclip` already provides saturating narrowing. The gap is specifically that RVV's narrowing operations only halve the element width (2:1 ratio), requiring two steps for i32->i8. The proposal is a 4:1 ratio narrowing.

5. **Highest-value genuine gaps** (MISSING, no equivalent):
   - 方案 4 `vfmacc.vv_lane` (lane-indexed FMA, 22-29% benefit)
   - 方案 5 `vmulacc.vv` (matrix outer product, 44-77% benefit)
   - 方案 18 `vfexp.v` (hardware exp, ~96% BB reduction, highest implementation cost)
   - 方案 14 `prefetch.v` (data prefetch, 5-15% benefit)
