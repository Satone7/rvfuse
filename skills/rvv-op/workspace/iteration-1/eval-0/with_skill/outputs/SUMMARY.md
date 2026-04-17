# Implementation Summary: ggml_gemv_q5_0_q8_0 RVV Vectorization

## Task

Implement RISC-V RVV vectorized `ggml_gemv_q5_0_q8_0` operator for llama.cpp with VLEN=512.

## Steps Taken

### Step 1: Read Application README

Read `/home/pren/wsp/cx/rvfuse/applications/llama.cpp/README.md`. Key findings:
- llama.cpp pinned at commit `e21cdc11` (b8783)
- Target: rv64gcv_zfh_zvfh_zicbop_zihintpause (lp64d ABI)
- Existing rvv-patches: gemm-q4_K-8x4-q8_K, gemv-q4_K-8x8-q8_K, quantize-q8_0-4x4
- Template directory exists at rvv-patches/_template/
- Build system: build.sh orchestrator, LLVM 22 cross-compilation
- Patch target: `ggml/src/ggml-cpu/arch/riscv/repack.cpp`

### Step 2: Discover Application Structure

| Item | Location |
|------|----------|
| RISC-V arch source | `vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/` |
| Arch files | quants.c, repack.cpp, cpu-feats.cpp, rvv_*.inl |
| Existing RVV Q5_0 | `rvv_vec_dot_q5_0_q8_0.inl` (vec_dot only, not GEMV) |
| Generic Q5_0 dot | `quants.c:308` (`ggml_vec_dot_q5_0_q8_0_generic`) |
| Q4_0 GEMV patterns | `repack.cpp` has `ggml_gemv_q4_0_8x8_q8_0`, `ggml_gemv_q4_0_16x1_q8_0` |
| GEMV function declarations | `repack.h` (no Q5_0 GEMV exists) |

### Step 3: Cross-Platform Analysis

**Critical finding**: `ggml_gemv_q5_0_q8_0` does NOT exist in upstream llama.cpp. Only `ggml_vec_dot_q5_0_q8_0` (a per-block vector dot product) exists. The GEMV pattern (processing multiple output columns) only exists for Q4_0 and Q4_K/Q5_K/Q6_K formats.

Cross-platform implementations of `ggml_vec_dot_q5_0_q8_0` (used as reference):

| Platform | File | Lines | Key Operations |
|----------|------|-------|----------------|
| **ARM NEON** | `arch/arm/quants.c:840` | ~60 lines | `vld1q_u8`, `vandq_u8`, `vshrq_n_u8`, `table_b2b_1` lookup for 5th bit, `ggml_vdotq_s32` (i8*i8 dot), `vmlaq_n_f32` FMA, processes 2 blocks at once |
| **x86 AVX2** | `arch/x86/quants.c:846` | ~40 lines | `bytes_from_nibbles_32`, `bytes_from_bits_32`, `_mm256_andnot_si256` for 5th bit, `_mm256_or_si256`, `mul_sum_i8_pairs_float` (i8*i8 pairs), `_mm256_fmadd_ps` FMA |
| **RISC-V RVV (existing)** | `arch/riscv/rvv_vec_dot_q5_0_q8_0.inl` | ~90 lines | Dual-path: VLEN>=256 (m1-based) and VLEN=128 (m2-based). Uses `vle8_v_u8mf2`, `vand_vx`, `vsrl_vx`, `vlm_v_b8` for bitmask, `vmnand_mm` for invert, `vsub_vx` masked subtract for sign extension, `vwmul_vv` widening MAC, `vwredsum_vs` reduction |
| **Generic (scalar)** | `quants.c:308` | ~30 lines | Bit extraction loops, nibble unpack, signed multiply-accumulate |

**Q5_0 format specifics**:
- `block_q5_0`: 32 elements per block (QK5_0=32)
- `d`: FP16 scale (2 bytes)
- `qh[4]`: 5th bit for each element (4 bytes = 32 bits)
- `qs[16]`: Packed 4-bit nibbles (16 bytes)
- Value decoding: lower nibble = (qs[j] & 0x0F) | (qh bit j << 4) - 16; upper nibble = (qs[j] >> 4) | (qh bit (j+16) << 4) - 16

### Step 4: Create Operator Package

Created 4 files in `rvv-patches/gemv-q5_0-q8_0/`:

#### File 1: `rvv_gemv_q5_0_q8_0.inl` (12.9 KB)
- **RVV implementation** (`ggml_gemv_q5_0_q8_0_rvv`):
  - 16x1 tile: processes 16 output columns simultaneously
  - Requires VLEN >= 512 (vlenb >= 64)
  - Uses `__riscv_zvfh` for FP16 scale operations
  - Per-block processing:
    1. Load packed nibbles via `vle8_v_u8mf2`
    2. Extract lower/upper nibbles via `vand_vx`/`vsrl_vx`
    3. Load qh bitmask via `vlm_v_b8`
    4. Invert mask via `vmnand_mm_b16`
    5. Apply sign extension via masked `vsub_vx` (subtract 0x10 where qh bit is 0)
    6. Compute dot product with Q8_0 activation block
    7. Reduce int16 partials to int32 via `vwadd_vv`
    8. Convert to float via `vfcvt_f_x_v`
    9. Apply combined scale via `vfmul_vf` + `vfmacc_vv`
  - Falls back to generic for VLEN < 512
- **Generic implementation** (`ggml_gemv_q5_0_q8_0_generic`):
  - Column-sequential scalar loop
  - Bit extraction per element, nibble unpack, signed MAC
- **Wrapper** (`ggml_gemv_q5_0_q8_0`):
  - Compile-time dispatch: RVV+Zvfh or generic fallback

#### File 2: `test.cpp` (12.0 KB)
- FP16 conversion helpers (IEEE 754 half <-> single)
- Scalar reference with `__attribute__((optnone))` for RVV targets
- LCG-based deterministic RNG (seed=42)
- 8 test cases:
  - 1x32, 4x32, 16x64, 32x64, 48x128, 1x128, 16x256, 64x128
- Tolerance: 1.0 (exact integer computation, small FP rounding in scale)

#### File 3: `patch.diff` (601 B)
- Targets: `ggml/src/ggml-cpu/arch/riscv/repack.cpp`
- Adds `#include "rvv_gemv_q5_0_q8_0.inl"` after `#define UNUSED GGML_UNUSED`
- 4 lines of context (standard git diff format)

#### File 4: `README.md` (4.3 KB)
- Function signature, parameter documentation
- Algorithm description (8 steps)
- Q5_0 and Q8_0 format specifications
- Cross-platform reference table
- Comparison with Q4_0 GEMV pattern
- Build and test instructions
- LLVM bug considerations

### Step 5: What the Gap Analysis Would Do (Phase 4, Step 7)

The `rvv-gap-analysis` skill would:

1. **Parse the RVV implementation** in `rvv_gemv_q5_0_q8_0.inl` to extract all RVV intrinsics used

2. **Launch parallel subagent analysis** for each reference platform:
   - x86 AVX2: Identify AVX2 intrinsics used in `arch/x86/quants.c:846` that have no RVV equivalent
   - ARM NEON: Identify NEON intrinsics used in `arch/arm/quants.c:840` that have no RVV equivalent
   - LoongArch LASX: Check for LoongArch vector extensions
   - PowerPC VSX: Check for PowerPC vector extensions
   - S390X VXE: Check for S390X vector extensions
   - WASM SIMD: Check for WebAssembly SIMD128

3. **Identify missing RVV instructions**, such as:
   - No native nibble-to-byte expansion (x86 `pshufb`/`pmovzxbw` equivalent)
   - No fused nibble-unpack-and-duplicate (ARM `table_b2b_1` equivalent)
   - No direct masked byte subtract with sign extension
   - No native int8xint8->int16 pairwise dot product with reduction (need separate `vwmul` + `vwredsum`)
   - Missing 4-bit shuffle capabilities

4. **Propose new RVV extensions** with benefit estimates:
   - E.g., a nibble-expand instruction that converts 16 packed nibbles to 16 sign-extended bytes
   - E.g., a 5th-bit-merge instruction for Q5 formats
   - Benefit: reduce instruction count by ~30% for Q5_0 GEMV inner loop

5. **Generate gap analysis report** at `docs/report/llama.cpp/rvv-gap-analysis-gemv-q5_0-q8_0-2026-04-17.md`

### Step 6: What the PDF Report Would Do (Phase 4, Step 8)

The `lovstudio-any2pdf` skill would:

1. Convert the gap analysis markdown report to a professionally typeset PDF
2. Theme: `github-light`
3. Skip cover page, skip watermark, skip frontispiece
4. Output: `docs/report/llama.cpp/pdf/rvv-gap-analysis-gemv-q5_0-q8_0-2026-04-17.pdf`

## Key Design Decisions

1. **No interleaved Q5_0 format**: Unlike Q4_0 which has `block_q4_0x16` (16 columns interleaved in memory), Q5_0 has no upstream interleaved format. The implementation processes columns sequentially but accumulates 16 results simultaneously using vector registers.

2. **16x1 tile**: Chosen to match the `ggml_gemv_q4_0_16x1_q8_0` pattern (requires Zvfh for FP16 scale operations). This tile size processes 16 output columns in parallel using m1/m2 vector registers.

3. **Zvfh requirement**: FP16 scale conversion (`GGML_CPU_FP16_TO_FP32`) needs hardware FP16 support. Alternative: compute scales on host and pass as FP32.

4. **Sign extension via masked subtract**: The existing `rvv_vec_dot_q5_0_q8_0.inl` uses this same technique (invert qh mask, then `vsub_vx_mu` with 0x10). This is the canonical RVV approach for Q5_0 sign extension.

5. **Column-sequential memory access**: Without interleaved Q5_0 blocks, weight data is accessed column-by-column. This is less cache-friendly than interleaved formats but is a limitation of the existing Q5_0 storage layout.

## Files Created

| File | Path |
|------|------|
| RVV Implementation | `/home/pren/wsp/cx/rvfuse/applications/llama.cpp/rvv-patches/gemv-q5_0-q8_0/rvv_gemv_q5_0_q8_0.inl` |
| Test | `/home/pren/wsp/cx/rvfuse/applications/llama.cpp/rvv-patches/gemv-q5_0-q8_0/test.cpp` |
| Patch | `/home/pren/wsp/cx/rvfuse/applications/llama.cpp/rvv-patches/gemv-q5_0-q8_0/patch.diff` |
| README | `/home/pren/wsp/cx/rvfuse/applications/llama.cpp/rvv-patches/gemv-q5_0-q8_0/README.md` |
| Summary | `/home/pren/wsp/cx/rvfuse/skills/rvv-op/workspace/iteration-1/eval-0/with_skill/outputs/SUMMARY.md` |

## Caveats

- This is a **plan-and-implement evaluation** -- no builds or QEMU execution were performed
- The implementation has not been compiled or tested against actual RISC-V hardware or QEMU
- The patch has not been verified with `git apply --check` against the actual source tree
- The RVV path in the .inl file contains a partially vectorized inner loop (per-column scalar extraction of dot product results) that could be further optimized with proper lane management