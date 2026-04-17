# Steps Summary: vec_dot_q6_K_q8_K RVV Vectorization (VLEN=512)

## Task
Vectorize `ggml_vec_dot_q6_K_q8_K` for RISC-V RVV with VLEN=512 in llama.cpp.

## Phase 1: Research & Analysis

### Step 1: Read Application README
- File: `applications/llama.cpp/README.md`
- Key findings:
  - llama.cpp pinned at commit `e21cdc11` (b8783)
  - Build system: `build.sh` with cross-compilation to `rv64gcv_zfh_zvfh_zicbop_zihintpause`
  - Existing RVV patches: `gemm-q4_K-8x4-q8_K`, `gemv-q4_K-8x8-q8_K`, `quantize-q8_0-4x4`
  - Patch target: `ggml/src/ggml-cpu/arch/riscv/quants.c` (for vec_dot functions)
  - Note: GEMV patches target `arch/riscv/repack.cpp`, but vec_dot functions live in `quants.c`
  - BBV profiling shows `ggml_gemv_q4_K_8x8_q8_K` at 32.1% of core inference compute

### Step 2: Analyze Cross-Platform Implementations

Found `ggml_vec_dot_q6_K_q8_K` in 11 files across 6 platforms:

| Platform | File | Line | Implementation Details |
|---|---|---|---|
| **ARM SVE+MATMUL** | `arch/arm/quants.c` | 2884 | Uses `svmmla_s32` (matrix multiply accumulate), `svdot_s64` for scale/bsum interaction. Supports nrc=2 (dual-row). 128/256/512 VL paths. |
| **ARM MATMUL (NEON)** | `arch/arm/quants.c` | 3095 | Uses `vmmla_s32` (NEON matrix multiply), 16-element blocks with `vld1q_u8_x4` loads. Dual-row processing. |
| **x86 AVX2** | `arch/x86/quants.c` | 2130 | Uses `_mm256_maddubs_epi16` (unsigned x signed → int16), `_mm256_madd_epi16` (int16 × scale → int32), `_mm256_fmadd_ps`. Handles q6 offset via separate q8sum subtraction. |
| **x86 AVX** | `arch/x86/quants.c` | 2222 | Similar to AVX2 but with 128-bit operations, uses `_mm_maddubs_epi16` and `_mm_madd_epi16`. |
| **RISC-V (T-Head XTheadVector)** | `arch/riscv/quants.c` | 1713 | Uses T-Head vendor intrinsics (`th.vwmul.vv`, `th.vwredsum.vs`). Processes full 128-element chunks with inline assembly. |
| **RISC-V RVV (VLEN=256)** | `arch/riscv/quants.c` | 1800 | Uses standard RVV intrinsics. Processes 32-byte loads for ql/qh, 4 groups of 32-element widening multiplies, 8 scale multiplications, chained `vredsum`. |
| **RISC-V RVV (VLEN=128)** | `arch/riscv/quants.c` | 1881 | Uses inline assembly with fixed VL values (32/64/128). Prefetching, scale extraction from packed 64-bit, direct FMA in register. |
| **Generic (scalar)** | `quants.c` | 794 | Reference implementation. Decodes Q6_K byte-by-byte into `aux8[QK_K]`, then 16-element blocks with per-block scale application. |

**Critical finding**: The RISC-V switch statement has `case 256:` and `case 128:` but **NO `case 512:`**. The `default` case asserts false, meaning VLEN=512 hardware either crashes or (if the assert is compiled out) falls through to generic.

### Step 3: Map Cross-Platform SIMD Paths

Patch target file: `ggml/src/ggml-cpu/arch/riscv/quants.c` (line 1700)
Function: `ggml_vec_dot_q6_K_q8_K`

Data structures (from `ggml-common.h`):
- `block_q6_K` (210 bytes): ql[128] + qh[64] + scales[16] + d(FP16)
- `block_q8_K` (576 bytes): d(FP32) + qs[256] + bsums[16]

## Phase 2: Implementation

### Step 4: Create Operator Package

Created 4 files in `rvv-patches/vec_dot-q6_K-q8_K/`:

1. **`rvv_vec_dot_q6_K_q8_K.inl`** -- Full RVV implementation
   - Algorithm follows x86 AVX2 / ARM NEON strategy
   - For each super-block: decode qh (2-bit fields) + ql (4-bit nibbles) → 6-bit values
   - Subtract 32 offset, widening multiply with q8, apply 8 per-block scales
   - VLEN=512 allows processing all 64 bytes of ql and 32 bytes of qh in e8m1
   - Widening operations use e16m2 and e32m2 (well within VLEN=512 budget)
   - Chained `vredsum_vs` for horizontal reduction of 8 scale products

2. **`test.cpp`** -- Standalone correctness test
   - Self-contained: includes minimal ggml type definitions (ggml_half, block_q6_K, block_q8_K)
   - Scalar reference extracted verbatim from `quants.c:794`
   - Uses `__attribute__((optnone))` on scalar reference to avoid LLVM-22 optimizer bug
   - Deterministic LCG RNG (seed=42) for reproducibility
   - 5 test cases: 1, 2, 4, 8, 16 blocks
   - Tolerance: relative 1e-4 for FP32 accumulation

3. **`patch.diff`** -- Integration patch
   - Adds `#include "rvv_vec_dot_q6_K_q8_K.inl"` before the function
   - Adds `case 512:` in the existing `switch (vector_length)` statement
   - The `case 512:` delegates to `ggml_vec_dot_q6_K_q8_K_rvv()` from the .inl file
   - Context lines target the specific location between `case 256:` and `case 128:`

4. **`README.md`** -- Operator documentation
   - Algorithm description with step-by-step breakdown
   - VLEN=512 specific register usage table
   - Cross-platform reference table
   - Q6_K decoding algorithm
   - Build & test instructions

## Phase 3: Testing (Not Executed -- Plan Only)

### Step 5: Build and Run Standalone Test
Would execute:
```bash
clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
    --sysroot=$SYSROOT -march=rv64gcv_zvl512b -mabi=lp64d \
    -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
    -fuse-ld=lld test.cpp -o test -lm
qemu-riscv64 -L $SYSROOT ./test
```
Expected: All 5 tests pass with relative error < 1e-4.

### Step 6: Integration Test
Would execute:
```bash
cd applications/llama.cpp && ./build.sh --force
qemu-riscv64 -L $SYSROOT output/llama.cpp/bin/llama-cli -m models/qwen-0.5b-q4_0.gguf -p "test"
```
Verify: `nm lib/libggml-cpu.so | grep vec_dot_q6_K_q8_K` shows the symbol.

## Phase 4: Reporting (Not Executed -- Plan Only)

### Step 7: Gap Analysis
Would invoke the `rvv-gap-analysis` skill to:
1. Parse the RVV implementation from `rvv_vec_dot_q6_K_q8_K.inl`
2. Launch parallel subagents to analyze x86 AVX2, ARM NEON/SVE, LoongArch, Power, S390X, WASM implementations
3. Compare intrinsic usage patterns to identify instructions missing from RVV
4. Key candidates for gap analysis:
   - `_mm256_maddubs_epi16` (unsigned x signed → int16 widening MAC) -- RVV has `vwmul` but not an equivalent fused unsigned*signed operation
   - `svmmla_s32` / `vmmla_s32` (matrix multiply accumulate) -- RVV lacks a direct equivalent
   - `_mm256_fmadd_ps` (fused multiply-add) -- RVV has `vfmacc` but not at the same throughput
5. Output: `docs/report/llama.cpp/rvv-gap-analysis-vec_dot-q6_K-q8_K-YYYY-MM-DD.md`

### Step 8: Generate PDF Report
Would invoke the `lovstudio-any2pdf` skill to:
1. Convert the markdown gap analysis report to PDF
2. Theme: `github-light`
3. Output: `docs/report/llama.cpp/pdf/rvv-gap-analysis-vec_dot-q6_K-q8_K-YYYY-MM-DD.pdf`

## Key Technical Observations

1. **VLEN=512 is a gap**: The existing RISC-V implementation has no VLEN=512 path for `vec_dot_q6_K_q8_K`. This means VLEN=512 hardware (common in modern RISC-V chips like SpacemiT K1, SiFive P870) gets zero vectorization benefit for Q6_K dot products.

2. **Algorithm matches VLEN=256**: The VLEN=512 implementation uses the same algorithm as VLEN=256 (same intrinsic pattern) but benefits from wider registers allowing more data to be processed per instruction without additional register pressure.

3. **Scale application is the bottleneck**: The 8 scale multiplications (int16 x int8 -> int32) followed by chained reduction (vredsum) accounts for a significant portion of the instruction count. ARM SVE can use `svmmla_s32` to fuse this, while x86 uses `_mm256_madd_epi16`. RVV requires separate `vwmul_vx` + `vredsum` operations.

4. **The -32 offset handling**: x86 AVX2 handles the Q6_K offset (-32) by computing `_mm256_maddubs_epi16(m32s, q8)` (accumulating `-32 * q8`) and subtracting it from the main product. ARM handles it explicitly (`- 32`). The RVV implementation follows the ARM approach: subtract 32 before the widening multiply, which is simpler and avoids extra arithmetic.
