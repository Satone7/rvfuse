# vec_dot-q6_K-q8_K

RVV implementation of `ggml_vec_dot_q6_K_q8_K` -- dot product of Q6_K quantized weights with Q8_K quantized activations.

## Status

Plan-and-implement evaluation. Not yet built or tested.

## Files

| File | Purpose |
|------|---------|
| `rvv_vec_dot_q6_K_q8_K.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);
```

## Motivation

The existing RISC-V RVV implementation of `ggml_vec_dot_q6_K_q8_K` in `arch/riscv/quants.c` has optimized paths for VLEN=256 and VLEN=128, but **no VLEN=512 path**. On VLEN=512 hardware, the `switch (vector_length)` falls through to the `default` case, which asserts false on RVV builds or falls back to the generic scalar implementation. This patch adds a VLEN=512 case that uses RVV intrinsics, filling the performance gap.

## Algorithm

Based on the x86 AVX2 and ARM NEON implementations from llama.cpp, adapted for RVV:

1. For each super-block (QK_K=256 elements):
   a. Compute block-level scale: `d = x[i].d * y[i].d`
   b. For each half-block (QK_K/128 = 2 iterations, 128 elements each):
      - Load `qh` (32 bytes): extract 2-bit upper fields into 4 groups
      - Load `q6` (64 bytes, lower 4 bits): extract nibbles into 4 groups
      - Combine q6 nibbles with qh fields to reconstruct 6-bit values
      - Subtract 32 (Q6_K offset) to get signed values in [-32, 31]
      - Load `q8` (128 bytes): int8 activation values in 4 groups of 32
      - Widening multiply: `vwmul(q6_signed, q8)` produces int16 partial products
      - Apply 8 per-block scales via `vwmul_vx_i32m2` (int16 x int8 -> int32)
      - Horizontal reduction: `vredsum` across int32 groups -> single sum per half-block
   c. Final: `sumf += d * sum_t`

## VLEN=512 Specifics

VLEN=512 provides 64 bytes per vector register, which is well-suited for this algorithm:

| Data Element | Size | LMUL Group | VL Value |
|---|---|---|---|
| qh | 32 bytes | e8m1 | 32 |
| q6 low/high | 32 bytes each | e8m1 | 32 |
| q8 groups | 32 bytes each | e8m1 | 32 |
| Widening products | 64 bytes | e16m2 | 32 |
| Scale products | 64 bytes | e32m2 | 16 |
| Reduction output | 4 bytes | e32m1 | 1 |

With VLEN=512, all q6/qh/q8 loads fit naturally in `e8m1` registers, and widening operations use `e16m2` (2 registers) without exceeding VLEN budget. The 8 scale multiplies use `e32m2` with vl=16, well within the VLEN=512 capacity.

## Data Layout

### block_q6_K (210 bytes total)

```cpp
struct block_q6_K {
    uint8_t ql[128];     // Lower 4 bits, 2 per byte (128 bytes)
    uint8_t qh[64];      // Upper 2 bits, 4 per byte (64 bytes)
    int8_t  scales[16];  // Per-block scales, 8-bit (16 bytes)
    ggml_half d;         // Super-block FP16 scale (2 bytes)
};
```

### block_q8_K (576 bytes total)

```cpp
struct block_q8_K {
    float   d;           // FP32 block scale (4 bytes)
    int8_t  qs[256];     // Int8 quantized values (256 bytes)
    int16_t bsums[16];   // Block sums (32 bytes)
};
```

### Q6_K Decoding

Each 6-bit quantized value is reconstructed from two sources:

```
For element index e in half-block (0..127):
  ql_byte  = ql[e/2]          // packed: lower nibble = elem(2k), upper = elem(2k+1)
  ql_nibble = (e % 2 == 0) ? (ql_byte & 0x0F) : (ql_byte >> 4)
  qh_byte  = qh[e/4]          // packed: 4 × 2-bit fields per byte
  qh_field = (qh_byte >> (2 * (e % 4))) & 0x03
  q6_value = ql_nibble | (qh_field << 4)   // 6-bit unsigned [0, 63]
  signed    = q6_value - 32                 // signed [-32, 31]
```

## Cross-Platform Reference Implementations

| Platform | File | Key Intrinsics |
|---|---|---|
| ARM NEON/SVE | `arch/arm/quants.c:2884` | `vmmla_s32`, `svdot`, `vld1q_u8_x4` |
| x86 AVX2 | `arch/x86/quants.c:2130` | `_mm256_maddubs_epi16`, `_mm256_madd_epi16`, `_mm256_fmadd_ps` |
| x86 AVX | `arch/x86/quants.c:2222` | `_mm_maddubs_epi16`, `_mm_madd_epi16` |
| RISC-V RVV (VLEN=256) | `arch/riscv/quants.c:1800` | `__riscv_vwmul_vv_i16m2`, `__riscv_vredsum_vs` |
| RISC-V (T-Head XTheadVector) | `arch/riscv/quants.c:1713` | `th.vwmul.vv`, `th.vwredsum.vs` |
| Generic (scalar) | `quants.c:794` | Direct integer arithmetic |

## Key Differences from VLEN=256 Path

The existing VLEN=256 implementation processes data in two passes (lower nibbles then upper nibbles of each ql byte) due to limited register space. The VLEN=512 implementation can process both nibbles simultaneously by loading all 64 bytes of `ql` in two 32-byte loads, enabling:

1. **Fewer iterations**: Both ql halves (64 bytes) and qh (32 bytes) loaded in a single pass
2. **Wider parallelism**: All 4 q6/q8 groups processed simultaneously
3. **Simpler scale application**: All 8 scales applied in a single reduction chain

## Build & Test

```bash
# Build llama.cpp with this patch
cd applications/llama.cpp && ./build.sh --force

# Build standalone test (cross-compile for rv64gcv VLEN=512)
clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
    --sysroot=/path/to/sysroot \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
    -fuse-ld=lld test.cpp -o test_vec_dot_q6_K_q8_K -lm

# Run standalone test under QEMU
qemu-riscv64 -L /path/to/sysroot ./test_vec_dot_q6_K_q8_K

# Scalar build (for reference on x86)
g++ -std=c++17 -O2 test.cpp -o test_scalar -lm && ./test_scalar
```

## Gap Analysis

See: `docs/report/llama.cpp/rvv-gap-analysis-vec_dot-q6_K-q8_K-YYYY-MM-DD.md`

## References

- Generic implementation: `vendor/llama.cpp/ggml/src/ggml-cpu/quants.c` (line 794)
- ARM NEON/SVE: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (line 2884)
- x86 AVX2/AVX: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (line 2130)
- RISC-V VLEN=256: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` (line 1800)
- Data structures: `vendor/llama.cpp/ggml/src/ggml-common.h` (line 348-366)
- RVV 1.0 spec: https://github.com/riscv/riscv-v-spec
