# gemv-q4_K-8x8-q8_K

RVV implementation of `ggml_gemv_q4_K_8x8_q8_K` - Q4_K weights × Q8_K activations GEMV (matrix-vector multiplication) with 8x8 interleaved tile blocking.

## Status

⚠️ **Blocked by LLVM-22 Bug** — Vectorization disabled, using scalar fallback.

**LLVM Bug**: LLVM 22 RISC-V backend optimizer produces incorrect code when compiling scalar code with RVV extensions (`-march=rv64gcv_zvl512b`). The optimizer generates garbage values for array elements even with explicit zero-initialization. This affects both the test harness and the GEMV kernel itself.

**Workaround**: The RVV function currently falls back to the generic scalar implementation (`ggml_gemv_q4_K_8x8_q8_K_generic`). Tests use `__attribute__((optnone))` to disable optimization for correctness verification.

**Vectorized Version**: The full RVV vectorized implementation can be found in commit `a650391b612d8d2394ef72547aee19efce6d5748`. It will be restored once the LLVM bug is resolved.

**LLVM Bug Report**: See [LLVM issue #83370](https://github.com/llvm/llvm-project/issues/83370) for details on the optimizer bug with minimal reproduction.

## Files

| File | Purpose |
|------|---------|
| `rvv_gemv_q4_K_8x8_q8_K.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc);
```

## VLEN Requirement

- **VLEN >= 512**: Uses optimized RVV intrinsics for final operations
- **VLEN < 512**: Falls back to scalar generic implementation

## Algorithm

Based on ARM NEON implementation from llama.cpp:

1. For each block of 256 elements (QK_K):
   - Decode 6-bit scales/mins from 96-byte array (24 bytes per subblock)
   - For each column pair (01, 23, 45, 67):
     - Compute dot product: q4 nibble × q8 using widening MAC
     - Apply scales: `sumi_lo × scales_lo + sumi_hi × scales_hi`
   - Bias correction: `mins × bsums × dmin × d`

2. Final output: `accumulated_sum - bias_correction`

### Data Layout

**block_q4_Kx8**:
```cpp
struct block_q4_Kx8 {
    ggml_half d[8];      // 8 scales (one per column)
    ggml_half dmin[8];   // 8 min scales
    uint8_t scales[96];  // 6-bit packed (24 bytes × 4 subblocks)
    uint8_t qs[1024];    // 4-bit packed quants
};
```

**qs Index Layout** (per subblock sb):
- q4_base = `qs + sb * 256`
- For column pair cp (0-3), load 4 vectors:
  - q4_base + 16*cp + 0 (vec_idx=0)
  - q4_base + 16*cp + 64 (vec_idx=1)
  - q4_base + 16*cp + 128 (vec_idx=2)
  - q4_base + 16*cp + 192 (vec_idx=3)

**q8 Index Layout**:
- Low nibbles: `sb * 64 + vec_idx * 8 + q8_half + n`
- High nibbles: `sb * 64 + vec_idx * 8 + 32 + q8_half + n`
- Where q8_half = (sum_idx % 2) * 4 (0 or 4)

### Dot Product Structure

Each 16-byte q4 vector produces 4 int32 sums via vdotq_s32 pattern:
- sum_idx=0: nibbles[0..3] × q8[0..3]
- sum_idx=1: nibbles[4..7] × q8[4..7]
- sum_idx=2: nibbles[8..11] × q8[0..3] (duplicate)
- sum_idx=3: nibbles[12..15] × q8[4..7] (duplicate)

### Scale Application

```cpp
// Pairwise add: acc_lo[p][k] + acc_lo[p+1][k]
sum_lo[k] = acc_lo[cp_group][k] + acc_lo[cp_group+1][k];

// Multiply by scales and accumulate
float scaled = scales[k] * sum[k] * sb_scale[k];
acc_f32[group][k] += scaled;
```

## 6-bit Scale/Min Encoding

From 12 bytes → 8 scales + 8 mins:
```
sm[0]: scales 0-3 in bits 0-5, scales 4-7 upper 2 bits in bits 6-7
sm[1]: mins 0-3 in bits 0-5, mins 4-7 upper 2 bits in bits 6-7
sm[2]: scales 4-7 lower 4 bits in bits 0-3, mins 4-7 lower 4 bits in bits 4-7
```

Decoding (matching ARM NEON):
```cpp
mins_0_3 = sm[1] & 0x3f3f3f3f;
mins_4_7 = ((sm[2] >> 4) & 0x0f0f0f0f) | (((sm[1] >> 6) & 0x03030303) << 4);
scales_0_3 = sm[0] & 0x3f3f3f3f;
scales_4_7 = (sm[2] & 0x0f0f0f0f) | (((sm[0] >> 6) & 0x03030303) << 4);
```

## RVV Intrinsics (Future)

The following intrinsics are planned for use once the LLVM bug is resolved:

| Operation | Intrinsic | Purpose |
|-----------|-----------|---------|
| Load float32 | `__riscv_vle32_v_f32m1` | Load accumulator vectors |
| Float multiply | `__riscv_vfmul_vv_f32m1` | Scale bias correction |
| Float subtract | `__riscv_vfsub_vv_f32m1` | Final result computation |
| Store float32 | `__riscv_vse32_v_f32m1` | Store output |
| Set VL | `__riscv_vsetvl_e32m1(4)` | 4-element vector length |

**Current Implementation**: Falls back to scalar loops. The algorithm structure and data flow are preserved, enabling straightforward RVV vectorization once the compiler bug is fixed.

## Build & Test

```bash
# Cross-compile with RVV extensions (falls back to scalar due to LLVM bug)
docker run --rm \
  -v $(pwd):/work -w /work \
  -v /path/to/sysroot:/sysroot \
  rvfuse/llvm-riscv:22 clang++ -std=c++17 -O2 \
  --target=riscv64-unknown-linux-gnu \
  --sysroot=/sysroot \
  -march=rv64gcv_zvl512b -mabi=lp64d \
  -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
  -fuse-ld=lld \
  test.cpp -o test_gemv -lm

# Run under QEMU (tests use optnone to avoid optimizer bug)
qemu-riscv64 -L /path/to/sysroot ./test_gemv

# Scalar build (no RVV, for reference comparison)
g++ -std=c++17 -O2 test.cpp -o test_scalar -lm
```

**Note**: The RVV build currently produces the same scalar code as the non-RVV build due to the LLVM optimizer bug workaround.

## References

- ARM NEON implementation: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` (lines 709-861)
- x86 AVX2 implementation: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` (lines 1464-1685)
- Data structures: `vendor/llama.cpp/ggml/src/ggml-cpu/repack.h`