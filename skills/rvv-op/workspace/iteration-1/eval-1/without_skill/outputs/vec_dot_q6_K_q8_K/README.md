# vec_dot_q6_K_q8_K

RVV implementation of `ggml_vec_dot_q6_K_q8_K` - Q6_K weight x Q8_K activation dot product (vec_dot) for RISC-V VLEN=512.

## Files

| File | Purpose |
|------|---------|
| `rvv_vec_dot_q6_K_q8_K.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp arch/riscv/quants.c |
| `test.cpp` | Correctness test (RVV vs scalar reference) |
| `README.md` | This file |

## Function Signature

```cpp
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);
```

Where:
- `n`: Total number of elements (must be multiple of QK_K=256)
- `s`: Output scalar dot product result
- `vx`: Array of `block_q6_K` weight blocks
- `vy`: Array of `block_q8_K` activation blocks
- `nrc`: Number of rows (must be 1 for this kernel)

## Data Structures

### block_q6_K (6.5625 bits per weight, 210 bytes)

```cpp
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];      // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits (16 bytes)
    ggml_half d;             // super-block scale (2 bytes)
} block_q6_K;
```

**6-bit encoding**: Each weight is 6 bits unsigned (0-63). The signed value used in computation is `decoded - 32` (range -32 to +31).

- **ql[n]**: Two 4-bit nibbles per byte
  - Low nibble: even-indexed elements (0, 2, 4, ...)
  - High nibble: odd-indexed elements (1, 3, 5, ...)
- **qh[n]**: Four 2-bit groups per byte
  - Bits [1:0]: elements 4n, 4n+2 (high bits of even-indexed)
  - Bits [3:2]: elements 4n+1, 4n+3 (high bits of odd-indexed)
  - Bits [5:4]: elements 4n+4, 4n+6
  - Bits [7:6]: elements 4n+5, 4n+7
- **scales[g]**: 8-bit signed scale for elements [16g, 16g+15]

### block_q8_K (intermediate quantization, 544 bytes)

```cpp
typedef struct {
    float   d;              // delta (4 bytes)
    int8_t  qs[QK_K];       // quants (256 bytes)
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16 (32 bytes)
} block_q8_K;
```

## Algorithm

The function computes the dot product of two quantized vectors:

```
result = sum_i (d_x[i] * d_y[i]) * sum_j (scales[j] * sum_k (q6_decoded[k] * q8[k]))
```

Per block of 256 elements (QK_K), processing 2 subblocks of 128 elements each:

1. **Decode Q6 values** (32 elements per vector register):
   - Extract 4 low bits from `ql` (low and high nibbles separately)
   - Extract 2 high bits from `qh` (4 different 2-bit fields)
   - Combine: `6bit = ql_nibble | (qh_bits << 4)`
   - Subtract 32 to convert to signed: `signed_val = 6bit - 32`

2. **Compute dot products** with Q8 activations:
   - `p = q6_signed * q8` (widening multiply: int8 x int8 -> int16)

3. **Apply per-16-element scales**:
   - Split 32-element product into 2 halves of 16
   - Multiply each half by its scale (widening: int16 x int8 -> int32)

4. **Reduce to scalar**:
   - Add 8 int32 scale-products pairwise
   - Horizontal sum across all scale-products

5. **Final accumulate**: `sumf += d_x * d_y * sum_integer`

## Cross-Platform Implementation Analysis

| Platform | File | Approach |
|----------|------|----------|
| **Generic** | `quants.c` (line 794) | Scalar loop with aux8/aux16/aux32 buffers |
| **RISC-V (XTheadVector)** | `arch/riscv/quants.c` (line 1713) | Inline assembly with T-Head vector extensions |
| **RISC-V (RVV, VLEN=256)** | `arch/riscv/quants.c` (line 1800) | RVV intrinsics, 32-element vectors at e8m1 |
| **RISC-V (RVV, VLEN=128)** | `arch/riscv/quants.c` (line 1881) | Inline assembly with all 32 vector registers |
| **ARM SVE+SME** | `arch/arm/quants.c` (line 2904) | SVE intrinsics with MMLA for nrc=2 |
| **ARM NEON** | `arch/arm/quants.c` (line 2884) | NEON intrinsics (for nrc=1) |
| **x86 AVX2** | `arch/x86/quants.c` (line 2143) | AVX2 intrinsics, 256-bit vectors |
| **x86 AVX** | `arch/x86/quants.c` (line 2222) | AVX intrinsics with bsums optimization |
| **LoongArch** | `arch/loongarch/quants.c` | LSX/LASX intrinsics |
| **PowerPC** | `arch/powerpc/quants.c` | VSX/ALTIVEC intrinsics |

### Key Findings from Cross-Platform Comparison

1. **AVX2 is the reference**: Uses `_mm256_maddubs_epi16` (unsigned-signed dot product), which directly handles the unsigned 6-bit x signed 8-bit multiply with offset subtraction. This is the most efficient approach.

2. **AVX (no VNNI)**: Uses the `bsums` optimization - precomputes `scales * bsums * 32` and subtracts from the main computation. This avoids explicit subtraction of 32 from each element.

3. **ARM NEON**: Similar to AVX2 but with 128-bit vectors and `_mm_maddubs_epi16` equivalent.

4. **ARM SVE+SME**: Uses `svdot_s64` (SVE2 integer dot product) for maximum throughput.

5. **RVV VLEN=256**: Already exists in upstream llama.cpp. Uses explicit nibble extraction and `vwmul` (widening multiply).

6. **RVV VLEN=128**: Uses all 32 vector registers with inline assembly for maximum register utilization.

### VLEN=512 Optimization Opportunity

The existing upstream code handles VLEN=256 and VLEN=128. This implementation adds VLEN=512. With VLEN=512 bytes:
- `e8m1` can hold 64 elements (up from 32 at VLEN=256)
- `e8m2` can hold 128 elements
- `e16m2` can hold 64 elements
- `e32m2` can hold 32 elements

This allows processing an entire 128-element subblock in fewer vector operations, but the core algorithm structure remains the same. The main difference is that VLEN=512 avoids the register pressure issues of VLEN=128 and has larger vectors than VLEN=256.

## VLEN Requirement

- **VLEN >= 512**: Uses this optimized RVV intrinsic path
- **VLEN == 256**: Falls through to existing upstream VLEN=256 path
- **VLEN == 128**: Falls through to existing upstream VLEN=128 path (inline assembly)
- **No RVV**: Falls through to generic scalar implementation

## RVV Intrinsics Used

| Operation | Intrinsic | Purpose |
|-----------|-----------|---------|
| Load uint8 | `__riscv_vle8_v_u8m1` | Load qh, ql data |
| Load int8 | `__riscv_vle8_v_i8m1` | Load q8 activations |
| Bit AND | `__riscv_vand_vx_u8m1` | Extract nibbles (mask 0x0F, 0x03) |
| Shift right | `__riscv_vsrl_vx_u8m1` | Extract high nibble, qh bit fields |
| Shift left | `__riscv_vsll_vx_u8m1` | Position qh bits (shift by 4) |
| Bit OR | `__riscv_vor_vv_u8m1` | Combine ql nibble + qh bits |
| Subtract | `__riscv_vsub_vx_i8m1` | Subtract 32 offset |
| Widen mul | `__riscv_vwmul_vv_i16m2` | int8 x int8 -> int16 |
| Widen mul | `__riscv_vwmul_vx_i32m2` | int16 x int8 -> int32 (scale apply) |
| Get half | `__riscv_vget_v_i16m2_i16m1` | Split vint16m2 into halves |
| Add | `__riscv_vadd_vv_i32m2` | Pair-wise add before reduction |
| Reduce sum | `__riscv_vredsum_vs_i32m2_i32m1` | Horizontal sum to scalar |
| Move to scalar | `__riscv_vmv_x_s_i32m1_i32` | Extract scalar from vector |

## Build & Test

```bash
# Cross-compile with RVV extensions (VLEN=512)
docker run --rm \
  -v $(pwd):/work -w /work \
  -v /path/to/sysroot:/sysroot \
  rvfuse/llvm-riscv:22 clang++ -std=c++17 -O2 \
  --target=riscv64-unknown-linux-gnu \
  --sysroot=/sysroot \
  -march=rv64gcv_zvl512b -mabi=lp64d \
  -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
  -fuse-ld=lld \
  test.cpp -o test_vec_dot -lm

# Run under QEMU
qemu-riscv64 -L /path/to/sysroot ./test_vec_dot

# Scalar build (no RVV, for reference comparison)
g++ -std=c++17 -O2 test.cpp -o test_scalar -lm
./test_scalar
```

## Known Issues

1. **LLVM 22 optimizer bug**: The existing GEMV implementation (`gemv-q4_K-8x8-q8_K`) is blocked by an LLVM RISC-V optimizer bug that produces incorrect code for scalar loops compiled with RVV extensions. This `vec_dot` implementation may be affected similarly - test thoroughly.

2. **VLEN dispatch**: The upstream switch-case only handles VLEN=256 and VLEN=128. The patch adds VLEN=512 support but requires proper integration into the dispatch logic.

## References

- Generic implementation: `vendor/llama.cpp/ggml/src/ggml-cpu/quants.c` (line 794)
- RVV VLEN=256: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` (line 1800)
- RVV VLEN=128: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` (line 1881)
- ARM NEON/SVE: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (line 2884)
- x86 AVX2: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (line 2143)
- Data structures: `vendor/llama.cpp/ggml/src/ggml-common.h` (line 352)
