# gemv-q4_K-8x8-q8_K

RVV implementation of `ggml_gemv_q4_K_8x8_q8_K` - Q4_K weights × Q8_K activations GEMV (matrix-vector multiplication) with 8x8 interleaved tile blocking.

## Files

| File | Purpose |
|------|---------|
| `rvv_gemv_q4_K_8x8_q8_K.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp (refactor to .inl) |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc);
```

## Algorithm

1. For each block of 256 elements (QK_K):
   - Decode 6-bit scales/mins from 96-byte scales array
   - RVV uses 24-byte layout: 4 subblocks × 24 bytes = 96 bytes
   - Each 24-byte block: 12 bytes for low nibble + 12 bytes for high nibble
2. For each column (8 columns interleaved):
   - Compute dot product: q4 × q8 for each iteration
   - Apply scales: `sumi_lo × scales_lo[j] + sumi_hi × scales_hi[j]`
3. Bias correction:
   - Accumulate: `mins × bsums × dmin × d`
   - Final output: `accumulated_sum - bias_correction`

## Key Implementation Details

### Scales/Mins Decoding (6-bit)

The 96-byte scales array encodes 8 scales and 8 mins per scalar subblock using 6-bit packing:

```
sm[0-2] = 3 × uint32 from 12 bytes
scales[0-3] = sm[0] & 0x3f3f3f3f
scales[4-7] = (sm[2] & 0x0f0f0f0f) | ((sm[0] >> 6) & 0x03030303) << 4
mins[0-3]   = sm[1] & 0x3f3f3f3f
mins[4-7]   = (sm[2] >> 4 & 0x0f0f0f0f) | ((sm[1] >> 6) & 0x03030303) << 4
```

### Layout Mapping

- Scalar: 8 subblocks × 12 bytes = 96 bytes
- RVV: 4 subblocks × 24 bytes = 96 bytes
- Mapping: RVV sb covers scalar sb×2 and sb×2+1

### mins Type

- mins are **uint8_t** (6-bit unsigned, range 0-63), NOT int16_t
- This fix was critical for correct output

## VLEN Requirement

- **VLEN >= 512**: Uses the optimized scalar-based accumulation
- **VLEN < 512**: Falls back to generic scalar implementation

## Build & Test

```bash
# Build llama.cpp with this patch
./build.sh --force --test

# Run standalone test (requires RVV toolchain)
riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b -mabi=lp64d \
    -DGGML_USE_RISCV_V -o test test.cpp -lm
./test

# Run standalone test (scalar mode, for reference)
g++ -std=c++17 -O2 -o test_scalar test.cpp -lm
./test_scalar
```

## Test Results

All tests pass with relative tolerance 1e-5:

| Test | Result |
|------|--------|
| seed=42, 1 block | PASS |
| seed=123, 1 block | PASS |
| seed=456, 1 block | PASS |
| seed=42, 4 blocks | PASS |
| seed=42, 8 blocks | PASS |

Max relative error: ~1.2e-6 (within floating-point rounding tolerance)