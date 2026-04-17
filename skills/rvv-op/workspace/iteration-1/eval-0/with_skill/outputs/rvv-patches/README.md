# gemv-q5_0-q8_0

RVV implementation of `ggml_gemv_q5_0_q8_0` -- matrix-vector multiplication with Q5_0 quantized weights and Q8_0 quantized activations (16x1 tile).

## Status

Planned -- requires interleaved Q5_0 block format support in llama.cpp upstream.

## Files

| File | Purpose |
|------|---------|
| `rvv_gemv_q5_0_q8_0.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp RISC-V repack.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void ggml_gemv_q5_0_q8_0(int n, float * s, size_t bs,
                          const void * vx, const void * vy,
                          int nr, int nc);
```

**Parameters:**
- `n`: Number of elements per column (must be multiple of QK8_0=32)
- `s`: Output array of FP32 values (one per column)
- `bs`: Byte stride between output elements
- `vx`: Weight matrix data (column-major, Q5_0 quantized)
- `vy`: Activation vector data (Q8_0 quantized)
- `nr`: Number of output rows (must be 1)
- `nc`: Number of output columns

## Algorithm

1. **Tile structure**: Processes 16 output columns simultaneously (16x1 tile)
2. **Per-column GEMV**: For each of the 16 columns, iterate over all blocks:
   a. Load 16 bytes of packed nibbles (`qs`) from Q5_0 weight block
   b. Load 4 bytes of high bits (`qh`) from Q5_0 weight block
   c. Unpack nibbles: extract lower 16 nibbles and upper 16 nibbles
   d. Apply 5th bit sign extension: create mask from `qh`, subtract 0x10 where bit is 0
   e. Load 32 int8 values from Q8_0 activation block
   f. Compute widening multiply-accumulate (int8 x int8 -> int16 -> int32)
3. **Reduction**: Sum int16 partials to int32, convert to float
4. **Scale**: Apply combined delta factors (weight delta x activation delta)
5. **Store**: Write 16 FP32 results to output

## Q5_0 Format

Q5_0 stores 32 quantized 5-bit signed integers per block:
- Values range from -16 to +15 (5 bits including sign)
- **qs[16]**: 16 bytes of packed 4-bit nibbles (lower 4 bits of each value)
- **qh[4]**: 32 bits, one bit per value (the 5th/significance bit)
- **d** (FP16): Scale factor

Sign extension decoding:
- If qh bit is 1: value = nibble (range 0..15)
- If qh bit is 0: value = nibble - 16 (range -16..-1)

## Q8_0 Format

Q8_0 stores 32 quantized 8-bit signed integers per block:
- Values range from -128 to +127
- **qs[32]**: 32 int8 values
- **d** (FP16): Scale factor

## VLEN Requirement

- VLEN >= 512 (vlenb >= 64): Uses RVV intrinsics with 16-element vectors
- VLEN < 512: Falls back to scalar generic implementation

## Cross-Platform Reference Implementations

| Platform | File | Status |
|----------|------|--------|
| ARM NEON | `arch/arm/quants.c:840` | `ggml_vec_dot_q5_0_q8_0` (vec_dot, not GEMV) |
| x86 AVX2 | `arch/x86/quants.c:846` | `ggml_vec_dot_q5_0_q8_0` (vec_dot, not GEMV) |
| RISC-V (existing) | `arch/riscv/rvv_vec_dot_q5_0_q8_0.inl` | `ggml_vec_dot_q5_0_q8_0_rvv` (vec_dot, not GEMV) |

Note: Q5_0 GEMV does not exist in upstream llama.cpp. Only the vec_dot variant exists.
This implementation creates a new GEMV kernel following the Q4_0 GEMV pattern (16x1 tile).

## Gap Analysis

See: `docs/report/llama.cpp/rvv-gap-analysis-gemv-q5_0-q8_0-YYYY-MM-DD.md`

## Key Differences from Q4_0 GEMV

| Aspect | Q4_0 GEMV | Q5_0 GEMV (this) |
|--------|-----------|-------------------|
| Bits per element | 4 | 5 |
| Extra bit storage | None | qh[4] (32 bits) |
| Value range | -8..+7 | -16..+15 |
| Sign extension | Shift + mask | qh bitmask + subtract |
| Interleaved format | block_q4_0x16 | Not interleaved (column-sequential) |
| Blocking | 16x1 (Zvfh) | 16x1 (Zvfh) |

## Build & Test

```bash
# Build llama.cpp with this patch
./build.sh --force --test

# Run standalone test
cd rvv-patches/gemv-q5_0-q8_0
riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b_zvfh -mabi=lp64d \
    -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
    test.cpp -o test -lm
qemu-riscv64 -L $SYSROOT ./test
```

## LLVM Bug Consideration

Due to LLVM 22 RISC-V backend optimizer bug (issue #83370), the RVV intrinsics path
uses `__riscv_v_intrinsic` guard and requires `__riscv_zvfh` for FP16 scale operations.
The test harness uses `__attribute__((optnone))` on the scalar reference to avoid
optimizer interference.