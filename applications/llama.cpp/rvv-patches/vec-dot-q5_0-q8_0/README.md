# vec-dot-q5_0-q8_0

RVV implementation of `ggml_vec_dot_q5_0_q8_0` - Q5_0 x Q8_0 dot product optimized for VLEN >= 256 (target: VLEN=512).

## Files

| File | Purpose |
|------|---------|
| `rvv_vec_dot_q5_0_q8_0.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void ggml_vec_dot_q5_0_q8_0(int n, float * s, size_t bs,
                              const void * vx, size_t bx,
                              const void * vy, size_t by, int nrc);
```

## Algorithm

1. Load 16 bytes of packed nibble data from `block_q5_0.qs`
2. Unpack lower and upper nibbles into 32 int8 values using mf2 -> m1 extension
3. Load 4 bytes of `block_q5_0.qh` bitmask, invert to create sign extension mask
4. Masked subtract 0x10 where original qh bit was 0 (negative value sign extension)
5. Load 32 int8 values from `block_q8_0.qs`
6. Widening multiply: i8 x i8 -> i16 (vwmul, m1 -> m2)
7. Widening reduction: sum i16 -> i32 (vwredsum, m2 -> m1)
8. Scale by combined delta factors and accumulate across blocks

## Optimization vs Original (VLEN=512)

| Aspect | Original | This Implementation |
|--------|----------|---------------------|
| Input LMUL | m2 (128 bytes) | m1 (64 bytes) |
| Multiply LMUL | m4 (256 bytes) | m2 (128 bytes) |
| Register pressure | High (4 groups for mul) | Low (2 groups for mul) |
| vlenb branching | Yes (16 vs else) | No (single path) |
| Nibble load | m1 (wasteful for 16 bytes) | mf2 (tight fit) |

## Gap Analysis

See: `docs/report/llama.cpp/rvv-gap-analysis-vec-dot-q5_0-q8_0.md`

## Build & Test

```bash
# Build llama.cpp with this patch
./build.sh --force --test

# Run standalone test
./build.sh --test
```
