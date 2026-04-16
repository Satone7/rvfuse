# gemm-q4_K-8x4

RVV implementation of `ggml_gemm_q4_K_8x4_q8_K` - Q4_K weights × Q8_K activations GEMM with 4x4 tile blocking.

## Files

| File | Purpose |
|------|---------|
| `rvv_gemm_q4_K_8x4.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void ggml_gemm_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc);
```

## VLEN Optimization

- **VLEN >= 512**: Dual-tile 4x8+4x8 → 4x16 output (2x throughput)
- **VLEN < 512**: Single-tile 4x8 output (original)

## Gap Analysis

See: `docs/report/llama.cpp/rvv-gap-analysis-gemm-q4_K-8x4-2026-04-16.md`

## Build & Test

```bash
# Build llama.cpp with this patch
./build.sh --force --test

# Run standalone test
./build.sh --test
```