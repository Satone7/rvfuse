# reduce-minmax-f32

RVV implementation of `MlasReduceMinimumMaximumF32Kernel` — Find minimum and maximum values in a float array.

## Status

✅ Implementation complete, pending standalone test verification

## Files

| File | Purpose |
|------|---------|
| `rvv_reduce_minmax_f32.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime MLAS compute.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void MLASCALL
MlasReduceMinimumMaximumF32Kernel(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
)
```

## Algorithm (VLEN=512, VL=16)

1. **64-wide main loop**: Use 4 vector accumulators (v_min0-3, v_max0-3)
   - Load 64 elements as 4 vectors (16 per vector)
   - Update each accumulator with `vfmin`/`vfmax`
   - Merge 4→2→1 accumulators after loop

2. **VL-wide secondary loop**: Process remaining elements in chunks of VL
   - Use `vsetvl` for tail handling

3. **Scalar tail**: Process remaining <VL elements one by one

4. **Vector reduction**: Use `vfredmin`/`vfredmax` to reduce vector to scalar

## VLEN Requirement

- **VLEN >= 128**: Uses RVV intrinsics with dynamic VL
- **VLEN < 128**: Falls back to scalar (via compile-time check)

## Performance Notes

- 4-accumulator pattern hides load latency (similar to AVX optimization)
- `vfredmin`/`vfredmax` are ordered reductions (deterministic result)
- VLEN=512 processes 64 elements per main loop iteration

## Build & Test

```bash
# Build standalone test (rv64gcv, VLEN=512)
clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -D__riscv_v_intrinsic -D__riscv_v_fixed_vlen=512 \
    --sysroot=/home/pren/wsp/rvfuse/output/cross-ort/sysroot \
    -I. test.cpp -o test -lm

# Run under QEMU
qemu-riscv64 -cpu rv64,v=true,vlen=512 \
    -L /home/pren/wsp/rvfuse/output/cross-ort/sysroot ./test

# Scalar build (for reference on x86)
g++ -std=c++17 -O2 test.cpp -o test_scalar -lm
```