# quantize-linear

RVV implementation of `MlasQuantizeLinear` — linear quantization from float32 to int8/uint8/int16/uint16.

## Status

✅ — Implementation complete, standalone test passing

## Files

| File | Purpose |
|------|---------|
| `rvv_quantize_linear.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime MLAS |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signatures

```cpp
void MlasQuantizeLinearU8KernelRVV(const float* Input, uint8_t* Output, size_t N, float Scale, uint8_t ZeroPoint);
void MlasQuantizeLinearS8KernelRVV(const float* Input, int8_t* Output, size_t N, float Scale, int8_t ZeroPoint);
void MlasQuantizeLinearKernelRVV<OutputType>(const float* Input, OutputType* Output, size_t N, float Scale, OutputType ZeroPoint);
```

## Algorithm

Formula: `Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)`

1. Load VL float32 elements from input
2. Divide by scale: `v_scaled = v_input / Scale`
3. Clamp to valid range: `[MinValue - ZeroPoint, MaxValue - ZeroPoint]`
4. Round to nearest even: `vfcvt_x_f`
5. Add zero point
6. Saturate and narrow: int32 → int16 → int8 (or int32 → uint16 → uint8)
7. Store results

## VLEN Requirement

- **VLEN >= 128**: Uses RVV intrinsics (e32m1 configuration)
- **VLEN < 128**: Falls back to scalar

## Build & Test

```bash
# Build standalone test (rv64gcv, VLEN=256)
riscv64-unknown-linux-gnu-clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gcv_zvl256b -mabi=lp64d \
    -D__riscv_v_intrinsic \
    test.cpp -o test -lm

# Run under QEMU (VLEN=256)
qemu-riscv64 -cpu rv64,v=true,vlen=256 -L <sysroot> ./test

# Build for VLEN=512
riscv64-unknown-linux-gnu-clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -D__riscv_v_intrinsic -D__riscv_v_fixed_vlen=512 \
    test.cpp -o test -lm

# Run under QEMU (VLEN=512)
qemu-riscv64 -cpu rv64,v=true,vlen=512 -L <sysroot> ./test
```
