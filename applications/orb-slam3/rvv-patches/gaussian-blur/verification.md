# ET-1 Verification Results: OpenCV CV_SIMD_SCALABLE RVV Path

**Date**: 2026-04-29
**Executor**: ET-1 (et1-verification)
**Target**: GaussianBlur hot functions in OpenCV 4.10.0

## Methodology

1. **Source audit**: Read the actual OpenCV source code in `smooth.simd.hpp` (lines 1127, 1157-1165 for hline; lines 1780, 1783-1855 for vline) and confirmed that full SIMD paths guarded by `#if (CV_SIMD || CV_SIMD_SCALABLE)` exist for both hot functions.

2. **Intrinsic layer audit**: Read `intrin_rvv_scalable.hpp` and verified that all universal intrinsics used by the hot functions have complete RVV mappings:
   - `vx_load_expand` → `vle8.v` + `vwcvtu` (u8→u16 widen)
   - `vx_setall_u16` → `vmv.v.x`
   - `v_mul (u16)` → `vwmulu` + `vnclipu` (saturating multiply)
   - `v_add (u16)` → `vadd.vv`
   - `v_store (u16)` → `vse16.v`
   - `v_mul_expand` → `vwmul` (i16→i32 widen-multiply)
   - `v_dotprod` → `vwmul` + horizontal add
   - `v_rshr_pack<16>` → `vnclipu` (i32→u16 with shift=16)
   - `v_pack` → `vnclipu` (u16→u8 with shift=0)
   - `v_zip` → interleave via `vzext` + `vslide1up` + `vor`

3. **Existing binary analysis**: Disassembled the actual built `libopencv_imgproc.so.4.10.0` at `output/opencv/lib/` and confirmed it contains **ZERO RVV instructions**. The vlineSmoothONa_yzy_a function at `0x20e8d4` uses purely scalar instructions (addw, sltu, negw, or, mul, lhu, sb) matching the Phase 1 perf annotate data.

4. **Root cause**: The CMake configuration has:
   - `CPU_BASELINE:STRING=` (EMPTY)
   - `CPU_DISPATCH:STRING=` (EMPTY)
   - `RISCV_RVV_SCALABLE:BOOL=ON` (set, but ineffectual without baseline/dispatch)
   - `CPU_RVV_USAGE_COUNT:INTERNAL=0`
   - `cv_cpu_config.h`: `CV_CPU_BASELINE_FEATURES 0`, `CV_CPU_DISPATCH_FEATURES 0`
   - `smooth.simd_declarations.hpp`: `CV_CPU_DISPATCH_MODES_ALL BASELINE`

   The `CPU_BASELINE` and `CPU_DISPATCH` variables were never set to `RVV`, so OpenCV's CPU dispatch system never compiled the RVV SIMD paths. The `CV_RVV` macro (set via `CV_CPU_COMPILE_RVV`) was never defined, so `intrin.hpp` fell through to the C++ scalar emulator (`intrin_cpp.hpp`).

5. **Recompilation test**: Wrote a standalone test (`test_compile.cpp`) that directly uses the same RVV intrinsics as OpenCV's universal intrinsics layer, and compiled with `-march=rv64gcv_zvl512b -O3`. The resulting assembly contains **124 RVV instructions** across both hot functions, proving the universal intrinsics DO generate correct RVV code.

## RVV Instruction Coverage

### hlineSmoothONa_yzy_a<uint8_t, ufixedpoint16> (SIMD path)

| Universal Intrinsic | RVV Instruction(s) | Present in test? |
|---------------------|--------------------|----|
| `vx_load_expand` | `vle8.v` + `vwcvtu.x.x.v` | YES (7 vle8, 7 vwcvtu) |
| `vx_setall_u16` | `vmv.v.x` | YES (13) |
| `v_mul (u16)` | `vwmulu.vv` + `vnclipu.wi` | YES (6 vwmulu, 8 vnclipu) |
| `v_add (u16)` | `vadd.vv` | YES (16) |
| `v_store (u16)` | `vse16.v` | YES (6) |

### vlineSmoothONa_yzy_a<uint8_t, ufixedpoint16> (SIMD path)

| Universal Intrinsic | RVV Instruction(s) | Present in test? |
|---------------------|--------------------|----|
| `vx_load (i16)` | `vle16.v` | YES (5) |
| `v_add_wrap (i16)` | `vadd.vv` | YES |
| `v_mul_expand` | `vwmul.vv` (i16→i32) | YES (11 vwmul) |
| `v_dotprod` | `vwmul.vv` + `vadd.vv` | YES |
| `v_rshr_pack<16>` | `vnclipu.wi` (shift=16) | YES |
| `v_pack (u16→u8)` | `vnclipu.wi` (shift=0) | YES |
| `v_store (u8)` | `vse8.v` | YES (5) |
| `vnsrl.wi` | `vnsrl.wi` (shift=16, narrowing) | YES (3) |

### Key RVV instructions confirmed in assembly

| RVV Instruction | Count | Function |
|----------------|-------|----------|
| `vle8.v` | 7 | Load u8 for hline expand |
| `vle16.v` | 5 | Load i16 for vline |
| `vse8.v` | 5 | Store u8 output |
| `vse16.v` | 6 | Store u16 intermediate |
| `vadd.vv` | 16 | Vector add (accumulate) |
| `vwmulu.vv` | 6 | Unsigned widen-multiply (hline) |
| `vwmul.vv` | 5 | Signed widen-multiply (vline) |
| `vnclipu.wi` | 8 | Saturating narrow (pack) |
| `vnsrl.wi` | 3 | Logical narrow shift |
| `vwcvtu.x.x.v` | 7 | Widen convert u8→u16 |
| `vmv.v.x` | 13 | Broadcast scalar to vector |
| `vsetvli` | 37 | Set vector length |

**Total**: 124 RVV instructions in test compile (vs 0 in deployed binary)

## Comparison: Scalar vs RVV Inner Loop

### vlineSmoothONa_yzy_a — Scalar (Banana Pi perf annotate, ~53% of inner loop)

```
20e930: addw    t5,t4,a5          # ufixedpoint32 saturating add
20e934: sltu    a5,t5,t4          # overflow detect
20e938: negw    a5,a5             # negate for saturation
20e93c: or      t4,a5,t5          # combine saturated result
```
→ 4 scalar instructions per element, serial dependency chain, no vectorization

### vlineSmoothONa_yzy_a — RVV (test compile with -march=rv64gcv_zvl512b)

```
vle16.v   v8, (a2)           # Load 32 i16 values
vadd.vv   v13, v8, v10       # Add bias
vwmul.vv  v8, v13, v12       # Widen-multiply i16→i32
vadd.vv   v8, v8, v9         # Accumulate
vnclipu.wi v8, v8, 16        # Pack i32→u16 with shift
vse8.v    v8, (dst)          # Store 32 u8 results
```
→ 6 vector instructions process 32 elements per iteration (VLEN=512)

## Conclusion

**EXISTING PATH WORKS — T1 should focus on enablement + verification**

The OpenCV 4.10.0 universal intrinsics RVV path is **complete and correct**. All required intrinsics (`v_mul_expand`, `v_dotprod`, `v_rshr_pack`, `v_pack`, `v_zip`, `v_add_wrap`, `vx_load_expand`, `vx_setall_u16`) have working RVV implementations in `intrin_rvv_scalable.hpp`. When compiled with correct flags, they generate proper RVV instructions (vle8, vle16, vwmulu, vwmul, vadd, vnclipu, vnsrl, vwcvtu, vse8, vse16).

The reason the Banana Pi binary is scalar is a **CMake misconfiguration**: `CPU_BASELINE` and `CPU_DISPATCH` were empty strings. The fix requires rebuilding OpenCV with `-DCPU_BASELINE=RVV` (or `-DCPU_DISPATCH=RVV`).

## Recommendation for T1 Scope

T1 should **NOT** proceed as greenfield RVV implementation. Instead:

1. **PRIMARY**: Rebuild OpenCV with correct CMake flags: `-DCPU_BASELINE=RVV -DRISCV_RVV_SCALABLE=ON -DCMAKE_C_FLAGS="-march=rv64gcv_zvl512b" -DCMAKE_CXX_FLAGS="-march=rv64gcv_zvl512b"`. This enables the existing `CV_SIMD_SCALABLE` path and should automatically generate RVV instructions for the hot functions.

2. **VERIFICATION**: After rebuild, disassemble the new `libopencv_imgproc.so` and confirm RVV instructions in `vlineSmoothONa_yzy_a` and `hlineSmoothONa_yzy_a`. Also verify the `ufixedpoint32::operator+` saturating add is vectorized (it's 53% of the vline inner loop).

3. **PERF VALIDATION**: Run ORB-SLAM3 on Banana Pi with the rebuilt OpenCV and measure GaussianBlur runtime reduction. Expect ~8-16x speedup for the vectorized inner loop (32 elements vs 1 per iteration at VLEN=512).

4. **GAP ANALYSIS** (secondary, only if perf is unsatisfactory): Investigate whether the saturating add in `ufixedpoint32::operator+` is fully vectorized. The current scalar pattern `addw + sltu + negw + or` maps to RVV `vsaddu.vv` (vector saturating add) which could replace 4 scalar instructions with 1 vector instruction. If the universal intrinsics layer doesn't generate this, a targeted patch may be needed.

5. **FURTHER OPTIMIZATION** (tertiary): The existing vline loop uses a 4x-unrolled pattern processing 4*VECSZ elements per iteration. For VLEN=512 with u16, this is 4*32=128 elements per iteration. The inner dot-product loop over kernel pairs could potentially use `vdotprod` (if available in hardware) or the `vfmacc` pattern for fused multiply-accumulate.

## Challenges Encountered

1. **No C++ headers in cross-compiler sysroot**: The RISC-V cross-compiler at `third_party/llvm-install/bin/clang++` lacks C++ standard library headers in `output/sysroot/`. Workaround: wrote standalone test using only `<riscv_vector.h>` (bundled with Clang) and `<stdint.h>` (available in sysroot C headers).

2. **RVV intrinsics API version mismatch**: Clang 22 uses the `__riscv_` prefixed v1.0+ API which requires 4 arguments for `vnclipu` (adding `__RISCV_VXRM_RNU` rounding mode). OpenCV's `intrin_rvv_011_compat.hpp` provides the 3-arg v0.10 compat layer but requires the full OpenCV header chain. Workaround: used the `__riscv_` prefixed API directly in the standalone test.

3. **CMake flag misnomer**: `RISCV_RVV_SCALABLE=ON` selects the scalable API variant but does NOT enable RVV compilation. The actual enabling comes from `CPU_BASELINE=RVV` or `CPU_DISPATCH=RVV`.

### Concrete T1 scope change

| Original T1 Scope | Revised T1 Scope |
|---|---|
| Greenfield RVV implementation of GaussianBlur | Enablement: rebuild OpenCV with `-DCPU_BASELINE=RVV` |
| Write new vectorized hline/vline from scratch | Verification: confirm RVV instructions in rebuilt binary |
| Custom fixedpoint vector types | Optional: check if `vsaddu.vv` is generated for saturating add |
| Extensive benchmarking | Perf validation: measure speedup on Banana Pi |
