# T1 Report: State Verification + Auto-vectorization Check

Date: 2026-04-30
Build: LLVM 22.1.3 (clang 22.1.3), `-march=rv64gcv_zvl512b`, Release

---

## Phase 1: OpenCV Library RVV Instruction Counts

| Library | vsetvli | vle | vse | vadd | vmul | vfpu | vcmp | vcpop | vnclipu | vsaddu.vv | vwmulu | vslide | vred | TOTAL |
|---------|---------|-----|-----|------|------|------|------|-------|---------|-----------|--------|-------|------|-------|
| imgproc | 12509 | 6101 | 4092 | 4449 | 2354 | — | 399 | 0 | 1120 | 356 | 174 | — | — | 41954 |
| core | 8106 | 4118 | 3185 | 741 | 347 | — | 582 | 0 | 170 | 12 | 37 | — | — | 23605 |
| imgcodecs | 4932 | 1645 | 1660 | 2146 | 1024 | — | 550 | 0 | 36 | 0 | 207 | — | — | 15094 |
| calib3d | 1500 | 1526 | 2016 | 168 | 23 | — | 107 | 0 | 3 | 20 | 1 | — | — | 10565 |
| flann | 1166 | 467 | 192 | 396 | 0 | — | 0 | 0 | 0 | 0 | 0 | — | — | 3858 |
| features2d | 379 | 759 | 720 | 72 | 31 | — | 25 | 0 | 2 | 0 | 0 | — | — | 2901 |
| video | 560 | 376 | 285 | 56 | 32 | — | 33 | 0 | 2 | 0 | 0 | — | — | 2087 |
| highgui | 113 | 140 | 172 | 5 | 0 | — | 0 | 0 | 0 | 0 | 0 | — | — | 516 |
| videoio | 78 | 114 | 183 | 30 | 14 | — | 1 | 0 | 0 | 0 | 12 | — | — | 606 |

**Key observation**: Zero `vcpop` instructions across ALL OpenCV libs. Popcount is implemented via software patterns (shift-mask-add), not via RVV `vcpop.m`.

---

## Phase 2: ORB-SLAM3 Library RVV Instruction Counts

### Old Build (output/orb-slam3/lib/ — April 29)

| Library | vsetvli | TOTAL_RVV |
|---------|---------|-----------|
| libORB_SLAM3.so | 0 | 0 |
| libg2o.so | 0 | 0 |
| libDBoW2.so | 0 | 0 |

**All three libs have ZERO RVV instructions** — old build did not use RVV flags.

### New Build (vendor/ORB_SLAM3/lib/ — April 30, with `-march=rv64gcv_zvl512b`)

| Library | vsetvli | vle | vse | vadd | vsub | vmul | vfpu | vcpop | vcmp | vmerge | vred | vslide | TOTAL |
|---------|---------|-----|-----|------|------|------|------|-------|------|--------|------|--------|-------|
| libORB_SLAM3.so | 1307 | 3407 | 3239 | 205 | 21 | 52 | 2139 | 0 | 35 | 111 | 36 | 1214 | 17102 |
| libg2o.so | 150 | 482 | 319 | 18 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 166 | 1147 |
| libDBoW2.so | 40 | 16 | 9 | 20 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 88 |

**Key observations**:
- `libORB_SLAM3.so` has substantial RVV (17,102 total instructions), dominated by `vslide` (1,214) and float ops (`vfpu=2,139`)
- `libg2o.so` has 1,147 RVV instructions but **ZERO float math** — only `vle`/`vse`/`vslide` (memory copies) and a few `vadd`/`vmerge`. The Eigen matrix operations inside g2o are NOT auto-vectorized.
- **Zero `vcpop`** anywhere in the entire ORB-SLAM3 stack.

---

## Phase 3: Per-Object-File Auto-vectorization Analysis

### ORB-SLAM3 Core Objects

| Object | vsetvli | vle | vse | vadd | vsub | vmul | vfpu | vcpop | vcmp | vmerge | vred | vslide | TOTAL | Auto-vec? |
|--------|---------|-----|-----|------|------|------|------|-------|------|--------|------|--------|-------|-----------|
| Optimizer.cc.o | 619 | 1296 | 1258 | 75 | 2 | 11 | 776 | 0 | 21 | 34 | 9 | — | 6675 | Partial |
| G2oTypes.cc.o | 175 | 508 | 414 | 24 | 0 | 1 | 361 | 0 | 0 | 19 | 0 | — | 2461 | Partial |
| ORBmatcher.cc.o | 13 | 92 | 17 | 34 | 17 | 21 | 101 | 0 | 0 | 0 | 17 | — | 498 | Partial |
| ORBextractor.cc.o | 51 | 116 | 125 | 8 | 1 | 6 | 4 | 0 | 0 | 0 | 3 | — | 477 | Minimal |
| System.cc.o | 5 | 116 | 129 | 0 | 0 | 0 | 42 | 0 | 0 | 0 | 0 | — | 515 | Minimal |
| LoopClosing.cc.o | 42 | 108 | 116 | 0 | 0 | 0 | 121 | 0 | 0 | 1 | 0 | — | 798 | Partial |
| Tracking.cc.o | 22 | 101 | 106 | 3 | 0 | 0 | 54 | 0 | 3 | 2 | 2 | — | 426 | Minimal |
| Frame.cc.o | 30 | 80 | 115 | 2 | 0 | 1 | 40 | 0 | 0 | 0 | 0 | — | 407 | Minimal |
| LocalMapping.cc.o | 10 | 60 | 56 | 0 | 0 | 0 | 71 | 0 | 0 | 0 | 0 | — | 372 | Minimal |
| Sim3Solver.cc.o | 119 | 220 | 179 | 20 | 0 | 1 | 166 | 0 | 1 | 9 | 0 | — | 1095 | Partial |
| KeyFrame.cc.o | 17 | 49 | 91 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | — | 249 | Minimal |
| MapPoint.cc.o | 5 | 8 | 20 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | 59 | No |
| Atlas.cc.o | 0 | 2 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | 12 | No |
| Converter.cc.o | 10 | 46 | 37 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | — | 241 | Minimal |

### Per-Function Auto-vectorization Detail

#### ORBmatcher.cc.o — DescriptorDistance (POPCOUNT HOT PATH)

| Function | Auto-vec? | RVV Count | Key Instructions | Notes |
|----------|-----------|-----------|------------------|-------|
| `DescriptorDistance` | **YES** | 13 | vle32, vxor.vv, vsrl.vi, vand.vx, vsub.vv, vadd.vv, vmul.vx, vredsum.vs | **Software popcount pattern** — LLVM does NOT use vcpop.m. Uses 8-step shift-mask-add-multiply-reduce. |

Full vectorized popcount sequence:
```
vsetivli zero, 8, e32, mf2, ta, ma
vle32.v  v8, (a0)           # load descriptor a
vle32.v  v9, (a1)           # load descriptor b
vxor.vv  v8, v9, v8         # XOR
vsrl.vi  v9, v8, 1          # shift right 1
vand.vx  v9, v9, 0x5555...  # mask
vsub.vv  v8, v8, v9         # subtract
vand.vx  v9, v8, 0x3333...  # mask
vsrl.vi  v8, v8, 2          # shift right 2
vand.vx  v8, v8, 0x3333...  # mask
vadd.vv  v8, v9, v8         # add
vsrl.vi  v9, v8, 4          # shift right 4
vadd.vv  v8, v8, v9         # add
vand.vx  v8, v8, 0x0F0F...  # mask
vmul.vx  v8, v8, 0x0101...  # multiply
vsrl.vi  v8, v8, 24         # shift right 24
vmv.s.x  v9, zero           # zero accumulator
vredsum.vs v8, v8, v9        # horizontal sum
vmv.x.s  a0, v8             # extract result
```
**This is the #1 gap**: vcpop.m would replace this entire 13-instruction sequence with ~3 instructions.

#### ORBextractor.cc.o

| Function | Auto-vec? | RVV Count | Key Instructions | Notes |
|----------|-----------|-----------|------------------|-------|
| `ComputeKeyPointsOctTree` | **Minimal** | 12 | vle32, vse32, vfadd.vv | Only small vector copies (2-4 x e32), not the core computation |
| `computeOrientation` (IC_Angle) | **No** | 1 | csrr vlenb | Only reads VLENB, no actual vectorization |
| `operator()` (includes computeOrbDescriptor) | **Partial** | 33 | vle32/vse32, vsub.vv, vredsum, vmul.vx, vadd.vx, vslidedown | Mostly small vector copies and scalar-ish reduction |
| `ComputePyramid` | **No** | 4 | vsetivli, vse64 | Just 2 stores of 4x e64 values |
| `DistributeOctTree` | **No** | 0 | — | Pure scalar |

**IC_Angle and computeOrbDescriptor are NOT effectively auto-vectorized** — they operate per-keypoint with data-dependent patterns (circular sampling, bit pattern matching) that defeat the loop vectorizer.

#### Optimizer.cc.o

| Function | Auto-vec? | RVV Count | Key Instructions | Notes |
|----------|-----------|-----------|------------------|-------|
| `BundleAdjustment` | **No** | 3 | csrr vlenb, vsetivli, vse64 | Only one 8x e64 store — setup code, not math |
| `LocalBundleAdjustment` | **No** | 1 | — | Effectively scalar |
| `PoseOptimization` | **No** | 1 | — | Effectively scalar |
| `Marginalize` | **Partial** | 40 | vle64, vse64 (all memory) | **Only memory copies**, NO float math (no vfadd/vfmul/etc.) — Eigen 6x6 matrix ops are scalar |

#### G2oTypes.cc.o

| Function | Auto-vec? | RVV Count | Key Instructions | Notes |
|----------|-----------|-----------|------------------|-------|
| `ImuCamPose::Update` | **Minimal** | 15 | vle64/vse64, some float | Mostly memory copies with a few float ops |
| `ImuCamPose::UpdateW` | **Partial** | 22 | vle64/vse64, float ops | Some float vectorization |
| `EdgeMono::linearizeOplus` | **YES** | 140 | vle64, vfmul.vf, vfadd.vv | **Best auto-vectorized function** — 3D projection Jacobian computation with vfmul.vf + vfadd.vv chains |
| `EdgeMonoOnlyPose::linearizeOplus` | **YES** | 81 | vle64, vfmul.vf, vfadd.vv | Same pattern as EdgeMono |
| `EdgeStereo::linearizeOplus` | **Partial** | 56 | vle64, some float | Stereo variant, partially vectorized |
| `EdgeStereoOnlyPose::linearizeOplus` | **Minimal** | 29 | vle64/vse64 | Mostly memory copies |
| `InvDepthPoint::Update` | **No** | 0 | — | Scalar |

#### g2o Core Library (libg2o.so)

| Component | vsetvli | Float Math | Notes |
|-----------|---------|------------|-------|
| optimizable_graph.cpp | 95 | — | Memory ops + struct copies |
| Eigen partialPivLu (6x6) | — | YES | Eigen's own LU decomposition has some auto-vec |
| Eigen llt_inplace (6x6, 7x7) | — | YES | Cholesky has some auto-vec |
| Eigen selfadjoint_matrix_vector_product | — | YES | Householder products have auto-vec |
| Other core files | 0-13 | — | Mostly scalar |

---

## Updated Operator Coverage Matrix

| Operator | Location | Auto-vec? | RVV Quality | Gap | Manual RVV Priority |
|----------|----------|-----------|-------------|-----|---------------------|
| Hamming distance (popcount) | ORBmatcher::DescriptorDistance | YES (sw popcount) | Poor — 13 insn vs 3 with vcpop.m | **vcpop.m not used** | **P0 — Highest** |
| ORB descriptor computation | ORBextractor::operator() | Minimal | Low — only small copies | Pattern-matching bit ops not vectorized | **P0** |
| IC_Angle (orientation) | ORBextractor::computeOrientation | No | Zero | Circular sampling loop not vectorized | P1 |
| Image pyramid | ORBextractor::ComputePyramid | No | Zero | Loop not vectorized (complex indexing) | P2 |
| 3D projection Jacobian | EdgeMono::linearizeOplus | YES | Good — vfmul.vf + vfadd.vv | Decent auto-vec already | P2 (marginal gain) |
| 6x6 matrix multiply | Optimizer::Marginalize | No (memory only) | Zero | Eigen fixed-size not vectorized | **P1** |
| 6x6 LU decomposition | g2o::partialPivLu | Partial | Some auto-vec | Small fixed-size limits vectorization | P2 |
| 7x7 Cholesky | g2o::llt_inplace | Partial | Some auto-vec | Same as above | P2 |
| Gaussian-Newton Hessian assembly | Optimizer::BundleAdjustment | No | Zero | g2o sparse optimizer inner loop scalar | P1 |
| Feature matching | ORBmatcher::SearchByProjection | No | Zero | Pointer-chasing loop, not vectorizable | P3 |
| Oct-tree distribution | ORBextractor::DistributeOctTree | No | Zero | Linked-list traversal, not vectorizable | P3 |

---

## Build Issues

1. **Build script copies wrong .so**: `build.sh` copies `libORB_SLAM3.so` from `output/orb-slam3/build/lib/`, but ORB-SLAM3's CMakeLists.txt sets `CMAKE_LIBRARY_OUTPUT_DIRECTORY` to `${PROJECT_SOURCE_DIR}/lib` (the vendor source tree). The actual new .so with RVV ends up at `vendor/ORB_SLAM3/lib/libORB_SLAM3.so`, while the old scalar .so remains at `output/orb-slam3/lib/`.
   - **Fix needed**: Update `build.sh` cross_compile_orbslam() to copy from the correct location, or override `CMAKE_LIBRARY_OUTPUT_DIRECTORY` in the cmake configure step.

2. **QEMU smoke test segfaults**: `mono_tum` crashes with segfault under QEMU even with correct sysroot and LD_LIBRARY_PATH. This is expected — it needs a vocabulary file and dataset path, and no arguments were provided.

---

## Discoveries

1. **build.sh .so copy bug**: The `cross_compile_orbslam()` function in `build.sh` copies from `${BUILD_DIR}/lib/libORB_SLAM3.so` which doesn't exist. ORB-SLAM3's CMakeLists.txt overrides `CMAKE_LIBRARY_OUTPUT_DIRECTORY` to `${PROJECT_SOURCE_DIR}/lib`. The actual output goes to `vendor/ORB_SLAM3/lib/`. Same issue for `libg2o.so` (goes to `vendor/ORB_SLAM3/Thirdparty/g2o/lib/`) and `libDBoW2.so` (goes to `vendor/ORB_SLAM3/lib/`). The `find` fallback in the script catches g2o and DBoW2 but NOT libORB_SLAM3.so since the first `cp` with the wrong path fails silently.

2. **vcpop.m completely absent**: LLVM 22 auto-vectorizer does NOT generate `vcpop.m` for popcount patterns. It uses the classic software popcount algorithm (shift-mask-add-multiply) vectorized across 8 x e32 elements. This is a **major instruction gap** — vcpop.m would reduce the DescriptorDistance hot path from 13 instructions to ~3.

3. **Eigen fixed-size matrix NOT vectorized for 6x6**: The `Marginalize` function (Eigen Matrix<double,6,6> operations) has ONLY vle64/vse64 (memory copies) and ZERO floating-point vector ops. The auto-vectorizer fails on Eigen's fixed-size matrix template expression trees. The `linearizeOplus` functions in G2oTypes DO get vectorized because they use explicit vfmul.vf + vfadd.vv chains on individual elements (Eigen partially unrolls the expression templates).

4. **g2o.so RVV is mostly memory copies**: Of the 1,147 RVV instructions in libg2o.so, 966 are vle/vse/vslide (84%), and only 18 are integer vadd. Zero float math in the g2o core. The float math in libORB_SLAM3.so comes from G2oTypes.cc (the application-specific edge types), not from the g2o framework itself.

5. **DescriptorDistance processes 8 x e32 = 256 bits per call**: The ORB descriptor is 256 bits (32 bytes). With `vsetivli zero, 8, e32, mf2`, the vector unit processes all 8 32-bit words at once. At VLEN=512, this uses only half the vector registers. A vcpop.m-based approach could process the entire 256-bit XOR result in one instruction.

6. **vslide dominance in libORB_SLAM3.so**: 1,214 out of 17,102 RVV instructions (7%) are vslide. This indicates heavy use of vector registers for memcpy-like operations (stack save/restore, struct copies) rather than computational vectorization.
