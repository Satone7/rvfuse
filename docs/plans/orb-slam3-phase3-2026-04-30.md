# ORB-SLAM3 Phase 3 — 全热点向量化 + Gap 分析 + RVV 扩展提案

**Date**: 2026-04-30 | **Status**: Plan | **Batch**: orb-slam3-phase3

## Context

Phase 1+2 完成了 OpenCV 重编译（39,828 RVV 指令），GaussianBlur（25%）和 FAST corner（4.4%）已通过 OpenCV `CV_SIMD_SCALABLE` 自动向量化。g2o Eigen 6x6（16%）RVV 内核代码已编写但存在 bug、未验证、未集成。ORB descriptor（<1%）仅完成算法分析，未实现。

本阶段目标：**覆盖全部 4 个热点算子**（自动向量化优先，手动向量化补齐），完成逐算子跨平台 gap 分析，最终汇成以 RVV 指令扩展方案为重点的综合报告。

用户约束：所有任务串行执行，不设时间限制。若实现逻辑在 RVV256 与 RVV512 间相似，优先 RVV256 在 Banana Pi 验证后再扩展。

## Pre-built Artifacts (from Phase 1+2)

| Artifact | Path | Status |
|----------|------|--------|
| OpenCV RVV libs | `output/opencv/lib/` | Rebuilt, 39,828 RVV instructions |
| ORB-SLAM3 libs | `output/orb-slam3/lib/` | libORB_SLAM3.so + libg2o.so + libDBoW2.so (scalar build, need rebuild) |
| LLVM 22 | `third_party/llvm-install/` | Ready |
| QEMU + BBV plugin | `third_party/qemu/build/` | tools/bbv/libbbv.so available |
| Sysroot | `output/orb-slam3/sysroot/` | Populated |
| Eigen 3.4.0 | `applications/orb-slam3/vendor/eigen-3.4.0/` | Header-only, no RVV backend |
| g2o Eigen RVV kernel | `applications/orb-slam3/rvv-patches/eigen-6x6/` | 4 files, unverified |
| Phase 1 report | `docs/report/orb-slam3/orb-slam3-consolidated-2026-04-29.md` | Reference (Chinese) |
| Phase 2 report | `docs/report/orb-slam3/orb-slam3-phase2-2026-04-29.md` | Reference (Chinese) |
| Gap analyses | `docs/report/orb-slam3/gap-analysis/` | 3 operators done, ORB desc pending |
| Banana Pi | `192.168.100.221`, VLEN=256 | Hardware verification |

## Operator Coverage Matrix (Current)

| Operator | Hotspot % | RVV Status | RVV Instructions | Method |
|----------|-----------|------------|-----------------|--------|
| GaussianBlur | ~25% | ✅ Auto-vectorized | 17,528 | OpenCV `CPU_BASELINE=RVV` |
| g2o Eigen 6x6 | ~16% | ❌ Code written, unverified | 0 (target: 48/op) | Manual RVV backend for Eigen |
| FAST corner | ~4.4% | ✅ Auto-vectorized | 1,525 | OpenCV `CPU_BASELINE=RVV` |
| ORB descriptor | <1% | ❌ Not implemented | 0 (target: ~200) | Manual `computeOrbDescriptor` RVV |

## Task Table

| ID | Task | Model | Status | Deps |
|----|------|-------|--------|------|
| T1 | State Verification + Auto-vectorization Check | sonnet | [ ] | None |
| T2 | LLVM 22 Bug Verification (accum.dispatch.cpp) | opus | [ ] | T1 |
| T3 | g2o Eigen 6x6 RVV — Fix, Verify, Integrate | opus | [ ] | T2 |
| T4 | ORB Descriptor RVV Implementation | opus | [ ] | T2 |
| T5 | QEMU BBV Profiling (all patches applied) | sonnet | [ ] | T3, T4 |
| T6 | Per-Operator Gap Analysis | sonnet | [ ] | T3, T4 |
| T7 | Consolidated Report + RVV Extension Proposals + PDF | opus | [ ] | T5, T6 |

**Execution order**: T1 → T2 → T3 → T4 → T5 → T6 → T7 (strictly serial)

## Task Details

### T1: State Verification + Auto-vectorization Check (sonnet)

验证 Phase 1+2 所有构建产物可用，确认识别哪些算子可自动向量化、哪些必须手动。

**Phases**:
1. Verify key artifacts exist and are valid RISC-V ELF:
   - `output/opencv/lib/libopencv_imgproc.so` — confirm RVV instructions in disassembly
   - `output/opencv/lib/libopencv_features2d.so` — confirm RVV instructions
   - `output/orb-slam3/lib/libORB_SLAM3.so` — confirm scalar or RVV status
   - `output/orb-slam3/lib/libg2o.so` — confirm scalar or RVV status
2. Verify toolchain and sysroot:
   - LLVM 22 clang/clang++ at `third_party/llvm-install/bin/`
   - Sysroot at `output/orb-slam3/sysroot/` with libc, libstdc++, OpenCV, Eigen headers
3. Auto-vectorization check — rebuild ORB-SLAM3 with `-march=rv64gcv_zvl512b` (toolchain already patched):
   - Rebuild `libORB_SLAM3.so` with `build.sh --skip-sysroot --skip-opencv --skip-source`
   - Disassemble `ORBextractor.cc.o` — does clang auto-vectorize `computeOrbDescriptor` or `IC_Angle`?
   - Disassemble `Optimizer.cc.o` / `G2oTypes.cc.o` — does clang auto-vectorize any Eigen 6x6 ops?
   - Report: which functions auto-vectorized, which need manual work
4. Update operator coverage matrix with auto-vectorization findings

**Output**: Verification report + updated coverage matrix in task SKILL

**Reference**: `cross-compile-app` skill, `build.sh`, `riscv64-linux-toolchain.cmake`

### T2: LLVM 22 Bug Verification (opus)

Phase 1 报告提到 `accum.dispatch.cpp` 在 `-march=rv64gcv_zvl512b` 下编译失败（scalable-to-fixed-width error），当前 workaround 是 `#undef CV_RVV`。本任务在 QEMU 和 Banana Pi 双平台验证此问题是 LLVM 真 bug 还是配置错误。

**Phases**:
1. **Reproduce**: 用 LLVM 22 clang++ 编译 `accum.dispatch.cpp`，记录完整错误信息（含 source line、intrinsic 名称、pass name）
2. **Isolate**: 若确认编译失败，从 `accum.dispatch.cpp` 中提取最小可复现用例（单个 RVV intrinsic + 最小模板上下文）。目标：<50 行 C++ 代码触发相同错误
3. **QEMU validation**: 用最小用例在不同 VLEN（128/256/512）下测试，确认错误是否与 VLEN 相关
4. **Banana Pi validation**: 在 Banana Pi（VLEN=256, real hardware）上用 LLVM 22 编译同一用例，确认硬件行为
5. **Root cause classification**:
   - If LLVM bug: 创建最小复现脚本（compile_cmd + test case），输出到 `output/orb-slam3/bug-reports/llvm22-accum-dispatch/`
   - If config error: 修复 `-march` 或 intrinsic 用法，更新 OpenCV toolchain
   - If VLA/VLEN mismatch: 文档化约束，更新 workaround
6. **Mitigation**: 无论结果如何，输出绕过方案（更新 `#undef CV_RVV` workaround 或移除 workaround）

**QEMU validation commands**:
```bash
# Test at different VLEN
for vlen in 128 256 512; do
  qemu-riscv64 -cpu rv64,v=true,vlen=${vlen} ./minimal_test
done
```

**Banana Pi validation command**:
```bash
ssh root@192.168.100.221 "clang++ -march=rv64gcv_zvl256b -O2 minimal_test.cpp -o /tmp/test && /tmp/test"
```

**Output**: 
- `output/orb-slam3/bug-reports/llvm22-accum-dispatch/` (minimal test case + repro script + platform matrix)
- Updated OpenCV toolchain or workaround in task SKILL

**Reference**: Phase 1 consolidated report §Challenges Encountered #1, `accum.dispatch.cpp` at `vendor/opencv/modules/core/src/`

### T3: g2o Eigen 6x6 RVV — Fix, Verify, Integrate (opus)

修复现有 `eigen_rvv.inl` 的 bug，QEMU 下编译验证正确性，apply patch.diff 到 Eigen 3.4.0，重编译 ORB-SLAM3，确认 RVV 指令生成。

**Phases**:
1. **Bug fix**: 修复 `eigen_6x6_triangular_solve_rvv` 的水平归约（当前空循环，应使用 `vfredusum.vs`），修复 `test.cpp` 中的 `#include` 路径
2. **Compile test**: 用 LLVM 22 交叉编译 `test.cpp`，链接到 Eigen headers：
   ```bash
   clang++ -march=rv64gcv_zvl512b -O3 \
     -I applications/orb-slam3/vendor/eigen-3.4.0 \
     applications/orb-slam3/rvv-patches/eigen-6x6/test.cpp \
     -o output/orb-slam3/bin/eigen6x6_test
   ```
3. **QEMU correctness**: 运行 3 个测试用例（multiply, add, triangular solve），对标 Eigen scalar reference，epsilon 1e-12
4. **Apply patch**: 将 `eigen_rvv.inl` 内容集成到 Eigen 3.4.0（添加 `Eigen/src/Core/arch/RVV/PacketMath.h`），按 `patch.diff` 修改 Eigen dispatch
5. **Rebuild ORB-SLAM3**: 用 patched Eigen 重编译 `libg2o.so` 和 `libORB_SLAM3.so`，确认原先的 0 RVV 指令变为 48+/op
6. **Disassembly verification**: 反汇编确认 `vle64.v`, `vfmacc.vf`, `vse64.v` 存在于 libg2o.so 的 `BlockSolver` 相关函数中
7. **Banana Pi verification (optional)**: 用 `-march=rv64gcv_zvl256b` 重编译（LMUL=2 变体），在 Banana Pi 上运行 correctness test

**Key RVV instructions for VLEN=512**:
- `vle64.v` — column loads (6×)
- `vfmacc.vf` — scalar-vector FMA (36×)
- `vse64.v` — column stores (6×)
- Total: 48 RVV instructions for 6×6 multiply (scalar: 216 fmadd.d → 6× speedup)

**VLEN=256 note**: f64 with VLEN=256 → VLMAX=4. Need LMUL=2 to hold 6 doubles. Different register allocation from VLEN=512 (LMUL=1). Target RVV512 first, document VLEN=256 adaptation.

**Output**: Verified working `eigen_rvv.inl` + passing `test.cpp` + patched Eigen + rebuilt libs with RVV instructions confirmed

**Reference**: `eigen_rvv.inl`, `test.cpp`, `patch.diff` at `applications/orb-slam3/rvv-patches/eigen-6x6/`, g2o gap analysis

### T4: ORB Descriptor RVV Implementation (opus)

实现 `computeOrbDescriptor` 的 RVV512 向量化。ORB descriptor 使用旋转后的像素对强度比较生成 256-bit 描述子。访问模式依赖 rotate 后的坐标，支持 gather 操作。

**Phases**:
1. **Read source**: `ORBextractor.cc` lines 107-146, understand the `GET_VALUE` macro and bit-packing loop
2. **Design RVV512 approach**:
   - 4 keypoints per iteration (VLEN=512, 8-bit pixels → 64 pixels/load)
   - `vluxei8.v` × 32: gather 32 pixel pairs (64 pixels) for each keypoint via indexed load
   - `vmslt.vv`: compare pairs → 16-bit mask per keypoint
   - `vcompress`/`vmandnot.mm`: pack 16 mask bits → 1 descriptor byte
   - `vse8.v`: store 32 descriptor bytes
3. **Implement RVV256 first** (logic identical to RVV512, just VLMAX=32 vs 64):
   - Write `rvv-patches/orb-descriptor/orb_descriptor_rvv.cpp`
   - Compile with `-march=rv64gcv_zvl256b`
   - Verify correctness against scalar reference (random keypoints, 1e-6 tolerance on descriptor bits)
   - Deploy to Banana Pi for hardware verification
4. **Scale to RVV512**:
   - Adjust VL parameter (64 elements vs 32)
   - Compile with `-march=rv64gcv_zvl512b`, test under QEMU
5. **Create deliverables**:
   - `orb_descriptor_rvv.inl` — inline RVV kernel
   - `test_orb_desc.cpp` — correctness test
   - `patch.diff` — ORBextractor.cc integration
   - `README.md` — documentation

**Key RVV instructions**:
- `vluxei8.v` — indexed gather for pixel pairs (rotation-dependent positions)
- `vmslt.vv` / `vmsgt.vv` — vector mask comparison
- `vmandnot.mm` — mask bit packing
- `vse8.v` — descriptor byte store

**VLEN=256 note**: u8 at VLEN=256 → VLMAX=32. Code logic identical to RVV512 (just VLMAX changes). **Implement RVV256 first, verify on Banana Pi, then scale.**

**Output**: `applications/orb-slam3/rvv-patches/orb-descriptor/` (4 files)

**Reference**: `ORBextractor.cc` at `vendor/ORB_SLAM3/src/`, existing `README.md` at `rvv-patches/orb-descriptor/`

### T5: QEMU BBV Profiling (sonnet)

对 T3+T4 全量补丁后的 ORB-SLAM3 运行 QEMU BBV profiling，获取各热点 RVV 指令执行占比。

**Phases**:
1. **Rebuild everything**: 确保 OpenCV + patched Eigen + ORB-SLAM3 全部以 `-march=rv64gcv_zvl512b` 编译
2. **Fix vocabulary loading** (if still broken): ORBvoc.txt format issue from Phase 2; re-download or convert
3. **Run BBV profiling**: QEMU BBV plugin at 10,000 instruction intervals, TUM Freiburg1 dataset 30+ frames:
   ```bash
   qemu-riscv64 -cpu rv64,v=true,vlen=512 \
     -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/orb-slam3/bbv/orb-slam3-p3.bbv \
     -L output/orb-slam3/sysroot \
     -E LD_LIBRARY_PATH=output/orb-slam3/sysroot/usr/local/lib:output/opencv/lib:output/orb-slam3/lib \
     output/orb-slam3/bin/mono_tum <vocab> <settings> <dataset>
   ```
4. **Generate hotspot report**: `python3 tools/analyze_bbv.py --bbv <bbv-file> --elf <binary> --sysroot <sysroot>`
5. **Extract RVV instruction execution %** per hot function:
   - GaussianBlur: what % of executed BBs are RVV?
   - g2o Eigen 6x6: confirm RVV execution in BlockSolver
   - FAST corner: confirm RVV execution
   - ORB descriptor: confirm RVV execution (if T4 complete)
6. **Compare static vs dynamic**: reconcile disassembly counts with actual execution counts

**Output**: `output/orb-slam3/bbv/orb-slam3-p3.bbv` + hotspot report with RVV execution percentages

**Reference**: `qemu-bbv-usage` skill, `tools/analyze_bbv.py`, Phase 2 BBV attempt

### T6: Per-Operator Gap Analysis (sonnet)

完成/细化全部 4 个算子的跨平台 RVV 指令级 gap 分析。

**Operators**:
1. **GaussianBlur** — existing analysis at `gap-analysis/gaussian-blur.md` is thorough; review and refine
2. **FAST corner** — existing analysis at `gap-analysis/fast-corner.md`; add Banana Pi hardware measurements
3. **g2o Eigen 6x6** — update `gap-analysis/g2o-eigen-6x6.md` with T3 verification results (actual RVV instruction count, QEMU performance)
4. **ORB descriptor** — NEW: compare RVV indexed gather (`vluxei8.v`) vs NEON `vtbl`, SSE `_mm_shuffle_epi8`, SVE `ld1b_gather`

**Per operator coverage**:
1. Map RVV instruction sequence to x86 (AVX2/AVX-512), ARM (NEON/SVE), LoongArch (LSX/LASX)
2. Count instructions per element for each ISA at native vector width
3. Estimate throughput (elements/cycle)
4. Identify RVV instructions with no direct equivalent → **extension proposal candidates**
5. Identify RVV instructions that are best-in-class (no extension needed)

**Expected extension candidates** (from Phase 2):
- `vdotprod` — integer dot-product for convolution accumulate (GaussianBlur)
- Combined shift-narrow-saturate (GaussianBlur)
- Mask-to-index conversion (FAST corner)
- SAD (sum of absolute differences) for corner score (FAST corner)
- Indexed store with mask (ORB descriptor)

**Output**: Updated `docs/report/orb-slam3/gap-analysis/<operator>.md` × 4

**Reference**: `rvv-gap-analysis` skill

### T7: Consolidated Report + RVV Extension Proposals + PDF (opus)

汇总 T5（BBV 数据）和 T6（gap 分析），形式化 RVV 指令扩展方案，生成中文综合报告 + PDF。

**Phases**:
1. **Synthesize findings**: per-operator vectorization status, BBV-weighted coverage, static vs dynamic RVV execution
2. **Operator coverage matrix** (updated with T1-T6 results)
3. **RVV extension proposals** (user's focus — formal and actionable):
   - Each proposal: instruction name, semantics, operand types, motivating operator, BBV-weighted benefit estimate
   - Cross-reference with x86/ARM/LoongArch equivalents
   - Priority ranking by benefit × coverage
4. **Cross-platform summary table**: RVV512 vs AVX2/AVX-512/NEON/SVE/LSX/LASX per operator
5. **LLVM 22 bug report summary** (from T2)
6. **Generate Chinese PDF** via `rvfuse-md2pdf` skill
7. **Cross-reference** with previous reports (Phase 1 consolidated, Phase 2)

**Output**: `docs/report/orb-slam3/orb-slam3-phase3-2026-04-30.md` + `.pdf`

## Execution Strategy

**Serial only** (machine constraint). The LLVM bug verification (T2) gates all subsequent C++ compilation work — if it's a real LLVM bug that blocks `-march=rv64gcv_zvl512b`, T3/T4 may need to target `zvl256b` exclusively.

**Worktree isolation**: Each worker teammate gets a `git worktree add --detach /tmp/worktrees/orb-slam3-p3-<task> main`. Teammates reuse main repo toolchain (LLVM 22 at `third_party/llvm-install/`, QEMU at `third_party/qemu/build/`, sysroot at `output/orb-slam3/sysroot/`). Do NOT rebuild toolchain from scratch.

**Guardian**: Deploy after T1 verification. Monitor serial pipeline, auto-approve permission prompts, wake Lead on stall.

## Acceptance Criteria

### Per-Teammate

- [ ] T1: All artifacts verified; auto-vectorization status documented for each hot function
- [ ] T2: LLVM bug confirmed or ruled out; minimal test case + repro script if bug; mitigation in place
- [ ] T3: test.cpp passes all 3 cases under QEMU (VLEN=512); libg2o.so contains RVV instructions; eigen_rvv.inl bug-free
- [ ] T4: ORB descriptor correctness verified against scalar (bit-exact); RVV256 passes on Banana Pi; RVV512 passes under QEMU
- [ ] T5: Non-empty .bbv file (≥ 1MB); hotspot report with per-function RVV execution %
- [ ] T6: 4 gap analysis files; each covers ≥ 4 ISA platforms; extension candidates identified
- [ ] T7: Chinese report covers all operators; ≥ 3 formal extension proposals; PDF generated

### Cross-Task

- [ ] All 4 hotspot operators have non-zero RVV coverage
- [ ] BBV data validates static disassembly counts
- [ ] LLVM bug resolved or documented with clear workaround
- [ ] Banana Pi hardware verification for at least RVV256-compatible operators
- [ ] No placeholder files, no "TODO-only" content
- [ ] All RVV patches follow convention: `.inl` + `test.cpp` + `patch.diff` + `README.md`

## References

- Phase 1 plan: `docs/plans/orb-slam3-2026-04-29.md`
- Phase 2 plan: `docs/plans/orb-slam3-phase2-2026-04-29.md`
- Phase 1+2 consolidated report: `docs/report/orb-slam3/orb-slam3-consolidated-2026-04-29.md`
- Phase 2 report: `docs/report/orb-slam3/orb-slam3-phase2-2026-04-29.md`
- Gap analyses: `docs/report/orb-slam3/gap-analysis/*.md`
- RVV patches: `applications/orb-slam3/rvv-patches/*/`
- Perf data: `output/perf/orb-slam3/`
- BBV data: `output/orb-slam3/bbv/`
- Build script: `applications/orb-slam3/build.sh`
- Toolchain: `applications/opencv/riscv64-linux-toolchain.cmake`, `applications/orb-slam3/riscv64-linux-toolchain.cmake`
