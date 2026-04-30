# ORB-SLAM3 Phase 2 — BBV Profiling, Gap Analysis, g2o Eigen RVV

**Date**: 2026-04-29 | **Status**: Plan | **Batch**: orb-slam3-phase2

## Context

Phase 1 confirmed: OpenCV rebuilt with RVV (39,828 RVV instructions), GaussianBlur + FAST fully vectorized.
g2o Eigen 6x6 (16% hotspot) identified as NOT auto-vectorized — needs greenfield RVV implementation.

This phase completes the deferred tasks: BBV profiling → Gap analysis → g2o Eigen RVV implementation.

## Pre-built Artifacts (from Phase 1)

| Artifact | Path | Status |
|----------|------|--------|
| OpenCV RVV libs | `output/opencv/lib/` | Rebuilt, 39,828 RVV instructions |
| ORB-SLAM3 RVV libs | `output/orb-slam3/lib/` | Rebuilt, libORB_SLAM3.so + libg2o.so + libDBoW2.so |
| LLVM 22 | `third_party/llvm-install/` | Ready |
| QEMU + BBV plugin | `third_party/qemu/build/` | tools/bbv/libbbv.so available |
| Sysroot | `output/orb-slam3/sysroot/` | Populated |
| Phase 1 report | `docs/report/orb-slam3/orb-slam3-consolidated-2026-04-29.md` | Reference |

## Task Table

| ID | Task | Model | Status | Deps |
|----|------|-------|--------|------|
| T8 | BBV Profiling (QEMU) | sonnet | [ ] | None |
| T9 | Cross-platform Gap Analysis | sonnet | [ ] | T8 |
| T10 | g2o Eigen 6x6 RVV Implementation | opus | [ ] | None (independent) |
| T11 | Consolidated Phase 2 Report + PDF | sonnet | [ ] | T8, T9, T10 |

Execution order: T8 + T10 in parallel → T9 → T11 (T10 is independent of T8/T9)

Wait — machine constraint: serial only. Order: T8 → T9 → T10 → T11

## Task Details

### T8: BBV Profiling via QEMU (sonnet)

Run QEMU BBV plugin on the rebuilt ORB-SLAM3 binary with real TUM dataset.

**Key questions to answer**:
- What % of executed basic blocks are RVV instructions in GaussianBlur, FAST, g2o?
- What is the actual RVV instruction mix (vadd/vwmul/vnclip/vsaddu vs vle/vse)?
- Confirmation that the 17,528 static RVV instructions in imgproc are actually executed

**Phases**:
1. Locate TUM test dataset (rgbd_dataset_freiburg1_xyz from Phase 0)
2. Build BBV command: `qemu-riscv64 -cpu rv64,v=true,vlen=512 -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/orb-slam3/bbv/orb-slam3.bbv -L output/orb-slam3/sysroot -E LD_LIBRARY_PATH=...`
3. Run profiling (expect 5-15 min QEMU runtime, use timeout=600000)
4. Generate hotspot report: `python3 tools/analyze_bbv.py --bbv <bbv-file> --elf <binary> --sysroot <sysroot>`
5. Extract RVV instruction mix per hot function using disassembly

**Output**: `output/orb-slam3/bbv/` with .bbv files + hotspot report

**Reference**: `qemu-bbv-usage` skill, `tools/analyze_bbv.py`

### T9: Cross-platform Gap Analysis (sonnet)

Per-operator gap analysis comparing RVV512 against x86 AVX2, ARM NEON/SVE, LoongArch LSX/LASX.

**Operators to analyze**:
1. GaussianBlur — fixed-point saturating convolution (hline+vline)
2. FAST corner — integer pixel threshold comparison
3. g2o Eigen 6x6 — dense matrix multiply/add

**Per operator**:
1. Map the RVV instruction sequence to equivalent x86/ARM/LoongArch sequences
2. Count instructions per element for each ISA
3. Estimate throughput (elements/cycle) at ISA-native vector width
4. Identify RVV instructions with no direct equivalent (gaps)
5. Estimate benefit of proposed new instructions

**Key comparison dimensions**:
- Saturating add: `vsaddu.vv` (RVV) vs `_mm_packus_epi16` (x86) vs `vqmovun` (NEON)
- Mask comparison: `vmslt.vx`/`vmsgt.vx` (RVV) vs `_mm_cmpgt_epi8` (x86) vs `vcgt` (NEON)
- Widen-multiply: `vwmulu.vv` (RVV) vs `_mm_mullo_epi16`+`_mm_mulhi_epu16` (x86)
- 6x6 matrix: `vle64.v`+`vfmacc.vv` (RVV) vs `_mm256_load_pd`+`_mm256_fmadd_pd` (AVX2)

**Output**: `docs/report/orb-slam3/gap-analysis/<operator>.md` per operator

**Reference**: `rvv-gap-analysis` skill

### T10: g2o Eigen 6x6 RVV Implementation (opus)

Implement RVV512 intrinsics for Eigen 3.4.0's 6x6 fixed-size dense matrix operations.

**Target functions** (from Phase 1 perf annotate):
- `Eigen::internal::dense_assignment_loop` — 6x6 matrix copy/add
- g2o `BlockSolver<...>` — uses Eigen 6x6 for Schur complement

**Approach**:
1. Read Eigen's 6x6 specialization (Eigen/src/Core/arch/ for NEON/SSE examples)
2. Create `rvv-patches/eigen-6x6/eigen_rvv.inl` with:
   - `eigen_6x6_load(vfloat64m1_t* dst, const double* src)` — 6 columns via `vle64.v` + strided load
   - `eigen_6x6_store(double* dst, vfloat64m1_t* src)` — 6 columns via `vse64.v`
   - `eigen_6x6_madd(vfloat64m1_t* C, vfloat64m1_t* A, vfloat64m1_t* B)` — 6x6 FMA via `vfmacc.vv`
   - `eigen_6x6_triangular_solve(...)` — back-substitution with `vfdiv`+`vfmacc`
3. Create `rvv-patches/eigen-6x6/test.cpp` — correctness against Eigen scalar
4. Create `rvv-patches/eigen-6x6/patch.diff` — integration into Eigen's dispatch
5. Expected: 6x6 multiply in ~36 FMAs (scalar: 6^3=216 FMAs, ~6x speedup)

**Output**: `applications/orb-slam3/rvv-patches/eigen-6x6/` (4 files)

**VLEN=512**: 64 bytes/vector = 1 double (64-bit) per vector. For 6 doubles, use 6 vector registers. With LMUL=1 (vfloat64m1), VLEN=512 holds exactly 1 double per register → need 6 registers for a 6x1 column.

Alternative: Use `vfloat64m8` (LMUL=8) to hold multiple columns, or use `vfloat64m1` with 6 separate loads.

### T11: Consolidated Phase 2 Report + PDF (sonnet)

Merge T8 (BBV data), T9 (gap analysis), and T10 (g2o Eigen RVV impl) into final report.

**Phases**:
1. Merge per-operator gap analyses into consolidated report
2. Add BBV-weighted priority table with execution percentages
3. Add g2o Eigen RVV implementation details and expected benefit
4. Update operator coverage matrix from Phase 1
5. Generate PDF via `md2pdf` skill

**Output**: `docs/report/orb-slam3/orb-slam3-phase2-2026-04-29.md` + `.pdf`

## Timeline Estimate

| Task | Est. Duration |
|------|--------------|
| T8 (BBV) | 15-25 min |
| T9 (Gap Analysis) | 25-40 min |
| T10 (g2o Eigen RVV) | 30-50 min |
| T11 (Report) | 10-15 min |
| **Total** | **1.3-2.2 hrs** |
