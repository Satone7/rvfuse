---
name: rvv512-optimization-pipeline
description: |
  End-to-end RVV512 optimization pipeline for any RISC-V application:
  perf profiling on dev board → RVV512 vectorization → QEMU-BBV profiling →
  cross-platform gap analysis → PDF reports → consolidated optimization report.
  Trigger when: the user mentions "RVV512 optimization pipeline", "full optimization",
  "端到端优化", "完整优化流水线", "profile and vectorize", "热点分析到向量化",
  or wants to run the entire optimization workflow for any application on RISC-V.
  This skill orchestrates the rvv-op, qemu-bbv-usage, rvv-gap-analysis, and md2pdf
  skills in strict phase order with data dependencies enforced.
---

# RVV512 Full Optimization Pipeline

## Purpose

Orchestrate the complete RISC-V RVV512 optimization workflow for any application:

1. **Perf profiling** on real hardware (dev board) to identify hotspots
2. **RVV512 vectorization** of operators consuming >1% execution time
3. **BBV profiling** via QEMU to capture per-operator instruction-level data
4. **Gap analysis** comparing RVV against cross-platform SIMD (AVX, NEON, LSX, VSX, etc.)
5. **Consolidated report** merging all findings into a single deliverable

Each phase depends on the output of the previous phase. The pipeline enforces strict ordering:
gap analysis MUST NOT run until BBV profiling data exists for the target operator.

## Input

The user provides:
- **Application path** — e.g., `applications/onnxrt/ort`, `applications/llama.cpp`
- **Model/workload** — e.g., `yolo11n_int8.ort`, `qwen-0.5b-q4_0.gguf`
- (Optional) **VLEN** — Vector register length, default 512-bit
- (Optional) **Dev board config** — SSH host, user, password for perf profiling

## Pipeline Overview

```
Phase 1: Perf Profiling (real hardware)
    │
    ├─ perf stat   → global metrics (cycles, IPC, cache-misses)
    ├─ perf record → sampling data
    └─ perf report → function hotspots with % time
    │
    ▼ identify operators >1% execution time
Phase 2: RVV512 Vectorization (per operator)
    │
    ├─ rvv-op skill → implement .inl + test.cpp + patch.diff
    ├─ standalone test under QEMU
    └─ rebuild application with RVV512 patches
    │
    ▼ each operator has a working RVV implementation
Phase 3: BBV Profiling (QEMU + BBV plugin)  ← MUST complete before Phase 4
    │
    ├─ qemu-bbv-usage skill → per-operator function-scoped BBV
    └─ produces .bb + .disas files for each operator
    │
    ▼ .bb and .disas data available for every operator
Phase 4: Gap Analysis (cross-platform comparison)
    │
    ├─ rvv-gap-analysis skill (with BBV data) → per-operator report
    ├─ md2pdf skill → per-operator PDF
    └─ reports in docs/report/<app-name>/
    │
    ▼ individual reports complete
Phase 5: Consolidated Report
    │
    ├─ merge all operator reports
    ├─ md2pdf skill → consolidated PDF
    └─ deliver complete optimization package
```

---

## Phase 1: Perf Profiling on Dev Board

### Goal

Obtain function-level hotspot distribution for the target workload on real RISC-V hardware.

### Prerequisites

- Cross-compiled binary and shared libraries (use `cross-compile-app` skill if needed)
- SSH access to the dev board
- `perf` installed on the board with `perf_event_paranoid <= 1`

### Workflow

#### Step 1: Upload artifacts to dev board

```bash
# Upload binary, shared libs, sysroot, and model
scp -r output/<app>/bin/<runner> root@<host>:/root/<remote-dir>/
scp -r output/<app>/lib/*.so* root@<host>:/root/<remote-dir>/lib/
scp <model-file> root@<host>:/root/<remote-dir>/
```

For ONNX Runtime chroot deployments, upload `rootfs.tar.gz` and extract on the board.

#### Step 2: Run perf stat (global metrics)

```bash
ssh root@<host> "cd /root/<remote-dir> && \
  perf stat -d -- ./<runner> <model> <iterations> 2>&1 | tee perf_stat.txt"
```

Capture: cycles, instructions, IPC, cache-misses, branch-misses.

**Note**: On SpacemiT K1 (Banana Pi), use `-e cpu-clock` instead of `-e cycles` for
`perf record` because SBI PMU does not reliably support hardware cycle sampling.

#### Step 3: Run perf record (sampling)

```bash
ssh root@<host> "cd /root/<remote-dir> && \
  perf record -e cpu-clock -g -F 999 -o perf.data -- ./<runner> <model> <iterations> 2>&1"
```

#### Step 4: Run perf report (function hotspots)

```bash
ssh root@<host> "cd /root/<remote-dir> && \
  perf report --stdio -n --percent-limit 0.5 -i perf.data > perf_report.txt 2>&1"
```

#### Step 5: Download results

```bash
mkdir -p <app-path>/data/perf-<workload-name>/
scp root@<host>:/root/<remote-dir>/perf_stat.txt <app-path>/data/perf-<workload-name>/
scp root@<host>:/root/<remote-dir>/perf_report.txt <app-path>/data/perf-<workload-name>/
```

#### Step 6: Parse hotspots and create summary

Extract the function hotspot table from `perf_report.txt`. Identify all functions with
>1% self-time. Create `<app-path>/data/perf-<workload-name>/summary.md` with:

```markdown
# Perf Profiling Summary: <workload-name>

## Global Metrics
| Metric | Value |
|--------|-------|
| Total time | X seconds |
| IPC | X.XX |
| Cache miss rate | X.X% |

## Function Hotspots (>1%)
| Function | Self % | Type |
|----------|--------|------|
| func_A | XX.XX% | description |
| func_B | XX.XX% | description |
| ... | ... | ... |

## Operators to Vectorize
List of operators meeting the >1% threshold, sorted by execution share.
```

### Output

- `<app-path>/data/perf-<workload-name>/perf_stat.txt`
- `<app-path>/data/perf-<workload-name>/perf_report.txt`
- `<app-path>/data/perf-<workload-name>/summary.md`

### Decision Gate

From the summary, extract the list of operators to vectorize (all with >1% share).
Skip memory-bound operators where vectorization provides negligible benefit (e.g., memcpy,
memset, im2col). Proceed to Phase 2 with the filtered list.

---

## Phase 2: RVV512 Vectorization

### Goal

For each hotspot operator from Phase 1, implement an RVV512 vectorized version and
integrate it into the application build.

### Per-Operator Workflow

For each operator in priority order (highest % time first):

#### Step 1: Invoke rvv-op skill

```
Use the rvv-op skill to implement the <operator-name> operator.
Application path: <app-path>
Operator name: <operator-name>
VLEN: 512
```

The rvv-op skill will:
1. Discover the application's structure and cross-platform SIMD implementations
2. Create the operator package: `rvv_<operator>.inl` + `test.cpp` + `patch.diff` + `README.md`
3. Build and run the standalone test under QEMU

#### Step 2: Validate standalone test

```bash
# Compile and run test
<cross-compiler> -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -D__riscv_v_intrinsic -D__riscv_v_fixed_vlen=512 \
    -I<operator-dir> <test.cpp> -o test -lm

third_party/qemu/build/qemu-riscv64 -L <sysroot> -cpu max,vlen=512 ./test
```

All tests must PASS before proceeding.

#### Step 3: Rebuild application with RVV512 patches

After all operators have standalone tests passing, rebuild the application once
with all patches integrated:

```bash
cd <app-path> && ./build.sh --toolchain <zvl512b-toolchain> --force
```

### zvl512b Toolchain

If the application does not already have a zvl512b toolchain file, create one:

```bash
# Copy existing toolchain and modify march
cp <app-path>/riscv64-linux-toolchain.cmake \
   <app-path>/riscv64-linux-zvl512b-toolchain.cmake

# Edit: change zvl256b → zvl512b, add -D__riscv_v_fixed_vlen=512
```

### Output

- `<app-path>/rvv-patches/<operator>/rvv_<operator>.inl` (for each operator)
- `<app-path>/rvv-patches/<operator>/test.cpp`
- `<app-path>/rvv-patches/<operator>/patch.diff`
- Rebuilt application binary with RVV512 patches

### Decision Gate

All operator standalone tests must pass. The application must build successfully
with all patches integrated. Only then proceed to Phase 3.

---

## Phase 3: BBV Profiling via QEMU

### Goal

For each RVV512 operator, obtain function-scoped BBV profiling data that captures
instruction-level execution frequencies.

### Critical Dependency

**Phase 3 MUST complete for all operators before Phase 4 (gap analysis) begins.**

The rvv-gap-analysis skill requires BBV `.bb` and `.disas` files to compute
hotspot-weighted benefit analysis. Without BBV data, gap analysis can only report
per-BB instruction reduction — not overall benefit. This severely limits the
report's usefulness.

### Per-Operator Workflow

#### Step 1: Get function offset

```bash
nm -D -S <output-lib> | grep <function_name>
# Output: 0000000000XXXXXX 0000000000000YYY T <function_name>
#         ^offset                         ^size
```

#### Step 2: Run QEMU with BBV plugin (function-scoped)

```bash
third_party/qemu/build/qemu-riscv64 \
  -L <sysroot> \
  -E LD_LIBRARY_PATH=<lib-dir> \
  -cpu max,vlen=512 \
  -plugin tools/bbv/libbbv.so,lib_name=<lib_name>,func_offset=<offset>,func_size=<size>,interval=1000,outfile=<output-dir>/<operator> \
  -- <runner> <model> <iterations>
```

**Parameter reference**:
- `lib_name`: Library name without `.so` suffix (e.g., `libonnxruntime`, `libggml-cpu`)
- `func_offset`: Hex offset from `nm -D -S` output
- `func_size`: Hex size from `nm -D -S` output
- `interval`: 1000 for function-scoped profiling (fine granularity)
- `outfile`: Output prefix, produces `<outfile>.0.bb` and `<outfile>.disas`

#### Step 3: Verify BBV output

```bash
# Check that BB data was captured
wc -l <output-dir>/<operator>.0.bb
# Should have multiple lines (BB execution frequency data)

# Check disassembly
head -20 <output-dir>/<operator>.disas
# Should show function-scoped disassembly with instruction counts
```

If output is empty (0 BBs):
- Verify the function name matches `nm` output (may differ after rebuild)
- Ensure the program actually exercises the operator's code path
- Check QEMU VLEN matches the binary's compile-time VLEN

### Output

- `<output-dir>/<operator>.0.bb` — BB execution frequencies
- `<output-dir>/<operator>.disas` — BB disassembly with instruction counts

### Decision Gate

Every target operator must have non-empty `.bb` and `.disas` files.
If any operator has empty output, debug and fix before proceeding to Phase 4.

---

## Phase 4: Gap Analysis

### Goal

For each RVV512 operator, produce a cross-platform gap analysis report that compares
the RVV implementation against other SIMD architectures and proposes new instructions.

### Critical Dependency

**Phase 4 REQUIRES BBV profiling data from Phase 3.**

Each gap analysis invocation must receive:
- The RVV implementation file (`.inl`)
- The BBV data directory containing `.bb` and `.disas` files
- The VLEN and SEW parameters

Without BBV data, the report can only express benefits as BB-scoped percentages
("BB内减少X%"), which cannot be translated to overall benefit. **Do NOT proceed
with gap analysis until BBV data is confirmed available.**

### Per-Operator Workflow

#### Step 1: Invoke rvv-gap-analysis skill

```
Use the rvv-gap-analysis skill to analyze the <operator-name> operator.
RVV source: <app-path>/rvv-patches/<operator>/rvv_<operator>.inl
BBV data: <output-dir>/  (contains <operator>.0.bb and <operator>.disas)
VLEN: 512
SEW: 32
```

The skill will:
1. Parse the RVV implementation
2. Launch parallel platform analysis subagents (x86 AVX, ARM NEON, LoongArch LSX, Power VSX, S390X, WASM SIMD)
3. Integrate BBV profiling data for hotspot-weighted benefit analysis
4. Run review-fix cycles (minimum 2 rounds)
5. Output a Markdown report with priority-ordered extension proposals

#### Step 2: Generate PDF from gap analysis report

Invoke the project's md2pdf skill:

```bash
python3 .claude/skills/md2pdf/scripts/md2pdf.py \
  <report-md-path> \
  <report-pdf-path> \
  --title "<operator> Cross-Platform Gap Analysis"
```

Where:
- `<report-md-path>`: `docs/report/<app-name>/rvv-gap-analysis-<operator>-YYYY-MM-DD.md`
- `<report-pdf-path>`: `docs/report/<app-name>/pdf/rvv-gap-analysis-<operator>-YYYY-MM-DD.pdf`

#### Step 3: Verify report quality

Each report must contain:
- Summary table with priority-ordered extensions (P0, P1, P2, ...)
- Overall benefit figures (整体收益) derived from BBV data using:
  `整体收益 = BB指令减少数 / BB总指令数 × BB执行占比`
- Per-platform analysis sections
- Review log documenting the review-fix cycle

### Output

Per operator:
- `docs/report/<app-name>/rvv-gap-analysis-<operator>-YYYY-MM-DD.md`
- `docs/report/<app-name>/pdf/rvv-gap-analysis-<operator>-YYYY-MM-DD.pdf`

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| PDF parse error with nested HTML | Asterisks inside inline code blocks parsed as italic | Replace `*` with `×` in inline code |
| "无BBV profiling数据" warning | BBV files not found or empty | Re-run Phase 3 for that operator |
| Inflated benefit claims | Counting without register-width normalization | Normalize all platforms to RVV VLEN |

---

## Phase 5: Consolidated Report

### Goal

Merge all per-operator reports into a single optimization report with executive summary,
per-operator findings, and consolidated recommendations.

### Report Structure

```markdown
# <Workload-Name> RVV512 Optimization Report

**Date**: YYYY-MM-DD
**Target**: <application> on RISC-V
**Platform**: <dev-board> (real hardware) → RVV512 (QEMU)

---

## Executive Summary

### Analysis Scope
- Total functions analyzed, operators covering X% of execution time
- Number of operators vectorized

### Vectorization Results
| Operator | % Time | RVV Impl | Test | Gap Analysis |
|----------|--------|----------|------|--------------|
| ... | ... | ... | ... | ... |

### Key Findings
Top 3-5 most impactful findings across all operators.

---

## Phase 1: Perf Profiling Results

### Global Metrics
(table from perf_stat.txt)

### Function Hotspots
(table from perf_report.txt)

### Performance Characteristics
(IPC analysis, cache behavior, bottleneck identification)

---

## Phase 2: RVV512 Operator Implementations

### Implementation Overview
Per operator: algorithm, RVV strategy, key intrinsics, instruction count per iteration.

---

## Phase 3: BBV Profiling Results

### Per-Operator Instruction Mix
Summary of BBV findings for each operator.

---

## Phase 4: Gap Analysis Summary

### Consolidated Priority Table
All proposed extensions across all operators, sorted by overall benefit.

| Priority | Extension | Operators Affected | Overall Benefit | Source Platform |
|----------|-----------|-------------------|-----------------|-----------------|
| P0 | ... | ... | 整体减少X% | ... |
| P1 | ... | ... | 整体减少X% | ... |

### Top 5 Recommended Extensions
Detailed description of the 5 highest-impact proposed instructions.

---

## Appendix: Individual Operator Reports
Links to each operator's gap analysis PDF.
```

### Output

- `docs/report/<app-name>/<workload-name>-rvv512-optimization-YYYY-MM-DD.md`
- `docs/report/<app-name>/pdf/<workload-name>-rvv512-optimization-YYYY-MM-DD.pdf`

---

## Complete Output Directory Structure

After the pipeline finishes, the following files will exist:

```
<app-path>/
├── data/
│   └── perf-<workload>/
│       ├── perf_stat.txt
│       ├── perf_report.txt
│       └── summary.md
├── rvv-patches/
│   ├── <operator-1>/
│   │   ├── rvv_<op1>.inl
│   │   ├── test.cpp
│   │   ├── patch.diff
│   │   └── README.md
│   ├── <operator-2>/
│   │   └── ...
│   └── ...
├── riscv64-linux-zvl512b-toolchain.cmake
└── ...

docs/report/<app-name>/
├── rvv-gap-analysis-<op1>-YYYY-MM-DD.md
├── rvv-gap-analysis-<op2>-YYYY-MM-DD.md
├── ...
├── <workload>-rvv512-optimization-YYYY-MM-DD.md
└── pdf/
    ├── rvv-gap-analysis-<op1>-YYYY-MM-DD.pdf
    ├── rvv-gap-analysis-<op2>-YYYY-MM-DD.pdf
    ├── ...
    └── <workload>-rvv512-optimization-YYYY-MM-DD.pdf

<bbv-output-dir>/
├── <op1>.0.bb
├── <op1>.disas
├── <op2>.0.bb
├── <op2>.disas
└── ...
```

---

## Phase Dependencies (Critical)

```
Phase 1 (Perf)
    │
    ▼ hotspot list
Phase 2 (RVV impl)  ←── can run operators in parallel
    │
    ▼ all tests pass + app rebuilt
Phase 3 (BBV)       ←── MUST complete fully before Phase 4
    │
    ▼ .bb + .disas for every operator
Phase 4 (Gap)       ←── REQUIRES BBV data; can run operators in parallel
    │
    ▼ all MD + PDF reports
Phase 5 (Consolidate)
    │
    ▼ final report + PDF
```

**Never skip phases or reorder.** Each phase consumes the output of the previous one.
Gap analysis without BBV data produces reports of severely limited value.

---

## Autonomous Execution Guidelines

When the user instructs autonomous execution ("不要等待我的输入", "自动推进"):

1. **Do not ask for confirmation** between phases
2. **Handle errors inline**: if a standalone test fails, debug and fix it
3. **Skip operators with irrecoverable issues**: document why and move on
4. **Report progress** with brief one-line updates at phase transitions
5. **Commit artifacts** only when explicitly asked

### Error Handling

| Error | Action |
|-------|--------|
| SSH connection fails | Retry once, then abort Phase 1 and proceed with existing profiling data if available |
| Standalone test fails | Debug, fix code, re-test. If still failing after 2 attempts, skip operator with documentation |
| BBV output empty | Verify nm offset and function name. Rebuild if needed. Skip operator if unresolvable |
| Gap analysis PDF fails | Fix markdown formatting (common: nested bold/code tags), retry |
| Application build fails | Check patch conflicts. Fix and retry |

---

## Skill Integration Reference

| Phase | Skill | Purpose |
|-------|-------|---------|
| 1 | perf-profiling | Remote profiling commands (reference only, construct SSH commands manually) |
| 2 | rvv-op | Implement RVV512 operator, standalone test, patch integration |
| 3 | qemu-bbv-usage | Function-scoped BBV profiling via QEMU |
| 4 | rvv-gap-analysis | Cross-platform comparison with BBV-weighted benefits |
| 4 | md2pdf | Convert gap analysis Markdown → PDF |
| 5 | md2pdf | Convert consolidated report Markdown → PDF |

---

## Validation Checklist

Before reporting pipeline completion:

- [ ] Perf profiling data exists with hotspot summary
- [ ] Every operator >1% has an RVV implementation in `rvv-patches/`
- [ ] All standalone tests pass under QEMU (zvl512b)
- [ ] Application builds successfully with all RVV512 patches
- [ ] Every vectorized operator has BBV `.bb` + `.disas` files
- [ ] Every operator has a gap analysis report (MD + PDF)
- [ ] Gap analysis reports contain overall benefit figures (整体收益) from BBV data
- [ ] Consolidated report merges all findings with priority table
- [ ] Consolidated PDF generated successfully
