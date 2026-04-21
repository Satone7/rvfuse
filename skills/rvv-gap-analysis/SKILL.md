---
name: rvv-gap-analysis
description: |
  Cross-platform vector operator analysis that compares an RVV (RISC-V Vector) implementation
  against x86 AVX/AVX2, ARM NEON/SVE, LoongArch LASX, Power VSX, S390X Z-Vector, and WASM SIMD.
  Identifies instructions missing from RVV, estimates benefits of proposed new instructions,
  and generates a comprehensive report with QEMU-BBV profiling integration.
  Trigger when: the user mentions "RVV gap analysis", "vector comparison", "cross-platform vector",
  "RVV extension proposal", "vector instruction gap", or wants to analyze an RVV operator against
  other architectures. Also trigger when the user provides an RVV kernel/algorithm implementation
  and asks about potential improvements from new instructions, or wants to compare vector ISAs.
---

# RVV Cross-Platform Gap Analysis

## Purpose

Given an RVV implementation of an operator (kernel/algorithm), produce a detailed analytical report
that:

1. Compares the RVV implementation against vector implementations on other platforms
2. Identifies instructions present in other platforms but absent from RVV
3. Proposes new RVV instructions inspired by the best ideas across all platforms
4. Estimates performance benefits, normalized to the RVV register width
5. Integrates QEMU-BBV profiling data for real-world hotspot weighting

## Input

The user provides one or more of:

1. **RVV implementation** — A C/assembly file or code snippet containing the RVV operator
2. **Reference implementations** — (optional) Equivalent kernels from other platforms (e.g., MLAS sources)
3. **BBV profiling data** — (optional) Path to `.bbv` or `.disas` files from QEMU profiling
4. **Target VLEN** — (optional) RVV vector register length, default 512-bit
5. **Target SEW** — (optional) Selected element width, default 32-bit (float32)

If reference implementations are not provided, the skill will search the project's `third_party/`
and `tools/` directories for relevant source code, and research platform ISA documentation.

## Output

A Markdown report saved to `docs/` with the filename pattern `rvv-gap-analysis-<operator>-<date>.md`.

## Workflow

The skill uses a **subagent-driven pipeline** with specific model assignments:

| Role | Model | Purpose |
|------|-------|---------|
| Platform analyst (x86, ARM) | sonnet | Deep analysis of x86 AVX/AVX2 and ARM NEON/SVE |
| Platform analyst (other) | haiku | LoongArch LASX, Power VSX, S390X, WASM SIMD |
| Report assembler | sonnet | Merge all platform analyses into final report |
| Reviewer | opus | Accuracy review, cross-check against RVV spec |
| Fixer | sonnet | Fix issues found during review |
| BBV profiler | haiku | Parse and integrate profiling data |

### Step 1: Parse the RVV implementation

Read the RVV operator code and extract:

- Key operations (FMA, load/store, shuffle, reduction, etc.)
- Register usage pattern (LMUL, EMUL, mask usage)
- Loop structure (K-loop, M-loop, N-loop in GEMM; sliding window in convolution)
- Instruction count per iteration of each loop
- Data flow between operations (which outputs feed which inputs)

Produce a structured summary of the RVV baseline implementation.

### Step 2: Launch parallel platform analysis subagents

Spawn one subagent per platform. Each subagent receives:
- The RVV baseline summary from Step 1
- The target VLEN and SEW
- Instructions to read the reference template from `references/report-template.md`
- Instructions to read the benefit calculation methodology from `references/benefit-calculation.md`

**Model assignments** (use the `model` parameter when spawning Agent tools):
- x86 AVX/AVX2 analyst → `sonnet`
- ARM NEON/SVE analyst → `sonnet`
- LoongArch LASX/LSX analyst → `haiku`
- Power VSX (POWER10) analyst → `haiku`
- S390X Z-Vector analyst → `haiku`
- WASM SIMD analyst → `haiku`

Each analyst must produce output following the per-platform section format in
`references/report-template.md`.

### Step 3: Assemble the draft report

Spawn a report assembler subagent (`sonnet` model) that:

1. Reads all platform analysis outputs
2. Builds the summary table (see "Summary Table" section below)
3. Merges all sections into a single Markdown document
4. Adds the benefit calculation methodology note
5. Saves the draft to `docs/rvv-gap-analysis-<operator>-<date>.md`

### Step 4: BBV profiling integration (if data provided)

If the user provides BBV profiling data:

1. Spawn a BBV profiler subagent (`haiku` model) to:
   - Parse the `.bbv` file to get each BB's execution frequency (执行次数)
   - Parse the `.disas` file to get each BB's instruction count and content
   - Calculate each BB's execution share: `BB占比 = BB执行次数 × BB指令数 / Σ(所有BB执行次数 × BB指令数)`
   - Identify which BBs contain the target operator's hot loops
2. For each proposed instruction, compute:
   - **BB指令减少数**: How many instructions the extension removes from the target BB
   - **BB内减少比例**: `BB指令减少数 / BB总指令数 × 100%`
   - **整体收益**: `BB内减少比例 × BB占比` — this is the definitive benefit figure
3. Produce a "Hotspot-Weighted Benefit Analysis" section with:
   - BBV hotspot table (top N BBs by execution share, with instruction counts)
   - Per-extension benefit chain showing the calculation from BB-level to overall
   - Cumulative benefit estimate (sum of all extensions' overall benefits)

If no BBV data is provided, report benefits as BB-scoped percentages only, and note
that overall benefit estimation requires profiling via `./tools/profile_to_dfg.sh`.

### Step 5: Review-Fix cycle

This is a critical quality gate. The review cycle runs **at least 2 rounds** and continues
until the reviewer confirms no issues remain.

**Each review round:**

1. **Reviewer** (opus model): Reads the draft report and cross-checks every instruction
   description against:
   - RVV intrinsic spec in `third_party/riscv-rvv-intrinsic-doc/` (search the appropriate
     `.adoc` files for the exact intrinsic definition)
   - Platform ISA documentation (verify instruction names, operand formats, behavior)
   - Register width normalization (verify calculations use the correct RVV VLEN)
   - Benefit arithmetic (verify instruction count comparisons use equivalent workloads)

   The reviewer produces a structured issue list with:
   - Location in the report (section, paragraph, table row)
   - What is inaccurate
   - What the correct information should be
   - Severity (critical/major/minor)

2. **Fixer** (sonnet model): Receives the issue list and:
   - Fixes each issue in the report
   - Performs self-review of the fixes
   - Notes any ambiguities that need the reviewer's attention
   - Saves the updated report

3. **Reviewer** reviews the fixes:
   - If issues remain → send back to fixer with new issue list
   - If all clear → produce a "Review passed" confirmation

**Minimum 2 full review rounds**, even if the first round finds no issues (to ensure thoroughness).

### Step 6: Finalize and deliver

After the review cycle completes:

1. Add a "Review Log" section at the end of the report documenting:
   - Number of review rounds
   - Issues found and fixed per round
   - Final reviewer confirmation
2. Save the final report
3. Present the summary to the user

## Summary Table Format

The report MUST begin with a summary table after the overview section.
The "预期收益" column MUST express benefit as **overall instruction reduction percentage**
(整体指令减少比例), not vague multipliers or per-function percentages.

### When BBV profiling data is available

Every benefit figure is derived from concrete BBV data using this chain:

```
整体收益 = BB指令减少数 / BB总指令数 × BB执行占比
```

Where:
- **BB指令减少数**: How many instructions the proposed extension eliminates from the target BB,
  normalized to RVV VLEN/SEW
- **BB总指令数**: Total instruction count of the target BB (from `.disas` file)
- **BB执行占比**: The target BB's share of total execution (from `.bbv` frequency data)

Example table:

```markdown
## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | 整体收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0     | vmulacc.vv | Power VSX | 整体减少3.2% | 高 | 无矩阵级指令 |
| P1     | vfmacc.vv_lane | ARM NEON | 整体减少2.1% | 中 | 需标量广播 |

**收益计算方式**（基于QEMU-BBV profiling数据）：
- BB指令减少数 = 原BB指令数 - 扩展后BB指令数（归一化到RVV VLEN=512bit, SEW=32bit）
- 整体收益 = BB指令减少数 / BB总指令数 × BB执行占比
  - P0 vmulacc.vv: K循环BB从8条减至5条（-37.5%），该BB占总执行21.8% → 37.5% × 21.8% = 整体减少8.2%
  - P1 vfmacc.vv_lane: 广播BB从12条减至7条（-41.7%），该BB占总执行12.5% → 41.7% × 12.5% = 整体减少5.2%
- 各扩展指令的整体收益可叠加估算上限（假设无交互效应）：Σ 整体收益
```

### When BBV profiling data is NOT available

Use per-BB instruction reduction only, clearly stating the scope:

```markdown
| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0     | vmulacc.vv | Power VSX | BB内减少37.5%（K循环BB） | 高 | 无矩阵级指令 |

**注**: 无BBV profiling数据，上表仅反映单个BB范围内的指令减少比例，无法推算整体收益。
建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算。
```

### Prohibited expressions

The following benefit expressions are **not allowed** in the summary table:
- Vague multipliers like "1.3-2.0x" — 1.3-2.0x relative to what? The whole program? The function?
- Bare percentages like "-37.5%指令" without scope — is this per-BB, per-function, or overall?
- "性能提升" without a clear baseline and scope

Always express benefit as either "整体减少X%" (with BBV data) or "BB内减少X% (BB名称)"
(without BBV data).

## Register Width Normalization

When the reference platform uses a different register width than RVV:

1. **Calculate element count per register** for each platform
2. **Scale instruction throughput**: if RVV VLEN=512 and reference is 128-bit,
   one RVV instruction processes 4× the elements — scale the reference's per-instruction
   work by 4× before comparing
3. **Report both raw and normalized figures** in the analysis
4. **Example**: ARM NEON `vmlaq_lane_f32` processes 4 float32 per call (128-bit).
   At RVV VLEN=512 (16 float32), one lane-indexed FMA would process 16 float32.
   Instruction count saving = (16 scalar loads + 16 FMA) → (1 vector load + 16 lane-FMA) = -52.9%

## RVV Reference

When verifying RVV instruction capabilities, search these files:

- **General intrinsics**: `third_party/riscv-rvv-intrinsic-doc/auto-generated/intrinsic_funcs.adoc`
- **Overloaded intrinsics**: `third_party/riscv-rvv-intrinsic-doc/auto-generated/overloaded_intrinsic_funcs.adoc`
- **By category** (in `auto-generated/` directory):
  - `00_vector_loads_and_stores_intrinsics.adoc` — Loads and stores
  - `01_vector_loads_and_stores_segment_intrinsics.adoc` — Segment loads/stores
  - `02_vector_integer_arithmetic_intrinsics.adoc` — Integer arithmetic
  - `03_vector_fixed-point_arithmetic_intrinsics.adoc` — Fixed-point
  - `04_vector_floating-point_intrinsics.adoc` — Floating-point
  - `05_vector_reduction_operations.adoc` — Reductions
  - `06_vector_mask_intrinsics.adoc` — Mask operations
  - `07_vector_permutation_intrinsics.adoc` — Permutation/gather
  - `08_miscellaneous_vector_utility_intrinsics.adoc` — Utilities
  - `09_vector_crypto_intrinsics.adoc` — Crypto extensions
  - `10_zvfbfmin_intrinsics.adoc` — BFloat16
  - `11_zvfbfwma_intrinsics.adoc` — BFloat16 widening MA

Always verify proposed "missing" RVV instructions by searching these files first — some
functionality may exist under a different name or encoding.

## Per-Platform Analysis Template

Each platform analysis section must include:

1. **Platform overview**: Register file, key characteristics
2. **High-value instruction table**: Instructions with no RVV equivalent
3. **Benefit analysis with code comparison**: Side-by-side assembly showing current RVV vs
   proposed improvement, with instruction counts
4. **Proposed RVV extension**: Instruction name, semantics, encoding constraints
5. **Normalized benefit**: Scaled to RVV VLEN with calculation shown

## Subagent Spawning Guide

When spawning subagents for this skill, use these patterns:

```
Agent({
  description: "x86 AVX gap analysis",
  model: "sonnet",
  prompt: "You are analyzing x86 AVX/AVX2 vector operations for the RVV gap analysis skill.
           Read the skill references at .claude/skills/rvv-gap-analysis/references/report-template.md
           and .claude/skills/rvv-gap-analysis/references/benefit-calculation.md.
           [Include RVV baseline summary and specific analysis instructions]"
})
```

For the review cycle:

```
Agent({
  description: "Report accuracy review",
  model: "opus",
  prompt: "You are reviewing an RVV gap analysis report for accuracy.
           Read the report at [path].
           Cross-check every instruction against the RVV spec in
           third_party/riscv-rvv-intrinsic-doc/ by searching the relevant .adoc files.
           [Full review instructions from references/review-checklist.md]"
})
```

```
Agent({
  description: "Report fix pass",
  model: "sonnet",
  prompt: "You are fixing issues found in an RVV gap analysis report.
           Read the report at [path] and the reviewer's issue list.
           Fix each issue, then self-review your fixes.
           [Full fix instructions]"
})
```

## Important Constraints

- **Never claim an RVV instruction is missing without first searching the intrinsic doc.**
  Many operations exist under unexpected names (e.g., `vlse32.v stride=0` for load+broadcast).
- **Always normalize register widths** before comparing instruction counts across platforms.
- **Always specify the RVV VLEN and SEW** used for benefit calculations.
- **Distinguish assembly-level guarantees from intrinsic-level guarantees** (e.g., stride=0
  behavior differs between assembly and C intrinsics).
- **Count instructions for equivalent workloads**, not different workload sizes.
- **Document assumptions** — if a proposed instruction requires new hardware (accumulator
  registers, new execution units), state this explicitly.

## Benefit Calculation Rules

### Amdahl's Law Application

When calculating overall benefit, ALWAYS use Amdahl's Law:

```
整体加速百分比 = (原总时间 - 新总时间) / 原总时间 × 100%

新总时间 = 未优化部分时间 + 热点部分时间 / 加速倍数
```

Example:
```
原总时间 = 100s
热点占比 = 35%（35s）
未优化部分 = 65s
热点加速倍数 = 3.5（热点新时间 = 10s）

新总时间 = 65 + 10 = 75s
整体加速百分比 = (100 - 75) / 100 = 33%
```

### Benefit Expression Format

**MANDATORY: Express ALL benefits as percentages, NOT as multipliers.**

| Format | Example | Usage |
|--------|---------|-------|
| Correct | 加速33% | Always use this |
| Incorrect | 1.33× | Never use |
| Incorrect | 33% → 1.33× | Never use |

When showing calculation process, use:
```
加速约 X%（周期减少Y%）
整体加速 = (原时间 - 新时间) / 原时间 = Z%
```

### Benefit Scope Levels

Always annotate the scope level for each benefit figure:

| Scope | Meaning | Example Expression |
|-------|---------|-------------------|
| 整体 | Whole application | 整体加速15-25% |
| 函数 | Specific function | 函数加速43% |
| K-loop级 | Innermost loop | K-loop级周期减少35% |

**Priority**: 整体收益 is the most important metric. Always calculate and report it.

### Conservative Estimation

When calculating benefits, apply these conservative adjustments:

| Factor | Adjustment |
|--------|-----------|
| New instruction latency | May be higher than estimated (add ~1-2 cycles) |
| Pipeline efficiency | Actual throughput ~80-90% of theoretical |
| Compiler optimization | May not achieve perfect scheduling |
| K-loop function ratio | Use 85% (not 90%) for function-level calculation |
| Function hotspot ratio | Use slightly lower values for overall calculation |

Example conservative calculation chain:
```
乐观估计:
  K-loop周期减少: 43.75% → 加速60%
  函数加速: 65% → 整体加速34%

保守估计:
  K-loop周期减少: 35% → 加速约54%
  函数加速: 43% → 整体加速约21%
  最终报告范围: 15-25%整体加速
```

## Software Team Constraints

This skill is for a **software team**. The following constraints apply:

### Scope Limitations

- **Software perspective only**: Analyze benefits from instruction count and cycle reduction
- **No hardware design**: Do not propose instruction encoding, hardware implementation details,
  or microarchitecture modifications
- **No hardware complexity estimates**: Do not discuss "hardware complexity", "pipeline stages",
  or "execution unit requirements"
- **No register pressure analysis**: Focus on instruction count, not register allocation pressure

### Content to Exclude

| Excluded Content | Reason |
|-----------------|--------|
| 指令编码方案 | Hardware team responsibility |
| 硬件单元设计 | Hardware team responsibility |
| 累加器寄存器设计 | Hardware team responsibility |
| 流水线吞吐优化 | Hardware team responsibility |
| 微架构实现难度 | Hardware team responsibility |
| 提议指令预估延迟（硬件依据） | Hardware team estimates |

### Acceptable Content

| Acceptable Content | Reason |
|--------------------|--------|
| Current instruction latency (from existing documentation) | Software optimization reference |
| Instruction count comparison | Software benefit analysis |
| Cycle calculation using documented latencies | Software estimation |
| Amdahl's Law benefit calculation | Software-level analysis |
| Platform instruction reference (ARM NEON, x86 VNNI) | Cross-platform comparison |

### When Hardware Topics Appear

If hardware design topics are mentioned in the analysis:
- Replace "需设计新硬件单元" with "需提出新指令方案"
- Replace "硬件实现难度高" with "建议由硬件团队评估实现可行性"
- Delete any section discussing instruction encoding or hardware implementation
