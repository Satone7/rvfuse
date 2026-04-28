# Int4 Native Compute Instruction Design — Execution Plan

**Date**: 2026-04-28 | **Team**: `int4-instruction-design`

> **阅读本文档的 AI Agent 注意**：
>
> 本文档描述的是**团队协作模型**（Team Model）。所有 Worker 必须通过 `TeamCreate` + `Agent(team_name="int4-instruction-design", name="...")` 创建为**队友（teammate）**。
>
> **禁止**将 Worker 误解为普通 subagent（不带 `team_name` 的 `Agent` 调用）或 Task（`TaskCreate` 创建的工作项）。
>
> Teammate = `Agent(team_name=..., name=...)`。Subagent = `Agent(description=...)`。二者完全不同。

## 1. Objective

Evaluate whether native int4 compute instructions (int4×int4→int32) are viable for RISC-V Vector by:
1. Measuring precision loss when replacing int8 activation quantization with int4 in two real workloads (llama.cpp Q4_0 LLM inference, ONNX Runtime YOLO11n-INT8)
2. If precision loss is acceptable, analyzing the arithmetic requirements of the hot kernels and proposing RVV int4 instruction semantics from a software perspective

This batch answers: *"Should we ask the hardware team for native int4 compute instructions, and if so, what should they look like?"*

## 2. Team Structure

### 2.1 Teammates vs Subagents

| Model | Creation | Membership | Communication | Purpose |
|-------|----------|------------|---------------|---------|
| **Teammate** | `Agent(team_name="int4-instruction-design", name="...")` | Belongs to team | `SendMessage(to="name")` | Long-running serial worker |
| **Subagent** | `Agent()` or `Agent(subagent_type="...")` | No team membership | Result returned to caller | Fire-and-forget verification check |

**Correct spawn pattern:**

```python
Agent(
    team_name="int4-instruction-design",
    name="llama-int4",
    subagent_type="general-purpose",
    isolation="worktree",
    model="opus",
    mode="auto",
    run_in_background=True,
    prompt="""..."""
)
```

### 2.2 Roles

| Role | Name | Model | Responsibility |
|------|------|-------|---------------|
| **Lead** | (current session) | — | `TeamCreate`, spawn teammates serially, dispatch opus verification subagent after each, manage rework loops, merge worktrees, cancel Guardian cron last |
| **Task 1** | `llama-int4` | **opus** | llama.cpp int4 precision validation on Banana Pi |
| **Task 2** | `yolo-int4` | **sonnet** | ONNX Runtime YOLO int4 precision validation under QEMU |
| **Task 3** | `int4-report` | **opus** | Int4 instruction requirement analysis (zvl512b) |
| **Guardian** | `guardian` | haiku | Monitor team progress via tmux cron loop |

**Serial execution**: `llama-int4` → `yolo-int4` → `int4-report`. Guardian runs persistently.

### 2.3 Model Selection Rationale

| Teammate | Model | Rationale |
|----------|-------|-----------|
| `llama-int4` | opus | Novel int4 quantization design — no precedent in project. Must understand llama.cpp's Q4_0 unpacking, Q8_0 quantization pipeline, design int4 variant, and qualitatively judge LLM output degradation. |
| `yolo-int4` | sonnet | Follows established QGEMM kernel pattern at `rvv-patches/qgemm-kernel-vl16/`. The int4 modifications are mechanical extensions of the existing int8 path. |
| `int4-report` | opus | Synthesis requiring deep RISC-V vector ISA knowledge, CUDA int4 instruction set analysis, and the ability to propose well-formed RVV instruction semantics. |

## 3. Tasks

| # | Teammate | Scope | Model | VLEN | Priority |
|---|----------|-------|-------|------|----------|
| 1 | `llama-int4` | llama.cpp int4 precision validation | opus | zvl256b / scalar | High |
| 2 | `yolo-int4` | ONNX RT YOLO int4 precision validation | sonnet | zvl512b | High |
| 3 | `int4-report` | Int4 instruction requirement analysis | opus | zvl512b | High (depends on 1+2) |

### Task 1: llama.cpp Int4 Precision Validation

**Teammate**: `llama-int4` | **Model**: opus | **VLEN**: zvl256b or scalar (Banana Pi K1 native)

#### Background

The hot kernel `ggml_gemm_q4_K_8x4_q8_K` (RVV patch at `applications/llama.cpp/rvv-patches/gemm-q4_K-8x4-q8_K/`) implements Q4_K weights × Q8_K activations GEMM with int8×int8→int32 compute. The pipeline:

```
Q4_0 model → load 4-bit weights → unpack to 8-bit (q4_lo, q4_hi)
fp32 activations → quantize to Q8_0 (int8 with per-row scale/d)
int8 weights × int8 activations → int32 accumulation → float output
```

The quantization function `ggml_quantize_mat_q8_0_4x4` (RVV patch at `applications/llama.cpp/rvv-patches/quantize-q8_0-4x4/`) converts fp32 activations to Q8_0 format.

#### Task

1. **Verify hotspot** on Banana Pi via perf: confirm `ggml_gemm_q4_K_8x4_q8_K` is the dominant hot function
2. **Design int4 activation quantization**: Modify the activation quantization path so activations are quantized to int4 (2× fewer levels) but still stored in int8 containers. The teammate designs the quantization scheme (symmetric vs asymmetric, per-row vs coarser granularity) and documents the rationale.
3. **Inject int4 activations**: Feed the int4-quantized activations (stored in int8) through the existing, unmodified int8×int8→int32 GEMM kernel
4. **Run inference** on Banana Pi with Qwen2.5-0.5B-Instruct Q4_0 model
5. **Qualitative evaluation**: Compare output text against the unmodified int8 path. Is the output coherent? Are there severe factual deviations or catastrophic failures? Document representative examples.

#### Key Constraints

- **VLEN**: zvl256b or scalar implementation. No zvl512b requirement. Banana Pi K1 is zvl256b.
- **Hardware**: Banana Pi K1 at `root@192.168.100.221` (password: `bianbu`)
- **Model**: `applications/llama.cpp/models/Qwen2.5-0.5B-Instruct-Q4_0.gguf`
- **Build**: `applications/llama.cpp/build.sh` (cross-compile for rv64gcv)
- **Reference code**: `applications/llama.cpp/rvv-patches/gemm-q4_K-8x4-q8_K/rvv_gemm_q4_K_8x4.inl` (GEMM kernel), `applications/llama.cpp/rvv-patches/quantize-q8_0-4x4/rvv_quantize_q8_0_4x4.inl` (quantization)
- **Reference runner**: `applications/llama.cpp/qwen` (Qwen inference wrapper)

#### Phases

**Phase 0 — Setup**:
1. Rebuild llama.cpp with existing RVV patches for Banana Pi (zvl256b)
2. Verify unmodified Qwen2.5-0.5B-Instruct Q4_0 inference works correctly on Banana Pi
3. Record baseline output for comparison

**Phase 1 — Perf Verification**:
1. Run perf on Banana Pi to confirm `ggml_gemm_q4_K_8x4_q8_K` is the hottest function
2. Document the perf data (hot function % of runtime)

**Phase 2 — Int4 Quantization Design & Implementation**:
1. Analyze the existing `ggml_quantize_mat_q8_0_4x4` quantization function
2. Design int4 quantization: fp32→int4 with appropriate scale/zero-point scheme
3. Implement the int4 quantization function (stored in int8 containers, so the existing GEMM can consume it unchanged)
4. Integrate into the llama.cpp inference path (modify quantization call, not the GEMM kernel itself)

**Phase 3 — Precision Evaluation**:
1. Run Qwen2.5-0.5B-Instruct Q4_0 inference with int4-quantized activations on Banana Pi
2. Collect output for representative prompts (at least 3-5 prompts covering factual recall, reasoning, language fluency)
3. Compare with baseline int8 output
4. Document: coherence, factual accuracy, any catastrophic failures
5. Measure any quantitative metrics available (perplexity if feasible, token-level accuracy)

**Phase 4 — Report**:
1. Write findings to `docs/report/int4-instruction/llama-int4-precision-*.md`
2. Include: quantization scheme design rationale, perf data, representative output comparisons, overall assessment
3. Generate PDF via md2pdf

### Task 2: ONNX Runtime YOLO Int4 Precision Validation

**Teammate**: `yolo-int4` | **Model**: sonnet | **VLEN**: zvl512b (QEMU)

#### Background

The hot kernel `MlasQgemmKernel` (RVV patch at `applications/onnxrt/rvv-patches/qgemm-kernel-vl16/`) implements uint8×uint8→int32 quantized GEMM. The pipeline:

```
INT8 YOLO model → uint8 weights
fp32 activations → quantize to uint8
uint8 weights × uint8 activations → int32 accumulation → float output
```

#### Task

1. **Verify QGEMM hotspot** via QEMU BBV profiling: confirm MlasQgemmKernel is the dominant hot function
2. **Implement int4 activation quantization**: fp32→int4 stored in uint8 containers
3. **Implement int4 weight quantization**: int8→int4 stored in uint8 containers  
4. **Run inference** under QEMU (zvl512b) with int4-simulated data through the existing uint8×uint8→int32 QGEMM kernel
5. **Accuracy measurement**: Compare mAP/top-1 accuracy vs vanilla INT8 YOLO11n

#### Key Constraints

- **VLEN**: zvl512b (follows existing QGEMM patch convention)
- **Runtime**: QEMU (`qemu-riscv64 -cpu rv64,v=true,vlen=512`)
- **Model**: YOLO11n INT8 ONNX model
- **Reference code**: `applications/onnxrt/rvv-patches/qgemm-kernel-vl16/rvv_qgemm_kernel_vl16.inl`
- **Reference build**: `applications/onnxrt/ort/build.sh`

#### Phases

**Phase 0 — Setup**:
1. Rebuild ONNX Runtime with existing QGEMM RVV patch for zvl512b
2. Verify unmodified YOLO11n-INT8 inference works correctly under QEMU
3. Record baseline accuracy/performance metrics

**Phase 1 — Hotspot Verification**:
1. Run BBV profiling on YOLO11n-INT8 under QEMU to confirm QGEMM dominates
2. Document BBV data

**Phase 2 — Int4 Quantization Implementation**:
1. Implement fp32→int4 activation quantization (stored in uint8)
2. Implement int8→int4 weight quantization (stored in uint8)
3. Feed int4-quantized data through the existing MlasQgemmKernel (uint8×uint8→int32 path)
4. The kernel itself is unmodified — only the quantization/preprocessing changes

**Phase 3 — Accuracy Evaluation**:
1. Run YOLO11n with int4 activations + int4 weights under QEMU
2. Measure detection accuracy (mAP or top-1) on test images
3. Compare with baseline INT8 accuracy
4. Document precision loss

**Phase 4 — Report**:
1. Write findings to `docs/report/int4-instruction/yolo-int4-precision-*.md`
2. Include: quantization scheme, BBV data, accuracy comparison, overall assessment
3. Generate PDF via md2pdf

### Task 3: Int4 Instruction Requirement Analysis

**Teammate**: `int4-report` | **Model**: opus | **VLEN**: zvl512b

#### Background

This task synthesizes the precision loss findings from Tasks 1 and 2 and produces the final deliverable: an int4 native compute instruction requirements document for the hardware team.

#### Task

1. **Synthesize precision findings**: Summarize the precision loss measurements from both tasks. If loss is large, document why int4's dynamic range is insufficient for these workloads. If acceptable, document the margin.
2. **Analyze kernel arithmetic requirements**: For both `ggml_gemm_q4_K_8x4_q8_K` (llama.cpp) and `MlasQgemmKernel` (ONNX RT), describe what native int4×int4→int32 instructions would look like if they existed:
   - How many int4 elements per vector register?
   - What packing format? (interleaved vs planar)
   - What accumulation width? (int32 natural, but int16 for intermediate?)
   - What dequantization steps remain after int4 compute? (scale application, zero-point correction)
3. **Reference CUDA int4 instruction set design**: Study and reference CUDA's approach to int4 compute (e.g., `dp4a`-like packed dot products, `mma.sync` for tensor cores). What design choices did NVIDIA make and why?
4. **Propose RVV int4 instruction semantics**: From the software team's perspective, what would ideal RVV int4 instructions look like? Describe:
   - Instruction semantics (operands, operation, width)
   - Element packing format
   - Accumulation behavior
   - Interaction with existing RVV infrastructure (vsetvl, LMUL, etc.)
5. **Output the report** as the software team's instruction requirements for the hardware team. Do NOT analyze feasibility, implementation difficulty, hardware cost, or priority ordering. This is a requirements document — "here's what we need and why."

#### Key Constraints

- **VLEN**: zvl512b (analysis and proposals target 512-bit vector width)
- **Role**: Software team → Hardware team requirements. No hardware feasibility analysis.
- **References**: Tasks 1+2 reports, CUDA PTX ISA documentation, existing RVV 1.0 spec

#### Phases

**Phase 0 — Gather Inputs**:
1. Read the precision reports from Tasks 1 and 2 (already merged to master)
2. Read the RVV kernel code for both hot functions
3. Research CUDA int4 instruction set design (dp4a, mma, etc.)

**Phase 1 — Arithmetic Analysis**:
1. For each kernel: count operations by type (load, MAC, scale, convert)
2. Identify where int4-native would replace existing int8 operations
3. Estimate theoretical speedup if native int4×int4→int32 existed

**Phase 2 — Instruction Semantics Proposal**:
1. Draft proposed RVV int4 instruction semantics
2. Cover: int4 dot-product with accumulate, int4×int4→int32 widening multiply, or other relevant forms
3. Specify packing, element ordering, and interaction with LMUL

**Phase 3 — Report**:
1. Consolidated report: `docs/report/int4-instruction/int4-instruction-requirements-*.md`
2. Sections: precision findings summary, kernel analysis, CUDA reference, proposed RVV int4 instructions
3. Generate PDF via md2pdf

## 4. Execution Strategy

### 4.1 Serial Execution Loop

```
TeamCreate("int4-instruction-design")
  → Spawn Guardian
  → TaskCreate for all 3 tasks
  → Spawn llama-int4 (opus) → WAIT → Verify → Merge → Shutdown
  → Spawn yolo-int4 (sonnet) → WAIT → Verify → Merge → Shutdown
  → Spawn int4-report (opus) → WAIT → Verify → Merge → Shutdown
  → TeamDelete
  → Cancel Guardian cron (LAST)
```

### 4.2 Worktree Isolation

Each teammate runs in a git worktree. Shared resources symlinked from main repo:

| Resource | Path |
|----------|------|
| QEMU source + build | `third_party/qemu/` |
| LLVM source + install | `third_party/llvm-project/`, `third_party/llvm-install/` |
| BBV plugin | `tools/bbv/libbbv.so` |
| llama.cpp vendor | `applications/llama.cpp/vendor/` (Task 1) |
| ONNX RT vendor | `applications/onnxrt/<app>/vendor/` (Task 2) |

### 4.3 Banana Pi Access (Task 1 Only)

Only Task 1 uses the Banana Pi. Task 2 runs under QEMU. Task 3 is pure analysis.

**Dev Board**:
- Host: `192.168.100.221` (Banana Pi K1, SpacemiT K1 SoC)
- User: `root`, Password: `bianbu`
- VLEN: 256 (zvl256b)

## 5. Acceptance Criteria

### Task 1 (llama-int4)
- [ ] Perf confirms `ggml_gemm_q4_K_8x4_q8_K` is the hottest function
- [ ] Int4 activation quantization function implemented and documented
- [ ] Qwen2.5-0.5B Q4_0 inference runs with int4 activations on Banana Pi
- [ ] Qualitative comparison with baseline int8 output for 3+ prompts
- [ ] Report at `docs/report/int4-instruction/llama-int4-precision-*.md` + PDF

### Task 2 (yolo-int4)
- [ ] BBV confirms `MlasQgemmKernel` is the dominant hot function
- [ ] Int4 activation and weight quantization implemented
- [ ] YOLO11n-INT8 runs with int4 activations + weights under QEMU
- [ ] Accuracy comparison (mAP or top-1) vs baseline INT8
- [ ] Report at `docs/report/int4-instruction/yolo-int4-precision-*.md` + PDF

### Task 3 (int4-report)
- [ ] Precision findings from Tasks 1+2 synthesized
- [ ] Both kernels analyzed for int4-native arithmetic requirements
- [ ] CUDA int4 instruction design referenced
- [ ] Proposed RVV int4 instruction semantics documented
- [ ] Report at `docs/report/int4-instruction/int4-instruction-requirements-*.md` + PDF
- [ ] No hardware feasibility/priority/effort analysis (software team perspective only)

### Cross-Batch
- [ ] All worktrees merged with `--no-ff`
- [ ] Guardian cron cancelled as last action
- [ ] Team deleted

## 6. Risk & Timeline

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Int4 precision loss catastrophically large (gibberish output) | Medium | Low | Task 3 still runs — report documents *why* int4 is insufficient, which is still valuable for hardware team |
| Banana Pi unreachable during Task 1 | Low | High | Pre-check SSH before spawning teammate. Fallback: run llama.cpp under QEMU (zvl256b) if board is down |
| llama.cpp build failure with int4 modifications | Low | Medium | Modifications are to quantization function only, not the GEMM kernel. Isolated change surface. |
| QEMU QGEMM accuracy differs from hardware | Low | Low | QGEMM patch already validates correctness under QEMU. Int4 simulation is mechanical. |

### Timeline (Serial)

| Phase | llama-int4 | yolo-int4 | int4-report |
|-------|-----------|-----------|-------------|
| Setup | 30-45 min | 20-30 min | 15-20 min |
| Hotspot verification | 15-25 min (perf) | 20-30 min (BBV) | — |
| Int4 implementation | 45-90 min | 30-60 min | — |
| Precision evaluation | 20-40 min | 15-25 min | — |
| Analysis & Report | 15-25 min | 10-20 min | 60-120 min |
| **Per-task Total** | **~2-4 hours** | **~2-3 hours** | **~1.5-2.5 hours** |

| Overhead | Estimate |
|----------|----------|
| Verification (opus subagent per task) | ~5-10 min each |
| Rework buffer | ~15-30 min each |
| Worktree merge + next spawn | ~5 min each |
| **Total wall-clock** | **~7-11 hours** |

## 7. Teammate Prompt — Mandatory Fragment

Every teammate prompt MUST include this verbatim after phase requirements:

```
DISCOVERY REPORTING: After each phase, you MUST include a section in your
completion message:

  ## Discoveries
  - New: <any operation where existing skill was wrong/missing/incomplete.
    Describe what you expected, what actually happened, and what you did.>
  - Supplement: <any correction to project skill instructions. Include the
    skill name, the specific issue, and the corrected approach.>
  - None: <only if you genuinely discovered nothing new in this phase>

If you encounter an error or unexpected behavior, do NOT silently work around
it and continue. Report it as a Discovery even if you resolved it yourself.

If "None" is reported but verification finds an unreported issue, that
counts as a FAIL.
```

## 8. Task Creation

```python
TaskCreate(
    subject="llama-int4: llama.cpp Int4 Precision Validation",
    description="Design int4 activation quantization, inject into Qwen2.5-0.5B Q4_0 inference on Banana Pi K1, qualitatively compare output with baseline int8."
)
TaskCreate(
    subject="yolo-int4: ONNX Runtime YOLO Int4 Precision Validation",
    description="Implement int4 activation and weight quantization for QGEMM kernel, run YOLO11n-INT8 under QEMU, measure accuracy vs baseline."
)
TaskCreate(
    subject="int4-report: Int4 Instruction Requirement Analysis Report",
    description="Synthesize Tasks 1+2 precision findings, analyze kernel arithmetic requirements, reference CUDA int4 ISA, propose RVV int4 instruction semantics (zvl512b)."
)
```
