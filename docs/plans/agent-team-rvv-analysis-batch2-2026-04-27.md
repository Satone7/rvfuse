# Agent Team Design: Cross-Application RVV Analysis — Batch 2

**Date**: 2026-04-27 | **Status**: Draft | **Team**: rvfuse-analysis-b2

> **阅读本文档的 AI Agent 注意**：
>
> 本文档描述的是**团队协作模型**（Team Model）。所有 Worker 必须通过 `TeamCreate` + `Agent(team_name="rvfuse-analysis-b2", name="...")` 创建为**队友（teammate）**。
>
> **禁止**将 Worker 误解为普通 subagent（不带 `team_name` 的 `Agent` 调用）或 Task（`TaskCreate` 创建的工作项）。Task 只是任务跟踪条目，不是执行者。
>
> Teammate = `Agent(team_name=..., name=...)`。Subagent = `Agent(description=...)`。二者完全不同。
>
> 如果阅读本文档后，你首先想到的是 `TaskCreate` 而不是 `TeamCreate`，或者首先想到的是 `Agent(description=...)` 而不是 `Agent(team_name=..., name=...)`，说明你理解错了。重新阅读 §2.1。

## 1. Objective

Run the `rvv512-optimization-pipeline` (Phase 0–5) for **4 new applications** **serially** (one at a time). Each application is a separate ONNX-based model running on ONNX Runtime, assigned to an isolated Worker operating in a git worktree. Three workers use **sonnet** (established patterns, skill-guided); the `superglue` worker uses **opus** (novel cross-attention operator and complex Sinkhorn implementation).

**Execution model**: Due to limited device performance, teammates execute **serially**, not in parallel. The Lead spawns one teammate at a time, waits for it to complete all phases, dispatches an **opus verification subagent** to check the work quality, and only proceeds to the next application after the current one passes verification. If verification fails, the teammate is instructed to fix issues and re-submit — this loop repeats until verification passes.

This batch covers two complementary domains:
- **Image classification** (ViT-Base/16, ViT-Base/32): Transformer-based architecture with attention dominance
- **Feature matching** (SuperPoint, SuperGlue): CNN detection + GNN matching pair, covering Conv2d/BN/ReLU through to cross-attention and optimal transport

Together with Batch 1 (YOLO INT8, OSTrack, llama.cpp, PyTorch scout), these 4 apps expand the operator coverage matrix significantly, especially for attention variants (self-attention, cross-attention) and CNN feature extraction operators.

## 2. Team Structure

### 2.1 Team Creation — CRITICAL: Teammates vs Subagents

The `rvfuse-analysis-b2` team uses the **team model**: teammates are spawned with `TeamCreate` + `Agent(team_name=...)`. This is fundamentally different from the standalone subagent model.

| Model | Creation | Membership | Communication | Purpose |
|-------|----------|------------|---------------|---------|
| **Team Agent (teammate)** | `Agent(team_name="...", name="...")` | Belongs to team, appears in team config | `SendMessage(to="name")` | Long-running serial worker |
| **Standalone Agent (subagent)** | `Agent()` or `Agent(subagent_type="...")` | No team membership | Result returned to caller | Fire-and-forget research/check task |

**DO NOT use standalone subagents as teammates.** A subagent without `team_name` is invisible to the team, can't receive `SendMessage`, and can't coordinate via the shared TaskList. Teammates are spawned ONLY with `Agent(team_name="rvfuse-analysis-b2", name="<name>", ...)`.

**Correct pattern for spawning a teammate:**

```python
# Step 1: Create the team (done once by Lead)
TeamCreate(team_name="rvfuse-analysis-b2")

# Step 2: Spawn a teammate — NOTE: team_name is REQUIRED, name is REQUIRED
Agent(
    team_name="rvfuse-analysis-b2",  # ← THIS makes it a teammate, not a subagent
    name="vit-base-16",              # ← human-readable name for SendMessage(to="vit-base-16")
    description="ViT-Base/16 Full Pipeline",
    subagent_type="general-purpose",
    isolation="worktree",
    model="sonnet",
    mode="auto",
    run_in_background=True,
    prompt="""..."""
)

# Step 3: Create task AFTER spawning (task goes into the shared team task list)
TaskCreate(subject="ViT-Base/16: Full Pipeline (Phase 0-5)", ...)
```

**Anti-patterns — NEVER do these:**

```python
# WRONG: subagent without team_name — NOT a teammate!
Agent(description="ViT-Base/16 pipeline", prompt="...")

# WRONG: TaskCreate assumed to be a teammate — tasks are work items, not workers!
TaskCreate(subject="ViT-Base/16: Full Pipeline...")  # This creates a task, not a worker
Agent(description="do the task")                       # This is a subagent, not a teammate
```

### 2.2 Roles

| Role | Name | Count | Responsibility |
|------|------|-------|---------------|
| **Lead** | (current session) | 1 | `TeamCreate`, spawn teammates **serially** via `Agent(team_name="rvfuse-analysis-b2", name=...)`, dispatch opus verification subagent after each teammate completes, manage rework loops, shutdown teammate after PASS, then spawn next teammate. Merge worktrees, cross-app synthesis report. Cancel Guardian cron as the final action. |
| **Teammate** | per-application name | 1 active at a time (4 total) | Execute full pipeline (Phase 0–5) for one application in an isolated worktree. Spawned as `Agent(team_name="rvfuse-analysis-b2", name="<app>")` — NEVER as a standalone subagent. |
| **Guardian** | `guardian` | 1 (persistent) | Monitor ALL team members (including Lead) via tmux on a cron loop. Intervene when idle/stuck to keep the team progressing. Spawned as `Agent(team_name="rvfuse-analysis-b2", name="guardian", ...)`. See §2.6. |

**Important**: `TaskCreate` does NOT create workers. Tasks are work-tracking items in the shared task list; they describe what needs to be done, not who does it. Teammates are the actual workers, spawned via `Agent` with `team_name`.

**Serial constraint**: Only ONE app teammate is active at any time. The Lead spawns app teammate N+1 only after teammate N has passed verification and been shut down. The Guardian runs persistently alongside all teammates.

### 2.3 Model Selection

Model selection is differentiated by task difficulty. **Sonnet** handles the three tasks that follow established patterns with existing skill coverage; **Opus** is reserved for the one task with genuinely novel operators and zero reusable references.

| Teammate | Model | Rationale |
|----------|-------|-----------|
| `superpoint` | **sonnet** | CNN patterns well-established, NMS is a deterministic algorithm, Conv2d analysis methodology mature |
| `superglue` | **opus** | Novel cross-attention (first in project), Sinkhorn optimal transport implementation, complex ONNX export with dynamic shapes — requires deep architectural reasoning |
| `vit-base-16` | **sonnet** | Standard ViT architecture, SGEMM patch already exists, runner follows YOLO reference — skill-guided execution suffices |
| `vit-base-32` | **sonnet** | Heavy reuse from vit-base-16, primarily shape comparison (50 vs 197 tokens) — lowest complexity |

**Verification subagent model**: All verification checks use **opus**. Verification requires deep architectural reasoning to judge whether a teammate's output meets quality standards — this is not a mechanical checklist task. See §2.4.3.

### 2.4 Lead Responsibilities — Serial Execution with Quality Gates

The Lead (current Claude session) executes a strict **serial workflow**: spawn → wait → verify → rework (if needed) → shutdown → next.

#### 2.4.1 Serial Execution Loop (Core Workflow)

```
┌──────────────────────────────────────────────────────────┐
│                    SERIAL EXECUTION LOOP                   │
│                                                          │
│  For each application in [superpoint, superglue,         │
│                           vit-base-16, vit-base-32]:       │
│                                                          │
│  1. SPAWN: Agent(team_name, name, isolation="worktree")  │
│     └─ Teammate runs Phase 0→5 autonomously              │
│                                                          │
│  2. WAIT: Teammate reports completion via SendMessage    │
│                                                          │
│  3. VERIFY: Lead dispatches opus subagent to check       │
│     ┌─ PASS ───────────────────────────────────┐        │
│     │  4a. SendMessage shutdown_request         │        │
│     │  5a. Wait for teammate to exit            │        │
│     │  6a. TaskUpdate(status="completed")       │        │
│     │  7a. Merge worktree (--no-ff)             │        │
│     │  8a. Move to NEXT application             │        │
│     └───────────────────────────────────────────┘        │
│     ┌─ FAIL ───────────────────────────────────┐        │
│     │  4b. SendMessage with specific fix list   │        │
│     │  5b. Teammate fixes issues                │        │
│     │  6b. Go to step 2 (WAIT for re-submit)   │        │
│     └───────────────────────────────────────────┘        │
│                                                          │
│                                                          │
│  Guardian runs alongside via cron (5-min tmux checks)     │
│  After ALL 4 apps PASS: Phase D + Phase E + cancel       │
│  Guardian cron (LAST action)                              │
└──────────────────────────────────────────────────────────┘
```

#### 2.4.2 Teammate Lifecycle

1. **Spawn**: Lead creates teammate via `Agent(team_name="rvfuse-analysis-b2", name="<name>", isolation="worktree", ...)`. The `team_name` parameter is what distinguishes a teammate from a standalone subagent.
2. **Claim**: Teammate reads `TaskList`, claims task via `TaskUpdate(owner="<its-name>")`
3. **Execute**: Teammate runs all pipeline phases (0→5) autonomously in sequence. Guardian monitors progress via tmux.
4. **Report**: Teammate sends completion message to Lead via `SendMessage(to="Lead")` after finishing Phase 5
5. **Verify**: Lead dispatches an **opus verification subagent** (standalone, no `team_name`) to check all deliverables
6. **Rework loop**: If verification FAILS, Lead sends fix instructions via `SendMessage(to="<teammate-name>")`, teammate fixes and re-reports → go to step 5
7. **Shutdown**: After PASS, Lead sends `shutdown_request` via `SendMessage`
8. **Merge**: Lead merges worktree back to master (`--no-ff`)
9. **Next**: Lead spawns the next teammate (go to step 1)
10. **Final Cleanup**: After all 4 teammates complete and shut down: run Phase E (synthesis) → `TeamDelete` → **cancel Guardian cron last**

#### 2.4.3 Stage-Gate Verification (Opus Subagent)

When a teammate reports full pipeline completion (Phase 0–5 done), Lead dispatches a **standalone opus verification subagent** — this is a standalone `Agent` WITHOUT `team_name`, because it's a fire-and-forget quality check, not a team member:

```python
# NOTE: no team_name, no name — this is a standalone subagent, NOT a teammate
# Using opus model for deep architectural reasoning during verification
Agent(
    description="Verify <teammate-name> Phase 0-5 deliverables",
    subagent_type="general-purpose",
    model="opus",
    mode="default",
    prompt="""
    Verify ALL deliverables for <teammate-name> in worktree <path>.

    CHECKLIST:

    Phase 0 — Setup:
    - [ ] ONNX model exists and is valid (check file size > 0, verify with python onnx.checker if possible)
    - [ ] C++ runner compiles and links successfully
    - [ ] ONNX Runtime cross-compiled for rv64gcv
    - [ ] QEMU smoke test: runner produces valid output (correct classification/keypoints/matches)
    - [ ] Sysroot extracted and functional

    Phase 2 — RVV512 Vectorization:
    - [ ] RVV patches exist for target operators (check file existence + non-empty)
    - [ ] Patches applied to ONNX Runtime build
    - [ ] Correctness verified: QEMU output matches vanilla (non-RVV) build output

    Phase 3 — BBV Profiling:
    - [ ] Each target operator has a non-empty .bb file (check: file size > 100 bytes)
    - [ ] Each target operator has a .disas file with valid RISC-V instructions
    - [ ] Function-scoped profiling used (not whole-program), verified by checking .bb file has function-specific data

    Phase 4 — Gap Analysis:
    - [ ] Each target operator has a gap analysis .md report
    - [ ] Each report includes cross-platform comparison (RVV vs x86/ARM/etc.)
    - [ ] Each report includes BBV-weighted benefit figures (整体收益)
    - [ ] Each report has a corresponding .pdf (via md2pdf)

    Phase 5 — Consolidated Report:
    - [ ] Consolidated .md report exists and merges all per-operator findings
    - [ ] Priority table with BBV-weighted scores is present
    - [ ] Consolidated .pdf exists (via md2pdf)
    - [ ] Report is self-consistent (no contradictions between per-operator and consolidated data)

    OVERALL:
    - [ ] No empty or placeholder files (check for files with only "TODO" or "TBD")
    - [ ] File paths match the expected directory structure
    - [ ] No obvious quality issues (e.g., BBV data with < 10 basic blocks, gap analysis without actual instruction comparison)

    Report: PASS/FAIL with detailed issue list for each failed item.
    If FAIL: provide SPECIFIC, actionable fix instructions for the teammate.
    """
)
```

**Why opus for verification?** Verification requires:
- Deep understanding of RISC-V vector ISA semantics to judge whether gap analysis findings are technically sound
- Cross-referencing across multiple output files (BBV data → gap analysis → consolidated report) for consistency
- Judging whether "empty BBV output" is a methodology error or a genuine result
- Identifying subtle quality issues like incomplete instruction coverage in disassembly analysis

These are architectural reasoning tasks that require opus-level capability. Sonnet is insufficient for reliable quality judgment.

**Why not a teammate for verification?** Verification is a one-shot, stateless check that returns a result and exits immediately. It doesn't need team membership, shared task list, or `SendMessage`. Using a standalone `Agent` (no `team_name`) for verification keeps the team roster clean and avoids spawning long-lived agents for short tasks.

#### 2.4.4 Rework Protocol

If verification returns FAIL:

1. Lead reads the verification subagent's detailed issue list
2. Lead sends `SendMessage(to="<teammate-name>")` with:
   - Clear statement: "Verification FAILED. Fix the following issues:"
   - Numbered, specific, actionable fix items (copied from verification output)
   - Instruction: "After fixing all issues, re-report completion via SendMessage."
3. Teammate fixes issues and sends new completion message
4. Lead dispatches a NEW verification subagent (fresh instance, same checklist)
5. Repeat until PASS

**Rework iteration limit**: If a teammate fails verification 3 times, Lead escalates — manually intervenes to fix the most critical issues, then re-verifies.

#### 2.4.5 Tmux Monitoring (Lightweight)

The Guardian provides continuous cron monitoring (see §2.6). In addition, the Lead checks the teammate's tmux pane **on demand** for direct intervention:

1. **During WAIT**: If teammate hasn't reported completion within expected time (see §11 timeline), check tmux to see if it's stuck
2. **Intervention**: If teammate is idle/blocked (prompt showing `❯` with no spinner), diagnose and provide input:
   - **Waiting for confirmation** → Send `y` or the appropriate response
   - **Encountered an error** → Send debug/fix instructions
   - **Stuck in a loop** → Send `Ctrl-C` then redirect
3. **Log**: Record interventions in `docs/plans/tmux-automation-log-b2.md`:
   ```markdown
   ### YYYY-MM-DD HH:MM - <teammate-name> <status>
   - **当前阶段**: ...
   - **检测到的问题**: ...
   - **决策**: ...
   - **采取的操作**: ...
   - **操作结果**: ...
   ```

### 2.5 Worker Types

All 4 workers in this batch are **Full Pipeline** (Phase 0–5): new applications with no existing data, requiring complete setup from ONNX export through to consolidated report.

| Type | Trigger | Phases | This Batch |
|------|---------|--------|------------|
| **Full Pipeline** | New application, no existing data | Phase 0 → 5 | All 4 apps |

### 2.6 Guardian Teammate — Long-Run Progress Monitor

The Guardian is a **persistent, lightweight teammate** (haiku model) that monitors all team members via tmux on a 5-minute cron loop. It intervenes only when the Lead is idle and the team is stuck, keeping the batch progressing during long unsupervised runs.

**The Guardian specification is defined in the `guardian` skill** (`skills/guardian/SKILL.md`). This includes the full intervention protocol, logging rules, fundamental rules (Leader-Only, Idle-Only), continuity notes mechanism, and spawn instructions.

Refer to `skills/guardian/SKILL.md` for the complete Guardian behavior specification. The spawn invocation for this batch is in §6 Step B1b below, using the skill-derived prompt with batch-specific parameters (team name, task count, log path, plan path).

## 3. Application Tasks

| # | Teammate Name | Application | Model | Type | Phases | Priority | Execution Order |
|---|---------------|-------------|-------|------|--------|----------|-----------------|
| 1 | `superpoint` | onnxrt/SuperPoint | sonnet | Full Pipeline | 0–5 | High | **1st** |
| 2 | `superglue` | onnxrt/SuperGlue | opus | Full Pipeline | 0–5 | Medium-High | **2nd** |
| 3 | `vit-base-16` | onnxrt/ViT-Base/16 | sonnet | Full Pipeline | 0–5 | High | **3rd** |
| 4 | `vit-base-32` | onnxrt/ViT-Base/32 | sonnet | Full Pipeline | 0–5 | Medium-High | **4th** |

**Execution order rationale**:
- `superpoint` runs FIRST: small model (~5 MB), fast ONNX Runtime build, establishes CNN operator patterns (Conv2d, ReLU, Softmax). Its SuperPoint outputs can optionally feed into SuperGlue for real-data testing.
- `superglue` runs SECOND: most complex (opus), depends on SuperPoint conceptually (feature matching pipeline) and can reference its code on master. Self-attention analysis will later be referenced by ViT apps.
- `vit-base-16` runs THIRD: large model (~350 MB), Transformer architecture. Refers to superglue's self-attention analysis on master. Establishes SGEMM/MatMul patterns that vit-base-32 reuses.
- `vit-base-32` runs LAST: heavy reuse from vit-base-16 (identical architecture, different shapes). Quickest execution since all patterns are established.

### 3.1 Task Details

#### 3.1.3 onnxrt/ViT-Base/16 — Full Pipeline Worker

**Application**: Vision Transformer (ViT) Base with 16×16 patches, image classification on ImageNet-1K.

**Architecture**:
- Patch embedding: `Conv2d(3, 768, k=16, s=16)` → flatten → (1, 196, 768)
- CLS token + position embedding: (1, 197, 768)
- 12× Transformer Encoder blocks:
  - Pre-norm LayerNorm → Multi-head Self-Attention (12 heads, head_dim=64)
  - QKV projection: `Linear(768, 2304)` — MatMul (197×768) × (768×2304)
  - Attention scores: `Softmax(Q×K^T/√d)` — MatMul (197×64) × (64×197) per head
  - Output projection: `Linear(768, 768)` — MatMul (197×64) × (64×768) per head
  - MLP: `Linear(768, 3072)` → GELU → `Linear(3072, 768)`
  - Residual adds after attention and MLP
- Final LayerNorm → Classification head: `Linear(768, 1000)`

**Hot Operator Profile** (estimated):
| Operator | % Compute | Shapes | RVV Vectorization Potential |
|----------|-----------|--------|-----------------------------|
| MatMul (QKV) | ~35% | (197,768)×(768,2304) | High — SGEMM kernel |
| MatMul (Attn QK) | ~15% | (12,197,64)×(12,64,197) | High — BatchMatMul |
| MatMul (Attn V) | ~15% | (12,197,197)×(12,197,64) | High — BatchMatMul |
| MatMul (Output Proj) | ~10% | (197,768)×(768,768) | High — SGEMM kernel |
| Linear (MLP up) | ~8% | (197,768)×(768,3072) | High — SGEMM kernel |
| Linear (MLP down) | ~8% | (197,3072)×(3072,768) | High — SGEMM kernel |
| LayerNorm | ~3% | (197,768) × 25 times | Medium — Reduce + broadcast |
| GELU | ~3% | (197,3072) × 12 times | Medium — Element-wise (has RVV patch) |
| Softmax | ~2% | (12,197,197) × 12 times | Medium — Reduce + exp + div |
| Conv2d (patch embed) | <1% | (1,3,224,224) single invocation | Low impact |
| Add (residual) | ~1% | (197,768) × 24 times | High — Element-wise |

**Phase 0 — Setup**:
1. ONNX model export from HuggingFace `google/vit-base-patch16-224`:
   ```python
   from transformers import ViTForImageClassification
   model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
   model.eval()
   torch.onnx.export(model, dummy_input, "vit_base_patch16_224.onnx",
                     input_names=["pixel_values"],
                     output_names=["logits"],
                     dynamic_axes={"pixel_values": {0: "batch"}})
   ```
2. C++ runner: `vit_runner.cpp`
   - Input: JPEG/PNG image file path
   - Preprocessing: resize to 224×224, normalize (mean=0.5, std=0.5), HWC→CHW
   - ONNX Runtime inference
   - Postprocessing: softmax, top-5 class indices + confidence scores
   - Image loading: use `stb_image.h` (header-only, no OpenCV dependency) or OpenCV if already available
3. Cross-compile ONNX Runtime + runner for rv64gcv (reuse `applications/onnxrt/ort/build.sh` pattern)
4. Sysroot extraction via Docker
5. QEMU smoke test: verify correct output (reasonable classification result)

**Target directory**: `applications/onnxrt/vit-base-16/`

**Key dependencies**: ONNX Runtime source (clone to `vendor/`), transformers Python package (for export only)

**Existing references**:
- YOLO runner: `applications/onnxrt/yolo/runner/yolo_runner.cpp` — C++ ONNX Runtime API pattern
- ONNX Runtime build: `applications/onnxrt/ort/build.sh` — cross-compile script
- SGEMM RVV patch: `applications/onnxrt/rvv-patches/sgemm-kernel-vl16/` — directly applicable to ViT MatMul operators
- GELU RVV patch: `applications/onnxrt/rvv-patches/quick-gelu/` — applicable to ViT GELU (note: ViT uses standard GELU, not QuickGELU; may need adaptation)
- SuperGlue self-attention analysis: `docs/report/superglue/` (already merged to master) — self-attention RVV patterns are directly reusable for ViT attention
- OSTrack ViT analysis: if Batch 1 `onnxrt-ostrack` has completed, its ViT-Base attention operator analysis is directly reusable

#### 3.1.4 onnxrt/ViT-Base/32 — Full Pipeline Worker

**Application**: Vision Transformer (ViT) Base with 32×32 patches. Same architecture as ViT-Base/16 but with larger patch size, resulting in fewer tokens and different MatMul shapes.

**Architecture differences from ViT-Base/16**:
- Patch embedding: `Conv2d(3, 768, k=32, s=32)` → (1, 49, 768)
- With CLS: (1, 50, 768)
- All subsequent MatMul shapes use sequence_length=50 instead of 197

**Shape Comparison** (key research dimension):
| Operator | ViT-Base/16 Shape | ViT-Base/32 Shape | VL=16 Efficiency |
|----------|-------------------|-------------------|------------------|
| QKV MatMul | (197,768)×(768,2304) | (50,768)×(768,2304) | 197%13≠0 → tail; 50%16≠0 → tail |
| Attention QK^T | (12,197,64)×(12,64,197) | (12,50,64)×(12,64,50) | Both need tail handling |
| MLP up | (197,768)×(768,3072) | (50,768)×(768,3072) | K=768 % 16 = 0 → aligned |
| MLP down | (197,3072)×(3072,768) | (50,3072)×(3072,768) | K=3072 % 16 = 0 → aligned |
| LayerNorm | (197,768) | (50,768) | Both 768 % VL elements |

**Research value**: The shape difference (197 vs 50 tokens) directly tests vectorization tail handling efficiency. Operators that are efficient at one shape may degrade at the other. This provides quantitative data for RVV extension proposals targeting dynamic shape handling (e.g., tail-agnostic instructions).

**Phase 0 — Setup**: Same process as ViT-Base/16, different HuggingFace model (`google/vit-base-patch32-384`). Note: `google/vit-base-patch32-224` does not exist on HuggingFace; the closest available model is `google/vit-base-patch32-384` (384×384 input, 12×12=144 patches + CLS = 145 tokens).

**Target directory**: `applications/onnxrt/vit-base-32/`

**Reuse from vit-base-16**:
- Runner code: identical structure, different model file path and input size handling
- ONNX Runtime build: identical (can reference already-merged vit-base-16 build)
- Gap analysis methodology: directly applicable, focus on shape-difference impact

#### 3.1.1 onnxrt/SuperPoint — Full Pipeline Worker

**Application**: SuperPoint — CNN-based interest point detection and descriptor extraction (Magic Leap, CVPR 2018).

**Source**: https://github.com/magicleap/SuperPointPretrainedNetwork

**Architecture**:
- Shared Encoder (VGG-style):
  - `Conv2d(1,64,k=3,s=1,p=1)` → ReLU
  - `Conv2d(64,64,k=3,s=1,p=1)` → ReLU → `MaxPool2d(2,2)`
  - `Conv2d(64,64,k=3,s=1,p=1)` → ReLU
  - `Conv2d(64,64,k=3,s=1,p=1)` → ReLU → `MaxPool2d(2,2)`
  - `Conv2d(64,128,k=3,s=1,p=1)` → ReLU
  - `Conv2d(128,128,k=3,s=1,p=1)` → ReLU → `MaxPool2d(2,2)`
  - `Conv2d(128,128,k=3,s=1,p=1)` → ReLU
  - `Conv2d(128,128,k=3,s=1,p=1)` → ReLU
  - Output: (B, 128, H/8, W/8)
- Interest Point Decoder:
  - `Conv2d(128,256,k=3,s=1,p=1)` → ReLU
  - `Conv2d(256,65,k=1,s=1)` → `Softmax(dim=1)` (65 channels = 8×8 grid cells + dustbin)
  - Reshape to (B, H, W) heatmap → NMS for keypoint extraction
- Descriptor Decoder:
  - `Conv2d(128,256,k=3,s=1,p=1)` → ReLU
  - `Conv2d(256,256,k=1,s=1)` → L2 normalize (descriptor dim=256)
  - Output: (B, 256, H/8, W/8)

**Hot Operator Profile** (estimated):
| Operator | % Compute | Dominant Pattern | RVV Potential |
|----------|-----------|-----------------|---------------|
| Conv2d (3×3, s=1) | ~70% | Im2Col + GEMM, 128→128 channels | High — Conv2D RVV |
| Conv2d (1×1) | ~10% | GEMM-like, 256→65 or 256→256 | High — MatMul |
| ReLU | ~5% | Element-wise | High — has RVV patterns |
| MaxPool2d (2×2) | ~3% | Reduction per 2×2 window | Medium |
| Softmax (channel) | ~5% | Exp + reduce + div, 65 channels | Medium |
| L2 Normalize | ~2% | Reduce sum + sqrt + div, 256-dim | Medium |
| NMS (postproc) | ~5% | Non-neural, in C++ runner | N/A (CPU) |

**Phase 0 — Setup**:
1. Clone Magic Leap SuperPoint repo
2. ONNX model export:
   ```python
   # Load pretrained SuperPoint model
   from superpoint_pytorch import SuperPointNet
   model = SuperPointNet()
   model.load_state_dict(torch.load("superpoint_v1.pth"))
   model.eval()
   # Export with fixed input size (e.g., 480×640 for VGA)
   torch.onnx.export(model, dummy_input_gray, "superpoint.onnx",
                     input_names=["image"],
                     output_names=["semi", "desc"],
                     dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}})
   ```
3. C++ runner: `superpoint_runner.cpp`
   - Input: grayscale image (convert from color if needed)
   - Preprocessing: resize, normalize to [0,1], HWC→CHW
   - ONNX Runtime inference → semi (heatmap) + desc (descriptors)
   - Postprocessing: softmax on semi, extract keypoints via NMS on heatmap, sample descriptors at keypoint locations
   - Output: keypoints (x, y, score) + descriptors per keypoint
4. Cross-compile ONNX Runtime + runner for rv64gcv
5. QEMU smoke test: verify keypoint count and descriptor shapes are reasonable

**Target directory**: `applications/onnxrt/superpoint/`

**Key considerations**:
- ONNX export verified: all operators (Conv2d, ReLU, MaxPool, Softmax, ReduceL2) are well-supported in ONNX opset 17
- NMS postprocessing is in C++ runner code, NOT in ONNX model — keeps model export simple
- Input size can be dynamic for ONNX export; fix to 480×640 for reproducible BBV profiling
- SuperPoint is a dependency for SuperGlue's real-image pipeline, but SuperGlue can use synthetic data for independent BBV analysis

**Existing references**:
- Conv2D analysis from YOLO/ResNet gap reports
- Element-wise operator RVV patches (ReLU, GELU) — can adapt for ReLU
- Softmax and reduction patterns from existing RVV patch collection

#### 3.1.2 onnxrt/SuperGlue — Full Pipeline Worker

**Application**: SuperGlue — Graph Neural Network for feature matching with optimal transport (Magic Leap, CVPR 2020).

**Source**: https://github.com/magicleap/SuperGluePretrainedNetwork

**Architecture**:
- Keypoint Encoder (per image):
  - `Linear(3, d_model)` — encode (x, y, score) of each keypoint → d_model=256
- 9× Attentional GNN Layers (alternating):
  - **Self-Attention** (within image A, then within image B):
    - QKV: `Linear(256, 256)` each → MatMul (N,256)×(256,256)
    - Attention: `Softmax(Q×K^T/√d)` — MatMul (N,256)×(256,N) per head
    - Output: MatMul(attn, V) + `Linear(256, 256)`
    - Residual add + LayerNorm
  - **Cross-Attention** (between image A and B):
    - Q from image A, K/V from image B (and vice versa)
    - Same MatMul pattern as self-attention but with cross-image K/V
    - This is the key operator NOT present in standard ViT
  - **MLP**:
    - `Linear(256, 512)` → ReLU → `Linear(512, 256)`
    - Residual add + LayerNorm
- Final Projection:
  - `Linear(256, 256)` → compute pairwise matching scores
- **Sinkhorn** (Optimal Transport):
  - Augment score matrix with dustbin row/column
  - Iterative row normalization → column normalization (typically 100 iterations)
  - Converts matching scores to soft assignment matrix
  - **Decision**: implement Sinkhorn in C++ runner (not in ONNX), since iterative ops are hard to export

**Hot Operator Profile** (estimated, N=max_keypoints=1024):
| Operator | % Compute | Shapes | RVV Potential | Note |
|----------|-----------|--------|---------------|------|
| MatMul (QKV) | ~35% | (N,256)×(256,768) per attention | High — SGEMM | 18 attention blocks |
| MatMul (Attn QK) | ~20% | (4,N,64)×(4,64,N) | High — BatchMatMul | Self + Cross |
| MatMul (Attn V) | ~15% | (4,N,N)×(4,N,64) | High — BatchMatMul | |
| Cross-Attn QK^T | ~10% | (4,Na,64)×(4,64,Nb) | High — BatchMatMul with different N | **Unique operator** |
| Linear (MLP) | ~10% | (N,256)×(256,512) etc. | High — SGEMM | |
| LayerNorm | ~3% | (N,256) × 27 times | Medium — Reduce | |
| ReLU | ~2% | (N,512) × 9 times | High — Element-wise | |
| Softmax | ~3% | (4,N,N) × 18 times | Medium | |
| Sinkhorn | ~2% | (Na+1,Nb+1) × 100 iter | Medium — Row/col reduction | C++ runner |

**Phase 0 — Setup**:
1. Clone Magic Leap SuperGlue repo
2. **ONNX model export** (GNN part only, no Sinkhorn):
   ```python
   # Export only the GNN and final projection
   # Sinkhorn is implemented in the C++ runner
   class SuperGlueONNX(torch.nn.Module):
       def __init__(self, superglue_model):
           super().__init__()
           self.kenc = superglue_model.kenc
           self.gnn = superglue_model.gnn  # 9 layers
           self.final_proj = superglue_model.final_proj

       def forward(self, kpts0, scores0, desc0, kpts1, scores1, desc1):
           # Keypoint encoding
           # GNN layers (self + cross attention)
           # Final projection
           return scores_matrix  # before Sinkhorn

   torch.onnx.export(onnx_model, (kpts0, scores0, desc0, kpts1, scores1, desc1),
                     "superglue_gnn.onnx",
                     input_names=["kpts0","scores0","desc0","kpts1","scores1","desc1"],
                     output_names=["scores_matrix"],
                     dynamic_axes={...})  # dynamic keypoint count
   ```
3. **Critical export consideration**: Keypoint count varies per image. For ONNX export:
   - Export with fixed N_max (e.g., 1024) and use padding/masking for fewer keypoints
   - OR export with dynamic axes for keypoint dimension
   - The attention masks handle variable keypoint counts — keypoints beyond actual count are masked out
   - Fixed N_max approach is simpler and sufficient for BBV profiling
4. C++ runner: `superglue_runner.cpp`
   - Input: keypoints + descriptors + scores from SuperPoint (or synthetic data file)
   - Preprocessing: normalize coordinates, pad/mask to N_max
   - ONNX Runtime inference → matching scores matrix
   - Postprocessing: Sinkhorn algorithm (100 iterations of row/col normalization in C++)
   - Output: matches (pairs of matched keypoint indices + confidence)
5. Cross-compile ONNX Runtime + runner for rv64gcv
6. QEMU smoke test: verify match count and confidence distribution

**Target directory**: `applications/onnxrt/superglue/`

**Key considerations**:
- **Cross-attention is the highest-value operator**: This is NOT present in any Batch 1 application (YOLO, OSTrack, llama.cpp). Cross-attention involves Q from one set and K/V from another, producing asymmetric attention patterns. The RVV gap analysis for cross-attention is novel.
- **Variable keypoint count**: BBV profiling should use fixed N=1024 (full keypoints) for worst-case analysis, but also consider N=256 (typical real scenario) for representative data.
- **Sinkhorn in C++**: Row/column normalization with exp — simple element-wise ops but iterated 100×. This is a memory-bound operation, not compute-bound. Include in BBV profiling for completeness but it won't dominate.
- **SuperPoint dependency**: For real end-to-end testing, SuperGlue needs SuperPoint outputs. However, for BBV profiling, synthetic random keypoints + descriptors are sufficient — operator performance is independent of input semantics. The C++ runner should support both modes.

**Existing references**:
- SuperPoint code (from teammate `superpoint`, already merged) — available on master for reference
- Self-attention has NO existing analysis in the project at this point (ViT-Base/16 runs later) — SuperGlue is the FIRST to analyze attention operators
- Cross-attention has NO existing analysis in the project — this is a novel operator family
- Sinkhorn has NO existing analysis — novel operator, though element-wise heavy and memory-bound

**Relationship to SuperPoint**:
- SuperGlue and SuperPoint form a feature matching pipeline: Image → SuperPoint → (keypoints, descriptors) → SuperGlue → matches
- For Phase 0–5 analysis, they run independently (serial, SuperPoint runs first):
  - SuperPoint analyzes CNN operators → reports in `docs/report/superpoint/`
  - SuperGlue analyzes GNN + attention + Sinkhorn operators → reports in `docs/report/superglue/`
  - Cross-referenced in the consolidated report
- The Lead's cross-app synthesis (Phase E) will combine both to describe the complete matching pipeline

## 4. Worktree Isolation

### 4.1 Shared Resources (Symlinked to Main Repo)

These are read-only and symlinked from each worktree to the main repository:

| Resource | Path | Size |
|----------|------|------|
| QEMU source + build | `third_party/qemu/` | ~680MB |
| LLVM source | `third_party/llvm-project/` | ~3.6GB |
| LLVM install | `third_party/llvm-install/` | ~1.5GB |
| RVV intrinsic doc | `third_party/riscv-rvv-intrinsic-doc/` | ~141MB |
| BBV plugin | `tools/bbv/libbbv.so` | ~22KB |
| Local LLVM tools | `tools/local-llvm/` | ~5KB |

### 4.2 Independent Resources (Per-Worktree)

| Resource | Path | Reason |
|----------|------|--------|
| Sysroot | `output/cross-ort/sysroot/` | Each worktree builds its own via Docker extraction |
| onnxrt source | `applications/onnxrt/<app>/vendor/` | Independent clone per application |
| Build output | `output/cross-<app>/` | Independent per application |
| ONNX models | `applications/onnxrt/<app>/model/` | Per-application model files |
| RVV patches | `applications/onnxrt/rvv-patches/` | Shared across onnxrt apps within same worktree; worktree-isolated from others |
| BBV profiling data | `output/bbv_<app>/` | Per-application (e.g., `output/bbv_superpoint/`, `output/bbv_superglue/`) |
| Reports | `docs/report/<app>/` | Per-application analysis |

### 4.3 ONNX Runtime Sharing Strategy

Since execution is serial (one worktree active at a time), worktree isolation is simpler. Each worktree builds its own ONNX Runtime from source in `vendor/`. The previous worktree is already merged back to master, so each new worktree starts from the updated master (including all previous app code).

### 4.4 Worktree Init Script

Each worker runs this at startup to establish toolchain links:

```bash
#!/usr/bin/env bash
set -euo pipefail

MAIN_REPO=$(git worktree list | grep -E 'master|main' | head -1 | awk '{print $1}')
WORKTREE=$(pwd)

# Shared toolchain symlinks
mkdir -p third_party
for sub in qemu llvm-project riscv-rvv-intrinsic-doc llvm-install; do
    [ -d "$MAIN_REPO/third_party/$sub" ] && ln -sfn "$MAIN_REPO/third_party/$sub" third_party/$sub
done

mkdir -p tools/bbv
[ -f "$MAIN_REPO/tools/bbv/libbbv.so" ] && ln -sfn "$MAIN_REPO/tools/bbv/libbbv.so" tools/bbv/libbbv.so
[ -d "$MAIN_REPO/tools/local-llvm" ] && ln -sfn "$MAIN_REPO/tools/local-llvm" tools/local-llvm

# Prepare application directory
mkdir -p "applications/onnxrt/${APP_NAME}/vendor"
mkdir -p "applications/onnxrt/${APP_NAME}/model"
mkdir -p "applications/onnxrt/${APP_NAME}/runner"

# Clone ONNX Runtime source (independent copy)
# The build.sh script handles this, or do it here
```

## 5. Serial Execution Constraints

Since only one teammate runs at a time on limited hardware, resource contention is minimal:

| Resource | Execution | Notes |
|----------|----------|-------|
| QEMU BBV | **Serial** | Only one BBV process at a time |
| LLVM toolchain | **N/A** | Symlinked, no rebuild needed |
| ONNX Runtime build | **Serial** | One build at a time; each worktree has independent `vendor/` |
| ONNX model export | **Pre-exported** | Lead pre-exports all 4 models before spawning any teammate (see Phase A) |
| Sysroot build | **Serial** | One Docker extraction at a time |
| Dev board (perf) | **Serial** | Single SSH managed by Lead; no contention in serial execution; pre/post checks per §5.1 |
| HuggingFace download | **Pre-downloaded** | Lead downloads all models in Phase A |
| Report output | **Serial** | One report generation at a time |
| Git merge | **Serial** | Each worktree merged before next teammate spawns |

**Key constraint**: ONNX model export requires PyTorch + transformers Python packages. Strategy:
- **Pre-export in main repo**: Lead exports all 4 ONNX models before spawning the first teammate, stores them in `output/models/`, and each teammate copies from there. Simplest and avoids Python dependency issues in worktrees.

### 5.1 RISC-V Dev Board Perf Profiling Protocol

Since execution is serial (one teammate active at a time), there is NO resource contention for the RISC-V dev board (Banana Pi). **Perf profiling on real hardware is the foundation of hotspot analysis — Phase 1 MUST NOT be skipped.** The serialized Lead manages dev board access, and each teammate includes Phase 1 in its pipeline.

**Dev Board Connection**:
- **Host**: `192.168.100.221` (Banana Pi K1, SpacemiT K1 SoC, Bianbu OS)
- **User**: `root`
- **SSH Port**: `22` (standard, NOT 2222)
- **Password**: `bianbu`
- **SSH**: `ssh root@192.168.100.221`
- **RVV capability**: `zvl256b` (VLEN=256) — binaries for native perf MUST be compiled with `-march=rv64gcv_zvl256b`
- **QEMU (for BBV profiling)**: `zvl512b` (VLEN=512) — binaries for BBV MUST be compiled with `-march=rv64gcv_zvl512b` and run with `qemu-riscv64 -cpu rv64,v=true,vlen=512`

**Build Target Split**:
| Use Case | VLEN | Compiler Flag | Runtime |
|----------|------|---------------|---------|
| Phase 1 (Perf on dev board) | 256 | `-march=rv64gcv_zvl256b` | Native on Banana Pi K1 |
| Phase 3 (BBV profiling) | 512 | `-march=rv64gcv_zvl512b` | QEMU with `-cpu rv64,v=true,vlen=512` |
| Phase 2 (RVV patches) | 512 | `-march=rv64gcv_zvl512b` | QEMU (VLEN=512) |

Phase 2 and Phase 4-5 analysis targets zvl512b, since the gap analysis and extension proposals are aimed at wider vector lengths. The dev board zvl256b perf data provides real-hardware validation but the reference for RVV optimization is zvl512b.

#### 5.1.1 Pre-Perf Board Check (Teammate, Before Each Perf Session)

Before uploading any binaries or running perf, the teammate MUST check the dev board state:

```bash
# 1. Check disk space (ensure >2GB available for binaries + models + perf data)
ssh root@192.168.100.221 "df -h /"

# 2. Check CPU load (should be near idle before starting)
ssh root@192.168.100.221 "uptime && top -bn1 | head -5"

# 3. Check for zombie perf processes
ssh root@192.168.100.221 "ps aux | grep -E 'perf|ort_runner|superpoint|superglue|vit' | grep -v grep"

# 4. Kill any leftover perf/data collection processes from previous runs
ssh root@192.168.100.221 "pkill -9 perf 2>/dev/null; pkill -9 ort_runner 2>/dev/null; pkill -9 superpoint 2>/dev/null; pkill -9 superglue 2>/dev/null; pkill -9 vit 2>/dev/null; echo 'Cleaned'"

# 5. If zombie processes cannot be killed → REBOOT
#    ssh root@192.168.100.221 "reboot"
#    Then wait 3-5 minutes before attempting reconnection:
#    sleep 180  # minimum 3 minutes
#    while ! ssh -o ConnectTimeout=5 root@192.168.100.221 "echo 'Board ready'"; do
#        echo "Waiting for board to come back..."
#        sleep 30
#    done
```

#### 5.1.2 Perf Execution (Teammate)

Use the `perf-profiling` skill to:
1. Upload cross-compiled binary, shared libraries, sysroot, and ONNX model to the dev board via `scp`/`paramiko`
2. Run `perf stat` for overall instruction-level metrics
3. Run `perf record` + `perf annotate` for hot function identification
4. Pull perf data back to the development machine

#### 5.1.3 Post-Perf Cleanup (Teammate, After Perf Completes)

```bash
# 1. Remove uploaded binaries, models, and libraries from the dev board
ssh root@192.168.100.221 "rm -rf /tmp/perf_workdir/"

# 2. Kill any perf/data collection processes still running
ssh root@192.168.100.221 "pkill -9 perf 2>/dev/null; pkill -9 ort_runner 2>/dev/null; echo 'Cleaned'"

# 3. Verify no leftover processes
ssh root@192.168.100.221 "ps aux | grep -E 'perf|ort_runner' | grep -v grep || echo 'No leftover processes'"
```

#### 5.1.4 Timeline Adjustment

Phase 1 adds **15-25 min** per application (board check + upload + perf run + cleanup), included in the updated timeline (§11).

## 6. Execution Plan

### Phase A: Pre-Export ONNX Models (Lead, One-Time)

Before spawning any teammate, the Lead pre-exports all 4 ONNX models to avoid Python dependency duplication:

```bash
# In main repo:
# 1. SuperPoint
python3 applications/onnxrt/superpoint/export_model.py
# 2. SuperGlue (GNN part only)
python3 applications/onnxrt/superglue/export_model.py
# 3. ViT-Base/16
python3 applications/onnxrt/vit-base-16/export_model.py
# 4. ViT-Base/32
python3 applications/onnxrt/vit-base-32/export_model.py

# Store in shared location
mkdir -p output/models/
cp applications/onnxrt/superpoint/model/*.onnx* output/models/
cp applications/onnxrt/superglue/model/*.onnx* output/models/
cp applications/onnxrt/vit-base-16/model/*.onnx* output/models/
cp applications/onnxrt/vit-base-32/model/*.onnx* output/models/
```

### Phase B: Serial Teammate Execution (Lead + Teammates)

**CRITICAL**: Only ONE teammate is active at any time. The Lead follows the strict serial loop defined in §2.4.1.

#### Step B1: Create the Team and Spawn Guardian (Lead, Once)

```python
# B1a: Create the team
TeamCreate(team_name="rvfuse-analysis-b2")

# B1b: Spawn Guardian FIRST — before any app teammate
# Uses the guardian skill (skills/guardian/SKILL.md) for the full behavior specification.
# Batch-specific parameters:
#   team_name       = "rvfuse-analysis-b2"
#   log_file        = "docs/plans/guardian-log-b2.md"
#   notes_file      = "/tmp/guardian-rvfuse-analysis-b2-notes.txt"
#   plan_file       = "docs/plans/agent-team-rvv-analysis-batch2-2026-04-27.md"
#   task_count      = 4
#   expected_end    = "all 4 app tasks completed, all worktrees merged, synthesis report done"
Skill(skill="guardian", args="team_name=rvfuse-analysis-b2 log_file=docs/plans/guardian-log-b2.md plan_file=docs/plans/agent-team-rvv-analysis-batch2-2026-04-27.md task_count=4")
```

The `guardian` skill handles: CronCreate setup, log file initialization, the full intervention protocol (Rules 1–7), continuity notes, and the SendMessage confirmation to the Lead.

**Guardian is now online.** It monitors all tmux panes independently. The Lead proceeds to spawn app teammates — the Guardian runs in parallel with everything.

#### Step B2: Execute Applications Serially

For each application in order (superpoint → superglue → vit-base-16 → vit-base-32), follow the **Serial Execution Loop** (§2.4.1). Detailed teammate prompts are in §6.1–§6.4 below. The Guardian runs alongside all app teammates.

**Application execution order and model assignment:**

| Order | Teammate | Model | Expected Duration |
|-------|----------|-------|-------------------|
| 1st | `superpoint` | sonnet | ~120-180 min |
| 2nd | `superglue` | opus | ~165-230 min |
| 3rd | `vit-base-16` | sonnet | ~130-190 min |
| 4th | `vit-base-32` | sonnet | ~85-125 min |

**After EACH teammate completes and passes verification:**
1. Send `shutdown_request` to the teammate via `SendMessage`
2. Wait for teammate to exit cleanly
3. Mark task as completed via `TaskUpdate(status="completed")`
4. Merge worktree to master with `--no-ff`:
   ```bash
   git merge --no-ff <worktree-branch>
   ```
5. Proceed to next application (go to B2 with next teammate)

### 6.3 Teammate Prompt: vit-base-16 (3rd, sonnet)

```python
Agent(
    team_name="rvfuse-analysis-b2",
    name="vit-base-16",
    subagent_type="general-purpose",
    description="ViT-Base/16 Full Pipeline (Phase 0-5)",
    isolation="worktree",
    model="sonnet",
    mode="auto",
    run_in_background=True,
    prompt="""You are the ViT-Base/16 pipeline worker in the rvfuse-analysis-b2 team.

Your task: Execute the full rvv512-optimization-pipeline for onnxrt/ViT-Base/16.

APPLICATION: Vision Transformer Base with 16x16 patches for ImageNet classification.
TARGET DIRECTORY: applications/onnxrt/vit-base-16/
ONNX MODEL: output/models/vit_base_patch16_224.onnx (pre-exported by Lead)
REFERENCE RUNNER: applications/onnxrt/yolo/runner/yolo_runner.cpp (ONNX Runtime API pattern)
REFERENCE BUILD: applications/onnxrt/ort/build.sh (cross-compile script)
REFERENCE RVV PATCHES: applications/onnxrt/rvv-patches/sgemm-kernel-vl16/ (SGEMM is directly applicable to ViT MatMul)
MODEL: sonnet
MODE: auto (execute without asking for permission)

IMPORTANT: You are the THIRD active teammate. SuperPoint and SuperGlue have completed and their code is merged to master. Reference their self-attention and CNN operator analyses in docs/report/.

PHASE 0 — SETUP (cross-compile-app skill):
1. Symlink toolchain from main repo (QEMU, LLVM, BBV plugin) — see worktree init script in plan §4.4
2. Copy pre-exported ONNX model from main repo: output/models/vit_base_patch16_224.onnx
3. Implement C++ runner: vit_runner.cpp
   - Use ONNX Runtime C++ API (Ort::Session, Ort::Run)
   - Input: image file path → resize 224x224, normalize (mean=0.5, std=0.5), HWC→CHW
   - Image loading: prefer stb_image.h (header-only) or OpenCV if available
   - Output: print top-5 class indices + confidence scores
   - Reference: applications/onnxrt/yolo/runner/yolo_runner.cpp for API usage
4. Clone ONNX Runtime source into applications/onnxrt/vit-base-16/vendor/
5. Cross-compile ONNX Runtime + runner using applications/onnxrt/ort/build.sh pattern
   - Toolchain: tools/local-llvm/ (symlinked)
   - Flags: rv64gcv, zvl512b
6. Build sysroot via Docker extraction
7. QEMU smoke test: verify runner produces valid top-5 output
8. Report phase completion in your status updates

PHASE 1 — PERF PROFILING (perf-profiling skill):
Since execution is serial, dev board access is no longer contended. Perf on real hardware IS the foundation of hotspot analysis.
1. Follow the **dev board perf protocol** in §5.1: pre-check disk/CPU/zombie processes, clean up before starting
2. Upload binary + libs + sysroot + model to the Banana Pi dev board
3. Run `perf stat` + `perf record` + `perf annotate` for hot function identification
4. Pull perf data back to the worktree
5. Post-perf cleanup per §5.1.3: remove uploaded resources, kill any lingering perf processes

PHASE 2 — RVV512 VECTORIZATION (rvv-op skill):
For operators estimated >1% compute time:
1. SGEMM (MatMul): Already has RVV patch at rvv-patches/sgemm-kernel-vl16/ — verify applicability to ViT MatMul shapes (K=768,2304,3072; N=197)
2. QuickGELU: Patch exists at rvv-patches/quick-gelu/ — note: ViT uses standard GELU, not QuickGELU. Create adapted RVV patch for standard GELU if needed.
3. Softmax: Reference superglue's Softmax analysis in docs/report/superglue/; review if existing reduction patterns apply
4. LayerNorm: Reference superglue's LayerNorm analysis in docs/report/superglue/; review reduction + broadcast patterns
5. Self-Attention: Reference superglue's self-attention RVV patterns in docs/report/superglue/
6. Apply all RVV patches to the ONNX Runtime build
7. Verify correctness: QEMU test with same image, compare output to vanilla
8. Report each operator's RVV implementation status

PHASE 3 — BBV PROFILING (qemu-bbv-usage skill):
CRITICAL: This MUST complete before Phase 4.
1. For each hot operator function, run function-scoped BBV profiling:
   - Identify function names/symbols in the ONNX Runtime binary
   - Run qemu-riscv64 with BBV plugin, targeting specific function address ranges
   - Produce .bb and .disas files
2. Target operators (priority order):
   a. SGEMM kernel (MatMul QKV, attention, MLP)
   b. GELU activation
   c. Softmax
   d. LayerNorm
3. Verify: each .bb file is non-empty, .disas has valid instructions
4. Report: operator → .bb file path → instruction count summary

PHASE 4 — GAP ANALYSIS (rvv-gap-analysis skill):
For each operator with BBV data:
1. Run cross-platform comparison: RVV vs x86 AVX/AVX2, ARM NEON/SVE, etc.
2. Use BBV data to compute weighted benefit scores
3. Identify missing RVV instructions and quantify impact
4. Cross-reference superglue's self-attention gap analysis in docs/report/superglue/
5. Generate per-operator gap analysis reports (.md)
6. Generate per-operator PDF reports (md2pdf skill)

PHASE 5 — CONSOLIDATED REPORT:
1. Merge all per-operator reports into a single consolidated document
2. Include priority table with BBV-weighted benefit scores
3. Include cross-reference to superglue's self-attention findings for Transformer attention comparison
4. Output: docs/report/vit-base-16/vit-base-16-consolidated-*.md + .pdf

AFTER ALL PHASES COMPLETE:
- Send completion message to Lead via SendMessage: "vit-base-16: All phases (0-5) complete. Ready for verification."
- Wait for Lead's verification result
- If Lead requests fixes, fix the specific issues and re-report completion
- If Lead sends shutdown_request, exit cleanly

IMPORTANT NOTES:
- Always invoke skills when entering each phase (cross-compile-app, rvv-op, qemu-bbv-usage, rvv-gap-analysis, md2pdf)
- Read CLAUDE.md and the plan at docs/plans/agent-team-rvv-analysis-batch2-2026-04-27.md for full context
- Use TaskUpdate to mark progress as you complete each phase
- SuperPoint and SuperGlue analyses are on master — reference them freely for CNN and attention patterns
"""
)
```

### 6.4 Teammate Prompt: vit-base-32 (4th, sonnet)

```python
Agent(
    team_name="rvfuse-analysis-b2",
    name="vit-base-32",
    subagent_type="general-purpose",
    description="ViT-Base/32 Full Pipeline (Phase 0-5)",
    isolation="worktree",
    model="sonnet",
    mode="auto",
    run_in_background=True,
    prompt="""You are the ViT-Base/32 pipeline worker in the rvfuse-analysis-b2 team.

Your task: Execute the full rvv512-optimization-pipeline for onnxrt/ViT-Base/32.

APPLICATION: Vision Transformer Base with 32x32 patches for ImageNet classification.
TARGET DIRECTORY: applications/onnxrt/vit-base-32/
ONNX MODEL: output/models/vit_base_patch32_384.onnx (pre-exported by Lead; google/vit-base-patch32-224 does not exist)
REFERENCE RUNNER: applications/onnxrt/yolo/runner/yolo_runner.cpp (ONNX Runtime API pattern)
REFERENCE BUILD: applications/onnxrt/ort/build.sh (cross-compile script)
REFERENCE RVV PATCHES: applications/onnxrt/rvv-patches/sgemm-kernel-vl16/ (directly applicable)
CRITICAL REFERENCE: applications/onnxrt/vit-base-16/ (already merged to master — same architecture, different shapes: 50 vs 197 tokens)
MODEL: sonnet
MODE: auto (execute without asking for permission)

IMPORTANT: You are the FOURTH and final active teammate. SuperPoint, SuperGlue, and ViT-Base/16 have all completed — their code and reports are on master.

KEY DIFFERENCE FROM ViT-Base/16:
- Patch size 32x32 → 7x7=49 patches + CLS = 50 tokens (vs 197 for patch16)
- All attention MatMul shapes use SeqLen=50 instead of SeqLen=197
- MLP MatMul shapes are identical (K=768, 3072)
- The shape difference is the primary research value: how does VL=16 vectorization efficiency change between SeqLen=197 and SeqLen=50?

PHASE 0 — SETUP:
Same as vit-base-16, but:
- Model: google/vit-base-patch32-384 (224 variant does not exist; 384x384 input, 145 tokens)
- Runner: reuse vit-base-16 runner code as base, change model file path and resize target (384×384)
- The runner is almost identical — only the model file path, input size (224→384), and token count differ
- Reference applications/onnxrt/vit-base-16/ for runner code (already on master)

PHASE 1 — PERF PROFILING (perf-profiling skill):
Since execution is serial, dev board access is no longer contended. Perf on real hardware IS the foundation of hotspot analysis.
1. Follow the **dev board perf protocol** in §5.1: pre-check disk/CPU/zombie processes, clean up before starting
2. Upload binary + libs + sysroot + model to the Banana Pi dev board
3. Run `perf stat` + `perf record` + `perf annotate` for hot function identification
4. Pull perf data back to the worktree
5. Post-perf cleanup per §5.1.3: remove uploaded resources, kill any lingering perf processes

PHASE 2 — RVV512 VECTORIZATION:
1. SGEMM: Apply existing patch from rvv-patches/sgemm-kernel-vl16/
   - KEY ANALYSIS: Compare efficiency at N=197 vs N=50
   - N=50 % VL16 = 2 (50 = 3×16 + 2 tail) — 2-element tail
   - N=197 % VL16 = 5 (197 = 12×16 + 5 tail) — 5-element tail
   - Document the tail handling overhead difference
2. GELU: Apply adapted GELU patch (same as vit-base-16)
3. Softmax/LayerNorm: Apply same patches as vit-base-16

PHASE 3 — BBV PROFILING:
Same operators as vit-base-16, but pay special attention to:
- Sequence-length-dependent operators (attention MatMul)
- Compare BBV instruction distributions with vit-base-16 results (in docs/report/vit-base-16/)
- Document shape sensitivity for each operator

PHASE 4 — GAP ANALYSIS:
Standard gap analysis per operator, PLUS:
- For each operator, compare with vit-base-16's gap analysis results
- Quantify the shape-sensitivity of each operator's RVV efficiency
- Identify operators where tail handling dominates at small SeqLen

PHASE 5 — CONSOLIDATED REPORT:
1. Standard consolidated report for ViT-Base/32
2. Cross-reference with ViT-Base/16 findings
3. Include a dedicated section on "Shape Sensitivity" — how VL efficiency varies with SeqLen

AFTER ALL PHASES:
- Send completion message to Lead via SendMessage: "vit-base-32: All phases (0-5) complete. Ready for verification."
- Wait for Lead's verification result
- If Lead requests fixes, fix the specific issues and re-report completion
- If Lead sends shutdown_request, exit cleanly

IMPORTANT:
- vit-base-16 code is already on master — reference it freely
- The shape comparison section in Phase 4-5 is the unique contribution of this worker
"""
)
```

### 6.1 Teammate Prompt: superpoint (1st, sonnet)

```python
Agent(
    team_name="rvfuse-analysis-b2",
    name="superpoint",
    subagent_type="general-purpose",
    description="SuperPoint Full Pipeline (Phase 0-5)",
    isolation="worktree",
    model="sonnet",
    mode="auto",
    run_in_background=True,
    prompt="""You are the SuperPoint pipeline worker in the rvfuse-analysis-b2 team.

Your task: Execute the full rvv512-optimization-pipeline for onnxrt/SuperPoint.

APPLICATION: SuperPoint — CNN-based interest point detection and descriptor extraction (Magic Leap, CVPR 2018).
SOURCE: https://github.com/magicleap/SuperPointPretrainedNetwork
TARGET DIRECTORY: applications/onnxrt/superpoint/
ONNX MODEL: output/models/superpoint.onnx (pre-exported by Lead)
REFERENCE RUNNER: applications/onnxrt/yolo/runner/yolo_runner.cpp (ONNX Runtime API pattern)
REFERENCE BUILD: applications/onnxrt/ort/build.sh (cross-compile script)
MODEL: sonnet
MODE: auto (execute without asking for permission)

IMPORTANT: You are the FIRST active teammate. No previous teammates have run — your work will establish patterns for subsequent teammates.

ARCHITECTURE SUMMARY:
- Shared VGG-style encoder: 8× Conv2d(3×3) + 3× MaxPool2d → (B,128,H/8,W/8)
- Interest Point Decoder: Conv2d → Softmax(65-dim) → NMS
- Descriptor Decoder: Conv2d → L2 Normalize(256-dim)
- Dominant operators: Conv2d (~70%), Softmax (~5%), ReLU (~5%)

PHASE 0 — SETUP (cross-compile-app skill):
1. Symlink toolchain from main repo (QEMU, LLVM, BBV plugin)
2. Copy pre-exported ONNX model: output/models/superpoint.onnx
3. Clone Magic Leap SuperPoint repo for reference (https://github.com/magicleap/SuperPointPretrainedNetwork)
   - This is just for code reference; the ONNX model is already exported
   - Clone to applications/onnxrt/superpoint/reference/
4. Implement C++ runner: superpoint_runner.cpp
   - Input: grayscale image file path
   - Preprocessing: convert to grayscale (if color), resize to 480x640, normalize to [0,1]
   - ONNX Runtime inference: two outputs (semi: heatmap, desc: descriptors)
   - Postprocessing:
     a. Softmax on semi (along channel dim, 65 classes)
     b. Reshape to 2D heatmap (H/8 × W/8)
     c. NMS on heatmap: local maxima within 4-pixel radius
     d. Extract keypoint coordinates + confidence scores
     e. Sample descriptors at keypoint locations from desc output
   - Output: print keypoint count, top-10 keypoint (x,y,score), descriptor statistics
5. Clone ONNX Runtime source into applications/onnxrt/superpoint/vendor/
6. Cross-compile ONNX Runtime + runner (same build pattern as YOLO)
7. QEMU smoke test: verify keypoint count > 0, descriptors have valid L2 norm (~1.0)
8. Report phase completion in status updates

PHASE 1 — PERF PROFILING (perf-profiling skill):
Since execution is serial, dev board access is no longer contended. Perf on real hardware IS the foundation of hotspot analysis.
1. Follow the **dev board perf protocol** in §5.1: pre-check disk/CPU/zombie processes, clean up before starting
2. Upload binary + libs + sysroot + model to the Banana Pi dev board
3. Run `perf stat` + `perf record` + `perf annotate` for hot function identification
4. Pull perf data back to the worktree
5. Post-perf cleanup per §5.1.3: remove uploaded resources, kill any lingering perf processes

PHASE 2 — RVV512 VECTORIZATION (rvv-op skill):
Focus on the dominant CNN operators:
1. Conv2d (3×3, stride=1): HIGHEST PRIORITY (~70% compute)
   - Im2Col + GEMM pattern
   - Check if existing ONNX Runtime Conv2d has RVV path; if not, create RVV patch
   - Key shapes: Cin=Cout=64 or 128, K=3, spatial=60×80 (H/8×W/8)
2. Conv2d (1×1): GEMM-like, lower priority (~10%)
3. ReLU: Element-wise, straightforward RVV implementation
4. Softmax (channel-wise, 65-dim): Reduction + exp + div
   - 65 is not aligned to VL=16 (65 = 4×16 + 1) — tail issue
5. L2 Normalize (256-dim): Reduction + sqrt + div
   - 256 = 16×16 — perfectly aligned with VL=16!
6. Apply RVV patches to ONNX Runtime build
7. Verify correctness: compare keypoint count with vanilla

PHASE 3 — BBV PROFILING (qemu-bbv-usage skill):
Target operators by priority:
1. Conv2d 3×3 kernel (Im2Col + GEMM path)
2. Conv2d 1×1 kernel
3. ReLU activation
4. Softmax (channel-wise, 65-dim)
5. L2 Normalize (256-dim)
Each must produce non-empty .bb + .disas files.

PHASE 4 — GAP ANALYSIS (rvv-gap-analysis skill):
Standard cross-platform analysis per operator with BBV data.
Key comparisons:
- Conv2d 3×3: RVV vs AVX2 Im2Col, NEON Im2Col, LSX/LASX
- Softmax: RVV vs NEON SVE reduction

PHASE 5 — CONSOLIDATED REPORT:
1. Merge per-operator reports
2. Priority table with BBV-weighted benefits
3. Output: docs/report/superpoint/superpoint-consolidated-*.md + .pdf

AFTER ALL PHASES:
- Send completion message to Lead via SendMessage: "superpoint: All phases (0-5) complete. Ready for verification."
- Wait for Lead's verification result
- If Lead requests fixes, fix the specific issues and re-report completion
- If Lead sends shutdown_request, exit cleanly

IMPORTANT:
- SuperPoint is the CNN representative in this batch. Focus on Conv2d operators as they complement the attention-heavy ViT analysis.
- NMS postprocessing is in C++ runner, not ONNX — include in BBV profiling if it's hot, but it likely won't be.
"""
)
```

### 6.2 Teammate Prompt: superglue (2nd, opus)

```python
Agent(
    team_name="rvfuse-analysis-b2",
    name="superglue",
    subagent_type="general-purpose",
    description="SuperGlue Full Pipeline (Phase 0-5)",
    isolation="worktree",
    model="opus",
    mode="auto",
    run_in_background=True,
    prompt="""You are the SuperGlue pipeline worker in the rvfuse-analysis-b2 team.

Your task: Execute the full rvv512-optimization-pipeline for onnxrt/SuperGlue.

APPLICATION: SuperGlue — GNN-based feature matching with optimal transport (Magic Leap, CVPR 2020).
SOURCE: https://github.com/magicleap/SuperGluePretrainedNetwork
TARGET DIRECTORY: applications/onnxrt/superglue/
ONNX MODEL: output/models/superglue_gnn.onnx (pre-exported by Lead, GNN part only without Sinkhorn)
REFERENCE RUNNER: applications/onnxrt/yolo/runner/yolo_runner.cpp (ONNX Runtime API pattern)
REFERENCE BUILD: applications/onnxrt/ort/build.sh (cross-compile script)
MODEL: opus
MODE: auto (execute without asking for permission)

IMPORTANT: You are the SECOND active teammate. SuperPoint has completed and its code is merged to master — reference it for CNN patterns and runner structure. No ViT analysis exists yet.

ARCHITECTURE SUMMARY:
- Keypoint Encoder: Linear(3→256) per keypoint (x, y, score)
- 9× GNN Layers (alternating self-attention and cross-attention):
  - Self-Attention: QKV from same image → MatMul(Softmax(QK^T/√d))V
  - Cross-Attention: Q from image A, K/V from image B (and vice versa)
  - MLP: Linear(256→512)→ReLU→Linear(512→256)
  - Residual + LayerNorm after each sub-layer
- Final Projection: Linear(256→256) → pairwise score matrix
- Sinkhorn: Iterative row/column normalization (100 iter) — implemented in C++ runner, NOT in ONNX

KEY RESEARCH VALUE:
- Cross-Attention is a NOVEL operator in this project — not present in YOLO, llama.cpp, or ViT self-attention
- Sinkhorn optimal transport is a unique algorithmic pattern (iterative normalization)
- GNN message passing pattern with alternating self/cross attention
- Self-attention analysis here will be the FIRST in Batch 2 — ViT apps will reference your results later

PHASE 0 — SETUP (cross-compile-app skill):
1. Symlink toolchain from main repo (QEMU, LLVM, BBV plugin)
2. Copy pre-exported ONNX model: output/models/superglue_gnn.onnx
3. Clone Magic Leap SuperGlue repo for reference (https://github.com/magicleap/SuperGluePretrainedNetwork)
   - Clone to applications/onnxrt/superglue/reference/
4. Implement C++ runner: superglue_runner.cpp
   - TWO INPUT MODES:
     a. Synthetic mode: generate random keypoints + descriptors + scores (N=1024 per image)
        - Use fixed random seed for reproducibility
        - This is the PRIMARY mode for BBV profiling (operator performance is independent of input semantics)
     b. File mode: load keypoints + descriptors + scores from file (for real-image testing)
   - Preprocessing: pad/mask keypoints to N_max, normalize coordinates
   - ONNX Runtime inference: output matching scores matrix (Na+1 × Nb+1)
   - Postprocessing: Sinkhorn algorithm in C++
     - Augment matrix with dustbin row/column (score = learnable parameter or fixed value)
     - Iterate 100 times: row softmax → column softmax
     - Extract matches: mutual argmax with confidence threshold
   - Output: match count, top-10 match (idx_a, idx_b, confidence)
5. Clone ONNX Runtime source into applications/onnxrt/superglue/vendor/
6. Cross-compile ONNX Runtime + runner (same build pattern as YOLO)
7. QEMU smoke test with synthetic data: verify match count > 0, confidence in [0,1]

PHASE 1 — PERF PROFILING (perf-profiling skill):
Since execution is serial, dev board access is no longer contended. Perf on real hardware IS the foundation of hotspot analysis.
1. Follow the **dev board perf protocol** in §5.1: pre-check disk/CPU/zombie processes, clean up before starting
2. Upload binary + libs + sysroot + model to the Banana Pi dev board
3. Run `perf stat` + `perf record` + `perf annotate` for hot function identification
4. Pull perf data back to the worktree
5. Post-perf cleanup per §5.1.3: remove uploaded resources, kill any lingering perf processes

PHASE 2 — RVV512 VECTORIZATION (rvv-op skill):
Priority-ordered operators:
1. SGEMM (MatMul in QKV projections): HIGHEST PRIORITY
   - Apply existing rvv-patches/sgemm-kernel-vl16/ patch
   - Shapes: (N,256)×(256,256) per projection, N=1024
2. BatchMatMul (Attention QK^T): Self-attention is same as ViT; Cross-attention is new
   - Cross-attention: Q from A (Na,256) × K from B (Nb,256)^T
   - Na and Nb can differ (different keypoint counts per image)
3. ReLU: Apply element-wise RVV pattern
4. LayerNorm: Apply reduction + broadcast RVV pattern
5. Softmax: Apply existing reduction pattern
6. Sinkhorn (C++ runner, not in ONNX):
   - Row/column normalization: exp + sum + div — element-wise heavy
   - 100 iterations × (Na+Nb) elements — memory-bound
   - Create RVV optimization for the normalization loop if profiling shows it's hot
7. Apply patches and verify correctness

PHASE 3 — BBV PROFILING (qemu-bbv-usage skill):
Target operators (priority order):
1. SGEMM kernel (QKV projections, MLP layers)
2. Self-Attention MatMul (QK^T, V projection) — this is the FIRST self-attention BBV in Batch 2, will be referenced by ViT apps later
3. Cross-Attention MatMul (QK^T with different N per image) — NOVEL
4. Softmax (attention weights)
5. LayerNorm (27 instances)
6. ReLU (9 MLP blocks)
7. Sinkhorn normalization loop (C++ runner, may be in separate binary)

CRITICAL: Cross-Attention BBV profiling is the highest-value contribution. Ensure the ONNX model's cross-attention operations are correctly identified and profiled.
Self-attention BBV data here sets the baseline for ViT-Base/16 comparisons later.

PHASE 4 — GAP ANALYSIS (rvv-gap-analysis skill):
Standard analysis per operator, PLUS:
- Cross-Attention: Compare RVV cross-attention vs NEON SVE cross-attention, AVX2, etc.
  - This is a NOVEL operator family in the project
  - Key challenge: asymmetric sequence lengths (Na ≠ Nb)
  - Document instructions that would help with variable-length K/V access
- Sinkhorn: Compare iterative normalization across platforms
  - Primarily memory-bound; document the RVV memory access pattern efficiency
- Self-Attention: This is the baseline analysis for Batch 2 — ViT apps will cross-reference it

PHASE 5 — CONSOLIDATED REPORT:
1. Merge per-operator reports
2. Highlight novel findings: Cross-Attention and Sinkhorn
3. Priority table with BBV-weighted benefits
4. Output: docs/report/superglue/superglue-consolidated-*.md + .pdf

AFTER ALL PHASES:
- Send completion message to Lead via SendMessage: "superglue: All phases (0-5) complete. Ready for verification."
- Wait for Lead's verification result
- If Lead requests fixes, fix the specific issues and re-report completion
- If Lead sends shutdown_request, exit cleanly

IMPORTANT NOTES:
- Cross-Attention is your unique contribution — invest extra analysis effort here
- Sinkhorn is in C++ runner, not ONNX — BBV profiling is on the runner binary, not the ONNX model
- Use synthetic data for BBV profiling; it's valid because operator performance is independent of input semantics
- SuperPoint code is on master — reference its runner pattern and CNN operator analysis
- Your self-attention analysis will be the Batch 2 baseline — document it thoroughly for ViT apps to reference
"""
)
```

### Phase C: Task Creation

After spawning each teammate, create its corresponding task in the shared task list:

```python
# Create all 4 tasks upfront (tasks track work, teammates execute it)
TaskCreate(
    subject="SuperPoint: Full Pipeline (Phase 0-5)",
    description="onnxrt/SuperPoint: VGG-style CNN for interest point detection + descriptor extraction. Conv2d-dominant (~70% compute). Magic Leap CVPR 2018."
)
TaskCreate(
    subject="SuperGlue: Full Pipeline (Phase 0-5)",
    description="onnxrt/SuperGlue: GNN feature matching with Cross-Attention + Sinkhorn optimal transport. Cross-Attention is a NOVEL operator family for this project. Magic Leap CVPR 2020."
)
TaskCreate(
    subject="ViT-Base/16: Full Pipeline (Phase 0-5)",
    description="onnxrt/ViT-Base/16: ONNX export + C++ runner + ORT build + BBV profiling + gap analysis + consolidated report. 12 Transformer layers, MatMul-dominant."
)
TaskCreate(
    subject="ViT-Base/32: Full Pipeline (Phase 0-5)",
    description="onnxrt/ViT-Base/32: Same architecture as ViT-Base/16 but patch_size=32 (50 tokens vs 197). Shape sensitivity analysis is key contribution."
)
```

### Phase D: Merge, Synthesis & Cleanup

After ALL 4 teammates have passed verification, been shut down, and their worktrees merged:

1. All worktrees already merged during the serial loop (step B2.4)
2. Generate cross-app synthesis report (Phase E)
3. `TeamDelete` to clean up team resources
4. **LAST ACTION — Cancel Guardian cron**: `CronDelete(id="<guardian-cron-id>")`
   - This MUST be the final action. The Guardian is the last thing stopped.
   - After this, the batch is fully complete. No processes remain.
   - Rationale: Guardian cancellation is irreversible in-session (no re-spawn without Lead). If cancelled early and the Lead gets stuck, no one can wake the Lead.

### Phase E: Cross-Application Synthesis Report (Lead)

Generate a synthesis report combining all 4 apps + cross-referencing Batch 1 findings:

```
docs/report/
├── onnxrt/                         # SuperPoint + SuperGlue reports
│   ├── rvv-gap-analysis-superpoint-*.md
│   └── superglue/                  # Per-operator + consolidated
├── vit-base-16/                    # Teammate output
│   └── vit-base-16-consolidated-*.md
├── vit-base-32/                    # Teammate output
│   └── vit-base-32-consolidated-*.md
└── rvv-extension-comprehensive-analysis-*.md  # Cross-app synthesis (Lead)
```

**Synthesis report key sections**:
1. **Operator Coverage Matrix**: operators × applications matrix showing which operators appear in which app
2. **Cross-App Hotspot Aggregation**: BBV-weighted priority across all 8 apps (Batch 1 + Batch 2)
3. **Novel Operator Deep-Dive**: Cross-Attention (SuperGlue), Sinkhorn (SuperGlue), Conv2d Im2Col (SuperPoint)
4. **Shape Sensitivity Analysis**: ViT-Base/16 vs /32 MatMul efficiency vs sequence length
5. **RVV Extension Priority Table**: Updated with Batch 2 data
6. **Updated Comprehensive PDF**: via md2pdf skill

## 7. Skill Reference

| Phase | Skill | Purpose |
|-------|-------|---------|
| 0 | `cross-compile-app` | App setup, ONNX Runtime cross-compile, sysroot, runner build |
| 1 | `perf-profiling` | Dev board perf profiling (required — foundation of hotspot analysis; pre/post checks per §5.1) |
| 2 | `rvv-op` | RVV512 operator implementation (SGEMM, Conv2d, Softmax, etc.) |
| 3 | `qemu-bbv-usage` | Function-scoped BBV profiling via QEMU |
| 4 | `rvv-gap-analysis` | Cross-platform comparison + BBV-weighted benefits |
| 5 | `md2pdf` | Report generation (per-operator + consolidated) |

Each worker invokes the relevant skill when entering each phase. The `rvv512-optimization-pipeline` skill orchestrates the full sequence.

## 8. Operator Coverage Matrix (Post-Analysis)

This matrix shows which operators are analyzed by which teammate. It guides the cross-app synthesis report.

| Operator Family | vit-base-16 | vit-base-32 | superpoint | superglue | Batch 1 Coverage |
|-----------------|-------------|-------------|------------|-----------|-------------------|
| MatMul / SGEMM | ● | ● | | ● | YOLO, llama.cpp |
| Self-Attention | ● | ● | | ● | OSTrack (ViT) |
| Cross-Attention | | | | ● | — NONE — |
| Softmax | ● | ● | ● | ● | YOLO |
| LayerNorm | ● | ● | | ● | — NONE — |
| GELU | ● | ● | | | YOLO (QuickGELU) |
| Conv2d (3×3) | | | ● | | ResNet |
| Conv2d (1×1) | | | ● | | ResNet |
| ReLU | | | ● | ● | — NONE — |
| L2 Normalize | | | ● | | — NONE — |
| Sinkhorn | | | | ● | — NONE — |
| MaxPool2d | | | ● | | ResNet |
| BN / GroupNorm | | | ● | | ResNet |

**Novel operators** (not in Batch 1): Cross-Attention, LayerNorm, ReLU, L2 Normalize, Sinkhorn.

## 9. Acceptance Criteria

Per teammate:
- [ ] Task claimed via `TaskUpdate(owner="<teammate-name>")` and status updated through lifecycle
- [ ] Phase 0: ONNX model runs under QEMU with correct output (smoke test passes)
- [ ] Phase 0: C++ runner produces valid results (classification top-5, keypoint detection, or feature matching)
- [ ] Phase 2: RVV patches applied and correctness verified (output matches vanilla)
- [ ] Phase 3: All target operators have non-empty BBV data (`.bb` + `.disas`)
- [ ] Phase 4: All target operators have gap analysis reports (MD + PDF)
- [ ] Phase 4: Reports contain BBV-weighted overall benefit figures (整体收益)
- [ ] Phase 5: Consolidated report merges findings with priority table
- [ ] Opus verification subagent returns PASS for all checklist items
- [ ] Worktree merges cleanly to master (no conflicts)
- [ ] Teammate responds to `shutdown_request` and exits cleanly

Guardian:
- [ ] Guardian spawned after TeamCreate, before first app teammate (via guardian SKILL)
- [ ] Guardian cron loop active (verified by `CronList` showing active job)
- [ ] Guardian notes file exists at `/tmp/guardian-rvfuse-analysis-b2-notes.txt`
- [ ] Guardian log at `docs/plans/guardian-log-b2.md` records interventions when executed (Rules 1-5); empty or sparse log is expected — normal ticks are not logged
- [ ] Guardian intervenes appropriately when team members are idle/stuck
- [ ] Guardian cron cancelled by Lead as the LAST action (after TeamDelete + synthesis)

Cross-batch:
- [ ] Cross-application synthesis report generated (Lead, Phase E)
- [ ] Operator coverage matrix updated with Batch 2 data
- [ ] RVV extension priority table updated
- [ ] Comprehensive analysis PDF regenerated

## 10. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SuperGlue ONNX export fails (Sinkhorn dynamic control flow) | Medium | High | Export GNN-only model (no Sinkhorn); implement Sinkhorn in C++ runner. This is the planned approach — risk is that even the GNN part has dynamic shapes from variable keypoint count. Fallback: export with fixed N_max=1024 and use masking. |
| Conv2d Im2Col not hot in SuperPoint | Low | Medium | Even if Conv2d is well-optimized by ORT's existing paths, the BBV profiling still provides valuable instruction-level data for gap analysis. The analysis focus shifts from "what to optimize" to "why is it already optimized." |
| ViT-Base/32 too similar to ViT-Base/16 | Medium | Low | The shape difference (50 vs 197 tokens) provides genuine research value even if operators are identical. The shape sensitivity analysis is the product, not the per-operator findings. |
| ONNX Runtime build failures | Low | High | The build.sh pattern is battle-tested (YOLO, ResNet). ViT uses standard ONNX ops — no custom ops needed. SuperPoint and SuperGlue also use standard ops. |
| Worktree merge conflicts | Low | Medium | Each app writes to isolated paths (applications/onnxrt/<app>/, docs/report/<app>/). No shared files. The only conflict surface is rvv-patches/ if multiple apps modify the same patch — but each app's patches are independent. Serial execution further reduces this risk since each worktree is merged before the next is created. |
| Teammate fails verification repeatedly | Medium | Medium | Rework limit of 3 iterations. If teammate cannot pass after 3 attempts, Lead manually intervenes to fix critical issues, then re-verifies. Serial execution means this blocks progress, but ensures quality. |
| Serial execution takes too long | Medium | Medium | Serial wall-clock time is ~10-14 hours total (sum of all 4 apps, including Phase 1 perf). Acceptable trade-off for device performance constraints and quality assurance. If needed, vit-base-32 can be deprioritized (lowest unique value). |
| Guardian malfunctions (false idle detection, over-intervention) | Low | Medium | Guardian intervention rules are priority-ordered with explicit conditions. False positives (e.g., intervening when Lead is actively thinking) are minimized by checking for the idle prompt pattern (❯ with no spinner). Guardian log provides audit trail — Lead can review and override any incorrect intervention. If Guardian becomes disruptive, Lead can temporarily pause its cron and resume later. |
| Guardian cron expires (7-day recurring limit) | Low | Low | Batch runs ~10-14 hours, well within the 7-day auto-expiry. Not a concern for this batch. |
| Lead or Guardian session terminated (power loss, crash) | Low | High | Guardian cron is `durable=False` (session-only). If Lead's session dies, Guardian dies with it. Worktrees persist on disk. Recovery: re-create team, re-spawn Guardian, resume from last completed app. Guardian log provides last-known state. |

## 11. Timeline Estimate (Serial Execution)

| Phase | superpoint (1st) | superglue (2nd) | vit-base-16 (3rd) | vit-base-32 (4th) |
|-------|-----------------|-----------------|-------------------|-------------------|
| 0 (Setup) | 30-45 min | 45-60 min (complex) | 30-45 min | 15-20 min (reuse) |
| 1 (Perf) | 15-25 min | 15-25 min | 15-25 min | 15-25 min |
| 2 (RVV) | 20-30 min | 30-45 min | 20-30 min | 10-15 min |
| 3 (BBV) | 20-30 min | 25-35 min | 20-30 min | 15-20 min |
| 4 (Gap) | 25-35 min | 35-50 min | 30-45 min | 20-30 min |
| 5 (Report) | 10-15 min | 10-15 min | 10-15 min | 10-15 min |
| **Per-app Total** | **~120-180 min** | **~165-230 min** | **~130-190 min** | **~85-125 min** |

| Overhead | Estimate |
|----------|----------|
| Verification (opus subagent per app) | ~5-10 min each |
| Rework buffer (1 round per app avg) | ~15-30 min each |
| Worktree merge + next spawn | ~5 min each |
| **Total wall-clock (serial)** | **~9-13 hours** |
| Cross-app synthesis (Lead, Phase E) | ~30-45 min |
| **Grand Total** | **~10-14 hours** |

**Guardian overhead**: Negligible. The Guardian runs a lightweight cron check every 5 minutes (tmux state inspection + log append). It does not consume significant CPU or memory, and the cron fires are sub-second operations. No adjustment to the timeline estimates is needed.

**Comparison with parallel execution**: Parallel would be ~2-3.5 hours wall-clock but requires 4× the device resources. Serial execution trades wall-clock time for device performance compatibility and built-in quality gates.
