# Agent Team Design: Cross-Application RVV Analysis

**Date**: 2026-04-26 | **Status**: Active | **Team**: rvfuse-analysis

> **阅读本文档的 AI Agent 注意**：
>
> 本文档描述的是**团队协作模型**（Team Model）。所有 Worker 必须通过 `TeamCreate` + `Agent(team_name="rvfuse-analysis", name="...")` 创建为**队友（teammate）**。
>
> **禁止**将 Worker 误解为普通 subagent（不带 `team_name` 的 `Agent` 调用）或 Task（`TaskCreate` 创建的工作项）。Task 只是任务跟踪条目，不是执行者。
>
> Teammate = `Agent(team_name=..., name=...)`。Subagent = `Agent(description=...)`。二者完全不同。
>
> 如果阅读本文档后，你首先想到的是 `TaskCreate` 而不是 `TeamCreate`，或者首先想到的是 `Agent(description=...)` 而不是 `Agent(team_name=..., name=...)`，说明你理解错了。重新阅读 §2.1。

## 1. Objective

Run the `rvv512-optimization-pipeline` across multiple applications in parallel. Each application is assigned to an isolated Worker operating in a git worktree, sharing read-only toolchain via symlinks but maintaining independent build artifacts and sysroot.

## 2. Team Structure

### 2.1 Team Creation — CRITICAL: Teammates vs Subagents

The `rvfuse-analysis` team uses the **team model**: teammates are spawned with `TeamCreate` + `Agent(team_name=...)`. This is fundamentally different from the standalone subagent model.

| Model | Creation | Membership | Communication | Purpose |
|-------|----------|------------|---------------|---------|
| **Team Agent (teammate)** | `Agent(team_name="...", name="...")` | Belongs to team, appears in team config | `SendMessage(to="name")` | Long-running parallel worker |
| **Standalone Agent (subagent)** | `Agent()` or `Agent(subagent_type="...")` | No team membership | Result returned to caller | Fire-and-forget research/code task |

**DO NOT use standalone subagents as teammates.** A subagent without `team_name` is invisible to the team, can't receive `SendMessage`, and can't coordinate via the shared TaskList. Teammates are spawned ONLY with `Agent(team_name="rvfuse-analysis", name="<name>", ...)`.

**Correct pattern for spawning a teammate:**

```python
# Step 1: Create the team (done once by Lead)
TeamCreate(team_name="rvfuse-analysis")

# Step 2: Spawn a teammate — NOTE: team_name is REQUIRED, name is REQUIRED
Agent(
    team_name="rvfuse-analysis",  # ← THIS makes it a teammate, not a subagent
    name="llama-cpp",             # ← human-readable name for SendMessage(to="llama-cpp")
    description="llama.cpp BBV + Gap Analysis",
    subagent_type="general-purpose",
    isolation="worktree",
    model="opus",
    mode="auto",
    run_in_background=True,
    prompt="""..."""
)

# Step 3: Create tasks AFTER spawning (tasks go into the shared team task list)
TaskCreate(subject="llama.cpp: BBV + Gap Analysis (Phase 3-5)", ...)
```

**Anti-patterns — NEVER do these:**

```python
# WRONG: subagent without team_name — NOT a teammate!
Agent(description="llama.cpp BBV", prompt="...")

# WRONG: TaskCreate assumed to be a teammate — tasks are work items, not workers!
TaskCreate(subject="llama.cpp: BBV...")  # This creates a task, not a worker
Agent(description="do the task")         # This is a subagent, not a teammate
```

### 2.2 Roles

| Role | Name | Count | Responsibility |
|------|------|-------|---------------|
| **Lead** | (current session) | 1 | `TeamCreate`, spawn teammates via `Agent(team_name="rvfuse-analysis", name=...)`, monitor via `SendMessage`, worktree merge, cross-app report |
| **Teammate** | per-application name | N (4 initially) | Execute pipeline for one application in an isolated worktree. Spawned as `Agent(team_name="rvfuse-analysis", name="<app>")` — NEVER as a standalone subagent. |

**Important**: `TaskCreate` does NOT create workers. Tasks are work-tracking items in the shared task list; they describe what needs to be done, not who does it. Teammates are the actual workers, spawned via `Agent` with `team_name`.

### 2.3 Model Selection

Choose between `opus` and `sonnet` per teammate based on task complexity:

| Teammate | Model | Rationale |
|----------|-------|-----------|
| `llama-cpp` | **opus** | Gap analysis requires deep cross-platform reasoning |
| `onnxrt-ostrack` | **opus** | Full pipeline: ONNX export, runner design, gap analysis — high complexity |
| `onnxrt-yolo-fp32` | **sonnet** | Supplement task: BBV + Gap for operators that already have INT8 analysis as reference |
| `pytorch-scout` | **sonnet** | Research-only task: web search + synthesis, no code generation |

**Guidelines**:
- **opus**: Full pipeline workers, complex gap analysis, tasks requiring architectural decisions
- **sonnet**: Supplement workers with clear reference data, research-only scouts, BBV profiling execution

### 2.4 Lead Responsibilities

The Lead (current Claude session) is NOT idle while teammates work. It actively monitors, verifies, and drives progress.

#### 2.4.1 Cron-Based Monitoring

Lead creates a recurring cron job via `CronCreate` to periodically check teammate status:

```python
CronCreate(
    cron="*/5 * * * *",           # every 5 minutes
    prompt="Check the status of all rvfuse-analysis teammates. "
           "For each teammate: read ~/.claude/teams/rvfuse-analysis/config.json "
           "to get member info, then check TaskList for task progress. "
           "If any teammate is idle or blocked, use tmux to inspect its terminal "
           "and provide appropriate input to unblock or advance the task.",
    recurring=True,
    durable=False                  # session-only, dies when Lead exits
)
```

#### 2.4.2 Tmux Monitoring Protocol

Lead monitors each teammate's terminal via tmux panes:

1. **Detect idle state**: Teammate prompt showing `❯` with no spinner means idle/blocked
2. **Diagnose**: Read the teammate's last output to understand why it stopped
3. **Intervene**: Send appropriate text input to the teammate's tmux pane:
   - **Waiting for confirmation** → Send `y` or the appropriate response
   - **Encountered an error** → Send debug/fix instructions
   - **Completed a phase** → Send instructions to proceed to next phase
   - **Stuck in a loop** → Send `Ctrl-C` then redirect

4. **Log**: Record interventions in `docs/plans/tmux-automation-log.md`:
   ```markdown
   ### YYYY-MM-DD HH:MM - <teammate-name> <status>
   - **当前阶段**: ...
   - **检测到的问题**: ...
   - **决策**: ...
   - **采取的操作**: ...
   - **操作结果**: ...
   ```

#### 2.4.3 Stage-Gate Verification

When a teammate reports phase completion (via `SendMessage` or `TaskUpdate`), Lead dispatches a **lightweight verification subagent** — this is a standalone `Agent` WITHOUT `team_name`, because it's a fire-and-forget check, not a team member:

```python
# NOTE: no team_name, no name — this is a standalone subagent, NOT a teammate
Agent(
    description="Verify llama-cpp Phase 3 results",
    subagent_type="general-purpose",
    mode="default",
    prompt="""
    Verify the Phase 3 results for <teammate-name> in worktree <path>.
    ...checklist...
    Report: PASS/FAIL with details.
    """
)
```

**Why not a teammate for verification?** Verification is a one-shot, stateless check that returns a result and exits immediately. It doesn't need team membership, shared task list, or `SendMessage`. Using a standalone `Agent` (no `team_name`) for verification keeps the team roster clean and avoids spawning long-lived agents for short tasks.

Based on verification result:
- **PASS** → Send `SendMessage` to teammate: "Phase N verified. Proceed to Phase N+1." or start shutdown
- **FAIL** → Send `SendMessage` to teammate with specific issues to fix, e.g. "Phase 3 verification failed: operator X has empty BBV output. Re-run BBV with corrected function offset."

#### 2.4.4 Teammate Lifecycle

1. **Spawn**: Lead creates teammate via `Agent(team_name="rvfuse-analysis", name="<name>", isolation="worktree", ...)`. The `team_name` parameter is what distinguishes a teammate from a standalone subagent.
2. **Claim**: Teammate reads `TaskList`, claims task via `TaskUpdate(owner="<its-name>")`
3. **Execute**: Teammate runs pipeline phases, updates task status
4. **Monitor**: Lead's cron job checks teammate status every 5 minutes via tmux
5. **Verify**: On each phase completion, Lead dispatches a standalone verification subagent (no `team_name`)
6. **Advance**: Lead sends next-phase instructions or fix requests via `SendMessage(to="<teammate-name>")`
7. **Shutdown**: After final phase passes verification, Lead sends `shutdown_request` via `SendMessage`
8. **Merge**: Lead merges worktree back to master (`--no-ff`)
9. **Cleanup**: `TeamDelete` after all teammates shut down

### 2.5 Worker Types

| Type | Trigger | Phases | Example |
|------|---------|--------|---------|
| **Full Pipeline** | New application, no existing data | Phase 0 → 5 | OSTrack |
| **Supplement** | Partial data exists, needs completion | Specified phases only | YOLO FP32, llama.cpp |
| **Scout** | Feasibility unknown | Research only, no code | PyTorch |

## 3. Application Tasks

| # | Teammate Name | Application | Model | Type | Phases | Priority |
|---|---------------|-------------|-------|------|--------|----------|
| 1 | `llama-cpp` | llama.cpp | opus | Supplement | 3–5 | High |
| 2 | `onnxrt-ostrack` | onnxrt/ostrack | opus | Full Pipeline | 0–5 | Medium-High |
| 3 | `onnxrt-yolo-fp32` | onnxrt/yolo (FP32) | sonnet | Supplement | 3–4 | High |
| 4 | `pytorch-scout` | pytorch | sonnet | Scout | Research | Medium |

### 3.1 Task Details

#### onnxrt/yolo (FP32) — Supplement Worker

**Status**: INT8 pipeline complete (7 operators). FP32 path only has SGEMM analyzed.

**What's needed**:
- Phase 3: BBV profiling for FP32-path operators that have RVV patches (Logistic, ReduceMinMax, QuantizeLinear — these are shared between FP32/INT8 paths but need FP32-specific BBV data)
- Phase 4: Gap Analysis with FP32 BBV data
- Update existing consolidated report

**Existing resources**:
- Perf data: `applications/onnxrt/yolo/data/perf/`
- RVV patches: `applications/onnxrt/rvv-patches/` (7 operators)
- Cross-compiled binary: `output/cross-ort/`

#### llama.cpp — Supplement Worker

**Status**: 5 RVV patches implemented with standalone tests passing. No BBV or Gap Analysis data.

**What's needed**:
- Phase 3: BBV profiling for all 5 operators
  - `gemm-q4_K-8x4-q8_K` (GEMM, 52.9% of matrix ops)
  - `gemv-q4_K-8x8-q8_K` (GEMV, 32.1% of core compute)
  - `quantize-q8_0-4x4` (Quantize, 49.8% of core)
  - `vec-dot-q5_0-q8_0` (Dot product, Q5_0 path)
  - `cmake-vlen-config` (build tool, skip profiling)
- Phase 4: Gap Analysis for each profiled operator
- Phase 5: Consolidated report

**Existing resources**:
- RVV patches: `applications/llama.cpp/rvv-patches/`
- Cross-compiled binary: `output/llama.cpp/`
- Perf data: documented in `applications/llama.cpp/README.md`

#### onnxrt/ostrack — Full Pipeline Worker

**Status**: New application, needs full setup under `applications/onnxrt/ostrack/`.

**What's needed**:
- Phase 0: ONNX model export + C++ runner + cross-compile onnxrt
  - OSTrack (ECCV 2022): Vision Transformer single-object tracker
  - Architecture: ViT-Base (embed_dim=768, 12 heads, 12 blocks)
  - Key operators: Multi-head Self-Attention (MatMul-dominant), LayerNorm, GELU MLP, Conv2d
  - ONNX export via `torch.onnx.export` (no built-in export script)
  - C++ runner: template (128×128) + search (256×256) images → bounding box
  - Target directory: `applications/onnxrt/ostrack/`
- Phase 1–5: Standard rvv512-optimization-pipeline

**Model variants**:
- OSTrack-256 (search 256×256) — recommended for initial analysis
- OSTrack-384 (search 384×384) — optional follow-up

**Important**: Must clone its own onnxrt source into `applications/onnxrt/ostrack/vendor/` to avoid parallel conflicts with YOLO's onnxrt build.

#### pytorch — Scout Worker

**Status**: Unknown feasibility.

**Research questions**:
1. Can PyTorch be cross-compiled for rv64gcv?
2. What are the build dependencies and toolchain requirements?
3. If PyTorch cannot build: which models can be exported to ONNX and analyzed via onnxrt?
4. What is the recommended fallback path?

**Output**: `docs/report/pytorch/feasibility-YYYY-MM-DD.md`

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
| Sysroot | `output/*/sysroot/` | Each worktree builds its own via Docker extraction |
| onnxrt source | `applications/onnxrt/<app>/vendor/` | Must clone independently to avoid parallel conflicts |
| Build output | `output/cross-<app>/` | Independent per application |
| RVV patches | `applications/<app>/rvv-patches/` | Per-application implementations |
| BBV profiling data | `output/bbv_rvv512/` | Work-specific |
| Reports | `docs/report/<app>/` | Per-application analysis |

### 4.3 Worktree Init Script

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

# Sysroot: NOT symlinked. Each worktree builds its own via application build.sh.
```

## 5. Parallel Scheduling Constraints

| Resource | Parallelism | Notes |
|----------|------------|-------|
| QEMU BBV | **Parallel** | Each process is independent |
| LLVM toolchain | **N/A** | Symlinked, no rebuild needed |
| onnxrt build | **Parallel** | Independent `vendor/` per app |
| Sysroot build | **Parallel** | Independent Docker extraction per app |
| Dev board (perf) | **Serial** | Single SSH; Lead serializes Phase 1 |
| Report output | **Parallel** | Paths isolated by app name |

## 6. Execution Plan

### Phase A: Spawn Teammates (Team Model)

**Step order matters.** Create the team first, then spawn all teammates as team members (with `team_name`), then create tasks in the shared task list:

```python
# Step 1: Create the team (Lead only, once)
TeamCreate(team_name="rvfuse-analysis")

# Step 2: Spawn ALL teammates simultaneously as TEAM MEMBERS
# Each has team_name, name, isolation="worktree", run_in_background=True
Agent(
    team_name="rvfuse-analysis",   # ← REQUIRED: makes this a teammate
    name="llama-cpp",              # ← REQUIRED: identity for SendMessage
    subagent_type="general-purpose",
    description="llama.cpp BBV + Gap Analysis",
    isolation="worktree",
    model="opus",
    mode="auto",
    run_in_background=True,
    prompt="""...full pipeline instructions..."""
)
Agent(name="onnxrt-ostrack", model="opus", team_name="rvfuse-analysis",
      subagent_type="general-purpose", isolation="worktree", run_in_background=True, ...)
Agent(name="onnxrt-yolo-fp32", model="sonnet", team_name="rvfuse-analysis",
      subagent_type="general-purpose", isolation="worktree", run_in_background=True, ...)
Agent(name="pytorch-scout", model="sonnet", team_name="rvfuse-analysis",
      subagent_type="general-purpose", isolation="worktree", run_in_background=True, ...)

# Step 3: Create tasks in the shared task list
TaskCreate(subject="llama.cpp: BBV + Gap Analysis (Phase 3-5)", ...)
TaskCreate(subject="onnxrt/ostrack: Full Pipeline (Phase 0-5)", ...)
TaskCreate(subject="onnxrt/yolo FP32: Supplement (Phase 3-4)", ...)
TaskCreate(subject="pytorch: Feasibility Research", ...)

# Step 4: Teammates self-assign tasks via TaskUpdate(owner="<their-name>")
```

**Key rule**: `Agent(team_name="rvfuse-analysis", name="...")` → teammate. `Agent(description="...", prompt="...")` without team_name → standalone subagent (used only for verification, not for pipeline work).

### Phase B: Dev Board Coordination

If any worker requires Phase 1 (perf profiling on real hardware):
- Lead serializes access via single SSH connection
- Workers without perf data use QEMU-only analysis or request Lead to run perf

### Phase C: Ongoing Monitoring & Verification (Lead)

After spawning, Lead's cron job runs every 5 minutes:
1. Check `TaskList` for teammate progress
2. Check tmux panes for idle/blocked teammates
3. Intervene as needed (see §2.4.2)
4. Log interventions in `docs/plans/tmux-automation-log.md`

When a teammate completes a phase:
1. Lead dispatches verification subagent (see §2.4.3)
2. If PASS: Send advance instruction via `SendMessage`
3. If FAIL: Send fix instructions via `SendMessage`
4. Repeat until all phases complete

### Phase D: Merge & Consolidate

After all teammates complete and pass final verification:
1. Send `shutdown_request` to each teammate via `SendMessage`
2. Merge each worktree to master with `--no-ff`
3. Delete cron monitoring job via `CronDelete`
4. `TeamDelete` to clean up team resources
5. Generate cross-application consolidated report
6. Update `docs/report/rvv-extension-comprehensive-analysis-*.md`

## 7. Skill Reference

| Phase | Skill | Purpose |
|-------|-------|---------|
| 0 | `cross-compile-app` | App setup, cross-compile, sysroot |
| 1 | `perf-profiling` | Dev board perf profiling |
| 2 | `rvv-op` | RVV512 operator implementation |
| 3 | `qemu-bbv-usage` | Function-scoped BBV profiling |
| 4 | `rvv-gap-analysis` | Cross-platform comparison + BBV-weighted benefits |
| 5 | `md2pdf` | Report generation |

Each worker follows the `rvv512-optimization-pipeline` skill for its applicable phases.

## 8. Acceptance Criteria

Per teammate:
- [ ] Task claimed via `TaskUpdate(owner="<teammate-name>")` and status updated through lifecycle
- [ ] All target operators have non-empty BBV data (`.bb` + `.disas`)
- [ ] All target operators have gap analysis reports (MD + PDF)
- [ ] Reports contain overall benefit figures (整体收益) from BBV data
- [ ] Consolidated report merges findings with priority table
- [ ] Worktree merges cleanly to master (no conflicts)
- [ ] Teammate responds to `shutdown_request` and exits cleanly
