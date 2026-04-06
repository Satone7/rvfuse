# Design: setup.sh Redesign — Automate README Steps 0-6

**Date**: 2026-04-06
**Branch**: `worktree-setup-script`
**Status**: Approved

## Problem

PR #8 added `setup.sh` based on an old README (phase `001-riscv-fusion-setup`) where "steps" meant repository setup (clone, review docs, init submodules). The README has since evolved to a 7-step build/profiling pipeline (Steps 0-6). The script automates completely different things and the `--force` flag doesn't work properly.

## Decision

Rewrite `setup.sh` from scratch as a thin orchestrator that automates README Steps 0-6 plus a report step. Delete all files from PR #8 that are no longer relevant.

## Approach: Thin Orchestrator

A single `setup.sh` (~300-400 lines) that delegates actual work to existing scripts, adds artifact detection, and handles `--force` with artifact cleanup.

## Step Definitions

| Step | Name | Action | Artifacts (skip check) | Force Cleanup |
|------|------|--------|----------------------|---------------|
| 0 | Init Submodules | `git submodule update --init [--depth 1]` | `third_party/qemu/.git`, `third_party/llvm-project/.git` | `git submodule deinit -f` + remove dirs |
| 1 | Prepare Model | `./prepare_model.sh` | `output/yolo11n.ort`, `output/test.jpg` | Remove `output/yolo11n.onnx`, `output/yolo11n.ort`, `output/test.jpg` |
| 2 | Build QEMU | `./verify_bbv.sh --force-rebuild` | `third_party/qemu/build/contrib/plugins/libbbv.so` | Remove `third_party/qemu/build/` |
| 3 | Docker Build | `./tools/docker-onnxrt/build.sh` | `output/yolo_inference`, `output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1` | Remove `output/yolo_inference`, `output/sysroot/` |
| 4 | BBV Profiling | Run QEMU with BBV plugin directly | `output/yolo.bbv.*.bb` (glob) | Remove `output/yolo.bbv.*` |
| 5 | Hotspot Report | `python3 tools/analyze_bbv.py --json-output ...` | `output/hotspot.json` | Remove `output/hotspot.json` |
| 6 | DFG Generation | `python3 -m tools.dfg ...` | `dfg/` directory (non-empty) | Remove `dfg/` |
| 7 | Generate Report | Write `setup-report.txt` | *(always runs)* | N/A |

## CLI Interface

```
usage: setup.sh [--force <steps>] [--force-all] [--shallow] [--bbv-interval <N>] [--top <N>] [--coverage <N>] [--help]

Options:
  --force <steps>       Re-execute specified steps (comma-separated, e.g., "2" or "3,5")
  --force-all           Re-execute all steps from scratch (deletes artifacts)
  --shallow             Use --depth 1 for submodule clones (Step 0)
  --bbv-interval <N>    BBV sampling interval for Step 4 (default: 100000)
  --top <N>             Top N blocks for Steps 5-6 (default: 20)
  --coverage <N>        Coverage threshold % for Steps 5-6 (default: 80)
  --help                Show this help message
```

## Execution Flow

1. Parse CLI arguments
2. Validate project root (`git rev-parse --show-toplevel`)
3. Check prerequisites (git >= 2.30, python3, Docker, disk space >= 20GB)
4. For each step 0-7:
   a. If forced: delete that step's artifacts, then execute
   b. If not forced and all artifacts exist: record SKIPPED
   c. Else: execute step, record PASS/FAIL
5. Step 7 always runs (report generation)
6. Exit 0 if all steps 0-6 PASS, exit 1 otherwise

## Error Handling

- **Prerequisites fail**: exit 1 immediately before any step
- **Step fails**: record FAIL, continue to next step (collect diagnostics)
- **Step 7**: always runs, even if previous steps failed
- **Exit code**: 0 if all steps 0-6 PASS, 1 if any step FAIL

## Prerequisites

- git >= 2.30
- python3
- Docker (for Step 3)
- Disk space >= 20GB (warning only)

## Report Format

```
RVFuse Setup Report
Generated: 2026-04-06T14:30:00+08:00
Options: force=2,5 shallow

Step 0: Init Submodules          [SKIPPED] (artifacts exist)
Step 1: Prepare Model            [PASS]   (3 artifacts)
Step 2: Build QEMU               [PASS]   (libbbv.so built)
Step 3: Docker Build             [PASS]   (yolo_inference + sysroot)
Step 4: BBV Profiling            [PASS]   (yolo.bbv.0.bb)
Step 5: Hotspot Report           [PASS]   (output/hotspot.json)
Step 6: DFG Generation           [FAIL]   (python3 -m tools.dfg exited 1)
Step 7: Generate Report          [PASS]

Overall: FAIL
```

## File Changes

### Delete
- `setup.sh` (rewrite)
- `specs/003-automated-setup-flow/` (entire directory — 6 files based on wrong requirements)
- `tests/setup/` (entire directory — 5 bats test files testing wrong steps)

### Create
- `setup.sh` (new thin orchestrator)

### Modify
- `.gitignore` — remove `tests/setup/test_helper/` entry

### Keep unchanged
- `prepare_model.sh`, `verify_bbv.sh`, `tools/docker-onnxrt/build.sh`
- `tools/profile_to_dfg.sh`, `tools/analyze_bbv.py`, `tools/dfg/`

## Testing

Deferred. The old 51-test bats-core suite tested wrong steps. Tests can be added later after the script is validated manually.
