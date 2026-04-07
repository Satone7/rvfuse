# setup.sh Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `setup.sh` as a thin orchestrator that automates README Steps 0-6 (init submodules through DFG generation) plus a report step, replacing the old 5-step repository-setup script.

**Architecture:** A single Bash script (~350 lines) that delegates actual work to existing scripts (`prepare_model.sh`, `verify_bbv.sh`, `tools/docker-onnxrt/build.sh`, `tools/profile_to_dfg.sh`), adds artifact-based skip detection, and handles `--force` with artifact cleanup. Each step checks for existing artifacts before running and can be selectively re-executed with `--force <steps>`.

**Tech Stack:** Bash 4.0+ (associative arrays, `declare -gA`), Git 2.30+, Python 3.10+, Docker, QEMU (via submodules)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `setup.sh` | Rewrite | Thin orchestrator — CLI parsing, prerequisites, 8 steps (0-7), report generation |
| `specs/003-automated-setup-flow/` | Delete | Old spec based on wrong requirements (6 files) |
| `tests/setup/` | Delete | Old bats-core tests for wrong steps (5 files, 705 lines) |
| `.gitignore` | Modify | Remove `tests/setup/test_helper/` entry |

**Files kept unchanged:** `prepare_model.sh`, `verify_bbv.sh`, `tools/docker-onnxrt/build.sh`, `tools/profile_to_dfg.sh`, `tools/analyze_bbv.py`, `tools/dfg/`

## Step Definitions (for reference)

| Step | Name | Delegates To | Artifacts (skip check) | Force Cleanup |
|------|------|-------------|----------------------|---------------|
| 0 | Init Submodules | `git submodule update --init [--depth 1]` | `third_party/qemu/.git`, `third_party/llvm-project/.git` | `git submodule deinit -f` + `rm -rf` dirs |
| 1 | Prepare Model | `./prepare_model.sh` | `output/yolo11n.ort`, `output/test.jpg` | `rm output/yolo11n.onnx output/yolo11n.ort output/test.jpg` |
| 2 | Build QEMU | `./verify_bbv.sh --force-rebuild` | `third_party/qemu/build/contrib/plugins/libbbv.so` | `rm -rf third_party/qemu/build/` |
| 3 | Docker Build | `./tools/docker-onnxrt/build.sh` | `output/yolo_inference`, `output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1` | `rm output/yolo_inference; rm -rf output/sysroot/` |
| 4 | BBV Profiling | QEMU CLI directly | `output/yolo.bbv.*.bb` (glob) | `rm output/yolo.bbv.*` |
| 5 | Hotspot Report | `python3 tools/analyze_bbv.py --json-output ...` | `output/hotspot.json` | `rm output/hotspot.json` |
| 6 | DFG Generation | `./tools/profile_to_dfg.sh` | `dfg/` directory (non-empty) | `rm -rf dfg/` |
| 7 | Generate Report | Inline (writes `setup-report.txt`) | *(always runs)* | N/A |

## New CLI Interface

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

---

### Task 1: Delete obsolete files from PR #8

**Files:**
- Delete: `specs/003-automated-setup-flow/` (entire directory — 6 files)
- Delete: `tests/setup/` (entire directory — 5 bats test files)
- Modify: `.gitignore:47` (remove `tests/setup/test_helper/` entry)

- [ ] **Step 1: Delete the old spec directory**

```bash
rm -rf specs/003-automated-setup-flow/
```

- [ ] **Step 2: Delete the old test directory**

```bash
rm -rf tests/setup/
```

- [ ] **Step 3: Remove `tests/setup/test_helper/` from `.gitignore`**

The current `.gitignore` line 47 is:
```
tests/setup/test_helper/
```
Remove this entire line. The surrounding lines 46 and 48 are blank, so remove one blank line too, leaving a single blank line before `# Pre-cloned vendor sources for Docker builds`.

- [ ] **Step 4: Verify no stale references remain**

```bash
grep -r "003-automated-setup-flow" --include="*.md" --include="*.sh" --include=".gitignore" . || echo "Clean"
grep -r "tests/setup/" --include="*.md" --include="*.sh" --include=".gitignore" . || echo "Clean"
```

Expected: "Clean" for both.

- [ ] **Step 5: Commit**

```bash
git add -A specs/ tests/setup/ .gitignore
git commit -m "chore: remove obsolete setup spec and tests from PR #8"
```

---

### Task 2: Write script skeleton — constants, state, logging, CLI parsing

**Files:**
- Create: `setup.sh` (overwrite existing — replace entire file)

This task writes the script header through the end of `parse_args()`. The script will be built incrementally across Tasks 2-8.

- [ ] **Step 1: Write the script skeleton (constants, state, logging, CLI, main entry)**

Overwrite `setup.sh` with this complete skeleton. This includes everything except the step functions and the `run_setup()` loop, which come in later tasks.

```bash
#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# RVFuse Setup Orchestrator — Automates README Steps 0-6
# =============================================================================
#
# Thin orchestrator that delegates actual work to existing scripts:
#   prepare_model.sh, verify_bbv.sh, tools/docker-onnxrt/build.sh,
#   tools/profile_to_dfg.sh
#
# Supports artifact-based skip detection and --force with cleanup.

# --- Script-level constants ---

REPORT_FILE="setup-report.txt"
readonly MIN_DISK_GB=20
readonly MIN_GIT_VERSION_MAJOR=2
readonly MIN_GIT_VERSION_MINOR=30
readonly TOTAL_STEPS=8  # Steps 0-7

# Step names (indexed 0-7)
readonly -A STEP_NAMES=(
    [0]="Init Submodules"
    [1]="Prepare Model"
    [2]="Build QEMU"
    [3]="Docker Build"
    [4]="BBV Profiling"
    [5]="Hotspot Report"
    [6]="DFG Generation"
    [7]="Generate Report"
)

# Step artifact paths (relative to PROJECT_ROOT)
# Steps 4 (BBV) uses a glob pattern — handled specially in check function
readonly -a STEP0_ARTIFACTS=(
    "third_party/qemu/.git"
    "third_party/llvm-project/.git"
)
readonly -a STEP1_ARTIFACTS=(
    "output/yolo11n.ort"
    "output/test.jpg"
)
readonly -a STEP2_ARTIFACTS=(
    "third_party/qemu/build/contrib/plugins/libbbv.so"
)
readonly -a STEP3_ARTIFACTS=(
    "output/yolo_inference"
    "output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1"
)
# Step 4 (BBV Profiling): glob output/yolo.bbv.*.bb — checked specially
readonly -a STEP5_ARTIFACTS=(
    "output/hotspot.json"
)
# Step 6 (DFG Generation): non-empty dfg/ directory — checked specially
readonly -a STEP7_ARTIFACTS=()  # always runs

# --- Global state ---

declare -g PROJECT_ROOT=""
declare -g SHALLOW_CLONE=0
declare -g BBV_INTERVAL=100000
declare -g TOP_N=20
declare -g COVERAGE_PCT=80
declare -ga FORCE_STEPS=()

# Per-step result tracking (indexed 0-7)
declare -gA STEP_STATUS=()
declare -gA STEP_MESSAGE=()

# Script options string for report
SCRIPT_OPTIONS=""

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

# =============================================================================
# CLI Argument Parsing
# =============================================================================

print_usage() {
    cat <<'EOF'
usage: setup.sh [--force <steps>] [--force-all] [--shallow] [--bbv-interval <N>] [--top <N>] [--coverage <N>] [--help]

Options:
  --force <steps>       Re-execute specified steps (comma-separated, e.g., "2" or "3,5")
  --force-all           Re-execute all steps from scratch (deletes artifacts)
  --shallow             Use --depth 1 for submodule clones (Step 0)
  --bbv-interval <N>    BBV sampling interval for Step 4 (default: 100000)
  --top <N>             Top N blocks for Steps 5-6 (default: 20)
  --coverage <N>        Coverage threshold % for Steps 5-6 (default: 80)
  --help                Show this help message
EOF
}

parse_args() {
    local args=("$@")
    local i=0
    while (( i < ${#args[@]} )); do
        case "${args[$i]}" in
            --force)
                (( i++ )) || true
                if (( i >= ${#args[@]} )); then
                    log_error "--force requires a value (e.g., --force 3 or --force 2,4)"
                    return 1
                fi
                local force_val="${args[$i]}"
                IFS=',' read -ra force_nums <<< "$force_val"
                for num in "${force_nums[@]}"; do
                    if ! [[ "$num" =~ ^[0-7]$ ]]; then
                        log_error "Invalid step number: $num (must be 0-7)"
                        return 1
                    fi
                    FORCE_STEPS+=("$num")
                done
                ;;
            --force-all)
                FORCE_STEPS=(0 1 2 3 4 5 6 7)
                ;;
            --shallow)
                SHALLOW_CLONE=1
                ;;
            --bbv-interval)
                (( i++ )) || true
                if (( i >= ${#args[@]} )); then
                    log_error "--bbv-interval requires a numeric value"
                    return 1
                fi
                if ! [[ "${args[$i]}" =~ ^[0-9]+$ ]]; then
                    log_error "--bbv-interval must be a positive integer: ${args[$i]}"
                    return 1
                fi
                BBV_INTERVAL="${args[$i]}"
                ;;
            --top)
                (( i++ )) || true
                if (( i >= ${#args[@]} )); then
                    log_error "--top requires a numeric value"
                    return 1
                fi
                if ! [[ "${args[$i]}" =~ ^[0-9]+$ ]]; then
                    log_error "--top must be a positive integer: ${args[$i]}"
                    return 1
                fi
                TOP_N="${args[$i]}"
                ;;
            --coverage)
                (( i++ )) || true
                if (( i >= ${#args[@]} )); then
                    log_error "--coverage requires a numeric value"
                    return 1
                fi
                if ! [[ "${args[$i]}" =~ ^[0-9]+$ ]]; then
                    log_error "--coverage must be a positive integer: ${args[$i]}"
                    return 1
                fi
                COVERAGE_PCT="${args[$i]}"
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: ${args[$i]}"
                print_usage >&2
                return 1
                ;;
        esac
        (( i++ )) || true
    done

    # Build options string for report
    local -a opt_parts=()
    if (( ${#FORCE_STEPS[@]} > 0 )); then
        opt_parts+=("force=${FORCE_STEPS[*]}")
    fi
    if (( SHALLOW_CLONE == 1 )); then
        opt_parts+=("shallow")
    fi
    if (( BBV_INTERVAL != 100000 )); then
        opt_parts+=("bbv-interval=${BBV_INTERVAL}")
    fi
    if (( TOP_N != 20 )); then
        opt_parts+=("top=${TOP_N}")
    fi
    if (( COVERAGE_PCT != 80 )); then
        opt_parts+=("coverage=${COVERAGE_PCT}")
    fi
    SCRIPT_OPTIONS="${opt_parts[*]:-none}"
}

is_step_forced() {
    local step_num="$1"
    for fs in "${FORCE_STEPS[@]+"${FORCE_STEPS[@]}"}"; do
        if [[ "$fs" == "$step_num" ]]; then
            return 0
        fi
    done
    return 1
}

# =============================================================================
# Project Root Detection
# =============================================================================

detect_project_root() {
    local root
    root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
        log_error "Not inside a git repository. Navigate to the RVFuse repo root and run ./setup.sh"
        return 1
    }
    PROJECT_ROOT="$root"
    log_info "Project root: ${PROJECT_ROOT}"
}

# =============================================================================
# Prerequisite Checking
# =============================================================================

check_prerequisites() {
    # Check git version
    if ! command -v git &>/dev/null; then
        log_error "git is not installed. Install git >= ${MIN_GIT_VERSION_MAJOR}.${MIN_GIT_VERSION_MINOR} and retry."
        return 1
    fi

    local git_version
    git_version="$(git --version 2>/dev/null | grep -oP '\d+\.\d+' || true)"
    local git_major git_minor
    git_major="${git_version%%.*}"
    git_minor="${git_version#*.}"

    if (( git_major < MIN_GIT_VERSION_MAJOR )) || \
       (( git_major == MIN_GIT_VERSION_MAJOR && git_minor < MIN_GIT_VERSION_MINOR )); then
        log_error "git version ${git_version} is below minimum ${MIN_GIT_VERSION_MAJOR}.${MIN_GIT_VERSION_MINOR}"
        return 1
    fi
    log_info "git version: ${git_version} (ok)"

    # Check python3
    if ! command -v python3 &>/dev/null; then
        log_error "python3 is not installed. Install Python 3.10+ and retry."
        return 1
    fi
    log_info "python3: $(python3 --version 2>&1) (ok)"

    # Check Docker (for Step 3)
    if ! command -v docker &>/dev/null; then
        log_warn "Docker is not installed. Step 3 (Docker Build) will fail."
    else
        log_info "Docker: $(docker --version 2>&1) (ok)"
    fi

    # Check disk space (warning only)
    local available_kb
    available_kb="$(df -k --output=avail "${PROJECT_ROOT}" 2>/dev/null | tail -1 | tr -d ' ')"
    if [[ -z "$available_kb" ]] || ! [[ "$available_kb" =~ ^[0-9]+$ ]]; then
        log_warn "Could not determine available disk space"
    else
        local available_gb
        available_gb=$((available_kb / 1024 / 1024))
        if (( available_gb < MIN_DISK_GB )); then
            log_warn "Available disk space: ~${available_gb}GB (recommended: ${MIN_DISK_GB}GB)"
            log_warn "Submodule clones and Docker builds may fail. Consider freeing disk space."
        else
            log_info "Available disk space: ~${available_gb}GB (ok)"
        fi
    fi

    return 0
}

# =============================================================================
# Artifact Checking
# =============================================================================

# Standard artifact check: all paths must exist and be non-empty (files only)
check_artifacts() {
    local -a artifacts=("$@")
    for artifact in "${artifacts[@]}"; do
        local full_path="${PROJECT_ROOT}/${artifact}"
        if [[ ! -e "$full_path" ]]; then
            return 1
        fi
        if [[ -f "$full_path" && ! -s "$full_path" ]]; then
            return 1
        fi
    done
    return 0
}

# Step 4 special check: glob for output/yolo.bbv.*.bb
check_bbv_artifacts() {
    local bbv_files
    bbv_files=( "${PROJECT_ROOT}"/output/yolo.bbv.*.bb )
    # If glob didn't match, bash returns the literal pattern
    if [[ ! -e "${bbv_files[0]}" ]]; then
        return 1
    fi
    # Check at least one .bb file is non-empty
    for f in "${bbv_files[@]}"; do
        if [[ -s "$f" ]]; then
            return 0
        fi
    done
    return 1
}

# Step 6 special check: dfg/ directory must exist and be non-empty
check_dfg_artifacts() {
    local dfg_dir="${PROJECT_ROOT}/dfg"
    if [[ ! -d "$dfg_dir" ]]; then
        return 1
    fi
    # Check for any files (not just directories)
    local count
    count="$(find "$dfg_dir" -type f 2>/dev/null | wc -l)"
    if (( count == 0 )); then
        return 1
    fi
    return 0
}

# =============================================================================
# Step Result Recording
# =============================================================================

record_step_result() {
    local step_num="$1"
    local status="$2"  # PASS, FAIL, SKIPPED
    local message="${3:-}"
    STEP_STATUS[$step_num]="$status"
    STEP_MESSAGE[$step_num]="$message"

    case "$status" in
        PASS)   log_info "Step ${step_num}: PASS" ;;
        FAIL)   log_error "Step ${step_num}: FAIL — ${message}" ;;
        SKIP*)  log_info "Step ${step_num}: SKIPPED" ;;
    esac
}

# Get artifacts array for a step number
get_artifact_names() {
    case "$1" in
        0) echo "STEP0_ARTIFACTS" ;;
        1) echo "STEP1_ARTIFACTS" ;;
        2) echo "STEP2_ARTIFACTS" ;;
        3) echo "STEP3_ARTIFACTS" ;;
        4) echo "__bbv__" ;;          # special glob check
        5) echo "STEP5_ARTIFACTS" ;;
        6) echo "__dfg__" ;;          # special dir check
        7) echo "STEP7_ARTIFACTS" ;;
    esac
}

# Check if all artifacts exist for a given step
step_artifacts_exist() {
    local step_num="$1"
    local art_key
    art_key="$(get_artifact_names "$step_num")"
    case "$art_key" in
        __bbv__)  check_bbv_artifacts ;;
        __dfg__)  check_dfg_artifacts ;;
        *)        check_artifacts "${!art_key}" ;;
    esac
}

# =============================================================================
# Placeholder — step functions and run_setup() will be added by Tasks 3-8
# =============================================================================

main() {
    parse_args "$@"
    detect_project_root
    check_prerequisites

    cd "$PROJECT_ROOT"

    run_setup

    # Determine overall exit code (steps 0-6; step 7 is always report)
    local has_fail=0
    for i in 0 1 2 3 4 5 6; do
        if [[ "${STEP_STATUS[$i]:-}" == "FAIL" ]]; then
            has_fail=1
            break
        fi
    done

    log_info "Setup complete. Report: ${PROJECT_ROOT}/${REPORT_FILE}"

    if (( has_fail == 1 )); then
        log_error "One or more steps failed. Review ${REPORT_FILE} for details."
        return 1
    fi
    return 0
}

# Only run main if not being sourced for testing
if [[ "${1:-}" != "--source-only" && "${SETUP_TEST_MODE:-}" != "1" ]]; then
    main "$@"
fi
```

- [ ] **Step 2: Make the script executable**

```bash
chmod +x setup.sh
```

- [ ] **Step 3: Smoke test — verify `--help` works**

```bash
bash setup.sh --help
```

Expected: usage text printed, exit 0.

- [ ] **Step 4: Smoke test — verify unknown flag is rejected**

```bash
bash setup.sh --bogus 2>&1; echo "exit: $?"
```

Expected: "Unknown option: --bogus" on stderr, exit code 1.

- [ ] **Step 5: Smoke test — verify `--force` validates step range**

```bash
bash setup.sh --force 8 2>&1; echo "exit: $?"
```

Expected: "Invalid step number: 8 (must be 0-7)", exit code 1.

- [ ] **Step 6: Commit**

```bash
git add setup.sh
git commit -m "feat(setup): write script skeleton — constants, state, logging, CLI parsing"
```

---

### Task 3: Implement Steps 0-1 (Init Submodules, Prepare Model)

**Files:**
- Modify: `setup.sh` — add `step0_init_submodules()`, `step1_prepare_model()`, and their force cleanup functions

Insert these functions before the `main()` function. Also add the `cleanup_step()` dispatcher and the start of `run_setup()`.

- [ ] **Step 1: Add step functions and cleanup logic**

Insert the following block in `setup.sh` **between the `step_artifacts_exist()` function and the `# Placeholder` comment** (just before the `main()` function):

```bash
# =============================================================================
# Force Cleanup Functions
# =============================================================================

cleanup_step() {
    local step_num="$1"
    case "$step_num" in
        0)
            log_info "Cleaning Step 0 artifacts (submodules)..."
            for dep_path in third_party/qemu third_party/llvm-project; do
                if [[ -d "${PROJECT_ROOT}/${dep_path}" ]]; then
                    git submodule deinit -f "$dep_path" 2>/dev/null || true
                    rm -rf "${PROJECT_ROOT}/${dep_path}"
                fi
            done
            ;;
        1)
            log_info "Cleaning Step 1 artifacts (model files)..."
            rm -f "${PROJECT_ROOT}/output/yolo11n.onnx"
            rm -f "${PROJECT_ROOT}/output/yolo11n.ort"
            rm -f "${PROJECT_ROOT}/output/test.jpg"
            ;;
        2)
            log_info "Cleaning Step 2 artifacts (QEMU build)..."
            rm -rf "${PROJECT_ROOT}/third_party/qemu/build"
            ;;
        3)
            log_info "Cleaning Step 3 artifacts (Docker build)..."
            rm -f "${PROJECT_ROOT}/output/yolo_inference"
            rm -rf "${PROJECT_ROOT}/output/sysroot"
            ;;
        4)
            log_info "Cleaning Step 4 artifacts (BBV output)..."
            rm -f "${PROJECT_ROOT}"/output/yolo.bbv.*
            ;;
        5)
            log_info "Cleaning Step 5 artifacts (hotspot report)..."
            rm -f "${PROJECT_ROOT}/output/hotspot.json"
            ;;
        6)
            log_info "Cleaning Step 6 artifacts (DFG output)..."
            rm -rf "${PROJECT_ROOT}/dfg"
            ;;
        7)
            # No cleanup for report step
            ;;
    esac
}

# =============================================================================
# Step 0: Init Submodules
# =============================================================================

step0_init_submodules() {
    local step=0
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    local clone_args=("--init")
    if (( SHALLOW_CLONE == 1 )); then
        clone_args+=("--depth" "1")
    fi

    local failed_deps=()
    for dep_path in "third_party/qemu" "third_party/llvm-project"; do
        local dep_marker="${PROJECT_ROOT}/${dep_path}/.git"
        if [[ -e "$dep_marker" ]]; then
            log_info "  ${dep_path}: already initialized"
            continue
        fi
        log_info "  Initializing ${dep_path}..."
        if ! git submodule update "${clone_args[@]}" "$dep_path" 2>&1; then
            failed_deps+=("$dep_path")
            log_error "  Failed to initialize ${dep_path}"
        else
            log_info "  ${dep_path}: initialized"
        fi
    done

    if (( ${#failed_deps[@]} > 0 )); then
        record_step_result "$step" "FAIL" "failed to initialize: ${failed_deps[*]}"
        return 1
    fi

    record_step_result "$step" "PASS" "all submodules initialized"
    return 0
}

# =============================================================================
# Step 1: Prepare Model
# =============================================================================

step1_prepare_model() {
    local step=1
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    if ! bash "${PROJECT_ROOT}/prepare_model.sh" 2>&1; then
        record_step_result "$step" "FAIL" "prepare_model.sh exited with error"
        return 1
    fi

    record_step_result "$step" "PASS" "yolo11n.ort + test.jpg ready"
    return 0
}

# =============================================================================
# run_setup() — Main Execution Loop
# =============================================================================

run_setup() {
    for step_num in 0 1 2 3 4 5 6 7; do
        # Step 7 always runs (report generation)
        if (( step_num == 7 )); then
            step7_report
            continue
        fi

        # Check if forced
        if is_step_forced "$step_num"; then
            cleanup_step "$step_num"
        # Check if artifacts exist (skip if they do)
        elif step_artifacts_exist "$step_num"; then
            STEP_STATUS[$step_num]="SKIPPED"
            STEP_MESSAGE[$step_num]="artifacts exist"
            log_info "=== Step ${step_num}: ${STEP_NAMES[$step_num]} ==="
            log_info "Step ${step_num}: SKIPPED (artifacts exist)"
            continue
        fi

        # Run the step
        case "$step_num" in
            0) step0_init_submodules ;;
            1) step1_prepare_model ;;
            # Steps 2-6: added by Tasks 4-7
            2) log_info "=== Step 2: ${STEP_NAMES[2]} ===" ; record_step_result 2 "FAIL" "not yet implemented" ;;
            3) log_info "=== Step 3: ${STEP_NAMES[3]} ===" ; record_step_result 3 "FAIL" "not yet implemented" ;;
            4) log_info "=== Step 4: ${STEP_NAMES[4]} ===" ; record_step_result 4 "FAIL" "not yet implemented" ;;
            5) log_info "=== Step 5: ${STEP_NAMES[5]} ===" ; record_step_result 5 "FAIL" "not yet implemented" ;;
            6) log_info "=== Step 6: ${STEP_NAMES[6]} ===" ; record_step_result 6 "FAIL" "not yet implemented" ;;
        esac || true
    done
}
```

Also **remove** the `# Placeholder` comment line:
```
# =============================================================================
# Placeholder — step functions and run_setup() will be added by Tasks 3-8
# =============================================================================
```

- [ ] **Step 2: Smoke test — `--help` still works after edits**

```bash
bash setup.sh --help
```

Expected: usage text, exit 0.

- [ ] **Step 3: Smoke test — invalid step 2-6 gives "not yet implemented"**

```bash
bash setup.sh --force 2 2>&1 | tail -3
```

Expected: "FAIL — not yet implemented".

- [ ] **Step 4: Commit**

```bash
git add setup.sh
git commit -m "feat(setup): add Steps 0-1 (init submodules, prepare model) and run_setup loop"
```

---

### Task 4: Implement Steps 2-3 (Build QEMU, Docker Build)

**Files:**
- Modify: `setup.sh` — add `step2_build_qemu()`, `step3_docker_build()`, update `run_setup()` placeholders

- [ ] **Step 1: Add Step 2 and Step 3 functions**

Insert after `step1_prepare_model()` (after its closing `}` and blank line) and before `# run_setup()`:

```bash
# =============================================================================
# Step 2: Build QEMU
# =============================================================================

step2_build_qemu() {
    local step=2
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    if ! bash "${PROJECT_ROOT}/verify_bbv.sh" --force-rebuild 2>&1; then
        record_step_result "$step" "FAIL" "verify_bbv.sh --force-rebuild exited with error"
        return 1
    fi

    record_step_result "$step" "PASS" "libbbv.so built"
    return 0
}

# =============================================================================
# Step 3: Docker Build
# =============================================================================

step3_docker_build() {
    local step=3
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    if ! bash "${PROJECT_ROOT}/tools/docker-onnxrt/build.sh" 2>&1; then
        record_step_result "$step" "FAIL" "tools/docker-onnxrt/build.sh exited with error"
        return 1
    fi

    record_step_result "$step" "PASS" "yolo_inference + sysroot ready"
    return 0
}
```

- [ ] **Step 2: Update `run_setup()` — replace Step 2 and Step 3 placeholders**

In the `run_setup()` function, find the case block:
```bash
            2) log_info "=== Step 2: ${STEP_NAMES[2]} ===" ; record_step_result 2 "FAIL" "not yet implemented" ;;
            3) log_info "=== Step 3: ${STEP_NAMES[3]} ===" ; record_step_result 3 "FAIL" "not yet implemented" ;;
```

Replace with:
```bash
            2) step2_build_qemu ;;
            3) step3_docker_build ;;
```

- [ ] **Step 3: Smoke test — `--help` still works**

```bash
bash setup.sh --help
```

Expected: usage text, exit 0.

- [ ] **Step 4: Commit**

```bash
git add setup.sh
git commit -m "feat(setup): add Steps 2-3 (build QEMU, Docker build)"
```

---

### Task 5: Implement Step 4 (BBV Profiling)

**Files:**
- Modify: `setup.sh` — add `step4_bbv_profiling()`, update `run_setup()` placeholder

This step runs QEMU with the BBV plugin directly (not via a delegate script). It needs to locate the QEMU binary, the BBV plugin `.so`, the inference binary, model, test image, and sysroot — all of which are produced by earlier steps.

- [ ] **Step 1: Add Step 4 function**

Insert after `step3_docker_build()` and before `# run_setup()`:

```bash
# =============================================================================
# Step 4: BBV Profiling
# =============================================================================

step4_bbv_profiling() {
    local step=4
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    local qemu_bin="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
    local plugin_so="${PROJECT_ROOT}/third_party/qemu/build/contrib/plugins/libbbv.so"
    local inference_bin="${PROJECT_ROOT}/output/yolo_inference"
    local ort_model="${PROJECT_ROOT}/output/yolo11n.ort"
    local test_image="${PROJECT_ROOT}/output/test.jpg"
    local sysroot="${PROJECT_ROOT}/output/sysroot"

    # Verify dependencies from earlier steps
    local missing_deps=()
    for dep in "$qemu_bin" "$plugin_so" "$inference_bin" "$ort_model" "$test_image"; do
        if [[ ! -e "$dep" ]]; then
            missing_deps+=("$(basename "$dep")")
        fi
    done
    if [[ ! -d "$sysroot" ]]; then
        missing_deps+=("sysroot/")
    fi

    if (( ${#missing_deps[@]} > 0 )); then
        record_step_result "$step" "FAIL" "missing dependencies: ${missing_deps[*]} — run earlier steps first"
        return 1
    fi

    # Clean old BBV output
    rm -f "${PROJECT_ROOT}"/output/yolo.bbv.*

    log_info "  Running QEMU BBV profiling (interval=${BBV_INTERVAL})..."
    if ! "${qemu_bin}" \
        -L "${sysroot}" \
        -plugin "${plugin_so}",interval="${BBV_INTERVAL}",outfile="${PROJECT_ROOT}/output/yolo.bbv" \
        "${inference_bin}" "${ort_model}" "${test_image}" 1 \
        2>&1; then
        record_step_result "$step" "FAIL" "qemu-riscv64 profiling exited with error"
        return 1
    fi

    # Verify BBV output was created
    local bbv_files=("${PROJECT_ROOT}"/output/yolo.bbv.*.bb)
    if [[ ! -e "${bbv_files[0]}" ]]; then
        record_step_result "$step" "FAIL" "no BBV output files generated"
        return 1
    fi

    local bbv_name
    bbv_name="$(basename "${bbv_files[0]}")"
    record_step_result "$step" "PASS" "${bbv_name}"
    return 0
}
```

- [ ] **Step 2: Update `run_setup()` — replace Step 4 placeholder**

Find:
```bash
            4) log_info "=== Step 4: ${STEP_NAMES[4]} ===" ; record_step_result 4 "FAIL" "not yet implemented" ;;
```

Replace with:
```bash
            4) step4_bbv_profiling ;;
```

- [ ] **Step 3: Smoke test — `--help` still works**

```bash
bash setup.sh --help
```

Expected: usage text, exit 0.

- [ ] **Step 4: Commit**

```bash
git add setup.sh
git commit -m "feat(setup): add Step 4 (BBV profiling via QEMU)"
```

---

### Task 6: Implement Steps 5-6 (Hotspot Report, DFG Generation)

**Files:**
- Modify: `setup.sh` — add `step5_hotspot_report()`, `step6_dfg_generation()`, update `run_setup()` placeholders

Step 5 runs `analyze_bbv.py` with `--json-output`. Step 6 runs `profile_to_dfg.sh` which chains Steps 5-6 of the README pipeline (BBV analysis + DFG generation).

- [ ] **Step 1: Add Step 5 and Step 6 functions**

Insert after `step4_bbv_profiling()` and before `# run_setup()`:

```bash
# =============================================================================
# Step 5: Hotspot Report
# =============================================================================

step5_hotspot_report() {
    local step=5
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    # Find BBV and disas files
    local bbv_files=("${PROJECT_ROOT}"/output/yolo.bbv.*.bb)
    if [[ ! -e "${bbv_files[0]}" ]]; then
        record_step_result "$step" "FAIL" "no BBV output files found — run Step 4 first"
        return 1
    fi
    local bbv_file="${bbv_files[0]}"

    local elf_bin="${PROJECT_ROOT}/output/yolo_inference"
    local sysroot="${PROJECT_ROOT}/output/sysroot"
    local json_output="${PROJECT_ROOT}/output/hotspot.json"

    if [[ ! -f "$elf_bin" ]]; then
        record_step_result "$step" "FAIL" "output/yolo_inference not found — run Step 3 first"
        return 1
    fi

    local analyze_args=(
        "--bbv" "$bbv_file"
        "--elf" "$elf_bin"
        "--json-output" "$json_output"
    )
    if [[ -d "$sysroot" ]]; then
        analyze_args+=("--sysroot" "$sysroot")
    fi
    if (( TOP_N > 0 )); then
        analyze_args+=("--top" "$TOP_N")
    fi

    log_info "  Running analyze_bbv.py..."
    if ! python3 "${PROJECT_ROOT}/tools/analyze_bbv.py" "${analyze_args[@]}" 2>&1; then
        record_step_result "$step" "FAIL" "analyze_bbv.py exited with error"
        return 1
    fi

    if [[ ! -f "$json_output" ]]; then
        record_step_result "$step" "FAIL" "hotspot.json not created"
        return 1
    fi

    record_step_result "$step" "PASS" "output/hotspot.json"
    return 0
}

# =============================================================================
# Step 6: DFG Generation
# =============================================================================

step6_dfg_generation() {
    local step=6
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    # Find BBV and disas files
    local bbv_files=("${PROJECT_ROOT}"/output/yolo.bbv.*.bb)
    if [[ ! -e "${bbv_files[0]}" ]]; then
        record_step_result "$step" "FAIL" "no BBV output files found — run Step 4 first"
        return 1
    fi
    local bbv_file="${bbv_files[0]}"

    local elf_bin="${PROJECT_ROOT}/output/yolo_inference"
    local sysroot="${PROJECT_ROOT}/output/sysroot"

    if [[ ! -f "$elf_bin" ]]; then
        record_step_result "$step" "FAIL" "output/yolo_inference not found — run Step 3 first"
        return 1
    fi

    # Use profile_to_dfg.sh which chains analyze_bbv + DFG generation
    local dfg_args=(
        "--bbv" "$bbv_file"
        "--elf" "$elf_bin"
        "--top" "$TOP_N"
    )
    if [[ -d "$sysroot" ]]; then
        dfg_args+=("--sysroot" "$sysroot")
    fi
    if (( COVERAGE_PCT > 0 )); then
        dfg_args+=("--coverage" "$COVERAGE_PCT")
    fi

    log_info "  Running profile_to_dfg.sh (top=${TOP_N}, coverage=${COVERAGE_PCT})..."
    if ! bash "${PROJECT_ROOT}/tools/profile_to_dfg.sh" "${dfg_args[@]}" 2>&1; then
        record_step_result "$step" "FAIL" "profile_to_dfg.sh exited with error"
        return 1
    fi

    # Verify dfg/ directory has content
    if ! check_dfg_artifacts; then
        record_step_result "$step" "FAIL" "dfg/ directory is empty after DFG generation"
        return 1
    fi

    record_step_result "$step" "PASS" "dfg/ generated"
    return 0
}
```

- [ ] **Step 2: Update `run_setup()` — replace Step 5 and Step 6 placeholders**

Find:
```bash
            5) log_info "=== Step 5: ${STEP_NAMES[5]} ===" ; record_step_result 5 "FAIL" "not yet implemented" ;;
            6) log_info "=== Step 6: ${STEP_NAMES[6]} ===" ; record_step_result 6 "FAIL" "not yet implemented" ;;
```

Replace with:
```bash
            5) step5_hotspot_report ;;
            6) step6_dfg_generation ;;
```

- [ ] **Step 3: Smoke test — `--help` still works**

```bash
bash setup.sh --help
```

Expected: usage text, exit 0.

- [ ] **Step 4: Commit**

```bash
git add setup.sh
git commit -m "feat(setup): add Steps 5-6 (hotspot report, DFG generation)"
```

---

### Task 7: Implement Step 7 (Generate Report)

**Files:**
- Modify: `setup.sh` — add `step7_report()`, update `run_setup()` to call it

This is the final step, which always runs regardless of previous failures. It writes `setup-report.txt` with the format specified in the redesign doc.

- [ ] **Step 1: Add Step 7 report function**

Insert after `step6_dfg_generation()` and before `# run_setup()`:

```bash
# =============================================================================
# Step 7: Generate Report
# =============================================================================

step7_report() {
    local step=7
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    local report_path="${PROJECT_ROOT}/${REPORT_FILE}"
    local overall="PASS"

    {
        echo "RVFuse Setup Report"
        echo "Generated: $(date -Iseconds)"
        echo "Options: ${SCRIPT_OPTIONS}"
        echo ""

        for i in 0 1 2 3 4 5 6 7; do
            local status="${STEP_STATUS[$i]:-UNKNOWN}"
            local message="${STEP_MESSAGE[$i]:-}"
            local step_line="Step ${i}: ${STEP_NAMES[$i]}"

            # Pad step line to 36 chars for alignment
            local padded
            padded="$(printf '%-36s' "$step_line")"

            if [[ "$status" == "SKIPPED" ]]; then
                echo "${padded}[SKIPPED] (${message})"
            elif [[ "$status" == "PASS" ]]; then
                echo "${padded}[PASS]   ${message}"
            elif [[ "$status" == "FAIL" ]]; then
                echo "${padded}[FAIL]   ${message}"
                overall="FAIL"
            else
                echo "${padded}[${status}]"
                overall="FAIL"
            fi
        done

        echo ""
        echo "Overall: ${overall}"
    } > "$report_path" || true  # || true prevents set -e from exiting before file check

    if [[ ! -f "$report_path" ]]; then
        record_step_result "$step" "FAIL" "could not write ${REPORT_FILE}"
        return 1
    fi

    record_step_result "$step" "PASS" "report written to ${REPORT_FILE}"
    return 0
}
```

- [ ] **Step 2: Verify `run_setup()` already calls `step7_report`**

The `run_setup()` function from Task 3 already has:
```bash
        if (( step_num == 7 )); then
            step7_report
            continue
        fi
```
No changes needed.

- [ ] **Step 3: Smoke test — `--help` still works**

```bash
bash setup.sh --help
```

Expected: usage text, exit 0.

- [ ] **Step 4: Smoke test — full run with all steps missing produces FAIL report**

```bash
bash setup.sh 2>&1 | tail -5; echo "---"; cat setup-report.txt 2>/dev/null || echo "No report"
```

Expected: Steps 0-6 show FAIL (no artifacts), Step 7 shows PASS, Overall: FAIL.

- [ ] **Step 5: Commit**

```bash
git add setup.sh
git commit -m "feat(setup): add Step 7 (report generation)"
```

---

### Task 8: Final integration, cleanup, and CLAUDE.md update

**Files:**
- Modify: `setup.sh` — final review, shellcheck compliance
- Modify: `CLAUDE.md` — update Key Commands section with new `setup.sh` usage

- [ ] **Step 1: Run shellcheck on the final script**

```bash
shellcheck setup.sh 2>&1 || true
```

Fix any errors. Expected warnings (acceptable, suppress with `# shellcheck disable=...`):
- `SC2034` for unused `STEP0_ARTIFACTS` etc. referenced indirectly via nameref — suppress at declaration
- Any `set -e` interaction warnings from `|| true` patterns

- [ ] **Step 2: Verify the complete script line count**

```bash
wc -l setup.sh
```

Expected: ~350-420 lines. If significantly over 420, review for accidental duplication.

- [ ] **Step 3: Update CLAUDE.md — add `setup.sh` to Key Commands**

In `CLAUDE.md`, add the following entry to the `## Key Commands` section, after the "Run analyze_bbv.py tests" block:

```markdown
# Run the full setup pipeline (Steps 0-6) with report
./setup.sh

# Run with specific options
./setup.sh --shallow --bbv-interval 50000 --top 30 --coverage 85

# Force re-run specific steps (deletes artifacts first)
./setup.sh --force 2,3     # re-build QEMU and Docker image
./setup.sh --force-all      # re-run everything from scratch
```

- [ ] **Step 4: Update CLAUDE.md — update "Recent Changes" section**

Add to the Recent Changes list:

```markdown
- setup.sh redesign: Rewrote as thin orchestrator for README Steps 0-6 (was 5-step repo setup)
```

- [ ] **Step 5: Integration smoke test — full dry run**

```bash
bash setup.sh --help
bash setup.sh 2>&1 | head -20
```

Expected: `--help` prints usage; full run starts with prerequisites check and attempts Step 0.

- [ ] **Step 6: Commit**

```bash
git add setup.sh CLAUDE.md
git commit -m "feat(setup): finalize setup.sh redesign — thin orchestrator for README Steps 0-6"
```

---

## Self-Review

**1. Spec coverage check:**

| Spec Requirement | Task | Status |
|---|---|---|
| Rewrite setup.sh as thin orchestrator (~300-400 lines) | Tasks 2-7 | Covered — skeleton + all 8 steps |
| 8 steps (0-7) matching README pipeline | Tasks 3-7 | Covered — step 0 through step 7 |
| `--force <steps>` with comma-separated step numbers | Task 2 | Covered — validates 0-7 range |
| `--force-all` | Task 2 | Covered — sets all steps 0-7 |
| `--shallow` for Step 0 | Task 2, 3 | Covered — `--depth 1` flag |
| `--bbv-interval <N>` for Step 4 | Task 2, 5 | Covered — passed to QEMU `-plugin` |
| `--top <N>` for Steps 5-6 | Task 2, 6 | Covered — passed to analyze_bbv.py and profile_to_dfg.sh |
| `--coverage <N>` for Steps 5-6 | Task 2, 6 | Covered — passed to profile_to_dfg.sh |
| Artifact-based skip detection | Task 2 | Covered — `step_artifacts_exist()` |
| Force cleanup (delete artifacts before re-run) | Task 3 | Covered — `cleanup_step()` dispatcher |
| Prerequisites: git >= 2.30, python3, Docker, disk space | Task 2 | Covered — `check_prerequisites()` |
| Report format matching spec | Task 7 | Covered — `step7_report()` |
| Exit code 0/1 based on steps 0-6 | Task 2 | Covered — `main()` exit logic |
| Delete specs/003-automated-setup-flow/ | Task 1 | Covered |
| Delete tests/setup/ | Task 1 | Covered |
| Remove tests/setup/test_helper/ from .gitignore | Task 1 | Covered |

**2. Placeholder scan:** No TBD, TODO, "implement later", "add appropriate error handling", or "similar to Task N" found. All code blocks contain complete, copy-pasteable code.

**3. Type consistency check:**
- Step numbers consistently use 0-7 throughout
- `STEP_NAMES` associative array keys: 0-7 ✓
- `STEP_STATUS`/`STEP_MESSAGE` associative array keys: 0-7 ✓
- `TOTAL_STEPS=8` matches 8 steps (0-7) ✓
- `get_artifact_names()` returns correct variable names ✓
- `cleanup_step()` covers cases 0-7 ✓
- `record_step_result()` called with correct step numbers ✓
- `run_setup()` loop iterates `0 1 2 3 4 5 6 7` ✓
- `main()` exit check loops `0 1 2 3 4 5 6` (excludes step 7 report) ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-06-setup-script-redesign.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
