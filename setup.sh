#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# RVFuse Automated Setup Flow Script
# =============================================================================

# --- Script-level constants ---

REPORT_FILE="setup-report.txt"
readonly MIN_DISK_GB=20
readonly MIN_GIT_VERSION_MAJOR=2
readonly MIN_GIT_VERSION_MINOR=30
readonly TOTAL_STEPS=5

# Submodule URLs — documented here per ADR-004 (traceable sources);
# actual URLs come from .gitmodules at submodule-update time.
# shellcheck disable=SC2034
readonly QEMU_URL="https://github.com/XUANTIE-RV/qemu"
# shellcheck disable=SC2034
readonly LLVM_URL="https://github.com/XUANTIE-RV/llvm-project"

# --- Global state ---

declare -g PROJECT_ROOT=""
declare -g SHALLOW_CLONE=0
declare -ga FORCE_STEPS=()

# Per-step result tracking (indexed 1-5)
declare -gA STEP_STATUS=()
declare -gA STEP_MESSAGE=()
declare -gA STEP_STARTED=()
declare -gA STEP_FINISHED=()
declare -gA STEP_WARNINGS=()

# Step artifact definitions (relative to PROJECT_ROOT)
readonly -a STEP1_ARTIFACTS=(".git")
readonly -a STEP2_ARTIFACTS=(
    "docs/architecture.md"
    "memory/ground-rules.md"
    "specs/001-riscv-fusion-setup/spec.md"
)
readonly -a STEP3_ARTIFACTS=(
    "third_party/qemu/.git"
    "third_party/llvm-project/.git"
)
readonly -a STEP4_ARTIFACTS=()
readonly -a STEP5_ARTIFACTS=("${REPORT_FILE}")

# Step names
readonly -A STEP_NAMES=(
    [1]="Clone Repository"
    [2]="Review Project Scope"
    [3]="Initialize Dependencies"
    [4]="Verify Setup"
    [5]="Generate Report"
)

# Script options string for report
SCRIPT_OPTIONS=""

# =============================================================================
# Logging Functions (T009)
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
# CLI Argument Parsing (T004)
# =============================================================================

print_usage() {
    cat <<'EOF'
usage: setup.sh [--force <steps>] [--force-all] [--shallow]

Options:
  --force <steps>   Re-execute specified steps (comma-separated, e.g., "3" or "2,4")
  --force-all       Re-execute all steps from scratch
  --shallow         Use --depth 1 for submodule clones (faster, less disk)
  --help            Show this help message
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
                    if ! [[ "$num" =~ ^[1-5]$ ]]; then
                        log_error "Invalid step number: $num (must be 1-5)"
                        return 1
                    fi
                    FORCE_STEPS+=("$num")
                done
                ;;
            --force-all)
                FORCE_STEPS=(1 2 3 4 5)
                ;;
            --shallow)
                SHALLOW_CLONE=1
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
# Project Root Detection (T005)
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
# Prerequisite Checking (T006)
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

    # Check disk space
    local available_kb
    available_kb="$(df -k --output=avail "${PROJECT_ROOT}" 2>/dev/null | tail -1 | tr -d ' ')"
    if [[ -z "$available_kb" ]] || ! [[ "$available_kb" =~ ^[0-9]+$ ]]; then
        log_warn "Could not determine available disk space"
    else
        local available_gb
        available_gb=$((available_kb / 1024 / 1024))
        if (( available_gb < MIN_DISK_GB )); then
            log_warn "Available disk space: ~${available_gb}GB (recommended: ${MIN_DISK_GB}GB)"
            log_warn "Submodule clones may fail. Consider freeing disk space."
        else
            log_info "Available disk space: ~${available_gb}GB (ok)"
        fi
    fi

    return 0
}

# =============================================================================
# Artifact Checking (T007)
# =============================================================================

check_artifacts() {
    local -a artifacts=("$@")
    local all_exist=0
    local missing=()

    for artifact in "${artifacts[@]}"; do
        local full_path="${PROJECT_ROOT}/${artifact}"
        if [[ ! -e "$full_path" ]]; then
            all_exist=1
            missing+=("$artifact")
        elif [[ -f "$full_path" && ! -s "$full_path" ]]; then
            all_exist=1
            missing+=("$artifact (empty)")
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        log_info "Missing artifacts: ${missing[*]}"
    fi

    return "$all_exist"
}

# =============================================================================
# Step Result Recording (T008)
# =============================================================================

record_step_started() {
    local step_num="$1"
    # shellcheck disable=SC2034  # consumed by data model; future report enhancement
    STEP_STARTED[$step_num]="$(date -Iseconds)"
    log_info "=== Step ${step_num}: ${STEP_NAMES[$step_num]} ==="
}

record_step_result() {
    local step_num="$1"
    local status="$2"  # PASS, FAIL, SKIPPED
    local message="${3:-}"
    STEP_STATUS[$step_num]="$status"
    STEP_MESSAGE[$step_num]="$message"
    # shellcheck disable=SC2034  # consumed by data model; future report enhancement
    STEP_FINISHED[$step_num]="$(date -Iseconds)"

    case "$status" in
        PASS)   log_info "Step ${step_num}: PASS" ;;
        FAIL)   log_error "Step ${step_num}: FAIL — ${message}" ;;
        SKIP*)  log_info "Step ${step_num}: SKIPPED" ;;
    esac
}

record_step_warning() {
    local step_num="$1"
    local warning="$2"
    STEP_WARNINGS[$step_num]+="${warning}; "
}

# =============================================================================
# Step 1: Clone Repository (T013)
# =============================================================================

step1_clone() {
    local step=1
    record_step_started "$step"

    if [[ -e "${PROJECT_ROOT}/.git" ]]; then
        record_step_result "$step" "SKIPPED" "already cloned"
        return 0
    fi

    # Script is already running from inside the repo, so .git must exist.
    # If we reach here, something is wrong with the detection.
    record_step_result "$step" "FAIL" "not a git repository (cannot run setup.sh outside the repo)"
    return 1
}

# =============================================================================
# Step 2: Review Project Scope (T014)
# =============================================================================

step2_review_scope() {
    local step=2
    record_step_started "$step"

    local missing=()
    local empty=()
    for artifact in "${STEP2_ARTIFACTS[@]}"; do
        local full_path="${PROJECT_ROOT}/${artifact}"
        if [[ ! -f "$full_path" ]]; then
            missing+=("$artifact")
        elif [[ ! -s "$full_path" ]]; then
            empty+=("$artifact")
        fi
    done

    if (( ${#missing[@]} > 0 || ${#empty[@]} > 0 )); then
        local msg=""
        (( ${#missing[@]} > 0 )) && msg+="Missing: ${missing[*]}. "
        (( ${#empty[@]} > 0 )) && msg+="Empty: ${empty[*]}. "
        record_step_result "$step" "FAIL" "$msg"
        return 1
    fi

    record_step_result "$step" "PASS" "all scope documents present and non-empty"
    return 0
}

# =============================================================================
# Step 3: Initialize Mandatory Dependencies (T015)
# =============================================================================

step3_init_deps() {
    local step=3
    record_step_started "$step"

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
        record_step_warning "$step" "Check network connectivity and GitHub status"
        return 1
    fi

    record_step_result "$step" "PASS" "all mandatory dependencies initialized"
    return 0
}

# =============================================================================
# Step 4: Verify Setup Completion (T016)
# =============================================================================

step4_verify_setup() {
    local step=4
    record_step_started "$step"

    local passed=0
    local total=7
    local failed_checks=()

    # Check 1: Repository structure
    if [[ -d "${PROJECT_ROOT}/docs" && -d "${PROJECT_ROOT}/specs" && -d "${PROJECT_ROOT}/memory" ]]; then
        (( passed++ ))
    else
        failed_checks+=("repo structure: missing docs/, specs/, or memory/")
    fi

    # Check 2: Architecture readable
    if [[ -s "${PROJECT_ROOT}/docs/architecture.md" ]]; then
        (( passed++ ))
    else
        failed_checks+=("architecture readable: docs/architecture.md missing or empty")
    fi

    # Check 3: Setup guide readable
    if [[ -s "${PROJECT_ROOT}/specs/001-riscv-fusion-setup/quickstart.md" ]]; then
        (( passed++ ))
    else
        failed_checks+=("setup guide readable: specs/001-riscv-fusion-setup/quickstart.md missing or empty")
    fi

    # Check 4: Ground-rules readable
    if [[ -s "${PROJECT_ROOT}/memory/ground-rules.md" ]]; then
        (( passed++ ))
    else
        failed_checks+=("ground-rules readable: memory/ground-rules.md missing or empty")
    fi

    # Check 5: Mandatory deps documented
    if grep -q "Xuantie QEMU" "${PROJECT_ROOT}/docs/architecture.md" 2>/dev/null && \
       grep -q "Xuantie LLVM" "${PROJECT_ROOT}/docs/architecture.md" 2>/dev/null; then
        (( passed++ ))
    else
        failed_checks+=("mandatory deps documented: QEMU or LLVM not found in architecture.md")
    fi

    # Check 6: Optional deps documented
    if grep -q "optional" "${PROJECT_ROOT}/docs/architecture.md" 2>/dev/null && \
       grep -q "newlib" "${PROJECT_ROOT}/docs/architecture.md" 2>/dev/null; then
        (( passed++ ))
    else
        failed_checks+=("optional deps documented: newlib optional label not found")
    fi

    # Check 7: Dependency sources traceable
    if grep -q "github.com/XUANTIE-RV" "${PROJECT_ROOT}/docs/architecture.md" 2>/dev/null; then
        (( passed++ ))
    else
        failed_checks+=("dependency sources traceable: XUANTIE-RV URLs not found")
    fi

    if (( ${#failed_checks[@]} > 0 )); then
        record_step_result "$step" "FAIL" "(${passed}/${total} checks passed)"
        record_step_warning "$step" "Failed: ${failed_checks[*]}"
        return 1
    fi

    record_step_result "$step" "PASS" "(${passed}/${total} checks)"
    return 0
}

# =============================================================================
# Step 5: Generate Report (T017, T029, T030, T031)
# =============================================================================

step5_report() {
    local step=5
    record_step_started "$step"

    local report_path="${PROJECT_ROOT}/${REPORT_FILE}"
    local overall="PASS"

    # Pre-set step 5 status (will be updated to FAIL if write fails)
    STEP_STATUS[$step]="PASS"

    {
        echo "RVFuse Setup Report"
        echo "Generated: $(date -Iseconds)"
        echo "Options: ${SCRIPT_OPTIONS}"
        echo ""

        for i in $(seq 1 "$TOTAL_STEPS"); do
            local status="${STEP_STATUS[$i]:-UNKNOWN}"
            local message="${STEP_MESSAGE[$i]:-}"
            local step_line="Step ${i}: ${STEP_NAMES[$i]}"

            # Pad step line to 40 chars for alignment
            local padded
            padded="$(printf '%-40s' "$step_line")"

            if [[ "$status" == "SKIPPED" ]]; then
                echo "${padded}[SKIPPED] (${message})"
            elif [[ "$status" == "PASS" ]]; then
                echo "${padded}[PASS]  ${message}"
            elif [[ "$status" == "FAIL" ]]; then
                echo "${padded}[FAIL]  ${message}"
                local warnings="${STEP_WARNINGS[$i]:-}"
                if [[ -n "$warnings" ]]; then
                    # Remove trailing "; " from warnings
                    warnings="${warnings%; }"
                    echo "  Hint: ${warnings}"
                fi
                overall="FAIL"
            else
                echo "${padded}[${status}]"
                overall="FAIL"
            fi
        done

        echo ""
        echo "Overall: ${overall}"
    } > "$report_path"

    if [[ ! -f "$report_path" ]]; then
        record_step_result "$step" "FAIL" "could not write ${REPORT_FILE}"
        return 1
    fi

    record_step_result "$step" "PASS" "report written to ${REPORT_FILE}"
    return 0
}

# =============================================================================
# Main Execution Loop (T018, T023)
# =============================================================================

# Map step numbers to function suffix names
declare -gA step_funcs=(
    [1]="clone"
    [2]="review_scope"
    [3]="init_deps"
    [4]="verify_setup"
    [5]="report"
)

run_setup() {
    for step_num in $(seq 1 "$TOTAL_STEPS"); do
        local artifacts=()
        case "$step_num" in
            1) artifacts=("${STEP1_ARTIFACTS[@]}") ;;
            2) artifacts=("${STEP2_ARTIFACTS[@]}") ;;
            3) artifacts=("${STEP3_ARTIFACTS[@]}") ;;
            4) artifacts=("${STEP4_ARTIFACTS[@]}") ;;
            5) artifacts=("${STEP5_ARTIFACTS[@]}") ;;
        esac

        # Step 5 always runs (report generation)
        local should_skip=false
        if (( step_num != 5 )) && ! is_step_forced "$step_num"; then
            if (( ${#artifacts[@]} > 0 )) && check_artifacts "${artifacts[@]}"; then
                should_skip=true
            fi
        fi

        if [[ "$should_skip" == "true" ]]; then
            STEP_STATUS[$step_num]="SKIPPED"
            STEP_MESSAGE[$step_num]="artifacts exist"
            log_info "=== Step ${step_num}: ${STEP_NAMES[$step_num]} ==="
            log_info "Step ${step_num}: SKIPPED (artifacts exist)"
        else
            "step${step_num}_${step_funcs[$step_num]}" || true
        fi
    done
}

# =============================================================================
# Main Entry Point (T019)
# =============================================================================

main() {
    parse_args "$@"
    detect_project_root
    check_prerequisites

    cd "$PROJECT_ROOT"

    run_setup

    # Determine overall exit code
    local has_fail=0
    for i in $(seq 1 $((TOTAL_STEPS - 1))); do
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
