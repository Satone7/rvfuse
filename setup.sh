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
readonly TOTAL_STEPS=9  # Steps 0-8

# Step names (indexed 0-8)
readonly -A STEP_NAMES=(
    [0]="Init Submodules"
    [1]="Prepare Model"
    [2]="Build QEMU"
    [3]="YOLO Build"
    [4]="BBV Profiling"
    [5]="Hotspot Report"
    [6]="DFG Generation"
    [7]="Generate Report"
    [8]="Fusion Pipeline"
)

# Step artifact paths (relative to PROJECT_ROOT)
# Steps 4 (BBV) uses a glob pattern — handled specially in check function
# shellcheck disable=SC2034
readonly -a STEP0_ARTIFACTS=(
    "third_party/qemu/.git"
    "third_party/llvm-project/.git"
)
# shellcheck disable=SC2034
readonly -a STEP1_ARTIFACTS=(
    "output/yolo11n.ort"
    "output/test.jpg"
)
# shellcheck disable=SC2034
readonly -a STEP2_ARTIFACTS=(
    "third_party/qemu/build/contrib/plugins/libbbv.so"
    "tools/bbv/libbbv.so"
)
# Step 3 artifacts depend on target — checked specially in step_artifacts_exist
# shellcheck disable=SC2034
readonly -a STEP3_ARTIFACTS_INFERENCE=(
    "output/yolo_inference"
    "output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1"
)
# shellcheck disable=SC2034
readonly -a STEP3_ARTIFACTS_PREPROCESS=(
    "output/yolo_preprocess"
    "output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1"
)
# shellcheck disable=SC2034
readonly -a STEP3_ARTIFACTS_POSTPROCESS=(
    "output/yolo_postprocess"
    "output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1"
)
# shellcheck disable=SC2034
readonly -a STEP5_ARTIFACTS=(
    "output/hotspot.json"
)
# Step 6 (DFG Generation): non-empty output/dfg/ directory — checked specially
readonly -a STEP7_ARTIFACTS=()  # always runs
# Step 8 (Fusion Pipeline): fusion_candidates.json
readonly -a STEP8_ARTIFACTS=(
    "output/fusion_candidates.json"
)

# --- Global state ---

declare -g PROJECT_ROOT=""
declare -g SHALLOW_CLONE=0
declare -g BBV_INTERVAL=100000
declare -g TOP_N=20
declare -g COVERAGE_PCT=80
declare -g TARGET="inference"  # inference | preprocess | postprocess
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
usage: setup.sh [--force <steps>] [--force-all] [--shallow] [--target <T>] [--bbv-interval <N>] [--top <N>] [--coverage <N>] [--help]

Options:
  --force <steps>       Re-execute specified steps (comma-separated, e.g., "2" or "3,5")
                          Step IDs:
                            0=Init Submodules  1=Prepare Model  2=Build QEMU
                            3=YOLO Build       4=BBV Profiling  5=Hotspot Report
                            6=DFG Generation   7=Generate Report  8=Fusion Pipeline
  --force-all           Re-execute all steps from scratch (deletes artifacts)
  --shallow             Use --depth 1 for submodule clones (Step 0)
  --target <T>          Target binary for BBV profiling (default: inference)
                          inference  — YOLO inference runner (ORT)
                          preprocess — Video decode + resize + normalize
                          postprocess— YOLO output parsing + NMS + draw
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
                    if ! [[ "$num" =~ ^[0-8]$ ]]; then
                        log_error "Invalid step number: $num (must be 0-8)"
                        return 1
                    fi
                    FORCE_STEPS+=("$num")
                done
                ;;
            --force-all)
                FORCE_STEPS=(0 1 2 3 4 5 6 7 8)
                ;;
            --shallow)
                SHALLOW_CLONE=1
                ;;
            --target)
                (( i++ )) || true
                if (( i >= ${#args[@]} )); then
                    log_error "--target requires a value (inference|preprocess|postprocess)"
                    return 1
                fi
                local target_val="${args[$i]}"
                if [[ "$target_val" != "inference" && "$target_val" != "preprocess" && "$target_val" != "postprocess" ]]; then
                    log_error "Invalid target: $target_val (must be inference|preprocess|postprocess)"
                    return 1
                fi
                TARGET="$target_val"
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
    if [[ "$TARGET" != "inference" ]]; then
        opt_parts+=("target=${TARGET}")
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
    git_version="$(git --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' || true)"
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

# Step 4 special check: glob based on target
check_bbv_artifacts() {
    local pattern
    case "$TARGET" in
        inference)   pattern="yolo.bbv.*.bb" ;;
        preprocess)  pattern="bbv_pre.*.bb" ;;
        postprocess) pattern="bbv_post.*.bb" ;;
    esac
    local bbv_files
    bbv_files=( "${PROJECT_ROOT}"/output/${pattern} )
    if [[ ! -e "${bbv_files[0]}" ]]; then
        return 1
    fi
    for f in "${bbv_files[@]}"; do
        if [[ -s "$f" ]]; then
            return 0
        fi
    done
    return 1
}

# Step 6 special check: dfg/ directory must exist and be non-empty
check_dfg_artifacts() {
    # DFG output goes to output/dfg/ (default when --output-dir is not passed)
    local dfg_dir="${PROJECT_ROOT}/output/dfg"
    if [[ ! -d "$dfg_dir" ]]; then
        return 1
    fi
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
        3) echo "STEP3_ARTIFACTS_${TARGET^^}" ;;
        4) echo "__bbv__" ;;          # special target-based check
        5) echo "STEP5_ARTIFACTS" ;;
        6) echo "__dfg__" ;;          # special dir check
        7) echo "STEP7_ARTIFACTS" ;;
        8) echo "STEP8_ARTIFACTS" ;;
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
            log_info "Cleaning Step 3 artifacts (YOLO build)..."
            rm -f "${PROJECT_ROOT}/output/yolo_inference"
            rm -f "${PROJECT_ROOT}/output/yolo_preprocess"
            rm -f "${PROJECT_ROOT}/output/yolo_postprocess"
            rm -rf "${PROJECT_ROOT}/output/sysroot"
            ;;
        4)
            log_info "Cleaning Step 4 artifacts (BBV output)..."
            rm -f "${PROJECT_ROOT}"/output/yolo.bbv.*
            rm -f "${PROJECT_ROOT}"/output/bbv_pre.*
            rm -f "${PROJECT_ROOT}"/output/bbv_post.*
            ;;
        5)
            log_info "Cleaning Step 5 artifacts (hotspot report)..."
            rm -f "${PROJECT_ROOT}/output/hotspot.json"
            ;;
        6)
            log_info "Cleaning Step 6 artifacts (DFG output)..."
            rm -rf "${PROJECT_ROOT}/output/dfg"
            ;;
        7)
            # No cleanup for report step
            ;;
        8)
            log_info "Cleaning Step 8 artifacts (fusion output)..."
            rm -f "${PROJECT_ROOT}/output/fusion_patterns.json"
            rm -f "${PROJECT_ROOT}/output/fusion_candidates.json"
            rm -rf "${PROJECT_ROOT}/output/fusion_schemes"
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
# Step 2: Build QEMU
# =============================================================================

step2_build_qemu() {
    local step=2
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    if ! bash "${PROJECT_ROOT}/verify_bbv.sh" --force-rebuild 2>&1; then
        record_step_result "$step" "FAIL" "verify_bbv.sh --force-rebuild exited with error"
        return 1
    fi

    record_step_result "$step" "PASS" "QEMU + official and custom libbbv.so built"
    return 0
}

# =============================================================================
# Step 3: Docker Build
# =============================================================================

step3_docker_build() {
    local step=3
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    # TODO: ONNX Runtime Docker build needs adaptation for LLVM 22 toolchain.
    # The docker-onnxrt build.sh and Dockerfile were written for the previous
    # toolchain setup. After the LLVM toolchain migration, the ONNX Runtime
    # build (CMake config, compiler flags, sysroot extraction) needs to be
    # updated to use tools/docker-llvm/ instead.
    log_warn "Step 3 ONNX Runtime build requires LLVM 22 toolchain adaptation."
    log_warn "Skipping — run './tools/docker-onnxrt/build.sh' manually after adaptation."
    record_step_result "$step" "SKIP" "LLVM 22 toolchain adaptation needed"
    return 0
}

# =============================================================================
# Step 4: BBV Profiling
# =============================================================================

step4_bbv_profiling() {
    local step=4
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} (target=${TARGET}) ==="

    # Step 4 depends on Step 3 (yolo_inference binary). If Step 3 was skipped
    # due to LLVM 22 toolchain adaptation, Step 4 cannot run.
    if [[ ! -e "${PROJECT_ROOT}/output/yolo_inference" ]]; then
        log_warn "Step 4 requires yolo_inference from Step 3 — skipped."
        record_step_result "$step" "SKIP" "depends on Step 3 (ONNX Runtime build)"
        return 0
    fi

    local qemu_bin="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
    local plugin_so="${PROJECT_ROOT}/tools/bbv/libbbv.so"
    local inference_bin="${PROJECT_ROOT}/output/yolo_inference"
    local ort_model="${PROJECT_ROOT}/output/yolo11n.ort"
    local test_image="${PROJECT_ROOT}/output/test.jpg"
    local sysroot="${PROJECT_ROOT}/output/sysroot"
    local bbv_outfile="${PROJECT_ROOT}/output/yolo.bbv"

    local target_bin="" target_args=()

    case "$TARGET" in
        inference)
            target_bin="${PROJECT_ROOT}/output/yolo_inference"
            target_args=("${PROJECT_ROOT}/output/yolo11n.ort" "${PROJECT_ROOT}/output/test.jpg" "10")
            ;;
        preprocess)
            target_bin="${PROJECT_ROOT}/output/yolo_preprocess"
            target_args=("${PROJECT_ROOT}/output/test_video.mp4" "10")
            bbv_outfile="${PROJECT_ROOT}/output/bbv_pre"
            ;;
        postprocess)
            target_bin="${PROJECT_ROOT}/output/yolo_postprocess"
            target_args=("--synthetic" "${PROJECT_ROOT}/output/test.jpg")
            bbv_outfile="${PROJECT_ROOT}/output/bbv_post"
            ;;
    esac

    # Verify dependencies
    local missing_deps=()
    for dep in "$qemu_bin" "$plugin_so" "$target_bin"; do
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

    # Clean old BBV output for this target
    rm -f "${bbv_outfile}"*

    log_info "  Running QEMU BBV profiling: $(basename "$target_bin") (interval=${BBV_INTERVAL})..."
    if ! "${qemu_bin}" \
        -L "${sysroot}" \
        -plugin "${plugin_so}",interval="${BBV_INTERVAL}",outfile="${bbv_outfile}" \
        "${target_bin}" "${target_args[@]}" \
        2>&1; then
        record_step_result "$step" "FAIL" "qemu-riscv64 profiling exited with error"
        return 1
    fi

    # Verify BBV output was created
    local bbv_files=("${bbv_outfile}"*.bb)
    if [[ ! -e "${bbv_files[0]}" ]]; then
        record_step_result "$step" "FAIL" "no BBV output files generated"
        return 1
    fi

    local bbv_name
    bbv_name="$(basename "${bbv_files[0]}")"
    record_step_result "$step" "PASS" "${bbv_name}"
    return 0
}

# =============================================================================
# Step 5: Hotspot Report
# =============================================================================

step5_hotspot_report() {
    local step=5
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    # Select BBV file and ELF based on target
    local bbv_pattern elf_bin json_output
    case "$TARGET" in
        inference)
            bbv_pattern="${PROJECT_ROOT}/output/yolo.bbv.*.bb"
            elf_bin="${PROJECT_ROOT}/output/yolo_inference"
            json_output="${PROJECT_ROOT}/output/hotspot.json"
            ;;
        preprocess)
            bbv_pattern="${PROJECT_ROOT}/output/bbv_pre.*.bb"
            elf_bin="${PROJECT_ROOT}/output/yolo_preprocess"
            json_output="${PROJECT_ROOT}/output/bbv_pre_report.json"
            ;;
        postprocess)
            bbv_pattern="${PROJECT_ROOT}/output/bbv_post.*.bb"
            elf_bin="${PROJECT_ROOT}/output/yolo_postprocess"
            json_output="${PROJECT_ROOT}/output/bbv_post_report.json"
            ;;
    esac

    # Find BBV file
    local bbv_files=(${bbv_pattern})
    if [[ ! -e "${bbv_files[0]}" ]]; then
        record_step_result "$step" "FAIL" "no BBV output files found — run Step 4 first"
        return 1
    fi
    local bbv_file="${bbv_files[0]}"

    local sysroot="${PROJECT_ROOT}/output/sysroot"

    if [[ ! -f "$elf_bin" ]]; then
        record_step_result "$step" "FAIL" "$(basename "$elf_bin") not found — run Step 3 first"
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

    # Select BBV file and ELF based on target
    local bbv_pattern elf_bin dfg_dir
    case "$TARGET" in
        inference)
            bbv_pattern="${PROJECT_ROOT}/output/yolo.bbv.*.bb"
            elf_bin="${PROJECT_ROOT}/output/yolo_inference"
            dfg_dir="${PROJECT_ROOT}/output/dfg"
            ;;
        preprocess)
            bbv_pattern="${PROJECT_ROOT}/output/bbv_pre.*.bb"
            elf_bin="${PROJECT_ROOT}/output/yolo_preprocess"
            dfg_dir="${PROJECT_ROOT}/output/dfg_pre"
            ;;
        postprocess)
            bbv_pattern="${PROJECT_ROOT}/output/bbv_post.*.bb"
            elf_bin="${PROJECT_ROOT}/output/yolo_postprocess"
            dfg_dir="${PROJECT_ROOT}/output/dfg_post"
            ;;
    esac

    local bbv_files=(${bbv_pattern})
    if [[ ! -e "${bbv_files[0]}" ]]; then
        record_step_result "$step" "FAIL" "no BBV output files found — run Step 4 first"
        return 1
    fi
    local bbv_file="${bbv_files[0]}"

    local sysroot="${PROJECT_ROOT}/output/sysroot"

    if [[ ! -f "$elf_bin" ]]; then
        record_step_result "$step" "FAIL" "$(basename "$elf_bin") not found — run Step 3 first"
        return 1
    fi

    # Use profile_to_dfg.sh which chains analyze_bbv + DFG generation
    local dfg_args=(
        "--bbv" "$bbv_file"
        "--elf" "$elf_bin"
        "--top" "$TOP_N"
        "--output-dir" "$dfg_dir"
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

    # Verify dfg directory has content
    if [[ ! -d "$dfg_dir" ]] || (( $(find "$dfg_dir" -type f 2>/dev/null | wc -l) == 0 )); then
        record_step_result "$step" "FAIL" "DFG directory is empty after generation"
        return 1
    fi

    record_step_result "$step" "PASS" "$(basename "$dfg_dir")/ generated"
    return 0
}

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

        for i in 0 1 2 3 4 5 6 8; do
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

        # Step 7 (this step) - we know status is PASS if we reach here writing the report
        local step7_line="Step 7: ${STEP_NAMES[7]}"
        local padded7
        padded7="$(printf '%-36s' "$step7_line")"
        echo "${padded7}[PASS]   report written to ${REPORT_FILE}"

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

# =============================================================================
# Step 8: Fusion Pipeline (F1 + F2)
# =============================================================================

step8_fusion_pipeline() {
    local step=8
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} ==="

    local dfg_dir="${PROJECT_ROOT}/output/dfg/json"
    local hotspot_json="${PROJECT_ROOT}/output/hotspot.json"
    local patterns_json="${PROJECT_ROOT}/output/fusion_patterns.json"
    local candidates_json="${PROJECT_ROOT}/output/fusion_candidates.json"

    # Verify dependencies from earlier steps
    if [[ ! -d "$dfg_dir" ]]; then
        record_step_result "$step" "FAIL" "output/dfg/json not found — run Step 6 first"
        return 1
    fi
    if [[ ! -f "$hotspot_json" ]]; then
        record_step_result "$step" "FAIL" "output/hotspot.json not found — run Step 5 first"
        return 1
    fi

    # F1: Discover fusion patterns
    log_info "  Running F1: Pattern Mining..."
    if ! python3 -m tools.fusion discover \
        --dfg-dir "$dfg_dir" \
        --report "$hotspot_json" \
        --output "$patterns_json" \
        --no-agent \
        --top "$TOP_N" \
        2>&1; then
        record_step_result "$step" "FAIL" "F1 discover exited with error"
        return 1
    fi

    if [[ ! -f "$patterns_json" ]]; then
        record_step_result "$step" "FAIL" "fusion_patterns.json not created"
        return 1
    fi

    # F2: Score and rank candidates
    log_info "  Running F2: Scoring & Constraints..."
    if ! python3 -m tools.fusion score \
        --catalog "$patterns_json" \
        --output "$candidates_json" \
        --top "$TOP_N" \
        2>&1; then
        record_step_result "$step" "FAIL" "F2 score exited with error"
        return 1
    fi

    if [[ ! -f "$candidates_json" ]]; then
        record_step_result "$step" "FAIL" "fusion_candidates.json not created"
        return 1
    fi

    # Summary
    local pattern_count candidate_count
    pattern_count="$(python3 -c "import json; print(len(json.load(open('${patterns_json}'))['patterns']))")"
    candidate_count="$(python3 -c "import json; print(len(json.load(open('${candidates_json}'))['candidates']))")"

    record_step_result "$step" "PASS" "${pattern_count} patterns, ${candidate_count} candidates"
    return 0
}

# =============================================================================
# run_setup() — Main Execution Loop
# =============================================================================

run_setup() {
    for step_num in 0 1 2 3 4 5 6 7 8; do
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
            2) step2_build_qemu ;;
            3) step3_docker_build ;;
            4) step4_bbv_profiling ;;
            5) step5_hotspot_report ;;
            6) step6_dfg_generation ;;
            8) step8_fusion_pipeline ;;
        esac || true
    done
}

main() {
    parse_args "$@"
    detect_project_root
    check_prerequisites

    cd "$PROJECT_ROOT"

    run_setup

    # Determine overall exit code (steps 0-6,8; step 7 is always report)
    local has_fail=0
    for i in 0 1 2 3 4 5 6 8; do
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
