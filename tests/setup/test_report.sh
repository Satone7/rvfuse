#!/usr/bin/env bash
# Tests for step5_report()
# Spec: T027, T028

setup() {
    SETUP_TEST_MODE=1
    source "$BATS_TEST_DIRNAME/../../setup.sh"

    SANDBOX="$BATS_TMPDIR/report_sandbox_$$"
    mkdir -p "$SANDBOX"
    PROJECT_ROOT="$SANDBOX"

    # Ensure REPORT_FILE points inside sandbox (relative to PROJECT_ROOT).
    REPORT_FILE="setup-report.txt"

    # Reset step tracking state.
    STEP_STATUS=()
    STEP_MESSAGE=()
    STEP_STARTED=()
    STEP_FINISHED=()
    STEP_WARNINGS=()
}

teardown() {
    rm -rf "$SANDBOX"
}

# ---------------------------------------------------------------------------
# Helper: pre-populate step results so step5_report has data to write.
# ---------------------------------------------------------------------------

_populate_all_pass() {
    STEP_STATUS[1]="SKIPPED"; STEP_MESSAGE[1]="(already cloned)"
    STEP_STATUS[2]="PASS";    STEP_MESSAGE[2]=""
    STEP_STATUS[3]="PASS";    STEP_MESSAGE[3]=""
    STEP_STATUS[4]="PASS";    STEP_MESSAGE[4]="(7/7 checks)"
    STEP_STATUS[5]="PASS";    STEP_MESSAGE[5]=""
    STEP_STARTED[1]="" ; STEP_FINISHED[1]=""
    STEP_STARTED[2]="" ; STEP_FINISHED[2]=""
    STEP_STARTED[3]="" ; STEP_FINISHED[3]=""
    STEP_STARTED[4]="" ; STEP_FINISHED[4]=""
    STEP_STARTED[5]="" ; STEP_FINISHED[5]=""
    SCRIPT_OPTIONS="shallow"
}

_populate_with_failure() {
    STEP_STATUS[1]="PASS";    STEP_MESSAGE[1]=""
    STEP_STATUS[2]="FAIL";    STEP_MESSAGE[2]="Missing: specs/001-riscv-fusion-setup/spec.md"
    STEP_STATUS[3]="SKIPPED"; STEP_MESSAGE[3]=""
    STEP_STATUS[4]="SKIPPED"; STEP_MESSAGE[4]=""
    STEP_STATUS[5]="PASS";    STEP_MESSAGE[5]=""
    STEP_STARTED[1]="" ; STEP_FINISHED[1]=""
    STEP_STARTED[2]="" ; STEP_FINISHED[2]=""
    STEP_STARTED[3]="" ; STEP_FINISHED[3]=""
    STEP_STARTED[4]="" ; STEP_FINISHED[4]=""
    STEP_STARTED[5]="" ; STEP_FINISHED[5]=""
    SCRIPT_OPTIONS="none"
    STEP_WARNINGS[2]="Check missing documents and retry with --force 2"
}

# ---------------------------------------------------------------------------
# T027: Report structure
# ---------------------------------------------------------------------------

@test "report: writes to setup-report.txt (or REPORT_FILE)" {
    _populate_all_pass
    step5_report
    [[ -f "${PROJECT_ROOT}/${REPORT_FILE}" ]]
}

@test "report: contains 'RVFuse Setup Report' header" {
    _populate_all_pass
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"RVFuse Setup Report"* ]]
}

@test "report: contains 'Generated:' with ISO 8601 timestamp" {
    _populate_all_pass
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    # ISO 8601 pattern: YYYY-MM-DDTHH:MM:SS
    [[ "$output" =~ Generated:[[:space:]]*[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]
}

@test "report: contains 'Options:' line with flags used" {
    _populate_all_pass
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"Options:"* ]]
    [[ "$output" == *"shallow"* ]]
}

@test "report: each step line includes status PASS, FAIL, or SKIPPED" {
    _populate_all_pass
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"Step 1"*[[:space:]]*"SKIPPED"* ]]
    [[ "$output" == *"Step 2"*[[:space:]]*"PASS"* ]]
    [[ "$output" == *"Step 3"*[[:space:]]*"PASS"* ]]
    [[ "$output" == *"Step 4"*[[:space:]]*"PASS"* ]]
    [[ "$output" == *"Step 5"*[[:space:]]*"PASS"* ]]
}

@test "report: overall status line 'Overall: PASS' when all steps pass" {
    _populate_all_pass
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"Overall: PASS"* ]]
}

@test "report: Step 4 sub-check details appear (e.g. '(7/7 checks)')" {
    _populate_all_pass
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"7/7"* ]]
}

# ---------------------------------------------------------------------------
# T028: Failure format
# ---------------------------------------------------------------------------

@test "report: overall status is FAIL when any step fails" {
    _populate_with_failure
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"Overall: FAIL"* ]]
}

@test "report: includes indented 'Error:' line for failed steps" {
    _populate_with_failure
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"Error:"* ]]
}

@test "report: error line contains the error message" {
    _populate_with_failure
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"Missing"* ]]
}

@test "report: includes indented 'Hint:' line for failed steps" {
    _populate_with_failure
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"Hint:"* ]]
}

@test "report: hint line suggests recovery action with --force" {
    _populate_with_failure
    step5_report
    run cat "${PROJECT_ROOT}/${REPORT_FILE}"
    [[ "$output" == *"--force"* ]]
}
