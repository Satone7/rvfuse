#!/usr/bin/env bash
# Tests for force re-execution integration
# Spec: T022, T026
#
# Strategy: We test the force / skip decision logic by exercising the
# interaction between FORCE_STEPS, artifact existence, and step execution.
# Since the actual step functions may need network or real submodules, we stub
# them so the tests are deterministic.

setup() {
    SETUP_TEST_MODE=1
    source "$BATS_TEST_DIRNAME/../../setup.sh"

    # Create a sandbox that mimics a fully completed project (all artifacts
    # present).  Individual tests can remove artifacts as needed.
    SANDBOX="$BATS_TMPDIR/force_sandbox_$$"
    mkdir -p "$SANDBOX"
    PROJECT_ROOT="$SANDBOX"

    # Simulate a fully completed workspace.
    mkdir -p "$SANDBOX/.git"
    mkdir -p "$SANDBOX/docs"
    echo "# Architecture" > "$SANDBOX/docs/architecture.md"
    mkdir -p "$SANDBOX/memory"
    echo "# Ground-rules" > "$SANDBOX/memory/ground-rules.md"
    mkdir -p "$SANDBOX/specs/001-riscv-fusion-setup"
    echo "# Spec" > "$SANDBOX/specs/001-riscv-fusion-setup/spec.md"
    echo "# Quickstart" > "$SANDBOX/specs/001-riscv-fusion-setup/quickstart.md"
    mkdir -p "$SANDBOX/third_party/qemu/.git"
    mkdir -p "$SANDBOX/third_party/llvm-project/.git"

    # Reset globals.
    FORCE_STEPS=()
    SHALLOW_CLONE=0
    STEP_STATUS=([1]="" [2]="" [3]="" [4]="" [5]="")
    STEP_MESSAGE=([1]="" [2]="" [3]="" [4]="" [5]="")
    STEP_STARTED=([1]="" [2]="" [3]="" [4]="" [5]="")
    STEP_FINISHED=([1]="" [2]="" [3]="" [4]="" [5]="")
    STEP_WARNINGS=()
    REPORT_FILE="setup-report.txt"
}

teardown() {
    rm -rf "$SANDBOX"
}

# ---------------------------------------------------------------------------
# Helper: track which steps actually executed
# ---------------------------------------------------------------------------
# We re-define the step functions so they simply record that they were called.

_stub_step_functions() {
    # Arrays to track execution order.
    _EXECUTED_STEPS=()

    step1_clone()   { _EXECUTED_STEPS+=(1); STEP_STATUS[1]="PASS"; STEP_MESSAGE[1]=""; }
    step2_review_scope()   { _EXECUTED_STEPS+=(2); STEP_STATUS[2]="PASS"; STEP_MESSAGE[2]=""; }
    step3_init_deps()      { _EXECUTED_STEPS+=(3); STEP_STATUS[3]="PASS"; STEP_MESSAGE[3]=""; }
    step4_verify_setup()   { _EXECUTED_STEPS+=(4); STEP_STATUS[4]="PASS"; STEP_MESSAGE[4]="(7/7 checks)"; }
    step5_report()         { _EXECUTED_STEPS+=(5); STEP_STATUS[5]="PASS"; STEP_MESSAGE[5]=""; }
}

# ---------------------------------------------------------------------------
# T022: --force 3 only re-runs Step 3
# ---------------------------------------------------------------------------

@test "force: --force 3 causes only Step 3 and Step 4 to execute when all artifacts exist" {
    # Note: Step 4 has no artifacts defined, so it always executes regardless of force.
    _stub_step_functions
    FORCE_STEPS=(3)
    run_setup
    # Step 3 ran because forced
    [[ " ${_EXECUTED_STEPS[*]} " == *" 3 "* ]]
    # Step 4 always runs (no artifacts to skip on)
    [[ " ${_EXECUTED_STEPS[*]} " == *" 4 "* ]]
    # Step 5 always runs (report generation)
    [[ " ${_EXECUTED_STEPS[*]} " == *" 5 "* ]]

    # Steps with artifacts that were NOT forced should be SKIPPED.
    [[ " ${_EXECUTED_STEPS[*]} " != *" 1 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " != *" 2 "* ]]
}

@test "force: non-forced steps with existing artifacts are SKIPPED" {
    _stub_step_functions
    FORCE_STEPS=(3)
    run_setup
    [[ "${STEP_STATUS[1]}" == "SKIPPED" ]]
    [[ "${STEP_STATUS[2]}" == "SKIPPED" ]]
    # Step 4 has no artifacts, so it always runs (not SKIPPED)
    [[ "${STEP_STATUS[4]}" == "PASS" ]]
}

@test "force: forced step that already has artifacts still executes" {
    _stub_step_functions
    FORCE_STEPS=(3)
    run_setup
    [[ "${STEP_STATUS[3]}" == "PASS" ]]
    [[ " ${_EXECUTED_STEPS[*]} " == *" 3 "* ]]
}

# ---------------------------------------------------------------------------
# T022: --force 2,4
# ---------------------------------------------------------------------------

@test "force: --force 2,4 causes Steps 2, 4 and 5 to execute" {
    # Step 4 has no artifacts, so it always runs. Step 5 always runs.
    _stub_step_functions
    FORCE_STEPS=(2 4)
    run_setup
    [[ " ${_EXECUTED_STEPS[*]} " == *" 2 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " == *" 4 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " == *" 5 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " != *" 1 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " != *" 3 "* ]]
}

# ---------------------------------------------------------------------------
# T026: --force-all re-executes all 5 steps
# ---------------------------------------------------------------------------

@test "force: --force-all causes all 5 steps to execute regardless of artifacts" {
    _stub_step_functions
    FORCE_STEPS=(1 2 3 4 5)
    run_setup
    # All steps should have executed.
    [[ " ${_EXECUTED_STEPS[*]} " == *" 1 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " == *" 2 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " == *" 3 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " == *" 4 "* ]]
    [[ " ${_EXECUTED_STEPS[*]} " == *" 5 "* ]]
    # None should be SKIPPED.
    for i in 1 2 3 4 5; do
        [[ "${STEP_STATUS[$i]}" != "SKIPPED" ]]
    done
}

# ---------------------------------------------------------------------------
# Force list validated against valid step range (1-5)
# ---------------------------------------------------------------------------

@test "force: step numbers outside 1-5 are rejected" {
    # parse_args should return 1 for invalid step numbers; this is tested
    # more thoroughly in test_args.sh.  Here we verify the integration:
    # if FORCE_STEPS somehow contains an invalid value, run_setup should not
    # execute it.
    _stub_step_functions
    FORCE_STEPS=(0)
    run run_setup
    # Step 0 should never appear in executed steps.
    [[ " ${_EXECUTED_STEPS[*]} " != *" 0 "* ]]
}

# ---------------------------------------------------------------------------
# No force: steps with missing artifacts still execute
# ---------------------------------------------------------------------------

@test "force: with no force list, steps with missing artifacts execute" {
    _stub_step_functions
    FORCE_STEPS=()

    # Remove Step 3 artifacts so they are "missing".
    rm -rf "$SANDBOX/third_party/qemu"
    rm -rf "$SANDBOX/third_party/llvm-project"

    run_setup
    # Step 3 must execute because artifacts are missing.
    [[ " ${_EXECUTED_STEPS[*]} " == *" 3 "* ]]
}

@test "force: with no force list, steps with present artifacts are skipped" {
    _stub_step_functions
    FORCE_STEPS=()

    run_setup
    # Step 1 artifacts (.git/) exist, so it should be SKIPPED.
    [[ "${STEP_STATUS[1]}" == "SKIPPED" ]]
}
