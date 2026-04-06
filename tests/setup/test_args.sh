#!/usr/bin/env bash
# Tests for parse_args()
# Spec: T021

setup() {
    # Prevent main() from running when sourcing
    SETUP_TEST_MODE=1
    source "$BATS_TEST_DIRNAME/../../setup.sh"

    # Reset global state before each test.
    FORCE_STEPS=()
    SHALLOW_CLONE=0
}

# ---------------------------------------------------------------------------
# --force
# ---------------------------------------------------------------------------

@test "parse_args: --force 3 sets FORCE_STEPS=(3)" {
    parse_args --force 3
    [[ ${#FORCE_STEPS[@]} -eq 1 ]]
    [[ "${FORCE_STEPS[0]}" == "3" ]]
}

@test "parse_args: --force 2,4 sets FORCE_STEPS=(2 4)" {
    parse_args --force 2,4
    [[ ${#FORCE_STEPS[@]} -eq 2 ]]
    [[ "${FORCE_STEPS[0]}" == "2" ]]
    [[ "${FORCE_STEPS[1]}" == "4" ]]
}

@test "parse_args: --force 1,3,5 sets FORCE_STEPS=(1 3 5)" {
    parse_args --force 1,3,5
    [[ ${#FORCE_STEPS[@]} -eq 3 ]]
    [[ "${FORCE_STEPS[0]}" == "1" ]]
    [[ "${FORCE_STEPS[1]}" == "3" ]]
    [[ "${FORCE_STEPS[2]}" == "5" ]]
}

# ---------------------------------------------------------------------------
# --force-all
# ---------------------------------------------------------------------------

@test "parse_args: --force-all sets FORCE_STEPS to all five steps" {
    parse_args --force-all
    [[ ${#FORCE_STEPS[@]} -eq 5 ]]
    [[ "${FORCE_STEPS[0]}" == "1" ]]
    [[ "${FORCE_STEPS[1]}" == "2" ]]
    [[ "${FORCE_STEPS[2]}" == "3" ]]
    [[ "${FORCE_STEPS[3]}" == "4" ]]
    [[ "${FORCE_STEPS[4]}" == "5" ]]
}

# ---------------------------------------------------------------------------
# --shallow
# ---------------------------------------------------------------------------

@test "parse_args: --shallow sets SHALLOW_CLONE=1" {
    parse_args --shallow
    [[ "$SHALLOW_CLONE" -eq 1 ]]
}

@test "parse_args: no --shallow leaves SHALLOW_CLONE=0" {
    parse_args
    [[ "$SHALLOW_CLONE" -eq 0 ]]
}

# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

@test "parse_args: --help prints usage and exits 0" {
    run parse_args --help
    [[ $status -eq 0 ]]
    [[ "$output" == *"usage"* || "$output" == *"Usage"* || "$output" == *"--force"* ]]
}

# ---------------------------------------------------------------------------
# Invalid step numbers
# ---------------------------------------------------------------------------

@test "parse_args: --force 6 prints error and returns 1" {
    run --separate-stderr parse_args --force 6
    [[ $status -eq 1 ]]
    [[ "$stderr" == *"1-5"* || "$stderr" == *"Invalid"* || "$stderr" == *"6"* ]]
}

@test "parse_args: --force 0 prints error and returns 1" {
    run --separate-stderr parse_args --force 0
    [[ $status -eq 1 ]]
    [[ "$stderr" == *"1-5"* || "$stderr" == *"Invalid"* || "$stderr" == *"0"* ]]
}

@test "parse_args: --force 2,6 prints error and returns 1" {
    run parse_args --force 2,6
    [[ $status -eq 1 ]]
}

# ---------------------------------------------------------------------------
# No arguments
# ---------------------------------------------------------------------------

@test "parse_args: no arguments leaves FORCE_STEPS empty and SHALLOW_CLONE=0" {
    parse_args
    [[ ${#FORCE_STEPS[@]} -eq 0 ]]
    [[ "$SHALLOW_CLONE" -eq 0 ]]
}

# ---------------------------------------------------------------------------
# Combined flags
# ---------------------------------------------------------------------------

@test "parse_args: --force 3 --shallow sets both correctly" {
    parse_args --force 3 --shallow
    [[ ${#FORCE_STEPS[@]} -eq 1 ]]
    [[ "${FORCE_STEPS[0]}" == "3" ]]
    [[ "$SHALLOW_CLONE" -eq 1 ]]
}

@test "parse_args: --shallow --force 2,4 sets both correctly regardless of flag order" {
    parse_args --shallow --force 2,4
    [[ ${#FORCE_STEPS[@]} -eq 2 ]]
    [[ "${FORCE_STEPS[0]}" == "2" ]]
    [[ "${FORCE_STEPS[1]}" == "4" ]]
    [[ "$SHALLOW_CLONE" -eq 1 ]]
}
