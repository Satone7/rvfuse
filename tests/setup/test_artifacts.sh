#!/usr/bin/env bash
# Tests for check_artifacts()
# Spec: T012

setup() {
    # Prevent main() from running when sourcing
    SETUP_TEST_MODE=1
    source "$BATS_TEST_DIRNAME/../../setup.sh"

    # Create a temp project-root sandbox for artifact tests.
    SANDBOX="$BATS_TMPDIR/artifact_sandbox_$$"
    mkdir -p "$SANDBOX"
    PROJECT_ROOT="$SANDBOX"

    # Create some files and directories to test against.
    mkdir -p "$SANDBOX/docs"
    echo "architecture" > "$SANDBOX/docs/architecture.md"
    mkdir -p "$SANDBOX/memory"
    echo "ground-rules" > "$SANDBOX/memory/ground-rules.md"
    mkdir -p "$SANDBOX/third_party/qemu/.git"
    mkdir -p "$SANDBOX/third_party/llvm-project/.git"
}

teardown() {
    rm -rf "$SANDBOX"
}

# ---------------------------------------------------------------------------
# All paths exist
# ---------------------------------------------------------------------------

@test "check_artifacts: returns 0 when all paths exist" {
    run check_artifacts "docs/architecture.md" "third_party/qemu/.git"
    [[ $status -eq 0 ]]
}

@test "check_artifacts: returns 0 for a single existing path" {
    run check_artifacts "docs/architecture.md"
    [[ $status -eq 0 ]]
}

# ---------------------------------------------------------------------------
# Some paths missing
# ---------------------------------------------------------------------------

@test "check_artifacts: returns 1 when any path is missing" {
    run check_artifacts "docs/architecture.md" "nonexistent/file.txt"
    [[ $status -eq 1 ]]
}

@test "check_artifacts: returns 1 when the only path is missing" {
    run check_artifacts "does/not/exist"
    [[ $status -eq 1 ]]
}

@test "check_artifacts: logs which paths are missing" {
    run check_artifacts "docs/architecture.md" "missing_dep/.git"
    [[ $status -eq 1 ]]
    [[ "$output" == *"missing_dep"* ]]
}

@test "check_artifacts: logs multiple missing paths" {
    run check_artifacts "foo/bar" "baz/qux"
    [[ $status -eq 1 ]]
    [[ "$output" == *"foo"* ]]
    [[ "$output" == *"baz"* ]]
}

# ---------------------------------------------------------------------------
# Empty files
# ---------------------------------------------------------------------------

@test "check_artifacts: returns 1 when a file path exists but is empty (zero bytes)" {
    touch "$SANDBOX/empty_file.md"
    run check_artifacts "empty_file.md"
    [[ $status -eq 1 ]]
}

@test "check_artifacts: returns 0 when a directory path exists even if directory is empty" {
    # Directories should not be subject to the "empty file" check —
    # e.g. third_party/qemu/.git may be an empty directory placeholder.
    mkdir -p "$SANDBOX/empty_dir"
    run check_artifacts "empty_dir"
    [[ $status -eq 0 ]]
}

@test "check_artifacts: returns 0 for non-empty files" {
    run check_artifacts "docs/architecture.md"
    [[ $status -eq 0 ]]
}

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@test "check_artifacts: returns 0 when called with no arguments" {
    # Design: "Empty artifact checks (no artifacts defined) always cause the
    # step to execute" — but check_artifacts() itself should return 0 when
    # there is nothing to check (vacuously true: all zero paths exist).
    run check_artifacts
    [[ $status -eq 0 ]]
}
