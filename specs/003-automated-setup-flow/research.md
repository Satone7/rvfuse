# Research: Automated Setup Flow Script

**Date**: 2026-04-06 | **Branch**: `003-automated-setup-flow`

## R-001: Bash Argument Parsing for Long Options

**Decision**: Use manual `while/case` parsing with `--` prefix support

**Rationale**: The script needs `--force`, `--force-all`, and `--shallow` flags. POSIX `getopts` does not support long options. Manual parsing is simple for 3 flags, avoids adding dependencies, and is widely understood by bash developers. The flag count is small enough that a manual approach stays under 50 lines.

**Alternatives considered**:
- `getopt` (GNU enhanced): Supports long options but behavior differs between GNU and BSD implementations, reducing portability
- External argument parsing libraries (e.g., `argbash`): Adds a dependency for minimal benefit given only 3 flags

**Architecture alignment**: N/A (implementation detail)

---

## R-002: Bash Testing Framework

**Decision**: Use `bats-core` for unit and integration testing

**Rationale**: `bats-core` is the most widely adopted bash testing framework with TAP-compatible output. It supports setup/teardown hooks, parallel execution, and is available as a single-file install. It aligns with the ground-rules requirement for deterministic, well-named tests.

**Alternatives considered**:
- `shunit2`: Older, less active maintenance, xUnit-style assertions feel unnatural in bash
- Pure bash test scripts (no framework): Harder to organize, no built-in assertion helpers, harder to get TAP output
- `shellcheck` (static analysis): Complements but does not replace runtime testing

**Architecture alignment**: N/A (implementation detail)

---

## R-003: Report File Format

**Decision**: Plain text with structured section headers

**Rationale**: Plain text is universally readable without tools, easy to `grep`, `cat`, or share in messages. Structured section headers (e.g., `== Step 1: Clone Repository ==`) allow both human scanning and simple programmatic parsing. Markdown would add rendering dependency; JSON would reduce readability for the primary audience (contributors reading setup results).

**Alternatives considered**:
- Markdown: Better rendering in some contexts but adds complexity to generation and requires a viewer for best experience
- JSON: Easy to parse programmatically but poor readability for contributors checking setup status
- TAP format: Good for test output but unfamiliar to most contributors

**Architecture alignment**: Aligns with quality target "setup guidance is understandable without relying on future workflow assumptions"

---

## R-004: Project Root Detection

**Decision**: Check for the presence of a known marker file/directory at the script's parent directory or use `git rev-parse --show-toplevel`

**Rationale**: `git rev-parse --show-toplevel` is reliable inside a git repository and doesn't require marker files. Since the quickstart Step 1 produces a git clone, the workspace is always a git repo when the script runs. Falling back to checking for `docs/` directory handles edge cases.

**Alternatives considered**:
- Marker file (e.g., `.rvfuse-root`): Requires maintaining an extra file; fails if forgotten
- `pwd` basename check: Fragile if repo is cloned into a differently-named directory

**Architecture alignment**: Consistent with ADR-001 (Git-based workspace)

---

## R-005: Script File Location and Invocation

**Decision**: Place `setup.sh` in the repository root (`/setup.sh`), invoked as `./setup.sh` from the project root

**Rationale**: Contributors clone the repo, `cd` into it, and run `./setup.sh`. This is the simplest possible invocation path. No PATH setup, no subdirectory navigation, no install step. The script resolves its own location using `dirname` if needed for relative paths to library files.

**Alternatives considered**:
- `scripts/setup.sh`: Requires `./scripts/setup.sh` or adding `scripts/` to PATH; slightly less discoverable
- Install script to `/usr/local/bin`: Overly complex for a per-repo setup tool

**Architecture alignment**: Aligns with quality target "contributors can complete setup within 30 minutes"
