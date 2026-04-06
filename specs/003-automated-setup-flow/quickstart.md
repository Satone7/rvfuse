# Quickstart: Automated Setup Flow Script

**Target**: Contributors who want to set up RVFuse using the automated script
**Time Target**: <1 minute (script invocation; excluding network download and third-party build time)

---

## Prerequisites

- Linux x86_64 workstation
- Git installed (version 2.30+)
- ~20GB disk space (for submodule clones)
- Network access to GitHub

---

## Usage

### Full Setup (First Run)

```bash
./setup.sh
```

The script automatically detects which steps are already complete and skips them.

### Force Re-execute Specific Steps

```bash
# Re-run only Step 3 (Initialize Dependencies)
./setup.sh --force 3

# Re-run Steps 2 and 4
./setup.sh --force 2,4

# Re-run everything from scratch
./setup.sh --force-all
```

### Shallow Clone (Faster Downloads)

```bash
./setup.sh --shallow
```

Use `--depth 1` for submodule clones. Reduces download size significantly.

### Combine Flags

```bash
# Full re-execution with shallow clones
./setup.sh --force-all --shallow

# Re-run Step 3 only, using shallow clones
./setup.sh --force 3 --shallow
```

---

## What Each Step Does

| Step | Action | Artifact Checked |
|------|--------|-----------------|
| 1 | Clone Repository | `.git/` directory |
| 2 | Review Project Scope | `docs/architecture.md`, `memory/ground-rules.md`, `specs/001-riscv-fusion-setup/spec.md` |
| 3 | Initialize Mandatory Dependencies | `third_party/qemu/.git`, `third_party/llvm-project/.git` |
| 4 | Verify Setup Completion | All 7 quickstart verification checks |
| 5 | Generate Report | `setup-report.txt` |

---

## Output

After completion, check `setup-report.txt` in the project root for a summary of all step results, including any warnings or errors.

---

## Troubleshooting

### "Not in project root" error

The script must be run from the RVFuse repository root. Navigate to the repo directory first:

```bash
cd /path/to/RVFuse
./setup.sh
```

### Step 3 fails (network error)

Check your network connection and GitHub status (https://githubstatus.com), then retry:

```bash
./setup.sh --force 3
```

### Insufficient disk space

The script warns if available disk space is below ~20GB. Free up space and re-run.
