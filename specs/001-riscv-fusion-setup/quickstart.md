# Quickstart: RVFuse Project Setup Foundation

**Target**: New contributors preparing the RVFuse workspace
**Time Target**: 30 minutes (excluding network download and third-party build time)

---

## Prerequisites

- Linux x86_64 workstation
- Git installed (version 2.30+)
- ~20GB disk space (for submodule clones)
- Network access to GitHub

---

## Step 1: Clone Repository (5 min)

```bash
# Clone the main repository
git clone https://github.com/your-org/RVFuse.git
cd RVFuse
```

**Verification**: Repository root contains `docs/`, `specs/`, `memory/` directories.

---

## Step 2: Review Project Scope (5 min)

Read the following documents to understand current phase scope:

| Document | Purpose | Path |
|----------|---------|------|
| Architecture | Scope, ADRs, quality targets | `docs/architecture.md` |
| Ground-rules | Development principles | `memory/ground-rules.md` |
| Specification | Feature requirements | `specs/001-riscv-fusion-setup/spec.md` |

**Key Scope Points**:
- Current phase: Project structure and dependency access
- Deferred: Hotspot detection, DFG generation, fusion validation
- newlib is optional - setup completes without it

---

## Step 3: Initialize Mandatory Dependencies (15 min)

```bash
# Initialize Xuantie QEMU (mandatory)
git submodule add https://github.com/XUANTIE-RV/qemu third_party/qemu

# Initialize Xuantie LLVM (mandatory)
git submodule add https://github.com/XUANTIE-RV/llvm-project third_party/llvm-project

# Optional: Initialize Xuantie newlib (optional - skip if not needed)
git submodule add https://github.com/XUANTIE-RV/newlib third_party/newlib
```

**Shallow Clone Option** (for faster setup):
```bash
git submodule add --depth 1 https://github.com/XUANTIE-RV/qemu third_party/qemu
```

**Network Failure Recovery**:
If GitHub is temporarily unavailable:
1. Wait and retry (GitHub outages typically resolve within hours)
2. Check GitHub status: https://githubstatus.com
3. Manual fallback: Download release tarball from repository releases page

---

## Step 4: Verify Setup Completion (5 min)

Run the verification checklist:

| Check | Command | Expected |
|-------|---------|----------|
| Repository structure | `ls -la` | docs/, specs/, memory/, third_party/ |
| Architecture readable | `cat docs/architecture.md` | File content visible |
| Setup guide readable | `cat specs/001-riscv-fusion-setup/quickstart.md` | This file visible |
| Ground-rules readable | `cat memory/ground-rules.md` | Principles visible |
| Mandatory deps documented | `grep -A5 "Key Dependencies" docs/architecture.md` | QEMU, LLVM listed |
| Optional deps documented | `grep "newlib" docs/architecture.md` | "optional" label present |
| Dependency sources traceable | `grep "github.com/XUANTIE-RV" docs/architecture.md` | URLs present |

**Completion Criteria**: All 7 checks pass.

---

## What's NOT Required (Deferred)

The following are **not required** for current-phase setup completion:

- Building QEMU, LLVM, or newlib
- Running profiling workflows
- Generating DFGs
- Fusion candidate analysis
- Cycle comparison tests

These will be introduced in future feature specifications.

---

## Troubleshooting

### Submodule initialization fails

**Symptom**: `git submodule add` returns network error

**Solution**:
1. Check network connectivity
2. Verify GitHub status (https://githubstatus.com)
3. Retry with `--depth 1` for shallow clone
4. Manual fallback: Download from releases page

### Directory structure mismatch

**Symptom**: Expected directories not present

**Solution**:
1. Verify you're in repository root (`pwd`)
2. Check clone was successful (`git status`)
3. Re-clone if structure is incomplete

### newlib confusion

**Symptom**: Unsure whether newlib is required

**Solution**:
1. Check `docs/architecture.md` for "optional" label
2. Setup completes without newlib in current phase
3. newlib becomes mandatory only when bare-metal runtime support needed

---

## Next Steps

After setup completion:
1. Review `memory/ground-rules.md` for development principles
2. Check `specs/001-riscv-fusion-setup/` for design artifacts
3. Wait for future feature specs (profiling, DFG, validation)

---

## References

| Dependency | Source | Status |
|------------|--------|--------|
| Xuantie QEMU | https://github.com/XUANTIE-RV/qemu | Mandatory |
| Xuantie LLVM | https://github.com/XUANTIE-RV/llvm-project | Mandatory |
| Xuantie newlib | https://github.com/XUANTIE-RV/newlib | Optional |