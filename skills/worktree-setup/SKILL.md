---
name: worktree-setup
description: |
  Prepare a git worktree development environment by linking to the main repository's
  compiled tools and submodules. Use this skill whenever a worktree is created (via
  `git worktree add` or Claude's EnterWorktree tool) to avoid re-cloning submodules
  and re-compiling QEMU, LLVM, sysroot, and other read-only resources.
  Trigger when: the user mentions "worktree", "setup worktree", "new branch in worktree",
  "enter worktree", "prepare worktree", or after creating a worktree directory.
---

# Worktree Environment Setup

## Purpose

When working in a git worktree, you need access to the main repository's compiled
resources without duplicating them. This skill creates symlinks to:

- **Submodules** (`third_party/`) — Shared via git's natural worktree behavior
- **Compiled tools** — QEMU build, LLVM install, BBV plugin
- **Sysroot** — Cross-compilation environment (flexible: shared or independent)
- **Model/test data** — YOLO model, test images (read-only artifacts)

## Resource Categories

### 1. Always Shared (Read-only)

These resources are large and should always be symlinked:

| Resource | Size | Main Repo Path | Symlink Target |
|----------|------|---------------|----------------|
| QEMU build | ~250MB | `third_party/qemu/build/` | Same path |
| LLVM install | ~1.5GB | `third_party/llvm-install/` | Same path |
| BBV plugin | ~22KB | `tools/bbv/libbbv.so` | Same path |
| Local LLVM | ~5KB | `tools/local-llvm/` | Same path |
| RVV intrinsic doc | Reference | `third_party/riscv-rvv-intrinsic-doc/` | Git submodule |

### 2. Flexibly Shared

| Resource | Main Repo Path | Options |
|----------|---------------|---------|
| Sysroot | `output/sysroot-new/` or `output/cross-ort/sysroot/` | Shared symlink OR independent copy |
| Model files | `output/yolo11n.ort`, `output/test.jpg` | Shared symlink (read-only) |

### 3. Always Local (Per-worktree)

| Resource | Reason |
|----------|--------|
| `output/*.bbv.*` files | BBV profiling data is work-specific |
| `output/hotspot*.json` | Analysis results are work-specific |
| `output/dfg/` | DFG generation results are work-specific |
| `output/yolo_inference*` | Build artifacts are work-specific |
| `output/fusion_*.json` | Fusion analysis results are work-specific |

## Setup Procedure

### Step 1: Identify Main Repository

First, find the main repository location. Git worktrees share `.git` objects,
but the main repo is where the worktree was created from:

```bash
# Get the main repository path
MAIN_REPO=$(git worktree list | grep -E '^\S+\s+\S+\s+\[master|main\]' | head -1 | cut -d' ' -f1)

# Alternative: if you know you're in a worktree, find the common .git
WORKTREE_GIT=$(git rev-parse --git-common-dir)
MAIN_REPO=$(dirname "$WORKTREE_GIT")
```

If using Claude's EnterWorktree, the main repo is the original working directory
before the worktree was created.

### Step 2: Initialize Submodules

Git submodules are NOT automatically initialized in worktrees. Run this in the worktree:

```bash
cd <worktree-path>
git submodule update --init
```

This is fast because git worktrees share the `.git` objects — no re-download needed.

### Step 3: Create Symlinks for Read-only Resources

Create symlinks pointing to the main repo's compiled resources:

```bash
# Define paths
MAIN_REPO=/path/to/main/rvfuse  # Replace with actual path
WORKTREE=$(pwd)

# QEMU build (if main repo has it)
if [ -d "$MAIN_REPO/third_party/qemu/build" ]; then
    mkdir -p "$WORKTREE/third_party/qemu"
    ln -s "$MAIN_REPO/third_party/qemu/build" "$WORKTREE/third_party/qEMU/build"
fi

# LLVM install (if main repo has it)
if [ -d "$MAIN_REPO/third_party/llvm-install" ]; then
    ln -s "$MAIN_REPO/third_party/llvm-install" "$WORKTREE/third_party/llvm-install"
fi

# BBV plugin (always symlink)
if [ -f "$MAIN_REPO/tools/bbv/libbbv.so" ]; then
    mkdir -p "$WORKTREE/tools/bbv"
    ln -s "$MAIN_REPO/tools/bbv/libbbv.so" "$WORKTREE/tools/bbv/libbbv.so"
fi

# Local LLVM tools (always symlink)
if [ -d "$MAIN_REPO/tools/local-llvm" ]; then
    ln -s "$MAIN_REPO/tools/local-llvm" "$WORKTREE/tools/local-llvm"
fi
```

**Note**: Some symlinks may already exist if the worktree was created with files
already in place. Check before creating: `ls -la third_party/qemu/build 2>/dev/null`

### Step 4: Sysroot Setup (Choose Option)

#### Option A: Shared Sysroot (Recommended for most work)

Link to an existing sysroot from the main repo:

```bash
# Check available sysroots in main repo
ls -la "$MAIN_REPO/output/" | grep sysroot

# Choose one (e.g., sysroot-new for general use)
mkdir -p "$WORKTREE/output"
ln -s "$MAIN_REPO/output/sysroot-new" "$WORKTREE/output/sysroot"

# Or use the llama.cpp sysroot if that's your target
ln -s "$MAIN_REPO/output/llama.cpp/sysroot" "$WORKTREE/output/sysroot"
```

#### Option B: Independent Sysroot (For specialized work)

If your worktree needs a different sysroot (different app, different architecture):

```bash
# Create independent sysroot directory
mkdir -p "$WORKTREE/output/sysroot"

# Populate via the application's build script (e.g., llama.cpp build.sh)
# Or copy from Docker image if needed
```

### Step 5: Model/Test Data (Optional)

If working on YOLO or similar inference workloads:

```bash
mkdir -p "$WORKTREE/output"
ln -s "$MAIN_REPO/output/yolo11n.ort" "$WORKTREE/output/yolo11n.ort"
ln -s "$MAIN_REPO/output/test.jpg" "$WORKTREE/output/test.jpg"
ln -s "$MAIN_REPO/output/yolo11n.onnx" "$WORKTREE/output/yolo11n.onnx"
```

### Step 6: Verify Setup

Check that symlinks are working:

```bash
# Test QEMU
$WORKTREE/third_party/qemu/build/qemu-riscv64 --version

# Test BBV plugin exists
ls -la $WORKTREE/tools/bbv/libbbv.so

# Test sysroot
ls -la $WORKTREE/output/sysroot/usr/lib/riscv64-linux-gnu/

# Test model files (if linked)
ls -la $WORKTREE/output/yolo11n.ort
```

## Quick Setup Script

For convenience, the skill can generate a setup script. Ask the user if they want
an automated setup script, then create `setup-worktree.sh` in the worktree:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Auto-detect main repo
MAIN_REPO=$(git worktree list | grep -E 'master|main' | head -1 | awk '{print $1}')
WORKTREE=$(pwd)

echo "Main repo: $MAIN_REPO"
echo "Worktree: $WORKTREE"

# Initialize submodules
git submodule update --init

# Create symlinks
mkdir -p third_party/qemu output tools/bbv

[ -d "$MAIN_REPO/third_party/qemu/build" ] && ln -sf "$MAIN_REPO/third_party/qEMU/build" third_party/qemu/build
[ -d "$MAIN_REPO/third_party/llvm-install" ] && ln -sf "$MAIN_REPO/third_party/llvm-install" third_party/llvm-install
[ -f "$MAIN_REPO/tools/bbv/libbbv.so" ] && ln -sf "$MAIN_REPO/tools/bbv/libbbv.so" tools/bbv/libbbv.so
[ -d "$MAIN_REPO/tools/local-llvm" ] && ln -sf "$MAIN_REPO/tools/local-llvm" tools/local-llvm

# Sysroot (ask user or use default)
if [ -d "$MAIN_REPO/output/sysroot-new" ]; then
    ln -sf "$MAIN_REPO/output/sysroot-new" output/sysroot
fi

# Model files
[ -f "$MAIN_REPO/output/yolo11n.ort" ] && ln -sf "$MAIN_REPO/output/yolo11n.ort" output/
[ -f "$MAIN_REPO/output/test.jpg" ] && ln -sf "$MAIN_REPO/output/test.jpg" output/

echo "Worktree setup complete!"
```

## Important Notes

### Git Worktree Behavior

- Worktrees share `.git` objects — `git submodule update --init` is fast
- Each worktree has its own working directory and can checkout different branches
- Deleting a worktree (`git worktree remove`) does NOT affect the main repo

### Symlink Safety

- Symlinks to read-only resources are safe — no risk of accidental modification
- Never symlink directories that will be written to (like `output/dfg/`)
- If a symlink target doesn't exist, the symlink will be a dangling link

### When to NOT Use Symlinks

- If the main repo's sysroot is incomplete or broken for your use case
- If you need to modify files that would affect other worktrees
- If the main repo is on a different filesystem (symlinks may not work)

## Cleanup

When done with the worktree:

```bash
# Remove symlinks (optional, not required for worktree removal)
find . -type l -delete

# Remove the worktree
git worktree remove <worktree-path>
```