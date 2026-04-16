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

### 1. Always Shared (Read-only, Symlinked)

These resources are large and should always be symlinked to the main repo:

| Resource | Size | Main Repo Path | Symlink Target |
|----------|------|---------------|----------------|
| QEMU source | ~431MB | `third_party/qemu/` | Main repo path |
| LLVM source | ~3.6GB | `third_party/llvm-project/` | Main repo path |
| QEMU build | ~250MB | `third_party/qemu/build/` | Included in QEMU source symlink |
| LLVM install | ~1.5GB | `third_party/llvm-install/` | Same path |
| BBV plugin | ~22KB | `tools/bbv/libbbv.so` | Same path |
| Local LLVM | ~5KB | `tools/local-llvm/` | Same path |
| RVV intrinsic doc | ~141MB | `third_party/riscv-rvv-intrinsic-doc/` | Main repo path |

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

### Step 2: Create Symlinks for Submodules and Read-only Resources

Git submodules are NOT automatically initialized in worktrees. Instead of running
`git submodule update --init` (which would check out ~4GB of submodule files), create
symlinks to the main repo's submodule directories — this is instant and uses zero extra
disk space. Since submodules are treated as read-only in worktrees, symlinks are the
correct approach.

```bash
# Define paths
MAIN_REPO=/path/to/main/rvfuse  # Replace with actual path
WORKTREE=$(pwd)

# Submodule source directories (symlink entire directory)
for sub in qemu llvm-project riscv-rvv-intrinsic-doc; do
    if [ -d "$MAIN_REPO/third_party/$sub" ]; then
        mkdir -p "$WORKTREE/third_party"
        ln -sfn "$MAIN_REPO/third_party/$sub" "$WORKTREE/third_party/$sub"
    fi
done

# LLVM install (if main repo has it)
if [ -d "$MAIN_REPO/third_party/llvm-install" ]; then
    ln -sfn "$MAIN_REPO/third_party/llvm-install" "$WORKTREE/third_party/llvm-install"
fi

# BBV plugin (always symlink)
if [ -f "$MAIN_REPO/tools/bbv/libbbv.so" ]; then
    mkdir -p "$WORKTREE/tools/bbv"
    ln -sfn "$MAIN_REPO/tools/bbv/libbbv.so" "$WORKTREE/tools/bbv/libbbv.so"
fi

# Local LLVM tools (always symlink)
if [ -d "$MAIN_REPO/tools/local-llvm" ]; then
    ln -sfn "$MAIN_REPO/tools/local-llvm" "$WORKTREE/tools/local-llvm"
fi
```

**Note**: Use `ln -sfn` (force + no-dereference) to safely replace existing symlinks
or directories. Check existing symlinks: `ls -la third_party/qemu 2>/dev/null`

### Step 3: Sysroot Setup (Choose Option)

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

### Step 4: Model/Test Data (Optional)

If working on YOLO or similar inference workloads:

```bash
mkdir -p "$WORKTREE/output"
ln -s "$MAIN_REPO/output/yolo11n.ort" "$WORKTREE/output/yolo11n.ort"
ln -s "$MAIN_REPO/output/test.jpg" "$WORKTREE/output/test.jpg"
ln -s "$MAIN_REPO/output/yolo11n.onnx" "$WORKTREE/output/yolo11n.onnx"
```

### Step 5: Verify Setup

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

# Symlink submodule source directories (instant, zero disk overhead)
mkdir -p third_party
for sub in qemu llvm-project riscv-rvv-intrinsic-doc; do
    [ -d "$MAIN_REPO/third_party/$sub" ] && ln -sfn "$MAIN_REPO/third_party/$sub" third_party/$sub
done

# Symlink compiled tools and resources
mkdir -p output tools/bbv

[ -d "$MAIN_REPO/third_party/llvm-install" ] && ln -sfn "$MAIN_REPO/third_party/llvm-install" third_party/llvm-install
[ -f "$MAIN_REPO/tools/bbv/libbbv.so" ] && ln -sfn "$MAIN_REPO/tools/bbv/libbbv.so" tools/bbv/libbbv.so
[ -d "$MAIN_REPO/tools/local-llvm" ] && ln -sfn "$MAIN_REPO/tools/local-llvm" tools/local-llvm

# Sysroot (shared by default)
if [ -d "$MAIN_REPO/output/sysroot-new" ]; then
    ln -sfn "$MAIN_REPO/output/sysroot-new" output/sysroot
fi

# Model files (read-only)
[ -f "$MAIN_REPO/output/yolo11n.ort" ] && ln -sfn "$MAIN_REPO/output/yolo11n.ort" output/
[ -f "$MAIN_REPO/output/test.jpg" ] && ln -sfn "$MAIN_REPO/output/test.jpg" output/

echo "Worktree setup complete!"
```

## Important Notes

### Git Worktree Behavior

- Worktrees share `.git` objects — submodules are symlinked, not re-checked-out
- Each worktree has its own working directory and can checkout different branches
- Deleting a worktree (`git worktree remove`) does NOT affect the main repo

### Symlink Safety

- Symlinks to read-only resources are safe — no risk of accidental modification
- Submodule directories are symlinked as-is (including their build artifacts)
- Never symlink directories that will be written to (like `output/dfg/`)
- If a symlink target doesn't exist, the symlink will be a dangling link

### When to NOT Use Symlinks

- If you need to modify submodule source code in this worktree (changes would affect
  all worktrees sharing the symlink)
- If the main repo's sysroot is incomplete or broken for your use case
- If the main repo is on a different filesystem (symlinks may not work)

## Cleanup

When done with the worktree:

```bash
# Remove symlinks (optional, not required for worktree removal)
find . -type l -delete

# Remove the worktree
git worktree remove <worktree-path>
```