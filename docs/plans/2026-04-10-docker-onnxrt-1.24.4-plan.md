# docker-onnxrt-1.24.4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Docker-based RISC-V native build environment for ONNX Runtime v1.24.4 with LLVM 18 targeting riscv64gcv.

**Architecture:** Single-stage Dockerfile that compiles ONNX Runtime v1.24.4 from pre-cloned sources using LLVM 18 (Ubuntu 24.04 default clang). Build output stays in Docker image tagged `rvfuse-onnxrt-1.24.4`.

**Tech Stack:** Docker (riscv64/ubuntu:24.04), LLVM/Clang 18, CMake/Ninja, ONNX Runtime v1.24.4, Eigen (specific commit from ORT deps.txt)

---

## Task 1: Create Directory Structure and Update .gitignore

**Files:**
- Create: `tools/docker-onnxrt-1.24.4/` (directory)
- Modify: `.gitignore:48-49` (add vendor ignore for new directory)

**Step 1: Create directory**

```bash
mkdir -p tools/docker-onnxrt-1.24.4
```

**Step 2: Add vendor directory to .gitignore**

Edit `.gitignore` line 48-49, add the new vendor pattern:

```diff
 # Pre-cloned vendor sources for Docker builds
 tools/docker-onnxrt/vendor/
+tools/docker-onnxrt-1.24.4/vendor/
```

**Step 3: Commit**

```bash
git add tools/docker-onnxrt-1.24.4/.gitignore .gitignore
git commit -m "$(cat <<'EOF'
chore: Create docker-onnxrt-1.24.4 directory structure

Add directory and .gitignore pattern for vendor sources.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Write build.sh Script

**Files:**
- Create: `tools/docker-onnxrt-1.24.4/build.sh`

**Step 1: Write build.sh**

```bash
cat > tools/docker-onnxrt-1.24.4/build.sh << 'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor"

ONNXRUNTIME_REPO="https://github.com/microsoft/onnxruntime.git"
ONNXRUNTIME_VERSION="v1.24.4"
# Eigen commit from ORT deps.txt (eigen-mirror repo, commit 1d8b82b)
# This matches the archive: https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33
EIGEN_REPO="https://github.com/eigen-mirror/eigen.git"
EIGEN_COMMIT="1d8b82b0740839c0de7f1242a3585e3390ff5f33"

clone_if_missing() {
    local repo_url="$1" version="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        echo "=== Skipping ${dest} (already exists) ==="
    else
        echo "=== Cloning ${repo_url} @ ${version} (shallow, no submodules) ==="
        mkdir -p "$(dirname "${dest}")"
        git clone --depth=1 --branch "${version}" --no-recurse-submodules "${repo_url}" "${dest}" ||
        # If branch fails (commit hash), try direct checkout
        git clone --depth=1 --no-recurse-submodules "${repo_url}" "${dest}" &&
        cd "${dest}" && git checkout "${version}"
    fi
}

clone_eigen_commit() {
    local repo_url="$1" commit="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        echo "=== Skipping ${dest} (already exists) ==="
    else
        echo "=== Cloning ${repo_url} @ ${commit} (shallow) ==="
        mkdir -p "$(dirname "${dest}")"
        git clone --depth=1 --no-recurse-submodules "${repo_url}" "${dest}"
        cd "${dest}"
        # Fetch the specific commit if not in shallow clone
        if ! git rev-parse --verify "${commit}" >/dev/null 2>&1; then
            git fetch --depth=1 origin "${commit}"
        fi
        git checkout "${commit}"
        cd - > /dev/null
    fi
}

# Pre-clone dependencies on the host so Docker doesn't need network access.
clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${VENDOR_DIR}/onnxruntime"

# Fetch onnxruntime submodules with retries (these are much smaller than the main repo)
# Uses --recursive to get ALL nested submodules (e.g. onnx -> benchmark, pybind11)
if [ -d "${VENDOR_DIR}/onnxruntime" ]; then
    echo "=== Fetching onnxruntime submodules (recursive) ==="
    cd "${VENDOR_DIR}/onnxruntime"
    MAX_RETRIES=5
    for attempt in $(seq 1 ${MAX_RETRIES}); do
        echo "  Attempt ${attempt}/${MAX_RETRIES}..."
        if git submodule update --init --recursive --depth=1; then
            echo "  All submodules fetched."
            break
        fi
        echo "  Retrying in 5s..."
        sleep 5
    done
    cd - > /dev/null
fi

# Clone Eigen at the specific commit ORT requires
clone_eigen_commit "${EIGEN_REPO}" "${EIGEN_COMMIT}" "${VENDOR_DIR}/eigen"

echo "=== Building ONNX Runtime v1.24.4 for RISC-V (LLVM 18, riscv64gcv) ==="
echo "This is a FULL build (not minimal). Expected time: 6-12 hours under QEMU."

DOCKER_BUILDKIT=1 docker build \
    --platform riscv64 \
    --network=host \
    -t rvfuse-onnxrt-1.24.4 \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --progress=plain \
    "${PROJECT_ROOT}"

echo "=== Build complete ==="
echo "Docker image: rvfuse-onnxrt-1.24.4"
docker images rvfuse-onnxrt-1.24.4

# Verify build output exists in the image
echo "=== Verifying build output ==="
docker run --rm rvfuse-onnxrt-1.24.4 \
    ls -la /onnxruntime/build-output/

echo "=== Checking libonnxruntime.so architecture ==="
docker run --rm rvfuse-onnxrt-1.24.4 \
    file /onnxruntime/build-output/libonnxruntime.so
SCRIPT
chmod +x tools/docker-onnxrt-1.24.4/build.sh
```

**Step 2: Verify script syntax**

```bash
bash -n tools/docker-onnxrt-1.24.4/build.sh
```

Expected: No output (syntax OK)

**Step 3: Commit**

```bash
git add tools/docker-onnxrt-1.24.4/build.sh
git commit -m "$(cat <<'EOF'
feat: Add build.sh for docker-onnxrt-1.24.4

Pre-clones ORT v1.24.4 + submodules and Eigen (specific commit)
before Docker build. Uses LLVM 18 targeting riscv64gcv.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Write Dockerfile

**Files:**
- Create: `tools/docker-onnxrt-1.24.4/Dockerfile`

**Step 1: Write Dockerfile**

```dockerfile
cat > tools/docker-onnxrt-1.24.4/Dockerfile << 'DOCKERFILE'
# ONNX Runtime v1.24.4 RISC-V Native Build
# Uses LLVM 18 (Ubuntu 24.04 default) targeting riscv64gcv
#
# Build output stays in this image (tag: rvfuse-onnxrt-1.24.4)
# No extraction to host - use docker cp or docker run to access artifacts

FROM --platform=riscv64 riscv64/ubuntu:24.04 AS onnxrt-build

# Install LLVM 18 + build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang lld llvm cmake ninja-build python3 git ca-certificates make \
    && rm -rf /var/lib/apt/lists/

# Configure compiler environment
ENV CC=clang
ENV CXX=clang++

# Copy pre-cloned sources (managed by build.sh on host)
COPY tools/docker-onnxrt-1.24.4/vendor/onnxruntime/ /onnxruntime/
COPY tools/docker-onnxrt-1.24.4/vendor/eigen/ /eigen/

# Build ONNX Runtime v1.24.4 (FULL build, not minimal)
# Target: riscv64gcv (G: IMAFD base, C: Compressed, V: Vector)
RUN --mount=type=cache,target=/onnxruntime/build \
    cd /onnxruntime \
    && bash build.sh \
        --config Release \
        --build_shared_lib \
        --parallel 8 \
        --allow_running_as_root \
        --skip_tests \
        --cmake_extra_defines \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
            FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen \
            CMAKE_C_FLAGS="-march=riscv64gcv -mtune=riscv64" \
            CMAKE_CXX_FLAGS="-march=riscv64gcv -mtune=riscv64"

# Organize build output for easy access
RUN mkdir -p /onnxruntime/build-output \
    && cp /onnxruntime/build/Linux/Release/libonnxruntime.so* \
       /onnxruntime/build-output/ \
    && cp /onnxruntime/build/Linux/Release/onnxruntime_config.h \
       /onnxruntime/build-output/ \
    && cp -r /onnxruntime/include /onnxruntime/build-output/ 2>/dev/null || true \
    && echo "Build output organized in /onnxruntime/build-output/"
DOCKERFILE
```

**Step 2: Verify Dockerfile syntax**

```bash
docker build --check -f tools/docker-onnxrt-1.24.4/Dockerfile . 2>/dev/null || \
    grep -E '^FROM|^RUN|^COPY|^ENV' tools/docker-onnxrt-1.24.4/Dockerfile | head -20
```

Expected: Shows valid Dockerfile instructions

**Step 3: Commit**

```bash
git add tools/docker-onnxrt-1.24.4/Dockerfile
git commit -m "$(cat <<'EOF'
feat: Add Dockerfile for ONNX Runtime v1.24.4 RISC-V build

Single-stage build using LLVM 18 (clang) targeting riscv64gcv.
Full build (not minimal) with shared library output.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Verify Build Setup (Dry Run)

**Files:**
- None (verification only)

**Step 1: Verify directory structure**

```bash
ls -la tools/docker-onnxrt-1.24.4/
```

Expected:
```
Dockerfile
build.sh
```

**Step 2: Verify .gitignore pattern**

```bash
grep -E 'docker-onnxrt.*vendor' .gitignore
```

Expected:
```
tools/docker-onnxrt/vendor/
tools/docker-onnxrt-1.24.4/vendor/
```

**Step 3: Check script executable**

```bash
ls -l tools/docker-onnxrt-1.24.4/build.sh
```

Expected: `-rwxr-xr-x` (executable)

**Step 4: Final commit (if any uncommitted changes)**

```bash
git status
# If clean, skip this step
git add -A
git commit -m "$(cat <<'EOF'
chore: Finalize docker-onnxrt-1.24.4 setup

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update Design Document with Verification Results

**Files:**
- Modify: `docs/plans/2026-04-10-docker-onnxrt-1.24.4-design.md:150-157`

**Step 1: Mark verification items as complete**

Update the verification checklist in the design document:

```diff
 ## 验证清单

-- [ ] build.sh 成功克隆 ORT v1.24.4 和 Eigen 到 vendor/
-- [ ] Docker build 成功完成，无错误
-- [ ] 镜像中存在 /onnxruntime/build-output/libonnxruntime.so*
-- [ ] 镜像中存在 /onnxruntime/build-output/include/ 头文件
-- [ ] `file` 确认 .so 是 RISC-V ELF
-- [ ] ORT 版本信息可通过 header 或 cmake 配置确认是 v1.24.4
+- [x] build.sh 脚本和 Dockerfile 已创建
+- [ ] build.sh 成功克隆 ORT v1.24.4 和 Eigen 到 vendor/
+- [ ] Docker build 成功完成，无错误
+- [ ] 镜像中存在 /onnxruntime/build-output/libonnxruntime.so*
+- [ ] 镜像中存在 /onnxruntime/build-output/include/ 头文件
+- [ ] `file` 确认 .so 是 RISC-V ELF
+- [ ] ORT 版本信息可通过 header 或 cmake 配置确认是 v1.24.4
```

**Step 2: Commit**

```bash
git add docs/plans/2026-04-10-docker-onnxrt-1.24.4-design.md
git commit -m "$(cat <<'EOF'
docs: Update verification checklist for docker-onnxrt-1.24.4

Mark setup tasks complete, runtime verification pending.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Commit Message Prefix |
|------|-------------|----------------------|
| 1 | Directory + .gitignore | `chore:` |
| 2 | build.sh script | `feat:` |
| 3 | Dockerfile | `feat:` |
| 4 | Dry run verification | N/A (no changes) |
| 5 | Update design doc | `docs:` |

**Total commits: 4-5**

**Estimated runtime for actual Docker build: 6-12 hours** (QEMU emulation of full ORT build)