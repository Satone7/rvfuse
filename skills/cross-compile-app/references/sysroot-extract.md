# Sysroot Extraction Reference

This document provides the standardized Docker sysroot extraction procedure for RISC-V cross-compilation. The function extracts a minimal sysroot from an Ubuntu RISC-V Docker image for use with Clang cross-compilation.

## Function: `extract_sysroot`

A complete, copy-paste-ready bash function for sysroot extraction.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sysroot_dir` | Positional | Target directory for the sysroot (e.g., `${OUTPUT_DIR}/sysroot`) |
| `extra_packages` | Variadic | Additional `-dev` packages to install (optional) |

### Docker Image

Uses `riscv64/ubuntu:24.04` for the sysroot source.

### Base Packages

Always installed:
- `libc6-dev` - C standard library headers and development files
- `libstdc++-12-dev` - C++ standard library headers
- `libgcc-12-dev` - GCC runtime library headers

Additional packages can be passed as variadic arguments.

### Complete Implementation

```bash
# Extract a RISC-V sysroot from Docker for cross-compilation
# Arguments:
#   $1 - sysroot_dir: Target directory for the sysroot
#   $2... - extra_packages: Additional -dev packages to install (optional)
extract_sysroot() {
    local sysroot_dir="${1:-${OUTPUT_DIR}/sysroot}"
    shift
    local extra_packages=("$@")

    # Skip if requested and sysroot exists
    if [[ "${SKIP_SYSROOT}" == "true" ]]; then
        if [ ! -d "${sysroot_dir}/usr" ]; then
            error "Sysroot not found at ${sysroot_dir} (remove --skip-sysroot to extract)"
        fi
        declare -g SYSROOT="${sysroot_dir}"
        info "Using existing sysroot: ${SYSROOT}"
        return 0
    fi

    # Skip if already exists (unless forced)
    if [[ "${FORCE}" != "true" && -d "${sysroot_dir}/usr" ]]; then
        info "Sysroot already exists at ${sysroot_dir}. Use --force to re-extract."
        declare -g SYSROOT="${sysroot_dir}"
        return 0
    fi

    info "Extracting riscv64 sysroot from riscv64/ubuntu:24.04..."
    command -v docker &>/dev/null || error "Docker not found. Sysroot extraction requires Docker."

    # Clean and create target directory
    rm -rf "${sysroot_dir}"
    mkdir -p "${sysroot_dir}"

    # Start temporary container
    local tmp_container="rvfuse-sysroot-prep-$$"
    docker run --platform riscv64 --name "${tmp_container}" -d riscv64/ubuntu:24.04 tail -f /dev/null > /dev/null
    trap "docker rm -f ${tmp_container} 2>/dev/null || true" RETURN

    # Install packages
    docker exec "${tmp_container}" apt-get update -qq
    docker exec "${tmp_container}" apt-get install -y --no-install-recommends -qq \
        libc6-dev \
        libstdc++-12-dev \
        libgcc-12-dev \
        "${extra_packages[@]}" \
        > /dev/null

    info "Copying sysroot directories from container..."

    # Copy directories from container (use intermediate paths to avoid docker cp quirks)
    docker cp "${tmp_container}:/usr/lib" "${sysroot_dir}/usr_lib_tmp"
    docker cp "${tmp_container}:/usr/include" "${sysroot_dir}/usr_include_tmp"
    docker cp "${tmp_container}:/lib/riscv64-linux-gnu" "${sysroot_dir}/lib_riscv_tmp"

    # Move to final locations
    mkdir -p "${sysroot_dir}/usr" "${sysroot_dir}/lib"
    mv "${sysroot_dir}/usr_lib_tmp" "${sysroot_dir}/usr/lib"
    mv "${sysroot_dir}/usr_include_tmp" "${sysroot_dir}/usr/include"
    mv "${sysroot_dir}/lib_riscv_tmp" "${sysroot_dir}/lib/riscv64-linux-gnu"

    # Clean up container
    docker rm -f "${tmp_container}" > /dev/null
    trap - RETURN

    # Top-level dynamic linker symlink (required for -static-pie and dynamic linking)
    ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot_dir}/lib/ld-linux-riscv64-lp64d.so.1"

    # Create symlinks for CRT and shared libs at usr/lib/ for Clang/lld
    # (Clang searches -L$SYSROOT/usr/lib but not the riscv64-linux-gnu/ subdirectory by default)
    local multilib="riscv64-linux-gnu"
    local base="${sysroot_dir}/usr/lib"
    if [ -d "${base}/${multilib}" ]; then
        # CRT (C Runtime) files
        for f in crt1.o crti.o crtn.o Scrt1.o; do
            [ -f "${base}/${multilib}/${f}" ] && ln -sf "${multilib}/${f}" "${base}/${f}"
        done
        # Shared libraries (avoid overwriting existing symlinks)
        for f in libc.so libc.so.6 libm.so libm.so.6 libdl.so libdl.so.2 \
                 librt.so librt.so.1 libpthread.so libpthread.so.0 \
                 libgcc_s.so libgcc_s.so.1 libstdc++.so libstdc++.so.6; do
            [ -e "${base}/${multilib}/${f}" ] && [ ! -e "${base}/${f}" ] && \
                ln -sf "${multilib}/${f}" "${base}/${f}"
        done
    fi

    # Also symlink CRT files and shared libs to lib/riscv64-linux-gnu/
    # (ld-linux looks for shared libs in /lib/triplet/)
    local lib="${sysroot_dir}/lib/riscv64-linux-gnu"
    if [ -d "${sysroot_dir}/usr/lib/${multilib}" ]; then
        # CRT files
        for f in crt1.o crti.o crtn.o Scrt1.o; do
            [ -f "${sysroot_dir}/usr/lib/${multilib}/${f}" ] && [ ! -e "${lib}/${f}" ] && \
                ln -sf "../../usr/lib/${multilib}/${f}" "${lib}/${f}"
        done
        # Shared libraries
        for f in libc.so.6 libm.so.6 libdl.so.2 librt.so.1 libpthread.so.0 \
                 libgcc_s.so libgcc_s.so.1 libstdc++.so libstdc++.so.6; do
            [ -e "${sysroot_dir}/usr/lib/${multilib}/${f}" ] && [ ! -e "${lib}/${f}" ] && \
                ln -sf "../../usr/lib/${multilib}/${f}" "${lib}/${f}"
        done
    fi

    # Remove problematic static libs (libm.a has __frexpl requiring long double support)
    find "${sysroot_dir}" -name "libm.a" -delete 2>/dev/null || true

    declare -g SYSROOT="${sysroot_dir}"
    info "Sysroot ready at ${SYSROOT}"
    echo "  $(du -sh "${SYSROOT}" | cut -f1)"
}
```

## Extra Packages Guide

Common library dependencies and their corresponding Ubuntu `-dev` packages:

| Library | Link Flag | Package |
|---------|-----------|---------|
| OpenSSL (`ssl`, `crypto`) | `-lssl -lcrypto` | `libssl-dev` |
| libcurl | `-lcurl` | `libcurl4-openssl-dev` |
| zlib | `-lz` | `zlib1g-dev` |
| pthread | `-lpthread` | (included in `libc6-dev`) |
| dl | `-ldl` | (included in `libc6-dev`) |
| SQLite3 | `-lsqlite3` | `libsqlite3-dev` |
| libxml2 | `-lxml2` | `libxml2-dev` |
| ncurses | `-lncurses` | `libncurses-dev` |
| readline | `-lreadline` | `libreadline-dev` |
| OpenMP | `-fopenmp` | `libgomp-dev` |

### Usage Example

```bash
# Basic sysroot with just C/C++ standard libraries
extract_sysroot "${OUTPUT_DIR}/sysroot"

# Sysroot with additional packages
extract_sysroot "${OUTPUT_DIR}/sysroot" libssl-dev libcurl4-openssl-dev zlib1g-dev
```

## Deduplication Strategies

If multiple applications share the same sysroot base:

### Option 1: Shared Sysroot Directory

Applications can share a single sysroot by pointing to a common location:

```bash
# In each application's build.sh
SYSROOT="${PROJECT_ROOT}/output/sysroot"  # Shared location

# Or use a project-level variable
extract_sysroot "${PROJECT_ROOT}/output/sysroot"
```

### Option 2: Skip After First Extraction

Use `--skip-sysroot` flag after the first extraction:

```bash
# First application extracts
./applications/yolo/ort/build.sh

# Subsequent applications skip extraction
./applications/llama.cpp/build.sh --skip-sysroot
```

### Option 3: Symlink to Shared Sysroot

```bash
# Create symlink to shared sysroot
ln -sf "${PROJECT_ROOT}/output/sysroot" "${OUTPUT_DIR}/sysroot"
```

## Integration in build.sh

### Command-Line Flags

```bash
SKIP_SYSROOT="false"
FORCE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sysroot)
            SKIP_SYSROOT="true"
            shift
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        *)
            shift
            ;;
    esac
done
```

### Step 1: Sysroot Extraction

```bash
main() {
    # Step 1: Extract sysroot (or use existing)
    extract_sysroot "${OUTPUT_DIR}/sysroot"

    # Step 2: Toolchain setup
    setup_toolchain

    # Step 3: Build application
    build_application

    # Step 4: Verify output
    verify_output
}
```

### Complete Build.sh Skeleton

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

SKIP_SYSROOT="false"
FORCE="false"

info() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sysroot) SKIP_SYSROOT="true"; shift ;;
        --force) FORCE="true"; shift ;;
        *) shift ;;
    esac
done

# Include extract_sysroot function here (copy from above)
extract_sysroot() {
    # ... (see complete implementation above)
}

main() {
    # Step 1: Sysroot extraction
    extract_sysroot "${OUTPUT_DIR}/sysroot"

    # Step 2-N: Application-specific build steps
    # ...
}

main "$@"
```

## Troubleshooting

### Docker Not Found

```
[ERROR] Docker not found. Sysroot extraction requires Docker.
```

**Solution**: Install Docker and ensure it's running:
```bash
# Check Docker is installed
docker --version

# Start Docker daemon if not running
sudo systemctl start docker
```

### Container Cleanup Failure

If the script was interrupted during extraction, a stale container may remain:

```bash
# List and remove stale containers
docker ps -a | grep rvfuse-sysroot-prep
docker rm -f rvfuse-sysroot-prep-<pid>
```

### Missing Libraries After Extraction

If your application fails to link with missing library errors:

1. Add the required `-dev` package to `extract_sysroot` call:
   ```bash
   extract_sysroot "${OUTPUT_DIR}/sysroot" libssl-dev
   ```

2. Re-extract with `--force`:
   ```bash
   ./build.sh --force
   ```

### Cross-Compilation Errors

If Clang cannot find headers or libraries:

1. Verify sysroot structure:
   ```bash
   ls -la ${SYSROOT}/usr/include
   ls -la ${SYSROOT}/usr/lib/riscv64-linux-gnu
   ```

2. Check symlinks exist:
   ```bash
   ls -la ${SYSROOT}/usr/lib/crt*.o
   ls -la ${SYSROOT}/lib/ld-linux-riscv64-lp64d.so.1
   ```