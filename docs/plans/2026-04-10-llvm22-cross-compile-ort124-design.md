# LLVM 22 Host Cross-Compile ONNX Runtime v1.24.4 (rv64gcv)

**Date**: 2026-04-10
**Status**: Approved

## Goal

Cross-compile ONNX Runtime v1.24.4 (full build) for rv64gcv on the host machine using the LLVM 22 toolchain compiled from `third_party/llvm-project`. Output: `libonnxruntime.so` shared library only. No Docker build step — cmake + ninja run directly on the host.

## Scope

- Create a self-contained build pipeline under `tools/cross-compile-ort/`
- Upgrade ORT from v1.17.3 to v1.24.4
- Use lld as the linker (no GCC cross-compiler dependency)
- Docker is used only for sysroot extraction from `riscv64/ubuntu:24.04`
- This branch does not integrate into `setup.sh` or affect other project components

## Approach

**Chosen: Pure LLVM toolchain + container-exported sysroot** (vs. Docker container build or GCC-linked approach).

Rationale: Eliminates GCC cross-compiler dependency; lld 22 has mature RISC-V support; fastest build (no QEMU emulation); fully self-contained.

## Architecture

```
tools/cross-compile-ort/
├── build.sh                      # Main build script (sysroot + ORT compile)
└── riscv64-linux-toolchain.cmake  # CMake toolchain file
```

Build flow:
1. Check prerequisites (LLVM toolchain, Docker for sysroot extraction)
2. Extract sysroot from `riscv64/ubuntu:24.04` container (Docker needed here only)
3. Ensure ORT v1.24.4 source is ready (auto-clone if missing)
4. cmake + ninja full build of `libonnxruntime.so`

Output: `output/cross-ort/lib/libonnxruntime.so*`

## Sysroot Export

**Source image**: `riscv64/ubuntu:24.04`

**Installed packages**:
- `libc6-dev` — C headers + CRT objects (crt1.o, crti.o, crtn.o)
- `libstdc++-12-dev` — C++ headers + libstdc++.so

**Exported paths**:
- `/usr/lib/riscv64-linux-gnu/` → CRT objects + shared libraries
- `/usr/include/riscv64-linux-gnu/` → triplet-specific headers
- `/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1` → dynamic linker

**Post-processing**:
- Create top-level symlinks in `usr/lib/` for CRT objects so lld can find them
- Create `lib/ld-linux-riscv64-lp64d.so.1` symlink for the dynamic linker
- Remove `libm.a` (contains `__frexpl` long double symbols not available on lp64d)

## CMake Toolchain File

Key settings:
- `CMAKE_SYSTEM_NAME`: Linux
- `CMAKE_C/CXX_COMPILER`: `$ENV{LLVM_INSTALL}/bin/clang` / `clang++`
- `CMAKE_C/CXX_COMPILER_TARGET`: `riscv64-unknown-linux-gnu`
- `CMAKE_SYSROOT`: `$ENV{SYSROOT}`
- Extra includes: `-isystem $SYSROOT/usr/include/riscv64-linux-gnu`
- Architecture flags: `-march=rv64gcv`
- Linker: `-fuse-ld=lld` (no GCC cross-ld dependency)

## CMake Configuration

```bash
cmake <ort-source>/cmake \
    -DCMAKE_TOOLCHAIN_FILE=tools/cross-compile-ort/riscv64-linux-toolchain.cmake \
    -DCMAKE_INSTALL_PREFIX=output/cross-ort \
    -DCMAKE_BUILD_TYPE=Release \
    -Donnxruntime_BUILD_SHARED_LIB=ON \
    -Donnxruntime_BUILD_UNIT_TESTS=OFF \
    -Donnxruntime_DISABLE_RTTI=ON \
    -DFETCHCONTENT_SOURCE_DIR_EIGEN=<eigen-path> \
    -DCMAKE_CXX_FLAGS='-Wno-stringop-overflow -Wno-unknown-warning-option' \
    -DIconv_IS_BUILT_IN=TRUE \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -G Ninja
```

Full build (no `--minimal_build` flag).

## Source Management

- ORT source: `tools/docker-onnxrt/vendor/onnxruntime/` (auto-clone v1.24.4 shallow + submodules if missing)
- Eigen: `tools/docker-onnxrt/vendor/eigen/` (version 3.4.0, auto-clone if missing)
- LLVM toolchain: `third_party/llvm-install/` (symlinked from main repo)

## Differences from c920-onnxrt

| Aspect | c920-onnxrt | This design |
|--------|-------------|-------------|
| Build execution | Inside Docker container | Directly on host |
| Ubuntu version | 22.04 | 24.04 |
| Linker | GCC cross-ld (`-B/riscv64-gcc-bin`) | lld (`-fuse-ld=lld`) |
| GCC dependency | Required (gcc-riscv64-linux-gnu) | None |
| ORT version | v1.17.3 | v1.24.4 |
| Output | Shared lib + YOLO runner | Shared lib only |
| llvm-install | Referenced but missing in worktree | Symlinked from main repo |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| lld fails on certain ORT object files | Try `--no-relax` or fall back to `--writable-sections`; worst case, install GCC cross-ld |
| ORT v1.24.4 CMake changes break existing flags | Check ORT release notes; adapt flags as needed |
| Sysroot packages differ in Ubuntu 24.04 | Pin specific package versions; test early |
| LLVM 22 RISC-V codegen bugs | Use released LLVM 22.1.x; report bugs if found |
