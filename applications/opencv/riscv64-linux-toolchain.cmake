# CMake toolchain file for RISC-V rv64gcv_zvl512b cross-compilation (OpenCV)
# RVV enabled via CPU_BASELINE=RVV — LLVM 22 __riscv_ prefixed intrinsics work correctly

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Use environment variables for paths (set by build.sh)
set(LLVM_INSTALL "$ENV{LLVM_INSTALL}")
set(SYSROOT "$ENV{SYSROOT}")

# Compiler settings
set(CMAKE_C_COMPILER "${LLVM_INSTALL}/bin/clang")
set(CMAKE_CXX_COMPILER "${LLVM_INSTALL}/bin/clang++")

# Target architecture
set(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

# Architecture flags - RVV enabled with VLEN=512
set(RISCV_FLAGS "-march=rv64gcv_zvl512b -mabi=lp64d -g")
set(CMAKE_C_FLAGS "${RISCV_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${RISCV_FLAGS}" CACHE STRING "" FORCE)

# Sysroot
set(CMAKE_SYSROOT "${SYSROOT}")

# Use lld linker
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# OpenCV-specific: enable RVV baseline for SIMD vectorization
set(CPU_BASELINE "RVV" CACHE STRING "Enable RVV baseline optimizations")

# OpenCV-specific settings
set(OPENCV_EXTRA_MODULES_PATH "$ENV{OPENCV_EXTRA_MODULES_PATH}" CACHE STRING "Path to opencv_contrib")