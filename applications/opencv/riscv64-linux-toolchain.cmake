# CMake toolchain file for RISC-V rv64gc_zvl256b cross-compilation (OpenCV)
# Note: RVV disabled due to LLVM 22 intrinsics compatibility issues (VXRM parameters)
# Will use scalar code and add RVV optimizations manually for KCF hotspots

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

# Architecture flags - scalar code only (RVV disabled due to intrinsics compatibility)
set(RISCV_FLAGS "-march=rv64gc -mabi=lp64d -g")
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

# OpenCV-specific: disable RVV baseline (use scalar code)
set(CPU_BASELINE "" CACHE STRING "Disable baseline optimizations")

# OpenCV-specific settings
set(OPENCV_EXTRA_MODULES_PATH "$ENV{OPENCV_EXTRA_MODULES_PATH}" CACHE STRING "Path to opencv_contrib")