# CMake toolchain file for RISC-V rv64gcv_zvl512b cross-compilation (ORB-SLAM3)
# For Phase 3 (BBV profiling), we target zvl512b.
# For Phase 1 (perf on dev board), rebuild with zvl256b.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(LLVM_INSTALL "$ENV{LLVM_INSTALL}")
set(SYSROOT "$ENV{SYSROOT}")

set(CMAKE_C_COMPILER "${LLVM_INSTALL}/bin/clang")
set(CMAKE_CXX_COMPILER "${LLVM_INSTALL}/bin/clang++")

set(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

set(RISCV_FLAGS "-march=rv64gcv_zvl512b -mabi=lp64d -g")
set(CMAKE_C_FLAGS "${RISCV_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${RISCV_FLAGS}" CACHE STRING "" FORCE)

set(CMAKE_SYSROOT "${SYSROOT}")

set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CPU_BASELINE "" CACHE STRING "Disable baseline optimizations (LLVM 22 RVV VXRM issue)")
