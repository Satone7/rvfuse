# Cross-compilation toolchain for RISC-V 64-bit using LLVM 22
# Vanilla: rv64gc only (NO V extension) — scalar fallback for all MLAS operations
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

SET(CMAKE_C_COMPILER $ENV{LLVM_INSTALL}/bin/clang)
SET(CMAKE_CXX_COMPILER $ENV{LLVM_INSTALL}/bin/clang++)

SET(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
SET(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

SET(CMAKE_SYSROOT $ENV{SYSROOT})
SET(CMAKE_FIND_ROOT_PATH $ENV{SYSROOT})

# Clang doesn't automatically add triplet-specific include paths like GCC does.
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")

# NO V extension — ensures MLAS_TARGET_RISCV is NOT defined, using scalar fallback paths
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gc")

# Use lld for linking
SET(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
SET(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")

# Debug symbols for profiling
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
