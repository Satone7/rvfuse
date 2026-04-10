# Cross-compilation toolchain for RISC-V 64-bit using LLVM
# Used inside Docker container where /riscv64-gcc-bin contains ld/as symlinks.

if(NOT DEFINED ENV{LLVM_INSTALL})
    message(FATAL_ERROR "LLVM_INSTALL environment variable not set")
endif()
if(NOT DEFINED ENV{SYSROOT})
    message(FATAL_ERROR "SYSROOT environment variable not set")
endif()

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

# Default -march (can be overridden via -DCMAKE_CXX_FLAGS on command line)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")

SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Use GCC's linker/assembler from the Docker container's /riscv64-gcc-bin
# (ld.lld has R_RISCV_ALIGN issues). Configurable via GCC_BIN_DIR env var.
SET(_GCC_BIN_DIR "$ENV{GCC_BIN_DIR}")
if(NOT _GCC_BIN_DIR)
    SET(_GCC_BIN_DIR "/riscv64-gcc-bin")
endif()
SET(CMAKE_EXE_LINKER_FLAGS "-B${_GCC_BIN_DIR} -Wl,-Bdynamic")
SET(CMAKE_SHARED_LINKER_FLAGS "-B${_GCC_BIN_DIR} -Wl,-Bdynamic")
