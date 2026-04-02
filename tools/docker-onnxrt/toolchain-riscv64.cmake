# toolchain-riscv64.cmake - CMake toolchain for RISC-V 64-bit Linux cross-compilation

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Cross-compiler toolchain
set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_AR riscv64-linux-gnu-ar)
set(CMAKE_RANLIB riscv64-linux-gnu-ranlib)
set(CMAKE_STRIP riscv64-linux-gnu-strip)

# Sysroot (will be set during Docker build)
set(CMAKE_SYSROOT /opt/riscv-sysroot)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# Search for programs in the host environment only
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target environment only
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Compiler flags
set(CMAKE_C_FLAGS "-O2 -g" CACHE STRING "C flags")
set(CMAKE_CXX_FLAGS "-O2 -g" CACHE STRING "CXX flags")

# Static linking preference
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
