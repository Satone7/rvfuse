# RVFuse Tools

This directory contains various tools and utilities for the RVFuse project.

## Docker LLVM Toolchain (`docker-llvm/`)

A containerized LLVM 13 RISC-V toolchain that serves as a lightweight alternative to compiling the `llvm-project` submodule from source. This is primarily designed for contributors with limited hardware resources.

### Available Tools

- `riscv-clang` / `riscv-clang++`
- `riscv-ld`
- `riscv-objdump`
- `riscv-strip`

See `docker-llvm/` for the containerized LLVM 13 RISC-V toolchain.
