# Docker LLVM RISC-V Toolchain Guide

## Overview

The Docker LLVM Toolchain provides a containerized version of LLVM 13 configured for cross-compiling RISC-V code. It is designed as an alternative for contributors whose machines may not have the performance to build the `llvm-project` submodule from source.

## Quick Start

Instead of calling `clang` directly, use the wrapper scripts located in `tools/docker-llvm/`:

```bash
# Set up your path
export PATH="$PWD/tools/docker-llvm:$PATH"

# Compile C code
riscv-clang -o program program.c

# Compile C++ code
riscv-clang++ -o program program.cpp
```

## Available Tools

- `riscv-clang`: C compiler
- `riscv-clang++`: C++ compiler
- `riscv-ld`: Linker
- `riscv-objdump`: Disassembler
- `riscv-strip`: Symbol stripper

## Version Compatibility (User Story 2)

This Docker toolchain is based on Debian Bullseye and uses the official `llvm-13` packages from `apt.llvm.org`. 

**Compatibility Note:**
The submodule LLVM is based on `llvmorg-13.0.0-rc1`. The Docker toolchain uses the stable release of LLVM 13. While they are highly compatible, there may be minor differences due to custom Xuantie patches present in the submodule that are not in upstream LLVM 13.

To verify the version:
```bash
riscv-clang --version
```
This will output the LLVM version matching upstream version 13.0.x.

## Advanced Usage

The wrapper scripts automatically handle mounting your current working directory and setting the correct user permissions (`--user $(id -u):$(id -g)`).

### Passing Custom Docker Options

You can pass custom arguments to the underlying `docker run` command using `--docker-opts`:

```bash
riscv-clang --docker-opts="-e DEBUG=1 -v /tmp:/tmp" -o myprog myprog.c
```

### Using a Custom Image

By default, the toolchain uses the image `rvfuse/llvm-riscv:13`. You can specify a different image using `--image`:

```bash
riscv-clang --image="my-custom-llvm:latest" -o myprog myprog.c
```
