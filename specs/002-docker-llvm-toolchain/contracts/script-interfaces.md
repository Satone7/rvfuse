# Script Interface Contracts

**Purpose**: Define the CLI interface for Docker LLVM wrapper scripts

## Overview

This feature provides command-line tools, not REST APIs. The contracts below define the expected behavior and interface of each wrapper script.

---

## Tool: riscv-clang

### Purpose
C compiler for RISC-V cross-compilation

### Usage

```bash
riscv-clang [options] <source files>
```

### Standard Options

| Option | Description |
|--------|-------------|
| `--version` | Display LLVM/Clang version |
| `--help` | Display usage information |
| `-c` | Compile only (produce .o file) |
| `-S` | Emit assembly only |
| `-o <file>` | Output file name |
| `-target <triple>` | Target triple (default: riscv64-unknown-elf) |
| `-march=<arch>` | RISC-V architecture (e.g., rv64gc, rv64imafdc) |
| `-O<level>` | Optimization level (0, 1, 2, 3, s, z) |
| `-I<dir>` | Include directory |
| `-D<macro>` | Define macro |

### Tool-Specific Options

| Option | Description |
|--------|-------------|
| `--docker-opts="<opts>"` | Additional Docker run options |
| `--image=<image>` | Override default Docker image |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Compilation error |
| 2 | Docker not available |
| 3 | Image pull failure |
| 4 | Invalid arguments |

### Example

```bash
riscv-clang -target riscv64-unknown-elf -march=rv64gc -O2 -c main.c -o main.o
```

---

## Tool: riscv-clang++

### Purpose
C++ compiler for RISC-V cross-compilation

### Usage

```bash
riscv-clang++ [options] <source files>
```

### Interface

Same as `riscv-clang`, with additional C++ specific options:
- `-std=<std>` - C++ standard (c++11, c++14, c++17, c++20)
- `-l<library>` - Link library

### Example

```bash
riscv-clang++ -std=c++17 -target riscv64-unknown-elf -c main.cpp -o main.o
```

---

## Tool: riscv-ld

### Purpose
Linker for RISC-V object files (using lld)

### Usage

```bash
riscv-ld [options] <object files>
```

### Standard Options

| Option | Description |
|--------|-------------|
| `-o <file>` | Output file name |
| `-L<dir>` | Library search path |
| `-l<library>` | Link library |
| `-T<script>` | Linker script |
| `--gc-sections` | Remove unused sections |
| `--strip-all` | Strip all symbols |

### Example

```bash
riscv-ld -T link.ld main.o utils.o -o program.elf
```

---

## Tool: riscv-objdump

### Purpose
Disassembler and object file inspector

### Usage

```bash
riscv-objdump [options] <file>
```

### Standard Options

| Option | Description |
|--------|-------------|
| `-d` | Disassemble |
| `-D` | Disassemble all sections |
| `-h` | Display section headers |
| `-t` | Display symbol table |
| `-r` | Display relocations |
| `-S` | Mix source with disassembly |

### Example

```bash
riscv-objdump -d program.elf
```

---

## Tool: riscv-strip

### Purpose
Strip symbols from binary

### Usage

```bash
riscv-strip [options] <file>
```

### Standard Options

| Option | Description |
|--------|-------------|
| `-o <file>` | Output file (default: modify in-place) |
| `--strip-all` | Remove all symbols |
| `--strip-debug` | Remove debug symbols only |
| `--keep-symbol=<sym>` | Keep specific symbol |

### Example

```bash
riscv-strip --strip-all program.elf
```

---

## Environment Variables

All tools respect these environment variables:

| Variable | Type | Description |
|----------|------|-------------|
| `RVFUSE_LLVM_IMAGE` | string | Default Docker image (default: `rvfuse/llvm-riscv:13`) |
| `RVFUSE_LLVM_TARGET` | string | Default target triple (default: `riscv64-unknown-elf`) |
| `RVFUSE_LLVM_DOCKER_OPTS` | string | Additional Docker options |

---

## Error Messages

Standard error messages for common failures:

| Condition | Message |
|-----------|---------|
| Docker not installed | `Error: Docker is not installed. Please install Docker first.` |
| Docker daemon not running | `Error: Cannot connect to Docker daemon. Is Docker running?` |
| Permission denied | `Error: Permission denied. Add user to docker group: sudo usermod -aG docker $USER` |
| Image not found | `Error: Image not found. Pull with: docker pull rvfuse/llvm-riscv:13` |
| Disk space insufficient | `Error: Insufficient disk space. Need ~2GB for Docker image.` |