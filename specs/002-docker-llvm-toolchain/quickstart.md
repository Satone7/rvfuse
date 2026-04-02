# Quickstart: Docker LLVM RISC-V Toolchain

**Target**: Contributors who need to compile RISC-V code without building LLVM
**Prerequisites**: Docker installed, ~2GB disk space for Docker image
**Time Target**: 5 minutes (excluding image pull)

---

## Prerequisites

- Linux x86_64 workstation
- Docker installed (version 20.10+)
- ~2GB disk space for Docker image
- Network access (for initial image pull)

### Docker Installation

If Docker is not installed:

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

---

## Step 1: Add Tools to PATH (1 min)

```bash
# Add RVFuse tools to PATH
export PATH="/path/to/RVFuse/tools/docker-llvm:$PATH"

# Add to shell config for persistence
echo 'export PATH="/path/to/RVFuse/tools/docker-llvm:$PATH"' >> ~/.bashrc
```

**Verification**: `which riscv-clang` should return the script path.

---

## Step 2: Pull Docker Image (2-5 min)

On first use, the Docker image is pulled automatically. To pull manually:

```bash
# Pull the image
docker pull rvfuse/llvm-riscv:13

# Or build from Dockerfile (if image not on Docker Hub)
cd tools/docker-llvm
docker build -t rvfuse/llvm-riscv:13 .
```

**Verification**: `docker images | grep llvm-riscv` should show the image.

---

## Step 3: Compile a Test Program (1 min)

Create a simple test program:

```c
// hello.c
int main() {
    return 0;
}
```

Compile for RISC-V:

```bash
riscv-clang -target riscv64-unknown-elf -c hello.c -o hello.o
```

**Verification**: `file hello.o` should show `ELF 64-bit LSB relocatable, UCB RISC-V`.

---

## Step 4: Check Toolchain Version (30 sec)

```bash
riscv-clang --version
```

Expected output should show `clang version 13.0.0` or similar.

---

## Available Tools

| Tool | Description |
|------|-------------|
| `riscv-clang` | C compiler |
| `riscv-clang++` | C++ compiler |
| `riscv-ld` | Linker (lld) |
| `riscv-objdump` | Disassembler |
| `riscv-strip` | Symbol stripper |

---

## Common Usage Patterns

### Compile C program

```bash
riscv-clang -target riscv64-unknown-elf -O2 -c source.c -o source.o
```

### Compile with specific RISC-V extension

```bash
riscv-clang -target riscv64-unknown-elf -march=rv64gc source.c -o source.o
```

### Link multiple objects

```bash
riscv-ld object1.o object2.o -o program.elf
```

### Disassemble binary

```bash
riscv-objdump -d program.elf
```

---

## Troubleshooting

### Docker not found

**Symptom**: `docker: command not found`

**Solution**:
1. Install Docker: `curl -fsSL https://get.docker.com | sh`
2. Add user to docker group: `sudo usermod -aG docker $USER`
3. Log out and back in

### Permission denied

**Symptom**: `permission denied while trying to connect to the Docker daemon`

**Solution**: Your user is not in the docker group. Run:
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Image pull fails

**Symptom**: `Error: image rvfuse/llvm-riscv:13 not found`

**Solution**:
1. Check network connectivity
2. Build image locally: `cd tools/docker-llvm && docker build -t rvfuse/llvm-riscv:13 .`
3. Use alternative image: `riscv-clang --image=alternative/llvm:13 file.c`

### Compilation fails silently

**Symptom**: No output or unclear error

**Solution**:
1. Check Docker logs: `docker logs $(docker ps -lq)`
2. Run with verbose output: `riscv-clang -v source.c`
3. Check disk space: `df -h`

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RVFUSE_LLVM_IMAGE` | Docker image name | `rvfuse/llvm-riscv:13` |
| `RVFUSE_LLVM_TARGET` | Default target triple | `riscv64-unknown-elf` |
| `RVFUSE_LLVM_DOCKER_OPTS` | Extra Docker options | (empty) |

Example:
```bash
export RVFUSE_LLVM_IMAGE=my-custom-llvm:14
riscv-clang --version  # Uses custom image
```

---

## Version Compatibility Note (User Story 2)

The Docker toolchain uses **upstream LLVM 13.0.0**, which may differ slightly from the **Xuantie LLVM** submodule (llvmorg-13.0.0-rc1 based).

**Potential differences**:
- Xuantie-specific patches not in upstream
- Different default target configurations
- Additional RISC-V extensions

**How to Verify Compatibility:**
To ensure your binaries are compatible across both environments, you can compile the same source code with both the Docker toolchain and the submodule LLVM, then use `readelf` or `llvm-objdump` to compare the resulting ABIs and disassembly.

For critical work requiring exact Xuantie compatibility, use the submodule LLVM when possible.

---

## Next Steps

After setup:
1. Compile your RISC-V application using `riscv-clang`
2. Use `riscv-objdump` to inspect generated code
3. Refer to LLVM documentation for advanced options
4. Compare output with submodule LLVM for compatibility verification