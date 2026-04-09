---
name: qemu-usage
description: |
  How to use QEMU (qemu-riscv64) for running RISC-V programs, profiling with the BBV plugin,
  and debugging within the RVFuse project. Use this skill whenever the user mentions running QEMU,
  qemu-riscv64, BBV profiling, RISC-V emulation, QEMU plugins, sysroot, running RISC-V binaries,
  QEMU CPU selection, vector extension programs, or QEMU debugging options — even if they don't
  explicitly ask for "QEMU help."
---

# QEMU Usage Guide (RVFuse Project)

This skill covers **runtime usage** of QEMU within the RVFuse project.
It does NOT cover building QEMU — see `verify_bbv.sh` for build instructions.

## Key Paths

| Item | Path |
|------|------|
| QEMU binary | `third_party/qemu/build/qemu-riscv64` |
| BBV plugin | `third_party/qemu/build/contrib/plugins/bbv.so` |
| Sysroot | `output/sysroot` (RISC-V shared libraries) |
| ELF binary | e.g. `output/yolo_inference` |
| BBV output | `output/yolo.bbv.<pid>.bb` + `output/yolo.bbv.disas` |

## Running RISC-V Programs (User Mode)

Basic execution requires a sysroot for dynamic linking:

```bash
qemu-riscv64 -L output/sysroot ./your_program [args...]
```

### CPU Selection (`-cpu`)

The `-cpu` flag selects the emulated CPU model and its default ISA extensions:

```bash
# Generic RV64 with specific extensions enabled
qemu-riscv64 -cpu rv64,v=true -L output/sysroot ./your_vec_program

# List all available CPU types
qemu-riscv64 -cpu help
```

**Common CPU types (upstream QEMU):**

| CPU | Features | Use case |
|-----|----------|----------|
| `rv64` | I (base) | Minimal; add extensions via flags |
| `any` | All supported | Feature exploration |
| `max` | All supported | Feature exploration |

**Enabling extensions on generic CPUs:**

```bash
# Enable vector + float + multiply
qemu-riscv64 -cpu rv64,v=true,f=true,m=true -L output/sysroot ./program
```

### Debugging Flags (`-d`)

```bash
# System call trace (see what syscalls the program makes)
qemu-riscv64 -d strace -L output/sysroot ./program

# CPU register state before each translation block
qemu-riscv64 -d cpu -L output/sysroot ./program

# Execution trace
qemu-riscv64 -d exec -L output/sysroot ./program

# Log unimplemented instructions (useful for extension coverage)
qemu-riscv64 -d unimp -L output/sysroot ./program

# Combine multiple debug options
qemu-riscv64 -d strace,unimp -L output/sysroot ./program
```

### GDB Remote Debugging

```bash
# Terminal 1: Start QEMU waiting for GDB connection on port 1234
qemu-riscv64 -g 1234 -L output/sysroot ./program

# Terminal 2: Connect with multiarch GDB
riscv64-unknown-elf-gdb -ex "target remote localhost:1234" ./program
# or with vanilla gdb
gdb-multiarch -ex "set architecture riscv:rv64" -ex "target remote localhost:1234" ./program
```

## BBV Plugin (Profiling)

The BBV (Basic Block Vector) plugin generates basic block execution counts for hotspot analysis.
This is the primary profiling mechanism in RVFuse.

### Basic Usage

```bash
qemu-riscv64 -L output/sysroot \
  -plugin third_party/qemu/build/contrib/plugins/bbv.so,interval=10000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg
```

**Plugin parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval` | 1000000 | Instructions per BBV sampling interval |
| `outfile` | `bbv` | Output file prefix (creates `.bb` and `.disas`) |

### Output Files

- `outfile.<pid>.bb` — Basic block execution counts (T:id:count format)
- `outfile.disas` — Disassembly of all basic blocks (for analysis tools)

### Choosing the Interval

- Smaller interval (1000-10000): finer granularity, larger output files
- Larger interval (100000+): coarser, suitable for long-running programs
- For YOLO inference profiling: `interval=10000` is a good starting point

### Downstream Pipeline

After BBV profiling, use the RVFuse tools:

```bash
# Generate hotspot report
python3 tools/analyze_bbv.py --bbv output/yolo.bbv.0.bb --elf output/yolo_inference --sysroot output/sysroot

# Generate DFG from hot basic blocks
./tools/profile_to_dfg.sh --bbv output/yolo.bbv.0.bb --elf output/yolo_inference --sysroot output/sysroot --top 50 --output-dir output/dfg

# Or generate DFG directly from .disas file
python -m tools.dfg --disas output/yolo.bbv.disas --isa I,F,M --top 20
```

## Other Built-in Plugins

QEMU ships with several useful plugins in `contrib/plugins/`:

```bash
# Hot block identification (find most-executed basic blocks)
qemu-riscv64 -plugin libhotblocks.so -L output/sysroot ./program

# Instruction execution logging
qemu-riscv64 -plugin libexeclog.so -L output/sysroot ./program

# Branch statistics
qemu-riscv64 -plugin libbranchstats.so -L output/sysroot ./program

# Cache simulation
qemu-riscv64 -plugin libcache.so -L output/sysroot ./program

# Coverage tracking
qemu-riscv64 -plugin libdrcov.so -L output/sysroot ./program
```

Plugin `.so` files are located in `third_party/qemu/build/contrib/plugins/`.

## Running Vector (V Extension) Programs

To run programs that use the RISC-V Vector extension:

1. Ensure the binary was compiled with `-march=rv64gcv`
2. Enable V explicitly on the CPU:

```bash
# Use generic CPU with V enabled
qemu-riscv64 -cpu rv64,v=true -L output/sysroot ./vec_program

# With BBV profiling on vector code
qemu-riscv64 -cpu rv64,v=true -L output/sysroot \
  -plugin third_party/qemu/build/contrib/plugins/bbv.so,interval=10000,outfile=output/vec.bbv \
  ./vec_program [args...]
```

## Common Patterns & Troubleshooting

### "No such file" for shared libraries

The program needs a sysroot with RISC-V shared libraries. Ensure `-L` points to the correct sysroot:

```bash
qemu-riscv64 -L output/sysroot ./program  # NOT just ./program
```

### Program hangs or no output

- Check that the program and arguments are correct
- Try with `-d strace` to see syscall activity
- For programs that read stdin, pipe input: `echo "input" | qemu-riscv64 ...`

### Performance is slow

- Plugins add overhead. Without profiling, omit `-plugin`
- Use larger `interval` values for BBV to reduce sampling overhead
- Native execution is always faster than emulation — QEMU is for analysis, not benchmarking absolute performance

### Check if a binary is RISC-V

```bash
file ./output/yolo_inference
# Should show: ELF 64-bit LSB executable, UCB RISC-V, ...
```
