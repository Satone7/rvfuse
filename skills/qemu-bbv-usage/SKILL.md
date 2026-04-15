---
name: qemu-bbv-usage
description: |
  How to use QEMU (qemu-riscv64) with the custom BBV plugin for RISC-V program profiling
  within the RVFuse project. Use this skill whenever the user mentions running QEMU,
  qemu-riscv64, BBV profiling, RISC-V emulation, QEMU plugins, sysroot, running RISC-V binaries,
  QEMU CPU selection, vector extension programs, or QEMU debugging options — even if they don't
  explicitly ask for "QEMU help."
---

# QEMU + BBV Plugin Usage Guide (RVFuse Project)

This skill covers **runtime usage** of QEMU with the custom BBV (Basic Block Vector) plugin.
For building QEMU and the plugin, see `verify_bbv.sh`.

## Key Paths

| Item | Path |
|------|------|
| QEMU binary | `third_party/qemu/build/qemu-riscv64` |
| Custom BBV plugin | `tools/bbv/libbbv.so` |
| Sysroot | `output/sysroot` (RISC-V shared libraries) |
| ELF binary | e.g. `output/yolo_inference` |
| BBV output | `outfile.<vcpu_index>.bb` + `outfile.disas` |

## BBV Plugin Overview

The BBV plugin generates basic block execution counts for hotspot analysis. This is a **custom implementation** located in `tools/bbv/` with these features:

- **Dual output**: Both `.bb` (execution counts) and `.disas` (disassembly) are generated
- **Exit flush**: Data is flushed at program exit, ensuring complete coverage
- **Per-vCPU tracking**: Supports multi-threaded programs with per-thread BBV data

### Source Files

| File | Purpose |
|------|---------|
| `tools/bbv/bbv.c` | Main plugin implementation |
| `tools/bbv/Makefile` | Build configuration |
| `tools/bbv/demo.c` | Test program for verification |

## Running RISC-V Programs (User Mode)

Basic execution requires a sysroot for dynamic linking:

```bash
qemu-riscv64 -L output/sysroot ./your_program [args...]
```

### CPU Selection (`-cpu`)

```bash
# Generic RV64 with specific extensions enabled
qemu-riscv64 -cpu rv64,v=true -L output/sysroot ./your_vec_program

# List all available CPU types
qemu-riscv64 -cpu help
```

### Debugging Flags (`-d`)

```bash
# System call trace
qemu-riscv64 -d strace -L output/sysroot ./program

# CPU register state before each translation block
qemu-riscv64 -d cpu -L output/sysroot ./program

# Log unimplemented instructions
qemu-riscv64 -d unimp -L output/sysroot ./program
```

### GDB Remote Debugging

```bash
# Terminal 1: Start QEMU waiting for GDB
qemu-riscv64 -g 1234 -L output/sysroot ./program

# Terminal 2: Connect with GDB
riscv64-unknown-elf-gdb -ex "target remote localhost:1234" ./program
```

## BBV Plugin Usage

### Basic Command

```bash
qemu-riscv64 -L output/sysroot \
  -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg
```

### Plugin Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `outfile` | **Yes** | - | Output file prefix (must be specified) |
| `interval` | No | 100000000 | Instructions per BBV sampling interval |

**Note**: `outfile` is **required** — the plugin fails if not specified.

### Output Files

The plugin generates two files:

1. **`outfile.<vcpu_index>.bb`** — Basic block execution counts
   - Format: `T:index:count` per line (one line per interval)
   - Example: `T:0:1000 1:500 2:250` means BB#0 executed 1000 times, BB#1 500 times

2. **`outfile.disas`** — Disassembly of all encountered basic blocks
   - Format per block:
     ```
     BB <index> (vaddr: 0x<vaddr>, <n> insns):
       0x<addr>: <disassembly>
       ...
     ```

### Choosing the Interval

| Interval | Granularity | Output Size | Use Case |
|----------|-------------|-------------|----------|
| 1000-10000 | Fine | Large | Short programs, detailed analysis |
| 10000-100000 | Medium | Moderate | Typical profiling (YOLO inference) |
| 100000+ | Coarse | Small | Long-running programs |

### Interval Behavior (from bbv.c)

The plugin counts instructions and triggers output when count reaches `interval`:

```c
// After each translation block execution:
qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
    tb, QEMU_PLUGIN_INLINE_ADD_U64, count_u64(), n_insns);

// When count >= interval, flush and reset:
qemu_plugin_register_vcpu_tb_exec_cond_cb(
    tb, vcpu_interval_exec, QEMU_PLUGIN_CB_NO_REGS,
    QEMU_PLUGIN_COND_GE, count_u64(), interval, NULL);
```

At program exit, `plugin_flush()` writes any remaining counts.

## Building the Custom BBV Plugin

```bash
# Build the custom plugin (requires QEMU headers from submodule)
make -C tools/bbv/

# Output: tools/bbv/libbbv.so
```

Requirements:
- `gcc` with `-shared -fPIC` support
- `glib-2.0` development package
- QEMU headers at `third_party/qemu/include/qemu/qemu-plugin.h`

## Verification

Run the verification script to build QEMU and test the BBV plugin:

```bash
# Build and verify (first time)
./verify_bbv.sh

# Force rebuild
./verify_bbv.sh --force-rebuild
```

The script:
1. Compiles a demo test program (`tools/bbv/demo.c`)
2. Builds QEMU with plugin support
3. Builds the custom BBV plugin
4. Runs the demo with BBV profiling
5. Validates output file generation

## Downstream Pipeline

After BBV profiling, use RVFuse analysis tools:

```bash
# Generate hotspot report
python3 tools/analyze_bbv.py --bbv output/yolo.bbv.0.bb --elf output/yolo_inference --sysroot output/sysroot

# Generate DFG from hot basic blocks
./tools/profile_to_dfg.sh --bbv output/yolo.bbv.0.bb --elf output/yolo_inference --sysroot output/sysroot --top 50 --output-dir output/dfg

# Or generate DFG directly from .disas file
python -m tools.dfg --disas output/yolo.bbv.disas --isa I,F,M --top 20
```

## Running Vector (V Extension) Programs

```bash
# Enable V extension explicitly
qemu-riscv64 -cpu rv64,v=true -L output/sysroot ./vec_program

# With BBV profiling
qemu-riscv64 -cpu rv64,v=true -L output/sysroot \
  -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/vec.bbv \
  ./vec_program [args...]
```

## Common Patterns & Troubleshooting

### "No such file" for shared libraries

Ensure `-L` points to the correct sysroot:

```bash
qemu-riscv64 -L output/sysroot ./program  # NOT just ./program
```

### "outfile unspecified" error

The custom BBV plugin requires `outfile` parameter:

```bash
# Wrong: missing outfile
qemu-riscv64 -plugin tools/bbv/libbbv.so,interval=10000 ./program

# Correct:
qemu-riscv64 -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/test ./program
```

### BBV output not generated

- Check plugin path: `tools/bbv/libbbv.so` (not upstream path)
- Verify plugin was built: `make -C tools/bbv/`
- Run `./verify_bbv.sh` to validate the setup

### Performance is slow

- Use larger `interval` values to reduce output frequency
- Without profiling, omit `-plugin` for faster execution
- QEMU is for analysis, not benchmarking absolute performance

### Check if a binary is RISC-V

```bash
file ./output/yolo_inference
# Should show: ELF 64-bit LSB executable, UCB RISC-V, ...
```

## Plugin Implementation Details

### Data Structures (from bbv.c)

```c
typedef struct Bb {
    uint64_t vaddr;              // Basic block virtual address
    struct qemu_plugin_scoreboard *count;  // Per-vCPU execution count
    unsigned int index;          // BB index for output
} Bb;

typedef struct Vcpu {
    uint64_t count;              // Instruction counter for interval
    FILE *file;                  // Output file handle
} Vcpu;
```

### Key Callbacks

| Callback | Purpose |
|----------|---------|
| `vcpu_init` | Open per-vCPU output file |
| `vcpu_tb_trans` | Register new BB, record disassembly |
| `vcpu_interval_exec` | Flush counts when interval reached |
| `plugin_exit` | Final flush and cleanup |