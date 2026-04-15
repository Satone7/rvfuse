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

## BBV Output Analysis

After BBV profiling, analyze the output with `tools/analyze_bbv.py`:

```bash
# Generate hotspot report
python3 tools/analyze_bbv.py \
  --bbv output/yolo.bbv.0.bb \
  --elf output/yolo_inference \
  --sysroot output/sysroot \
  --top 20 \
  --json-output output/hotspot.json \
  -o output/hotspot-report.txt
```

This resolves basic block addresses to source locations (function names) using addr2line, producing a ranked hotspot report. See **Hotspot Analysis Best Practices** below for correct usage.

## Hotspot Analysis Best Practices

### Correct Sysroot Path

**Critical**: The sysroot path must contain all shared libraries used by the profiled binary. Using an incorrect sysroot leads to misidentification of hotspots.

```bash
# Example: llama.cpp has its own sysroot with libllama.so, libggml*.so
python3 tools/analyze_bbv.py \
  --bbv output/llama.bbv.0.bb \
  --elf output/llama.cpp/bin/llama-cli \
  --sysroot output/llama.cpp/sysroot   # NOT output/sysroot!
```

**Verification**: Check that application-specific libraries exist in the sysroot:
```bash
# For llama.cpp
ls output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libllama*.so
ls output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libggml*.so

# For ONNX Runtime
ls output/sysroot/lib/riscv64-linux-gnu/libonnx*.so
```

### PIE Executable Address Resolution

Modern binaries are often PIE (Position Independent Executable). Runtime addresses differ from static file offsets, requiring base address detection.

The `analyze_bbv.py` script handles this automatically:
- Detects PIE executables via `file` command output
- Analyzes address distribution to estimate runtime base
- Converts runtime addresses to file offsets for addr2line

**Manual verification** (if needed):
```bash
# Check if binary is PIE
file output/llama.cpp/bin/llama-cli
# Should show: "DYN ... PIE executable"

# Estimated PIE base appears in stderr during analysis
python3 tools/analyze_bbv.py --bbv ... 2>&1 | grep -i "pie"
```

### Application Library Matching Priority

The analysis prioritizes application-specific libraries (llama, ggml, onnx, ort) over system libraries (libc, libcrypto). This prevents hotspots being incorrectly attributed to system libraries.

**Common misidentification patterns** (before fix):
- libcrypto.so showing 80%+ of execution (incorrect)
- libgnutls.so showing most hotspots (incorrect)

**Expected patterns** (after proper analysis):
- libllama.so showing majority of inference hotspots
- libggml-cpu.so showing quantization/computation hotspots
- libonnxruntime.so showing ML inference hotspots

### Verifying Analysis Results

#### 1. Library Distribution Check

```bash
python3 -c "
import json
with open('output/hotspot.json') as f:
    data = json.load(f)
libs = {}
for b in data['blocks']:
    loc = b['location']
    if '[' in loc:
        lib = loc.split('[')[1].split(']')[0]
        libs[lib] = libs.get(lib, 0) + b['count']
total = sum(libs.values())
for lib, cnt in sorted(libs.items(), key=lambda x: -x[1])[:5]:
    print(f'{lib}: {cnt/total*100:.2f}%')
"
```

**Expected**: Application libraries (libllama, libggml, libonnx) should dominate, not system libraries.

#### 2. Symbol Resolution Check

Hotspots should show meaningful function names, not `??`:

```bash
# Check top 10 hotspots for symbol quality
head -20 output/hotspot-report.txt | grep -E "^\s*[0-9]+"
```

**Good result**: `[libggml-cpu.so] ggml_quantize_mat_q8_0_4x4`
**Bad result**: `[libcrypto.so.3] ?? (??:0)` (wrong library + no symbol)

#### 3. Address-to-Library Mapping Verification

For suspicious hotspot addresses, manually verify which library they belong to:

```bash
# Check if address 0x7f290b083500 belongs to libggml-cpu.so
addr=0x7f290b083500
base=0x7f290b04f000  # Estimated from analysis
offset=$((addr - base))
addr2line -f -e output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libggml-cpu.so.0.9.11 $offset
```

If this gives a valid symbol, but the report shows libcrypto.so, the analysis has a matching error.

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Wrong sysroot | Hotspots in system libraries | Use application-specific sysroot |
| Missing .so files | `<unknown-so>@0x...` labels | Add libraries to sysroot |
| Stripped libraries | `??` for all symbols | Use debug builds or accept symbol unknowns |
| PIE not detected | Main binary addresses unmatched | Check `file` output shows "PIE" |
| Library overlap | Multiple libraries matched | Analysis uses symbol validation bonus |

### Analysis Algorithm Details

The `analyze_bbv.py` script uses these techniques for accurate matching:

1. **PIE base detection**: Analyzes low-address cluster (0x55* region) to estimate base
2. **Application library priority**: Processes llama/ggml/onnx libs before libc/libcrypto
3. **Execution count weighting**: Weights candidate bases by total execution count, not address count
4. **Symbol validation bonus**: Uses addr2line to verify symbols, giving 50% bonus to correct library

### Stripped Binary Handling

Most RISC-V libraries are stripped (no debug symbols). The analysis still works because:
- Dynamic symbol table (`nm -D`) provides function names
- Library matching uses LOAD segment address ranges
- Hotspots are attributed to correct libraries even without file/line info

```bash
# Check if library is stripped
file output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libllama.so.0.0.1
# Shows: "stripped" - but nm -D still shows exported symbols

nm -D output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libllama.so.0.0.1 | head
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