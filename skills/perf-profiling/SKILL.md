---
name: perf-profiling
description: |
  Profile RISC-V inference workloads on real hardware (Banana Pi) using Linux perf.
  Uploads cross-compiled binaries, shared libraries, sysroot, and models from the
  development machine via SSH, runs perf profiling, pulls results back.
  Supports both ONNX Runtime (generic_ort_runner) and llama.cpp frameworks.
  Auto-detects framework from model file extension (.onnx / .gguf).
  Use this skill when the user mentions: perf profiling, hardware profiling,
  香蕉派 profiling, real hardware hotspot, instruction fusion hotspot discovery,
  多模型 profiling, perf stat, perf record, perf annotate, remote profiling,
  远程 profiling, or replacing QEMU BBV with hardware profiling.
---

# Perf Profiling on RISC-V Hardware

Profile inference workloads on real RISC-V hardware (Banana Pi) using Linux `perf`.
Upload cross-compiled binaries from the development machine, run perf on the board,
pull results back. Unlike QEMU BBV profiling, hardware perf captures real cycle counts,
cache behavior, and branch prediction effects.

## Execution Model

This skill describes the profiling **workflow and commands**. Agents execute each step
directly using `ssh`, `scp`, and shell commands — no monolithic wrapper script.
This keeps each step observable, debuggable, and adaptable to edge cases (network
glitches, missing models, chroot mount failures, etc.).

Prefer `sshpass` for non-interactive SSH auth when the board uses password login.
For key-based auth, use `ssh -i <keyfile>`.

```bash
# Install sshpass if not already available
sudo apt-get install -y sshpass
```

## Supported Frameworks

### ONNX Runtime (generic_ort_runner)

| Artifact | Local Path | Remote Path |
|----------|-----------|-------------|
| Runner binary | `output/cross-ort/generic_ort_runner` | `<remote-dir>/rootfs/generic_ort_runner` |
| ORT shared lib | `output/cross-ort/lib/libonnxruntime.so*` | inside rootfs.tar.gz |
| Rootfs (chroot) | `output/cross-ort/rootfs.tar.gz` | `<remote-dir>/rootfs.tar.gz` |
| ONNX models | `*.onnx` or `*.ort` files | `<remote-dir>/rootfs/` |

### llama.cpp

| Artifact | Local Path | Remote Path |
|----------|-----------|-------------|
| CLI binary | `output/llama.cpp/bin/llama-cli` | `<remote-dir>/llama-cli` |
| Core library | `output/llama.cpp/lib/libllama.so*` | `<remote-dir>/lib/` |
| GGML backends | `output/llama.cpp/lib/libggml*.so*` | `<remote-dir>/lib/` |
| Sysroot | `output/llama.cpp/sysroot/` | `<remote-dir>/sysroot/` |
| GGUF models | `*.gguf` files | `<remote-dir>/` |

## Remote Profiling Workflow

Execute these steps in order. Each step is a standalone `ssh`/`scp`/shell command.

### Step 1: Resolve Models

Models can be local paths, short names, or URLs. If the model file doesn't exist
locally, download it to `output/models/`.

```bash
# ResNet50 (ONNX Model Zoo)
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"

# MobileNetV2
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx"

# SqueezeNet
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"

# ShuffleNet
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx"

# VGG-16
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-7.onnx"

# DenseNet-121
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-121.onnx"

# Inception v1
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx"

# EfficientNet-Lite4
wget -nc -P output/models/ \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"

# Qwen GGUF models (HuggingFace) — use huggingface-cli or wget
wget -nc -P output/models/ \
  "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_0.gguf"

wget -nc -P output/models/ \
  "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_0.gguf"
```

### Step 2: Check Remote Board

Verify SSH connectivity, perf availability, and kernel settings.

```bash
# Test connectivity
sshpass -p '<password>' ssh -o StrictHostKeyChecking=no root@<host> 'uname -m'

# Check perf
sshpass -p '<password>' ssh root@<host> 'perf --version'

# Check disk space (need ~2x model size free)
sshpass -p '<password>' ssh root@<host> 'df -h /'

# Lower paranoid level for perf sampling (required on most boards)
sshpass -p '<password>' ssh root@<host> \
  'cat /proc/sys/kernel/perf_event_paranoid && echo 0 > /proc/sys/kernel/perf_event_paranoid'
```

### Step 3: Upload Workload

Two modes: **chroot mode** (preferred for ORT — upload rootfs.tar.gz, extract, copy models in)
and **direct mode** (for llama.cpp — upload runner + libs + sysroot).

#### Chroot Mode (ONNX Runtime)

```bash
# Create remote work dir
sshpass -p '<password>' ssh root@<host> 'mkdir -p <remote-dir>'

# Upload rootfs.tar.gz
sshpass -p '<password>' scp output/cross-ort/rootfs.tar.gz root@<host>:<remote-dir>/

# Extract on remote
sshpass -p '<password>' ssh root@<host> "cd <remote-dir> && tar xzf rootfs.tar.gz"

# Upload models into rootfs
sshpass -p '<password>' scp output/models/resnet50.onnx root@<host>:<remote-dir>/rootfs/
sshpass -p '<password>' scp output/models/mobilenetv2.onnx root@<host>:<remote-dir>/rootfs/
```

#### Direct Mode (llama.cpp)

```bash
# Create remote work dir
sshpass -p '<password>' ssh root@<host> 'mkdir -p <remote-dir>/lib'

# Upload runner
sshpass -p '<password>' scp output/llama.cpp/bin/llama-cli root@<host>:<remote-dir>/

# Upload shared libs
sshpass -p '<password>' scp output/llama.cpp/lib/*.so* root@<host>:<remote-dir>/lib/

# Upload model
sshpass -p '<password>' scp output/models/Qwen2.5-0.5B-Instruct-Q4_0.gguf root@<host>:<remote-dir>/
```

### Step 4: Setup Chroot Mounts (ORT chroot mode only)

```bash
ROOTFS="<remote-dir>/rootfs"

# Create mount points
sshpass -p '<password>' ssh root@<host> "mkdir -p ${ROOTFS}/{proc,dev,sys,tmp}"

# Mount kernel filesystems
sshpass -p '<password>' ssh root@<host> "mount -t proc proc ${ROOTFS}/proc"
sshpass -p '<password>' ssh root@<host> "mount -t sysfs sysfs ${ROOTFS}/sys"
sshpass -p '<password>' ssh root@<host> "mount --bind /dev ${ROOTFS}/dev"
```

### Step 5: Run Perf Profiling

Run perf stat, perf record, perf report, and perf annotate for each model.

**IMPORTANT**: Use `-e cpu-clock` for RISC-V — the SBI PMU on Banana Pi does not reliably
support hardware cycle sampling. Use `chroot <rootfs>` to isolate the runtime environment
when using rootfs mode.

#### Build the runner command

For ORT (chroot mode):
```bash
RUN_CMD="chroot <remote-dir>/rootfs /generic_ort_runner /<model>.onnx <iterations>"
```

For llama.cpp (direct mode):
```bash
RUN_CMD="LD_LIBRARY_PATH=<remote-dir>/lib <remote-dir>/llama-cli -m <remote-dir>/<model>.gguf <extra-args>"
```

#### perf stat — global metrics

```bash
sshpass -p '<password>' ssh root@<host> \
  "perf stat -d -o /tmp/perf_stat.txt -- ${RUN_CMD}"
```

Key metrics: cycles, instructions, IPC, cache-misses, branch-misses.

#### perf record — sampling

```bash
sshpass -p '<password>' ssh root@<host> \
  "perf record -e cpu-clock -g -F <freq> -o /tmp/perf.data -- ${RUN_CMD}"
```

Default freq: 999 Hz. Lower to 99 for long-running models; raise to 9999 for short runs.

#### perf report — function hotspots

```bash
# With symfs for chroot (maps symbols from inside rootfs):
sshpass -p '<password>' ssh root@<host> \
  "perf report --stdio -n --percent-limit 0.5 --symfs <remote-dir>/rootfs -i /tmp/perf.data > /tmp/perf_report.txt"

# Without symfs (direct mode):
sshpass -p '<password>' ssh root@<host> \
  "perf report --stdio -n --percent-limit 0.5 -i /tmp/perf.data > /tmp/perf_report.txt"
```

#### perf annotate — instruction-level hotspots

```bash
sshpass -p '<password>' ssh root@<host> \
  "perf annotate --stdio --symfs <remote-dir>/rootfs -i /tmp/perf.data > /tmp/perf_annotate.txt"
```

### Step 6: Download Results

```bash
mkdir -p output/perf/<model_name>/

sshpass -p '<password>' scp root@<host>:/tmp/perf_stat.txt output/perf/<model_name>/
sshpass -p '<password>' scp root@<host>:/tmp/perf_report.txt output/perf/<model_name>/
sshpass -p '<password>' scp root@<host>:/tmp/perf_annotate.txt output/perf/<model_name>/
```

### Step 7: Cleanup Remote

```bash
# Remove temp perf files
sshpass -p '<password>' ssh root@<host> 'rm -f /tmp/perf_stat.txt /tmp/perf_report.txt /tmp/perf_annotate.txt /tmp/perf.data'

# If chroot mode, teardown mounts:
sshpass -p '<password>' ssh root@<host> 'umount <remote-dir>/rootfs/proc; umount <remote-dir>/rootfs/sys; umount <remote-dir>/rootfs/dev'

# Optionally remove all uploaded files:
sshpass -p '<password>' ssh root@<host> 'rm -rf <remote-dir>'
```

### Step 8: Generate Summary

After profiling all models, generate a Markdown summary table at `output/perf/summary.md`.

Parse `perf_stat.txt` for cycles, instructions, IPC, cache-miss percentage.
Parse `perf_report.txt` for the top function name and its overhead percentage.
Write a table:

| Model | Framework | Cycles | Instructions | IPC | Cache Miss % | Top Function | Top % |
|-------|-----------|--------|-------------|-----|-------------|-------------|-------|
| ... | ... | ... | ... | ... | ... | ... | ... |

Then append the top-10 functions from each model's `perf_report.txt`.

## Profiling Multiple Models

When profiling multiple models with the same runner, upload once then loop
through models repeating Steps 5-7 for each. This avoids re-uploading the
rootfs/runner for each model.

```bash
MODELS=("resnet50.onnx" "mobilenetv2.onnx" "squeezenet.onnx")

for model in "${MODELS[@]}"; do
    model_name="${model%.*}"
    echo "=== Profiling: ${model_name} ==="
    # scp model into rootfs
    sshpass -p '<password>' scp "output/models/${model}" root@<host>:<remote-dir>/rootfs/
    # perf stat + record + report + annotate (Step 5 with per-model output dirs)
    # download results (Step 6 to output/perf/${model_name}/)
    # cleanup remote perf temp files
done
```

## Prerequisites

### Remote Board

```bash
# Verify perf is installed
perf --version

# If permission denied, lower paranoid level
sudo sysctl -w kernel.perf_event_paranoid=0
```

### Local Machine

```bash
# Required for non-interactive SSH
sudo apt-get install -y sshpass

# Optional — for key-based auth (avoids password in shell history)
ssh-keygen -t ed25519 -f ~/.ssh/bananapi -N ''
ssh-copy-id -i ~/.ssh/bananapi.pub root@<host>
```

### Cross-Compilation

Use the **cross-compile-app** skill to build ONNX Runtime or llama.cpp for RISC-V
before profiling. Key flags for good profiling:
- `-g -fno-omit-frame-pointer` for perf call graph unwinding
- Do NOT strip shared libraries (symbols needed for `perf report`)
- Use `generic_ort_runner` for ORT — it auto-detects model input shape

## RISC-V Perf Notes

### Software Events Required

Banana Pi (SpacemiT K1) SBI PMU does not reliably support hardware cycle sampling.
Use `-e cpu-clock` (software event) which always works:

```bash
# This works on Banana Pi:
perf record -e cpu-clock -g -F 999 -- ./runner model.onnx

# This may fail (no samples):
perf record -e cycles -g -F 999 -- ./runner model.onnx
```

### Symbol Resolution

For meaningful function names in `perf report`:
1. Build with `-g` (debug info) and without stripping
2. The `generic_ort_runner` and `yolo_inference` binaries have debug info by default
3. **libonnxruntime.so** must NOT be stripped — build script uses `install` (not `install/strip`)
4. In chroot mode, use `--symfs <rootfs>` so perf finds symbols inside the rootfs
5. In direct mode, shared libs must be in `LD_LIBRARY_PATH` on the remote board

## Perf Commands Reference

### perf stat — Global Metrics

| Metric | Meaning | Fusion Relevance |
|--------|---------|------------------|
| `cycles` | Total CPU cycles | Baseline for speedup estimation |
| `instructions` | Total instructions retired | Instruction mix density |
| `IPC` | Instructions per cycle | Compute vs memory bottleneck |
| `cache-misses` | Cache miss rate | Memory-bound vs compute-bound |
| `branch-misses` | Mispredicted branches | Control-flow hotspot |

### perf annotate Output Format

```
 Percent |      Source code & Disassembly
---------------------------------------------------------
 12.30 :   106b4:   ld      a5,0(a0)
  8.72 :   106b8:   addiw   a5,a5,1
  7.45 :   106bc:   mulw    a5,a5,a4
  5.21 :   106c0:   sw      a5,0(a0)
```

Instructions with >5% are strong fusion candidates.

## Hotspot Analysis for Fusion Candidates

### Fusion Patterns to Look For

**1. Load-Compute-Store (Memory-ALU fusion)**
```
 15.2 :   ld      a5,0(a0)
  9.8 :   addw    a5,a5,a4
 12.1 :   sw      a5,0(a0)
```

**2. Multiply-Accumulate Chain (MAC fusion)**
```
  8.5 :   mulw    a5,a3,a4
  7.2 :   addw    a5,a5,a6
```

**3. Address Calculation (Address generation fusion)**
```
  5.1 :   slli    a5,a5,2
  4.3 :   add     a5,a5,a0
  4.8 :   ld      a5,0(a5)
```

### Analysis Checklist

1. **Top-10 functions**: Which consume >80% of cycles?
2. **IPC**: Compute-bound (IPC < 1.0) or memory-bound?
3. **Cross-model overlap**: Shared hot functions = highest priority fusion targets
4. **Cross-framework overlap**: Functions hot in both ORT and llama.cpp = universal candidates

## Output Directory Convention

```
output/perf/
├── ort/                       # ONNX Runtime results
│   ├── resnet50/
│   │   ├── perf_stat.txt
│   │   ├── perf_report.txt
│   │   └── perf_annotate.txt
│   ├── mobilenetv2/
│   └── summary.md
└── llama/                     # llama.cpp results
    ├── qwen-0.5b/
    └── summary.md
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Permission denied | perf_event_paranoid too high | `ssh root@<host> 'echo 0 > /proc/sys/kernel/perf_event_paranoid'` |
| No symbols (hex addresses) | Binary stripped or no debug info | Build with `-g -fno-omit-frame-pointer` |
| No symbols in chroot mode | perf can't find .so inside rootfs | Use `perf report --symfs <rootfs>` |
| No samples in perf record | Hardware PMU not supported on SBI | Use `-e cpu-clock` |
| Too few samples | Low freq or short run | Increase `-F 9999` or more iterations |
| High overhead | Frequency too high | Lower `-F 99` |
| SSH hangs | Network issue or board overloaded | Add `-o ConnectTimeout=10`, check board load |
| scp stalls | Large file transfer over slow link | Use `-o ServerAliveInterval=15` |
| chroot fails (ENOENT) | Missing /lib or /bin inside rootfs | Verify rootfs.tar.gz was built correctly |
| glibc version mismatch | Board glibc older than build sysroot | Use chroot mode with rootfs |
| Disk full on remote | Large rootfs or multiple models | Check `df -h /` before uploading; clean old runs |

## Model Registry (Reference)

| Short Name | Framework | URL |
|------------|-----------|-----|
| `resnet50` | ORT | `github.com/onnx/models/.../resnet/model/resnet50-v2-7.onnx` |
| `mobilenetv2` | ORT | `github.com/onnx/models/.../mobilenet/model/mobilenetv2-10.onnx` |
| `squeezenet` | ORT | `github.com/onnx/models/.../squeezenet/model/squeezenet1.1-7.onnx` |
| `shufflenet` | ORT | `github.com/onnx/models/.../shufflenet/model/shufflenet-9.onnx` |
| `vgg16` | ORT | `github.com/onnx/models/.../vgg/model/vgg16-7.onnx` |
| `densenet121` | ORT | `github.com/onnx/models/.../densenet-121/model/densenet-121.onnx` |
| `inception` | ORT | `github.com/onnx/models/.../inception_v1/model/inception-v1-9.onnx` |
| `efficientnet-lite4` | ORT | `github.com/onnx/models/.../efficientnet-lite4/model/efficientnet-lite4-11.onnx` |
| `qwen-0.5b-q4_0` | llama.cpp | `huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/...` |
| `qwen-1.5b-q4_0` | llama.cpp | `huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/...` |

Full URLs are in Step 1 above.

## Limitations

| Limitation | Mitigation |
|------------|------------|
| SBI PMU events vary by firmware | Check `perf list`; software events always work |
| perf is statistical, not exact | Increase `-F` for more resolution |
| Limited PMU counters | Profile events in separate runs |
| cpu-clock samples at lower rate | Use high freq (999+) and more iterations (30+) |
| chroot requires root | SSH as root user (default on Banana Pi) |
| Password in shell history | Use `sshpass` from env var or set up SSH keys |
