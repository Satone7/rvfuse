---
name: perf-scalar-profiling
description: |
  Profile RISC-V inference workloads on real hardware (Banana Pi) using Linux perf.
  Uploads cross-compiled binaries, shared libraries, sysroot, and models from the
  development machine via SSH/paramiko, runs perf profiling, pulls results back.
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

## Supported Frameworks

### ONNX Runtime (generic_ort_runner)

Cross-compiled via `applications/yolo/ort/build.sh` (see cross-compile-app skill).
The `generic_ort_runner` reads input shape from the ONNX model itself and generates
random test data — no need to write model-specific test cases.

| Artifact | Local Path | Remote Path |
|----------|-----------|-------------|
| Runner binary | `output/cross-ort/generic_ort_runner` | `<remote-dir>/rootfs/generic_ort_runner` |
| ORT shared lib | `output/cross-ort/lib/libonnxruntime.so*` | inside rootfs.tar.gz |
| Rootfs (chroot) | `output/cross-ort/rootfs.tar.gz` | `<remote-dir>/rootfs.tar.gz` |
| ONNX models | `*.onnx` or `*.ort` files | `<remote-dir>/rootfs/` |

### llama.cpp

Cross-compiled via `applications/llama.cpp/build.sh` (see cross-compile-app skill).

| Artifact | Local Path | Remote Path |
|----------|-----------|-------------|
| CLI binary | `output/llama.cpp/bin/llama-cli` | `<remote-dir>/llama-cli` |
| Core library | `output/llama.cpp/lib/libllama.so*` | `<remote-dir>/lib/` |
| GGML backends | `output/llama.cpp/lib/libggml*.so*` | `<remote-dir>/lib/` |
| Sysroot | `output/llama.cpp/sysroot/` | `<remote-dir>/sysroot/` |
| GGUF models | `*.gguf` files | `<remote-dir>/` |

## Remote Profiling Workflow

Run from your x86 development machine. The script handles everything: model download → upload → profile → download.

### Model Auto-Download

Models can be specified by short name, local path, or URL. If not found locally,
the script auto-downloads to `output/models/`.

```bash
# List all available models
python3 tools/perf_scalar_profile.py --list-models
```

| Short Name | Framework | Description |
|------------|-----------|-------------|
| `resnet50` | ORT | ResNet-50 v2 (98 MB) |
| `mobilenetv2` | ORT | MobileNetV2 (14 MB) |
| `squeezenet` | ORT | SqueezeNet 1.1 (5 MB) |
| `shufflenet` | ORT | ShuffleNet (9 MB) |
| `vgg16` | ORT | VGG-16 (528 MB) |
| `densenet121` | ORT | DenseNet-121 |
| `inception` | ORT | Inception v1 (27 MB) |
| `efficientnet-lite4` | ORT | EfficientNet-Lite4 |
| `qwen-0.5b-q4_0` | llama.cpp | Qwen2.5 0.5B Instruct Q4_0 (~350 MB) |
| `qwen-1.5b-q4_0` | llama.cpp | Qwen2.5 1.5B Instruct Q4_0 (~1 GB) |

Resolution order:
1. Exact local file path (e.g., `output/models/resnet50.onnx`)
2. `output/models/<basename>` (e.g., `resnet50` → `output/models/resnet50.onnx`)
3. Registry download (e.g., `mobilenetv2` → downloads from ONNX Model Zoo)
4. URL download (e.g., `https://example.com/model.onnx`)

### Connection Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | required | Banana Pi IP address |
| `--user` | `root` | SSH username |
| `--password` | required | SSH password |
| `--remote-dir` | `/root` | Remote working directory |

### Usage: ONNX Runtime

```bash
# Chroot mode (recommended) — isolated glibc, portable across boards
python3 tools/perf_scalar_profile.py \
  --host 192.168.1.22 --user root --password bianbu \
  --remote-dir /root/ort-perf \
  --runner output/cross-ort/generic_ort_runner \
  --rootfs output/cross-ort/rootfs.tar.gz \
  --models resnet50 mobilenetv2 squeezenet \
  --outdir output/perf/ort \
  --iterations 30 --freq 999

# Skip upload after first deploy
python3 tools/perf_scalar_profile.py \
  --host 192.168.1.22 --user root --password bianbu \
  --remote-dir /root/ort-perf \
  --skip-upload \
  --runner generic_ort_runner \
  --rootfs output/cross-ort/rootfs.tar.gz \
  --models resnet50 \
  --iterations 30 --freq 999
```

### Usage: llama.cpp

```bash
# Auto-downloads GGUF from HuggingFace
python3 tools/perf_scalar_profile.py \
  --host 192.168.1.22 --user root --password bianbu \
  --remote-dir /root/llama-perf \
  --runner output/llama.cpp/bin/llama-cli \
  --libs output/llama.cpp/lib \
  --sysroot output/llama.cpp/sysroot \
  --models qwen-0.5b-q4_0 \
  --input "-p \"Hello world\" -n 128" \
  --outdir output/perf/llama
```

### Workflow Diagram

```
┌──────────────────────────┐                  ┌──────────────────────┐
│  Dev Machine (x86_64)    │                  │  Banana Pi (rv64gcv) │
│                          │  1. Upload       │                      │
│  perf_scalar_profile.py  │─────────────────►│  /root/ort-perf/     │
│                          │     SFTP         │    generic_ort_runner│
│                          │                  │    lib/*.so          │
│                          │                  │    sysroot/          │
│                          │                  │    *.onnx            │
│                          │                  │                      │
│                          │  2. Profile      │                      │
│                          │─────────────────►│  perf stat -d        │
│                          │     SSH exec     │  perf record         │
│                          │                  │    -e cpu-clock -g   │
│                          │                  │  perf report         │
│                          │                  │  perf annotate       │
│                          │                  │                      │
│  output/perf/ort/        │  3. Download     │                      │
│    resnet50/             │◄──────────────────│  perf_stat.txt       │
│      perf_stat.txt       │     SFTP         │  perf_report.txt     │
│      perf_report.txt     │                  │  perf_annotate.txt   │
│    summary.md            │  4. Cleanup      │                      │
│                          │─────────────────►│  rm temp files       │
└──────────────────────────┘                  └──────────────────────┘
```

### Script Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--host` | Yes | - | Banana Pi IP address |
| `--user` | No | `root` | SSH username |
| `--password` | Yes | - | SSH password |
| `--runner` | Yes | - | Local path to inference binary |
| `--models` | Yes | - | Models: short names (`resnet50`), local paths, or URLs |
| `--models-dir` | No | `output/models` | Local directory for downloaded models |
| `--list-models` | No | - | List available models and exit |
| `--input` | No | `""` | Extra args for runner (llama.cpp prompt args) |
| `--iterations` | No | `30` | Inference iterations for ORT runner |
| `--libs` | No | - | Local directory with shared libraries (.so) |
| `--sysroot` | No | - | Local sysroot directory |
| `--outdir` | No | `output/perf` | Local output directory |
| `--remote-dir` | No | `/root` | Remote working directory |
| `--freq` | No | `999` | Sampling frequency (Hz) |
| `--upload-only` | No | false | Upload files only, skip profiling |
| `--skip-upload` | No | false | Skip upload (files already on board) |
| `--dry-run` | No | false | Print actions without executing |

## Prerequisites

### Remote Board

```bash
# Verify perf is installed
perf --version

# If permission denied, lower paranoid level
sudo sysctl -w kernel.perf_event_paranoid=0
```

The script automatically checks and sets `perf_event_paranoid=0` if needed.

### Local Machine

```bash
pip install paramiko
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
The script uses `-e cpu-clock` (software event) which always works:

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
4. Shared libs must be in `LD_LIBRARY_PATH` on the remote board

## Profiling Commands (Reference)

These are the commands the script runs remotely. Useful for manual debugging.

### perf stat — Global Metrics

```bash
perf stat -d -- ./generic_ort_runner model.onnx 30
```

| Metric | Meaning | Fusion Relevance |
|--------|---------|------------------|
| `cycles` | Total CPU cycles | Baseline for speedup estimation |
| `instructions` | Total instructions retired | Instruction mix density |
| `IPC` | Instructions per cycle | Compute vs memory bottleneck |
| `cache-misses` | Cache miss rate | Memory-bound vs compute-bound |
| `branch-misses` | Mispredicted branches | Control-flow hotspot |

### perf record — Sampling

```bash
perf record -e cpu-clock -g -F 999 -o perf.data -- ./generic_ort_runner model.onnx 30
```

### perf report — Function Hotspots

```bash
perf report --stdio -n --percent-limit 0.5 -i perf.data
```

### perf annotate — Instruction Hotspots

```bash
perf annotate --stdio -i perf.data
```

Output format:
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
| Permission denied | perf_event_paranoid too high | `sudo sysctl -w kernel.perf_event_paranoid=0` |
| No symbols (hex addresses) | Binary stripped or no debug info | Build with `-g -fno-omit-frame-pointer`; ensure `install` not `install/strip` |
| No symbols in chroot mode | perf can't find .so inside rootfs | Script uses `--symfs <rootfs>` automatically |
| No samples in perf record | Hardware PMU not supported on SBI | Use `-e cpu-clock` (script does this by default) |
| Too few samples | Low freq or short run | Increase `--freq 9999` or `--iterations 100` |
| High overhead | Frequency too high | Use `--freq 99` |
| SSH connection fails | Network or auth issue | Test: `python3 -c "import paramiko; ..."` |
| glibc version mismatch | Board glibc older than build sysroot | Use `--rootfs` for chroot isolation |
| chroot fails | No root on board | Script requires root SSH access for chroot/mount |

## Limitations

| Limitation | Mitigation |
|------------|------------|
| SBI PMU events vary by firmware | Check `perf list`; software events always work |
| perf is statistical, not exact | Increase `--freq` for more resolution |
| Limited PMU counters | Profile events in separate runs |
| cpu-clock samples at lower rate | Use high freq (999+) and more iterations (30+) |
| chroot requires root | SSH as root user (default on Banana Pi) |
