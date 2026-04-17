---
name: rvv-op
description: |
  Implement a RISC-V RVV vectorized operator for any target application, following the
  rvv-patches convention (single .inl source of truth + patch.diff + test.cpp + README.md).
  This skill guides you through the full development cycle: source analysis → cross-platform
  study → RVV implementation → patch integration → correctness test → integration test →
  gap analysis → PDF report.
  Trigger when: the user asks to implement, vectorize, or add an RVV/SIMD kernel for RISC-V
  in any application. Matches phrases like "RVV vectorize", "RISC-V vector", "rvv-patches",
  "算子向量化", "RVV 实现". The user specifies the target application path and the operator name.
---

# RVV Operator Implementation Workflow

## Purpose

Guide the complete development cycle for implementing a RISC-V RVV vectorized operator
in any target application:

1. Discover the application's structure and build system
2. Study cross-platform SIMD implementations (ARM NEON, x86 AVX, etc.)
3. Implement RVV version following the **rvv-patches convention**
4. Create patch, standalone test, and integrate into the application build
5. Run integration tests
6. Perform cross-platform gap analysis
7. Generate PDF report

## Input

The user provides:
- **Application path** — e.g., `applications/llama.cpp`, `applications/yolo`, or any directory
  containing a C/C++ project with multi-architecture SIMD code
- **Operator name** — e.g., `ggml_gemv_q5_0_q8_0`, `conv2d_3x3`, `softmax_fp16`
- (Optional) **VLEN** — Vector register length, default 512-bit

## Conventions

All RVV operator implementations in this project follow a unified directory layout.
The agent must discover and conform to these conventions by inspecting the target
application, or create them if they do not yet exist.

### Operator Package Layout

Each operator lives in its own directory under `rvv-patches/`:

```
<app-path>/rvv-patches/<operator-name>/
├── rvv_<operator>.inl    # RVV implementation (single source of truth)
├── patch.diff            # Patch to integrate into the target application
├── test.cpp              # Standalone correctness test (RVV vs scalar)
└── README.md             # Operator documentation
```

**Naming rules:**
- Directory name uses `kebab-case` from the function name (e.g., `ggml_gemv_q5_0_q8_0`)
- `.inl` filename: `rvv_<operator_name>.inl`
- The `.inl` file is the **single source of truth** — both the production build and the
  test binary include this same file

### Template Directory

When an application's `rvv-patches/` directory exists, check for `_template/` which
contains starter files:

```
rvv-patches/_template/
├── rvv_<name>.inl.template
├── patch.diff.template
├── test.cpp.template
└── README.md.template
```

Use these templates when creating a new operator. If no template exists, follow the
conventions described below.

### Report Output Layout

```
docs/report/<app-name>/
├── rvv-gap-analysis-<operator>-YYYY-MM-DD.md
└── pdf/
    └── rvv-gap-analysis-<operator>-YYYY-MM-DD.pdf
```

Where `<app-name>` is the last path component of the application directory (e.g., `llama.cpp`,
`yolo`).

## Application Discovery

Before starting, inspect the target application to understand its structure:

### Step 1: Read Application README

Read `<app-path>/README.md` to discover:
- Directory layout (`vendor/`, `output/`, `rvv-patches/`, etc.)
- Build commands (build scripts, makefiles, CMakeLists.txt)
- Cross-compilation toolchain and sysroot setup
- QEMU execution commands
- Existing RVV patches and their status
- Hotspot analysis data (which operators are worth vectorizing)

### Step 2: Discover Application Structure

Inspect the application directory to find:

| What to find | Where to look |
|---|---|
| Source code (with arch-specific SIMD) | `<app-path>/vendor/*/` or `<app-path>/src/` |
| Build script | `<app-path>/build.sh`, `Makefile`, `CMakeLists.txt` |
| Existing RVV patches | `<app-path>/rvv-patches/*/` |
| Patch application logic | `build.sh` (look for `apply_patches` or `git apply`) |
| Test runner | `build.sh --test` or dedicated test script |
| Integration test / smoke test | `<app-path>/qwen`, `<app-path>/run_test.sh`, etc. |
| Cross-compile toolchain | `riscv64-linux-toolchain.cmake`, `third_party/llvm-install/` |
| QEMU binary | `third_party/qemu/build/qemu-riscv64` |
| Sysroot | `output/<app>/sysroot/` or shared `output/sysroot/` |
| Model / test data | `<app-path>/models/` or `<app-path>/test_data/` |

Extract the **patch target file** — the source file in the application that the `.inl`
will be `#include`d into (e.g., `ggml/src/ggml-cpu/arch/riscv/repack.cpp`).

### Step 3: Map Cross-Platform SIMD Paths

Find the architecture-specific source files for the target operator:

| Platform | Typical path pattern |
|---|---|
| ARM NEON / SVE | `arch/arm/`, `arch/aarch64/`, `neon/`, `sve/` |
| x86 SSE / AVX / AVX2 | `arch/x86/`, `avx/`, `avx2/` |
| RISC-V RVV (existing) | `arch/riscv/`, `riscv/` |
| Scalar / generic | `ops.cpp`, `generic/`, `common/` |

## Workflow Phases

### Phase 1: Research & Analysis

#### Step 1: Read Project Context

Read `<app-path>/README.md` per the discovery procedure above.

#### Step 2: Analyze Cross-Platform Implementations

Spawn a subagent to analyze the target application's source code:

**For each platform found, extract:**
- Function signature matching the target operator
- Algorithm structure (loop nesting, blocking strategy, tiling)
- Key intrinsics used (FMA, shuffle, load/store, reduction)
- Data structure layouts (block types, quantization formats)
- Register usage and memory access patterns

**Report format:**
```
## Platform: ARM NEON
- Function: <name> (file:line range)
- Key operations: <intrinsics list>
- Blocking: <tile sizes>
- Data layout: <struct definitions>

## Platform: x86 AVX2
- Function: <name> (file:line range)
- Key operations: <intrinsics list>
...
```

### Phase 2: Implementation

#### Step 3: Create Operator Package

Create the operator directory and files following the convention:

```bash
mkdir -p <app-path>/rvv-patches/<operator-name>
```

**File 1: `rvv_<operator>.inl`** — RVV implementation (single source of truth)

Structure:
```cpp
// rvv_<operator>.inl — RVV implementation of <function_name>
//
// Single source of truth. Included by:
//   - <patch-target-file>  (via patch, production build)
//   - test.cpp             (correctness test)
//
// Prerequisites before including:
//   - Required type definitions must be defined
//   - GGML_RESTRICT / similar macros must be defined
//   - On RVV: <riscv_vector.h> must be included

#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// RVV implementation
inline void <function_name>_rvv(<parameters>) {
    // ... RVV intrinsics implementation
}

// Wrapper: selects RVV or generic at compile time
inline void <function_name>(<parameters>) {
#if defined(__riscv_v_intrinsic)
    <function_name>_rvv(/* args */);
#else
    <function_name>_generic(/* args */);
#endif
}
```

**RVV implementation guidelines:**
- Use `vsetvl` to handle tail elements (never assume full vector length)
- Prefer widening operations for narrow-to-wide MAC (int8 → int32)
- Use segment loads/stores for interleaved data (`vlseg*`, `vsseg*`)
- Document the algorithm with inline comments
- Provide compile-time fallback to scalar via `#if defined(__riscv_v_intrinsic)`

**File 2: `test.cpp`** — Standalone correctness test

Structure:
```cpp
// test.cpp — correctness test for RVV <function_name>
//
// Build (rv64gcv, VLEN=512):
//   <compiler> -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
//       -I<path-to-operator-dir> test.cpp -o test -lm

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Minimal type definitions (copied from application headers)
// ...

// Scalar (generic) reference
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void <function_name>_generic(<parameters>) {
    // Scalar implementation extracted from the application
}

// RVV implementation — single source of truth
#include "rvv_<operator>.inl"

// Deterministic pseudo-random data generator (LCG)
static uint32_t rng_state = 42;
static uint32_t rng_next() { ... }

// Test runner
static int run_test(<params>, const char * label) {
    // Generate input data
    // Run both RVV and scalar
    // Compare with tolerance
}

int main() {
    int failures = 0;
    failures += run_test(..., "test-1");
    // ...
    printf("Summary: %d test(s) failed\n", failures);
    return failures;
}
```

Key test design rules:
- Scalar reference must match the application's generic implementation
- Use `__attribute__((optnone))` on RVV targets to bypass LLVM optimizer bugs
- Use deterministic LCG RNG (seed = 42) for reproducibility
- Test multiple input sizes (small, medium, large)
- Tolerance depends on data type: FP32 → 1e-6, FP16 → 1e-3, INT8 quantization → 1

**File 3: `patch.diff`** — Integration into target application

```diff
diff --git a/<patch-target-file> b/<patch-target-file>
--- a/<patch-target-file>
+++ b/<patch-target-file>
@@ -<line>,6 +<line>,10 @@

 <context: existing code>

+// RVV implementation: single source of truth is rvv_<operator>.inl
+// (copied to this directory by build.sh before patch is applied)
+#include "rvv_<operator>.inl"
+
 <context: surrounding code>
```

The patch must:
- Target the correct source file discovered in Step 2
- Add the `#include` at the right location (after existing guards/defines, before functions)
- Use enough context lines (±6) for stable application
- Be applicable with `git apply --check`

**File 4: `README.md`** — Per-operator documentation

```markdown
# <operator-name>

RVV implementation of `<function_name>` — <brief description>.

## Status
✅ Tests passing / ⚠️ Blocked by <reason>

## Files
| File | Purpose |
|------|---------|
| `rvv_<operator>.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into <application> |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature
\```cpp
<function_signature>
\```

## Algorithm
1. <Step 1>
2. <Step 2>

## VLEN Requirement
- VLEN >= <threshold>: Uses RVV intrinsics
- VLEN < <threshold>: Falls back to scalar

## Build & Test
\```bash
# Build application with this patch
<build-command>
# Run standalone test
<test-command>
\```
```

#### Step 4: Verify Patch Application

```bash
# Check if patch applies cleanly
git -C <source-dir> apply --check <app-path>/rvv-patches/<operator-name>/patch.diff
```

If it fails:
- Check line numbers match the actual source file
- Verify the application source version matches what the patch was generated against
- Re-generate patch with correct context lines

### Phase 3: Testing

#### Step 5: Build and Run Standalone Test

Use the application's test infrastructure:

```bash
# If the application has a build script with --test flag:
cd <app-path> && ./build.sh --test

# Or compile and run manually:
<cross-compiler> -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
    -I<operator-dir> <test.cpp> -o <test-bin> -lm
<qemu-binary> -L <sysroot> <test-bin>
```

If tests fail:
- Check tolerance thresholds for the data type
- Verify scalar reference matches the application's generic implementation
- Try `-O0` if LLVM-22 optimizer bug is suspected (see Common Issues)

#### Step 6: Integration Test

Run the application end-to-end to verify the operator is actually invoked at runtime:

1. **Build** the application with the patch integrated:
   ```bash
   cd <app-path> && ./build.sh --force
   ```

2. **Run** a representative workload (discovered in Step 1):
   - This may be a dedicated script (e.g., `./qwen -p "test"`),
     a benchmark binary, or a test harness
   - Use the application's README to find the right command

3. **Verify** the RVV operator is called:
   - Check that the operator symbol exists in the shared library:
     `nm <output-lib> | grep <operator>`
   - Or use QEMU tracing: `-d in_asm` to observe execution
   - Or add temporary debug prints and check stdout

### Phase 4: Reporting

#### Step 7: Run Gap Analysis

Invoke the `rvv-gap-analysis` skill:

```
Use the rvv-gap-analysis skill to analyze the <operator> implementation.
RVV source: <app-path>/rvv-patches/<operator-name>/rvv_<operator>.inl
VLEN: 512-bit
Reference implementations:
- ARM NEON: <path-discovered-in-step-2>
- x86 AVX2: <path-discovered-in-step-2>
Output: docs/report/<app-name>/rvv-gap-analysis-<operator>-YYYY-MM-DD.md
```

The gap analysis skill will:
1. Parse the RVV implementation
2. Launch parallel platform analysis subagents (x86, ARM, LoongArch, Power, S390X, WASM)
3. Identify instructions missing from RVV
4. Propose new RVV extensions with benefit estimates

#### Step 8: Generate PDF Report

Invoke the `lovstudio-any2pdf` skill:

```
Use the lovstudio-any2pdf skill to convert the gap analysis report to PDF.
Input: docs/report/<app-name>/rvv-gap-analysis-<operator>-YYYY-MM-DD.md
Output: docs/report/<app-name>/pdf/rvv-gap-analysis-<operator>-YYYY-MM-DD.pdf
Theme: github-light
Skip cover page, skip watermark, skip frontispiece
```

## Common Issues

### LLVM Optimizer Bug (LLVM-22)

LLVM 22 RISC-V backend may produce incorrect code with `-O2` and RVV intrinsics.
- **Symptom**: Array elements contain garbage despite zero-initialization
- **Workaround**: Use `__attribute__((optnone))` for test harness code
- **Reference**: LLVM issue #83370

### VLEN-Dependent Paths

Some operators only benefit from RVV above a certain VLEN:
- Segment stores for interleaving: VLEN >= 256
- Full-row GEMV accumulation: VLEN >= 512
- Always provide a scalar fallback path for smaller VLEN configs

### Patch Context Mismatch

If the application pins a specific upstream version, the patch context lines must match:
- Check the actual source file before generating the patch
- Use `git -C <source-dir> log --oneline -1` to verify the version

## Validation Checklist

Before reporting completion, verify:

- [ ] `rvv_<operator>.inl` compiles with the RVV cross-compiler
- [ ] `test.cpp` passes under QEMU (RVV output matches scalar within tolerance)
- [ ] `patch.diff` applies cleanly with `git apply --check`
- [ ] Full application build succeeds with the patch integrated
- [ ] Integration test runs and the operator symbol is present in the output
- [ ] Gap analysis report generated in `docs/report/<app-name>/`
- [ ] PDF report generated in `docs/report/<app-name>/pdf/`

## References

- `skills/rvv-gap-analysis/SKILL.md` — Gap analysis methodology
- `/home/pren/.claude/skills/lovstudio-any2pdf/SKILL.md` — PDF conversion
- RVV 1.0 specification: https://github.com/riscv/riscv-v-spec
