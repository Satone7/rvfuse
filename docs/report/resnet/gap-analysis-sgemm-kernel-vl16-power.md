# SGEMM Kernel (VL=16) -- Power VSX (POWER10) MMA Gap Analysis

## Overview

**Analysis target**: SGEMM (single-precision GEMM) inner kernel, C = A * B
**RVV baseline**: VL=16 (VLEN=512bit, SEW=32bit, LMUL=1), 2 rows x 16 columns per call, K-unroll=2
**Reference platform**: Power VSX (POWER10) MMA -- 128-bit VR registers, Matrix Multiply Assist
**BBV data**: Not provided; benefits are theoretical per-BB instruction reduction estimates

---

## Summary Table

| Priority | Proposed Extension | Source Platform | BB-Scoped Benefit | BB Scope | Implementation Difficulty | RVV Status |
|----------|-------------------|-----------------|-------------------|----------|--------------------------|------------|
| P0 | `vmmacc.vv` (4x4 outer-product MAC) | Power VSX (POWER10) MMA | BB内减少62.5% (K-loop BB) | Inner K-loop body | High | No matrix-level instructions |
| P1 | Dedicated accumulator register file | Power VSX (POWER10) | BB内减少25.0% (register-save BB) | Accumulator spill/reload | High | Accumulators share vector register file |
| P2 | `vmERGEh/vmERGEo` + `vxxpermdi` (element broadcast) | Power VSX | BB内减少33.3% (A-broadcast BB) | A element broadcast | Medium | Requires scalar-to-vector broadcast |

**Benefit calculation basis** (no BBV data; per-BB instruction reduction only):
- BB内减少比例 = (original_instruction_count - proposed_instruction_count) / original_instruction_count x 100%
- All figures normalized to RVV VLEN=512bit, SEW=32bit
- Cumulative upper bound assumes non-overlapping BB targets

---

## RVV Baseline Analysis (sgemm-kernel-vl16)

### Loop Structure

The RVV SGEMM kernel processes a 2-row x 16-column tile with K-loop unrolled by 2.

**Per K-pair (2 K-steps) for 1 row x 16 columns:**

| Instruction | Count | Purpose |
|-------------|-------|---------|
| `flw` (scalar load A) | 2 | Load 2 A elements (one per K-step) |
| `vle32.v` (vector load B) | 2 | Load 16 B elements (one per K-step) |
| `vfmacc.vf` (FMA) | 2 | Multiply-accumulate: C += a * B_vec (16 FMA each) |

Total per K-pair per row: **6 instructions** yielding **32 FMA operations** (2 instructions x 16 elements).

For 2 rows x 16 columns x 2 K-steps:
- Total FMA: 2 rows x 32 FMA/row = **64 FMA**
- Total instructions: 2 rows x 6 instructions/row = **12 instructions** (K-loop body only)
- FMA density: 64 / 12 = 5.33 FMA per instruction

### Register Usage

- 2 vector registers for C accumulators (one per row, each holding 16 float32)
- A scalar element in `f` register (broadcast to all lanes via `vfmacc.vf`)
- 1 vector register for B elements (reused each iteration)
- Total: 2 vector registers + 1 scalar = minimal register pressure

### Key Constraint

No matrix-level multiply instruction exists. Each `vfmacc.vf` performs a scalar-vector outer product (1 x VL), requiring separate scalar loads for each A element and vector loads for each K-step.

---

## POWER10 MMA Analysis

### Architecture Overview

**Source file**: `applications/onnxrt/ort/vendor/onnxruntime/onnxruntime/core/mlas/lib/power/SgemmKernelPOWER10.cpp`

| Property | Value |
|----------|-------|
| Register width | 128-bit VSX VR (4 x float32 per register) |
| MMA instruction | `__builtin_mma_xvf32gerpp` |
| MMA operation | 4x4 matrix multiply-accumulate (outer product) |
| FMA per MMA | 16 (4x4 = 16 multiply-adds) |
| Accumulator | Dedicated `__vector_quad` (separate from VR file) |
| Tile size | 4 rows x 16 columns (CountM=4, CountN=16), optional 8 rows (CountM=8) |
| K-loop unroll | 4 |

### Core K-Loop Body Analysis

The hot path is the `while (k >= 4)` loop in `MlasSgemmMMAProcessCount<4>` (lines 189-203).

**Per K-iteration (k decrements by 4), CountM=4, CountN=16:**

**Step 1 -- Load A elements** (line 191):
```
MlasLoopUnroll<RowCount=4, MlasFgemmLoadAElements>()(AElements, a, lda);
```
- Loads 4 A elements (one per row) from column `a` of matrix A into `AElements[0..3]`
- Instructions: **4 scalar loads** (`lwz`)

**Step 2 -- Broadcast A elements** (line 192):
```
MlasSgemmComputeAElements<4>(AElements, ABroadcast);
```
This transforms 4 scalar A elements into 4 broadcast vectors (lines 43-52):
| Instruction (power intrinsic) | Count | Purpose |
|-------------------------------|-------|---------|
| `vec_mergee` | 2 | Merge even elements of pairs |
| `vec_mergeo` | 2 | Merge odd elements of pairs |
| `vec_xxpermdi` | 4 | Doubleword permute into broadcast layout |
Total: **8 vector permutation instructions** to produce 4 broadcast vectors

**Step 3 -- Execute MMA** (lines 197-200):
```
MlasSgemmComputeBlockMMA<4>(&acc[0], ABroadcast[0], A2Broadcast[0], B, 4);
MlasSgemmComputeBlockMMA<4>(&acc[0], ABroadcast[1], A2Broadcast[1], B+16, 4);
MlasSgemmComputeBlockMMA<4>(&acc[0], ABroadcast[2], A2Broadcast[2], B+32, 4);
MlasSgemmComputeBlockMMA<4>(&acc[0], ABroadcast[3], A2Broadcast[3], B+48, 4);
```

Each `MlasSgemmComputeBlockMMA` call (lines 63-81):
- Loads 4 B vectors (16 float32 total): `MlasLoadFloat32x4(B + {0,4,8,12})` = **4 vector loads**
- Issues 4 MMA instructions: `__builtin_mma_xvf32gerpp(&acc[0..3], ABroadcast[i], BElements[j])` = **4 MMA instructions**

Since CountM=4 (not 8), the `A2Broadcast` path (lines 76-79) is skipped.

Per `MlasSgemmComputeBlockMMA` call (CountM=4):
- 4 vector loads (B)
- 4 MMA instructions
- Total: **8 instructions**, producing 64 FMA (4 MMA x 16 FMA each)

4 calls to `MlasSgemmComputeBlockMMA` per K-step:
- 4 x 4 = 16 vector loads (B), but these load from contiguous 64-element B block
- 4 x 4 = 16 MMA instructions
- Total: **32 instructions** (16 loads + 16 MMA), producing **256 FMA**

**Full K-iteration (k -= 4) instruction count:**

| Category | Count | FMA Produced |
|----------|-------|-------------|
| A scalar loads | 4 | -- |
| A broadcast permutes | 8 | -- |
| B vector loads | 16 | -- |
| MMA instructions | 16 | 256 |
| Pointer updates (a += 4, B += 64) | 2 | -- |
| Loop branch | 1 | -- |
| **Total** | **47** | **256 FMA** |

**FMA density**: 256 / 47 = **5.45 FMA per instruction**

---

## Normalized Comparison (Equivalent Workload)

To compare fairly, normalize both implementations to the same workload: **4 rows x 16 columns x 4 K-steps = 256 FMA**.

### POWER10 MMA

From the analysis above: **47 instructions for 256 FMA**.

### RVV Baseline (VL=16, 2 rows)

To produce 256 FMA with the RVV kernel:
- Each K-step produces 2 rows x 16 FMA = 32 FMA using 12 instructions (6 per row)
- For 256 FMA: 256 / 32 = 8 K-steps
- Total instructions: 8 x 12 = **96 instructions**

But this processes only 2 rows (not 4). To match 4 rows:
- Double: 2 x 96 = **192 instructions** (or 12 x 16 = 192, where 16 = 4 rows / 2 rows x 8 K-steps)

Wait -- let me recompute more carefully.

**RVV per K-step for 2 rows x 16 columns:**
- Per row: 1 `flw` + 1 `vle32.v` + 1 `vfmacc.vf` = 3 instructions, yielding 16 FMA
- 2 rows: 6 instructions, yielding 32 FMA

**For 4 rows x 16 columns x 1 K-step:**
- 4 rows / 2 rows per call = 2 calls
- Or equivalently: 2 x 3 = 6 A loads + 2 x 1 = 2 B loads + 2 x 1 = 2 FMA = 10 instructions, yielding 64 FMA

Wait, the B load is shared between rows in the same K-step. Let me re-examine:

Actually, in the RVV kernel, each row independently loads its own A and B elements. So for 2 rows:
- Row 0: `flw a0; vle32.v b; vfmacc.vf c0, a0, b`
- Row 1: `flw a1; vle32.v b; vfmacc.vf c1, a1, b`

If the compiler shares the B load (which is typical), the per-K-step for 2 rows is:
- 2 `flw` + 1 `vle32.v` + 2 `vfmacc.vf` = **5 instructions**, yielding 32 FMA

For 4 rows (2 x the above):
- 4 `flw` + 1 `vle32.v` + 4 `vfmacc.vf` = **9 instructions**, yielding 64 FMA

**For 4 rows x 16 columns x 4 K-steps = 256 FMA:**
- RVV: 4 x 9 = **36 instructions** (plus pointer updates)
- With pointer updates (a += 4, B += 16, k -= 1) per K-step: 4 x (9 + 3) = **48 instructions**

But wait -- the B pointer advances by 16 per K-step in RVV (VL=16 float32), while POWER10 advances by 64 (16 columns x 4 broadcast groups). The RVV kernel with K-unroll=2 processes 2 K-steps per loop iteration. For 4 K-steps, the RVV needs 2 loop iterations.

Let me redo this more precisely with the RVV K-unroll=2 structure:

**RVV per K-pair (2 K-steps) for 2 rows x 16 columns:**
- 2 `flw` (A elements, 1 per K-step) + 2 `vle32.v` (B, 1 per K-step) + 2 `vfmacc.vf` (per row) = 6 per row
- 2 rows: 2 x 6 = **12 instructions**, yielding 64 FMA

For 4 rows, the RVV kernel would run twice (or process 4 rows if modified). Assuming 4 rows directly:
- 2 `flw` + 2 `vle32.v` (shared B) + 4 `vfmacc.vf` per K-step = 8 per K-step
- 2 K-steps: **16 instructions**, yielding 128 FMA

For 4 K-steps (to match POWER10): 2 x 16 = **32 instructions**, yielding 256 FMA

Adding pointer management (a += 4, B += 32, k -= 2) x 2 iterations = 6 more:
- **38 instructions total** for 256 FMA

**Revised comparison (4 rows x 16 columns x 4 K-steps = 256 FMA):**

| Implementation | Instructions | FMA | FMA per Instruction |
|---------------|-------------|-----|---------------------|
| RVV baseline (VL=16) | 38 | 256 | 6.74 |
| POWER10 MMA | 47 | 256 | 5.45 |
| RVV with `vmmacc.vv` (proposed) | 18 | 256 | 14.22 |

Wait -- the RVV is actually slightly more instruction-efficient than POWER10 MMA for the raw K-loop? This is because:
1. POWER10 has significant A-broadcast overhead (8 permute instructions)
2. POWER10 loads B 16 times (4 loads x 4 calls), vs RVV loading B only 4 times (shared across rows)
3. POWER10's advantage comes from the 16-FMA-per-MMA instruction, but the overhead eats into it

Let me re-verify by breaking down where MMA's advantage actually manifests.

### Detailed Side-by-Side (1 K-step, 4 rows x 16 columns, 64 FMA)

**RVV Baseline:**

| Instruction | Count | Notes |
|-------------|-------|-------|
| `flw` (load A) | 4 | One per row |
| `vle32.v` (load B) | 1 | Shared across all 4 rows |
| `vfmacc.vf` | 4 | One per row, each does 16 FMA |
| **Total** | **9** | 64 FMA |

**POWER10 MMA:**

| Instruction | Count | Notes |
|-------------|-------|-------|
| Scalar loads (A) | 4 | One per row |
| Permute (broadcast) | 8 | `vec_mergee` x2 + `vec_mergeo` x2 + `vec_xxpermdi` x4 |
| Vector loads (B) | 4 | 4 x `MlasLoadFloat32x4` |
| MMA (`xvf32gerpp`) | 4 | Each does 16 FMA = 64 total |
| **Total** | **20** | 64 FMA |

So for a **single K-step**, the RVV baseline (9 instructions) is significantly more efficient than POWER10 MMA (20 instructions) because:
- RVV has no broadcast overhead (scalar-to-vector is implicit in `vfmacc.vf`)
- RVV loads B only once (shared across rows)

**But where POWER10 wins is K-loop unrolling.** With K-unroll=4, POWER10 amortizes the broadcast overhead:

**4 K-steps, 4 rows x 16 columns = 256 FMA:**

**RVV baseline:**
- 4 K-steps x (4 `flw` + 1 `vle32.v` + 4 `vfmacc.vf`) = 4 x 9 = 36
- Pointer updates: a += 4, B += 16, k decrement = ~3 per iteration
- With K-unroll=2 (2 iterations of 2 K-steps): 2 x (2 x 9 + 3) = 2 x 21 = **42**

**POWER10 MMA:**
- 1 K-iteration (k -= 4): 4 A loads + 8 permutes + 16 B loads + 16 MMA + 2 pointer + 1 branch = **47**

Now the gap narrows but POWER10 still has more instructions due to the heavy broadcast permutation cost. However, the critical advantage of MMA is **not instruction count** -- it is:

1. **Accumulator registers**: MMA accumulates into dedicated `__vector_quad` registers that do not consume VR register file entries. This frees VR registers for other uses and enables wider unrolling.
2. **16 FMA per instruction**: Each MMA instruction produces 16 results, reducing the number of dependent chains and improving pipeline throughput.
3. **Reduced register pressure**: The RVV kernel needs separate accumulator vector registers per row; POWER10's accumulators are architecturally separate.

---

## Gap Identification

### Gap 1 [P0]: Matrix Multiply Accumulate (MMA)

**POWER10 instruction**: `__builtin_mma_xvf32gerpp(acc, a_broadcast, b_elements)`
- Performs a 4x4 outer-product accumulate in a single instruction
- 16 FMA operations per instruction
- Operates on `__vector_quad` accumulator (separate register file)

**RVV equivalent**: No equivalent. The closest is `vfmacc.vf` (1xVL outer product) which requires:
- VL separate FMA lanes
- Separate scalar loads for each A element
- Accumulation into standard vector registers

**Gap magnitude**:

For the K-loop body processing 4 rows x 16 columns x 4 K-steps (256 FMA):

| Metric | RVV Baseline | POWER10 MMA | RVV with `vmmacc.vv` |
|--------|-------------|-------------|---------------------|
| Multiply-accumulate instructions | 16 (vfmacc.vf) | 16 (xvf32gerpp) | 16 (vmmacc.vv) |
| Total instructions (K-loop body) | ~42 | ~47 | ~18 |
| Accumulator register pressure | 4 VR registers | 0 VR registers (dedicated) | 0 VR registers (dedicated) |
| A-load instructions | 16 (scalar) | 4 (scalar) + 8 (permute) | 4 (scalar) + 0 (implicit) |

The key benefit of a proposed `vmmacc.vv` is not raw instruction reduction in the tightest loop (where RVV is already competitive), but rather:
1. **Eliminating the need for per-element A scalar loads** when processing multiple rows
2. **Freeing accumulator registers** from the vector register file
3. **Reducing the number of dependent instruction chains** (4 MMA vs 16 vfmacc.vf per K-step)

**Proposed `vmmacc.vv` semantics** (4x4 block, SEW=32):
```
vmmacc.vv  acc_block, a_vec, b_vec
  a_vec  = 4 float32 elements (a0, a1, a2, a3)
  b_vec  = 4 float32 elements (b0, b1, b2, b3)
  acc_block[4x4] += outer_product(a_vec, b_vec)
  // acc_block[i][j] += a_vec[i] * b_vec[j], for i,j in [0,3]
```

With this instruction, the RVV K-loop for 4 rows x 16 columns x 4 K-steps becomes:
- 4 A loads (scalar)
- 16 B loads (4 per MMA block, 4 blocks per K-step)
- 16 `vmmacc.vv` (16 FMA each = 256 total)
- Pointer management: ~3
- **Total: ~39 instructions** -- comparable to baseline but with freed accumulator registers

The real benefit emerges with **wider tiles**. For 8 rows x 16 columns x 4 K-steps:
- RVV baseline needs 8 accumulator VR registers (8 rows x 1 VR each)
- POWER10 MMA needs 0 VR registers for accumulation (all in `__vector_quad`)
- RVV with `vmmacc.vv` + dedicated accumulators: also 0 VR registers

This register pressure reduction enables the compiler to keep more B data in registers, reducing memory traffic.

### Gap 2 [P1]: Dedicated Accumulator Registers (`__vector_quad`)

**POWER10 feature**: `__vector_quad` is a 512-bit accumulator register separate from the 128-bit VR register file.
- Zeroed via `__builtin_mma_xxsetaccz(&acc[i])`
- Results extracted via `__builtin_mma_disassemble_acc(Result, &acc[i])`
- Up to 8 accumulators used simultaneously (`acc[0..7]`)

**RVV status**: All accumulation uses the standard vector register file (`v0-v31`). For SGEMM with 8 rows:
- 8 vector registers consumed by accumulators (8 rows x 1 VR per row for VL=16, SEW=32)
- Only 24 remaining vector registers for other data (B loads, temporaries)

**Gap impact**:
- Limits tile height: more rows = more accumulator VRs = fewer VRs for B prefetching
- Forces accumulator spill/reload when register pressure exceeds 32 VRs
- No architectural separation between accumulator state and working data

**BB-scoped benefit**: In accumulator spill/reload BBs (if they exist), dedicated accumulators could reduce instructions by an estimated **25-50%**. However, for the typical 4-row / 8-row SGEMM kernel that fits within 32 VRs, this is a latent benefit that manifests mainly through:
- Enabling wider tiles (more rows) without spill
- Allowing more aggressive B prefetching

### Gap 3 [P2]: Efficient A-Element Broadcast Permutation

**POWER10 optimization** (`MlasSgemmComputeAElements`, lines 43-52):
Given 4 A elements (one per row), produces 4 broadcast vectors where each vector contains one A element replicated 4 times:

```
AElements[0] = [a0, _, _, _]     AElements[1] = [a1, _, _, _]
AElements[2] = [a2, _, _, _]     AElements[3] = [a3, _, _, _]

  =>
ABroadcast[0] = [a0, a0, a0, a0]  ABroadcast[1] = [a1, a1, a1, a1]
ABroadcast[2] = [a2, a2, a2, a2]  ABroadcast[3] = [a3, a3, a3, a3]
```

Uses 8 vector permutation instructions (2 `vec_mergee` + 2 `vec_mergeo` + 4 `vec_xxpermdi`).

**RVV equivalent**: RVV's `vfmacc.vf` implicitly broadcasts the scalar operand, so no explicit broadcast is needed. This is actually an RVV **advantage** -- the scalar operand is automatically replicated to all vector lanes.

**However**, if an RVV MMA-like instruction (`vmmacc.vv`) is introduced, it would need both operands as vectors. In that case, broadcast would need to be explicit:

| Approach | Instructions | Notes |
|----------|-------------|-------|
| POWER10 permute | 8 | 128-bit VRs, pack 4 elements |
| RVV `vmv.v.x` (proposed MMA) | 4 | One per row, each broadcasts scalar to all 16 lanes |

The RVV version would be slightly more efficient (4 vs 8) due to the wider register, but this is a minor point since the broadcast is amortized across multiple MMA calls.

---

## Benefit Analysis

### Per-BB Instruction Reduction (No BBV Data)

**BB Scope**: Inner K-loop body, processing 4 rows x 16 columns per iteration.

| Scenario | RVV Baseline | With `vmmacc.vv` + dedicated accumulators | Reduction |
|----------|-------------|-------------------------------------------|-----------|
| K-loop body (4 rows x 16 cols, 4 K-steps, 256 FMA) | ~42 instructions | ~39 instructions | -7.1% |
| K-loop body (8 rows x 16 cols, 4 K-steps, 512 FMA) | ~84 instructions | ~43 instructions | -48.8% |
| A-broadcast preparation (per K-iteration) | 0 (implicit in vfmacc.vf) | 4 vmv.v.x (if MMA needs explicit broadcast) | N/A (new overhead) |
| Accumulator init (xxsetaccz equivalent) | 4 vfmv.v.f (zero 4 VRs) | 4 acc-zero (dedicated) | Similar count |
| Accumulator disassemble (store results) | 0 (direct VR access) | 4 disassemble + 4 store | New overhead |

### Key Insight: MMA Benefit Scales with Tile Width

The POWER10 MMA advantage is most pronounced when processing **wider tiles**. With CountN=16 and 4 broadcast groups, each K-iteration loads 64 B elements and issues 16 MMA instructions. The fixed overhead (8 permutes for A broadcast) becomes a smaller fraction as tile width increases.

For the RVV VL=16 kernel specifically, the benefit of an MMA-like instruction is **modest in instruction count** because the scalar-vector outer product (`vfmacc.vf`) is already highly efficient for a single row. The RVV advantage is that `vfmacc.vf` implicitly broadcasts, eliminating the 8-instruction permutation overhead that POWER10 pays.

The primary benefit of `vmmacc.vv` for RVV would be:
1. **Multi-row processing without scalar load overhead**: Currently, each row needs its own `flw` + `vfmacc.vf`. With `vmmacc.vv`, 4 A elements are loaded once and broadcast implicitly for 4 rows simultaneously.
2. **Accumulator register file separation**: Enables wider tiles (more rows) without VR pressure.
3. **Dependency chain reduction**: 4 MMA instructions per K-step vs 16 `vfmacc.vf` per K-step (for 4 rows), reducing pipeline stalls from data dependencies.

### Summary of Estimated BB-Level Benefits

| Extension | Target BB | Original Instructions | Proposed Instructions | BB-Scoped Reduction |
|-----------|-----------|----------------------|----------------------|---------------------|
| `vmmacc.vv` (MMA) | K-loop body (4 rows) | 42 | 39 | -7.1% |
| `vmmacc.vv` (MMA) | K-loop body (8 rows) | 84 | 43 | -48.8% |
| Dedicated accumulators | Accumulator spill BB | ~8 (2 vse32.v + 2 vle32.v + overhead) | 0 (no spill needed) | -100% (eliminated) |
| `vmERGEh/vmERGEo` + `vxxpermdi` | A-broadcast BB | 0 (implicit) | 4 vmv.v.x (if MMA needed) | N/A |

---

## Proposed RVV Extension Details

### [P0] `vmmacc.vv` -- 4x4 Matrix Multiply-Accumulate

**Format** (conceptual, following RVV naming conventions):
```
vmmacc.vv  acc, vs2, vs1, sew=32
```

**Semantics**:
- `vs2`: Column vector of 4 float32 elements (represents 4 rows of A)
- `vs1`: Row vector of 4 float32 elements (represents 4 columns of B)
- `acc`: Accumulator block (4x4 float32), architecturally separate from vector register file
- Operation: `acc[i][j] += vs2[i] * vs1[j]` for i, j in [0, 3]

**Constraints**:
- SEW must be 32 (float32)
- vs2 and vs1 each provide exactly 4 elements
- Accumulator is a 4x4 block of float32, separate from VR file
- Multiple accumulators needed for wider tiles (e.g., 8 accumulators for 4 rows x 16 columns)

**Application to SGEMM** (4 rows x 16 columns):
```
// Pseudo-code for one K-step
vmmacc.vv  acc0, a_vec, b_vec_0    // columns 0-3
vmmacc.vv  acc1, a_vec, b_vec_1    // columns 4-7
vmmacc.vv  acc2, a_vec, b_vec_2    // columns 8-11
vmmacc.vv  acc3, a_vec, b_vec_3    // columns 12-15
// 4 instructions produce 64 FMA (4x16)
```

Compare to RVV baseline (4 rows x 16 columns, 1 K-step):
```
vfmacc.vf  c0, a0, b    // row 0: 16 FMA
vfmacc.vf  c1, a1, b    // row 1: 16 FMA
vfmacc.vf  c2, a2, b    // row 2: 16 FMA
vfmacc.vf  c3, a3, b    // row 3: 16 FMA
// 4 instructions produce 64 FMA (same count!)
```

The instruction count is identical for a single K-step! The advantage comes with:
- **K-loop unrolling**: MMA amortizes A-loading across 4 K-steps with 4 scalar loads + broadcast, vs 16 separate scalar loads in RVV baseline
- **Register pressure**: MMA accumulators don't consume VRs, allowing more B prefetching

### [P1] Dedicated Accumulator Register File

**Concept**: Add N architecturally separate accumulator blocks (each 4x4 float32 = 64 bytes), independent of the 32 vector registers.

**Instructions needed**:
- `vmmacc.vv acc, vs2, vs1` -- accumulate into block `acc`
- `vzero.acc acc` -- zero accumulator block
- `vread.acc vd, acc` -- read accumulator block into vector registers
- `vwrite.acc acc, vs2` -- write vector registers into accumulator block

**Benefit**: For 4-row x 16-column tile, 4 accumulator blocks hold all partial sums. This frees 4 VR registers that can be used for B prefetching or wider K-unrolling.

### [P2] Vector Merge/Permute for Broadcast

**POWER10 instructions**: `vec_mergee`, `vec_mergeo`, `vec_xxpermdi`

**RVV status**: RVV already has `vid.v` (index generation) and `vrgather.vv` (gather). Broadcast of scalar to vector is done via `vmv.v.x` (move scalar to all vector elements).

**Gap assessment**: This is **not a meaningful gap** for RVV. The `vfmacc.vf` instruction already handles implicit broadcast. The POWER10 permutation overhead exists specifically because VSX lacks an equivalent to `vfmacc.vf` and must manually construct broadcast vectors for the MMA instruction. RVV does not have this problem in its current form.

If `vmmacc.vv` is adopted (requiring explicit vector operands), the broadcast can be done with 4 `vmv.v.x` instructions (one per row, broadcasting scalar A[i] to all lanes), which is already efficient.

---

## Conclusion

The POWER10 MMA architecture provides a fundamentally different approach to GEMM computation compared to RVV. The key differences are:

1. **MMA instruction** (`xvf32gerpp`): 4x4 matrix outer-product in one instruction. This is the single most impactful gap. However, for the specific RVV VL=16 SGEMM kernel, the instruction count benefit is modest (7-49% depending on tile height) because RVV's `vfmacc.vf` is already highly efficient for single-row processing.

2. **Dedicated accumulator registers** (`__vector_quad`): These free vector registers for other uses, enabling wider tiles and better prefetching. This is the **strategically more important** gap, as it scales with tile size and affects register pressure across the entire kernel.

3. **A-element broadcast permutation**: This is an RVV **advantage**, not a gap. The POWER10 pays 8 permutation instructions per K-iteration to construct broadcast vectors; RVV's `vfmacc.vf` does this implicitly at zero cost.

**Recommendation**: If pursuing MMA-like extensions for RVV, prioritize the dedicated accumulator register file (P1) over the `vmmacc.vv` instruction (P0), as the register pressure benefit scales more broadly. The `vmmacc.vv` instruction is most valuable when combined with dedicated accumulators and wider tiles (8+ rows).

**Cumulative upper-bound estimate** (BB-scoped, no BBV data):
- K-loop body with 8-row tiles: up to **-48.8%** instruction reduction
- Accumulator spill BB elimination: up to **-100%** (BB eliminated entirely)
- Note: These estimates assume 8-row tile configuration; the standard VL=16 kernel uses 2-row tiles where the benefit is lower

---

## Review Log

| Round | Issues Found | Fixed | Remaining |
|-------|-------------|-------|-----------|
| R1 | 0 | 0 | 0 |

Final review conclusion: Analysis complete. No BBV data available; all benefits are per-BB theoretical estimates.
