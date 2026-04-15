# Review Checklist

This document guides the reviewer subagent in checking the accuracy of the gap analysis report.

## Review Instructions

You are the accuracy reviewer for an RVV gap analysis report. Your job is to verify every
factual claim in the report, especially around instruction names, behaviors, and benefits.

## Step 1: Read the Report

Read the full report carefully. As you read, note every instruction name mentioned.

## Step 2: Verify RVV Claims

For each RVV instruction or capability mentioned in the report:

1. **Search the RVV intrinsic doc**:
   - Use Grep to search `third_party/riscv-rvv-intrinsic-doc/auto-generated/intrinsic_funcs.adoc`
     for the instruction or intrinsic name
   - Also search the category-specific files in `auto-generated/`:
     - `04_vector_floating-point_intrinsics.adoc` for float operations
     - `07_vector_permutation_intrinsics.adoc` for shuffle/gather operations
     - `00_vector_loads_and_stores_intrinsics.adoc` for load/store operations
     - `05_vector_reduction_operations.adoc` for reductions
   - If the report says "RVV has no equivalent of X", verify by searching for related terms

2. **Verify instruction behavior**:
   - Check that the described semantics match the intrinsic definition
   - Check operand types (`.vf` = scalar float, `.vv` = vector-vector, `.vx` = scalar integer)
   - Check that LMUL/EMUL constraints are correctly stated

3. **Common mistakes to watch for**:
   - Claiming RVV lacks a feature that exists under a different name
   - Incorrectly describing operand types (e.g., saying `.vf` when it should be `.vv`)
   - Ignoring mask registers in RVV (most RVV instructions have masked variants)
   - Not accounting for VL (vector length) flexibility in RVV

## Step 3: Verify Platform-Specific Claims

For each non-RVV platform instruction mentioned:

1. **Instruction name**: Is it a real instruction? Common traps:
   - ARM: `vmlaq_lane_f32` is real, but verify the exact syntax
   - x86: `vunpcklps` vs `vunpckldq` — make sure the correct variant is cited
   - Power: `xvf32gerpp` is POWER10-specific, not available on POWER9
   - LoongArch: `xvldrepl.w` is LASX (256-bit), `vldrepl.w` is LSX (128-bit)

2. **Register width**: Verify the register width stated for each platform
   - ARM NEON: 128-bit Q registers
   - x86 AVX: 256-bit YMM, AVX2 also YMM, AVX-512: 512-bit ZMM
   - Power VSX: 128-bit (but MMA uses 4×128 = 512-bit accumulator)
   - LoongArch LASX: 256-bit, LSX: 128-bit
   - S390X: 128-bit VR registers

## Step 4: Verify Benefit Calculations

For each claimed benefit:

1. **Instruction counts**: Re-count the instructions in each code snippet. Do the numbers
   in the text match the actual code?

2. **Register width normalization**: If the report claims X% improvement, verify:
   - Was the normalization factor correctly calculated?
   - Was it applied to both sides of the comparison?
   - Does the normalized workload actually represent equivalent work?

3. **BBV weighting**: If BBV data is used, verify:
   - The hotspot percentages are from actual data, not guessed
   - The weighting math is correct

4. **Benefit scope and clarity**: Check that every benefit figure in the summary table
   has a clearly defined scope. Flag any of these problems:
   - Vague multipliers like "1.3-2.0x" without stating the baseline and scope
   - Bare percentages like "-37.5%指令" without specifying BB scope or overall scope
   - "性能提升" claims without a clear reference point

   The summary table MUST use one of these formats:
   - With BBV data: "整体减少X%" (with the calculation chain shown)
   - Without BBV data: "BB内减少X%（BB名称）"

5. **Overall benefit chain verification**: For each proposed extension in the summary table,
   verify the arithmetic:
   ```
   整体收益 = BB内减少比例 × BB执行占比
   ```
   - Does the BB内减少比例 match the before/after instruction counts?
   - Does the BB执行占比 match the BBV data?
   - Is the final multiplication correct?

## Step 5: Verify Proposed Instructions

For each proposed new RVV instruction:

1. **Naming convention**: Does it follow RVV naming patterns?
   - Operand suffix: `.vv`, `.vf`, `.vx`, `.vi` (immediate)
   - The proposed name should be consistent with existing RVV instructions

2. **Encoding feasibility**: Does it fit in RVV's encoding space?
   - RVV uses 6-bit funct6 + 3-bit vm + 5-bit vs2 + 5-bit vs1 + 7-bit opcode
   - New instructions must not collide with existing encodings

3. **Semantic clarity**: Is the instruction definition unambiguous?
   - What happens at boundary conditions (VL not a multiple of 4)?
   - What about masked variants?
   - How does it interact with LMUL > 1?

## Issue Report Format

Report issues in this format:

```
## Review Round N

### Issues Found: [count]

#### Issue 1 [CRITICAL/MAJOR/MINOR]
- **Location**: Section X, paragraph Y, table row Z
- **Problem**: [what is wrong]
- **Expected**: [what it should be]
- **Evidence**: [reference to spec/source]

#### Issue 2 ...
```

### Severity Definitions

- **CRITICAL**: Factual error about an instruction's existence, behavior, or operand format.
  Examples: claiming RVV lacks a feature it has, wrong instruction semantics.
- **MAJOR**: Incorrect benefit calculation, wrong normalization, or misleading comparison.
  Examples: comparing unequal workloads, wrong register width.
- **MINOR**: Typo in instruction name, unclear wording, or missing context.
  Examples: `vunpklps` instead of `vunpcklps`.

### Pass Criteria

The report passes review when:
- Zero CRITICAL issues remain
- Zero MAJOR issues remain
- Minor issues are either fixed or documented as accepted limitations
- The reviewer is confident the report is factually accurate
