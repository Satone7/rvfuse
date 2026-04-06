---
name: dfg-generate
description: "Generate a Data Flow Graph (DFG) for a RISC-V basic block, especially when the block contains instructions from ISA extensions not yet supported by the script parser (e.g., V-vector, F/D-float, M-multiply). Use this skill when the prompt mentions 'dfg-generate', 'generate DFG', 'build data flow graph', or provides a basic block that needs DFG construction."
---

# DFG Generate: Build a Data Flow Graph for a Basic Block

## What This Skill Does

You are given a RISC-V basic block (assembly listing). Your job is to generate a complete DFG (Data Flow Graph) by identifying RAW (Read-After-Write) dependencies between instructions, then output the result as JSON.

**You must respond with ONLY a JSON object.** No markdown fences, no explanation, no preamble.

## Input Format

The user provides a basic block in this text format:
```
BB 7 (vaddr: 0x12000, 4 insns):
  0x12000: vsetvli t0,a0,e8,m1
  0x12004: vle8.v v0,(a1)
  0x12008: vadd.vv v1,v0,v0
  0x1200c: vse8.v v1,(a2)
```

## How to Generate the DFG

### Step 1: Parse the Basic Block

Extract each instruction's mnemonic and operands. For every instruction, determine:
- **Destination registers**: which registers this instruction writes to
- **Source registers**: which registers this instruction reads from

### Step 2: Build RAW Dependencies

Walk instructions in order (index 0, 1, 2, ...). Maintain a map `last_writer[reg] = instruction_index`.

For each instruction at index `i`:
1. For each **source register** `r`:
   - If `r` is in `last_writer` and `last_writer[r] != i`, create an edge: `{src: last_writer[r], dst: i, register: r}`
2. For each **destination register** `r`:
   - Update `last_writer[r] = i`

**Skip `zero` / `x0`**: this constant register is never truly written, so never add it to `last_writer`.

### Step 3: Determine Register Flow for Each Instruction

Use these rules for known RV64I/RV64GCV instructions. For unfamiliar instructions, apply the general RISC-V convention described below.

#### RV64I (base integer)

| Category | Instructions | dst | src |
|----------|-------------|-----|-----|
| R-type ALU | add, sub, and, or, xor, sll, srl, sra, slt, sltu | rd | rs1, rs2 |
| I-type ALU | addi, andi, ori, xori, slti, sltiu | rd | rs1 |
| I-type shift | slli, srli, srai | rd | rs1 |
| W R-type | addw, subw, sllw, srlw, sraw | rd | rs1, rs2 |
| W I-type | addiw, slliw, srliw, sraiw | rd | rs1 |
| Loads | lb, lbu, lh, lhu, lw, lwu, ld | rd | rs1 (base) |
| Stores | sb, sh, sw, sd | - | rs2 (data), rs1 (base) |
| 2-reg branch | beq, bne, blt, bge, bltu, bgeu | - | rs1, rs2 |
| 1-reg branch | beqz, bnez, bgtz, blez | - | rs1 |
| Pseudo branch | bgt, ble, bgtu, bleu | - | rs1, rs2 |
| JAL | jal | rd | - |
| JALR | jalr | rd | rs1 |
| Pseudo | j, ret, nop, ecall, ebreak, fence, fence.i | - | - |
| Pseudo | mv | rd | rs1 |
| Pseudo | li | rd | - |
| Pseudo | la, auipc, lui | rd | - |
| Pseudo | not, neg, negw, seqz, snez, sltz, sgtz | rd | rs1 |

#### RV64M (multiply/divide)

| Instructions | dst | src |
|-------------|-----|-----|
| mul, mulh, mulhsu, mulhu | rd | rs1, rs2 |
| mulw | rd | rs1, rs2 |
| div, divu, rem, remu | rd | rs1, rs2 |
| divw, divuw, remw, remuw | rd | rs1, rs2 |

#### RV64F (single-precision float)

| Instructions | dst | src |
|-------------|-----|-----|
| fadd.s, fsub.s, fmul.s, fdiv.s, fmin.s, fmax.s | rd (float) | rs1, rs2 (float) |
| fadd.d, fsub.d, fmul.d, fdiv.d, fmin.d, fmax.d | rd (float) | rs1, rs2 (float) |
| feq.s, flt.s, fle.s, feq.d, flt.d, fle.d | rd (int) | rs1, rs2 (float) |
| fsgnj.s, fsgnjn.s, fsgnjx.s (+ .d variants) | rd (float) | rs1, rs2 (float) |
| fsqrt.s, fsqrt.d | rd (float) | rs1 (float) |
| fmv.s, fmv.d (integer variants: fmv.x.s, fmv.x.d, fmv.s.x, fmv.d.x) | rd | rs1 |
| fcvt.s.d, fcvt.d.s, fcvt.w.s, fcvt.s.w, etc. | rd | rs1 |
| flw, fld | rd (float) | rs1 (base) |
| fsw, fsd | - | rs2 (float data), rs1 (base) |
| fmadd.s, fmsub.s, fnmsub.s, fnmadd.s (+ .d variants) | rd (float) | rs1, rs2, rs3 (float) |

Float register names: f0-f31, or ABI: ft0-ft11, fa0-fa7, fs0-fs11. Treat float registers independently from integer registers.

#### RV64A (atomic)

| Instructions | dst | src |
|-------------|-----|-----|
| lr.w, lr.d | rd | rs1 (base) |
| sc.w, sc.d | rd | rs1 (base), rs2 (data) |
| amo*.w, amo*.d | rd | rs1 (base), rs2 (data) |

#### RV64V (vector)

| Instructions | dst | src |
|-------------|-----|-----|
| vsetvli, vsetvl | rd | rs1 |
| vadd.vv, vsub.vv, vand.vv, vor.vv, vxor.vv, vslideup.vi | vd | vs1, vs2 |
| vadd.vx, vsub.vx, vrsub.vx, vslideup.vx | vd | rs1, vs2 |
| vadd.vi | vd | vs2 |
| vmul.vv, vmul.vx, vdiv.vv, vdiv.vx | vd | vs1/rs1, vs2 |
| vmerge.vim, vmerge.vvm, vmerge.vxm | vd | vs1/rs1, vs2 |
| vle8.v, vle16.v, vle32.v, vle64.v (vector loads) | vd | rs1 (base) |
| vse8.v, vse16.v, vse32.v, vse64.v (vector stores) | - | vs2 (data), rs1 (base) |
| vlse8.v, vlse16.v, etc. (strided loads) | vd | rs1 (base), rs2 (stride) |
| vsse8.v, vsse16.v, etc. (strided stores) | - | vs2 (data), rs1 (base), rs2 (stride) |
| vredsum.vs, vredmaxu.vs, etc. (reductions) | vd | vs2, vs1 |
| vmv.v.x | vd | rs1 |
| vmv.v.i | vd | - |
| vmv.x.s, vmv.s.x (scalar move) | rd / vd | vs1 / rs1 |
| vlm.v, vsm.v (mask load/store) | vd | rs1 (base) |

Vector register names: v0-v31. Treat them independently from integer and float registers.

### Step 4: Output Result

**Respond with ONLY this JSON object:**

```json
{
  "bb_id": 7,
  "vaddr": "0x12000",
  "nodes": [
    {"index": 0, "address": "0x12000", "mnemonic": "vsetvli", "operands": "t0,a0,e8,m1"},
    {"index": 1, "address": "0x12004", "mnemonic": "vle8.v", "operands": "v0,(a1)"},
    {"index": 2, "address": "0x12008", "mnemonic": "vadd.vv", "operands": "v1,v0,v0"},
    {"index": 3, "address": "0x1200c", "mnemonic": "vse8.v", "operands": "v1,(a2)"}
  ],
  "edges": [
    {"src": 0, "dst": 1, "register": "a0"},
    {"src": 0, "dst": 1, "register": "a1"},
    {"src": 1, "dst": 2, "register": "v0"},
    {"src": 1, "dst": 2, "register": "v0"},
    {"src": 2, "dst": 3, "register": "v1"},
    {"src": 0, "dst": 3, "register": "a2"}
  ]
}
```

**Requirements:**
- `bb_id` and `vaddr` must match the input header
- `address` values must match the input exactly (use the same hex format)
- `mnemonic` and `operands` must match the input exactly
- `index` is the 0-based position in the instruction list
- `edges` use integer indices for `src` and `dst`, and register name strings for `register`
- Duplicate register dependencies across edges are acceptable (e.g., v0 used twice in `vadd.vv v1,v0,v0`)

## Important Constraints

- Do NOT modify any files. You are read-only.
- Do NOT output anything other than the JSON result.
- Address values must be preserved exactly as provided in the input.
- If you are unsure about an instruction's register flow, make your best judgment based on RISC-V conventions and document the edge. Partial results are better than no results.
- Only output RAW (Read-After-Write) dependencies. Do not output WAR, WAW, or control dependencies.
