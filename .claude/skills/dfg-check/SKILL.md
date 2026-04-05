---
name: dfg-check
description: "Verify the correctness of a Data Flow Graph (DFG) for a RISC-V basic block. Use this skill when you receive a basic block disassembly along with a DFG JSON and need to check whether the RAW dependency edges are correct. Always use this skill when the prompt mentions 'dfg-check', 'verify DFG', 'check data flow', or 'validate DFG edges'."
---

# DFG Check: Verify Data Flow Graph Correctness

## What This Skill Does

You are given a RISC-V basic block (assembly listing) and a DFG (Data Flow Graph) represented as JSON. Your job is to verify that every RAW (Read-After-Write) dependency edge in the DFG is correct, and that no edges are missing.

**You must respond with ONLY a JSON object.** No markdown fences, no explanation, no preamble.

## Input Format

The user provides:

1. A basic block in this text format:
```
BB 5 (vaddr: 0x1117c, 6 insns):
  0x1117c: lw a0,-28(s0)
  0x11180: srliw a1,a0,31
  0x11184: addw a1,a1,a0
  0x11186: andi a1,a1,-2
  0x11188: subw a0,a0,a1
  0x1118a: bnez a0,20
```

2. A DFG in JSON format:
```json
{
  "bb_id": 5,
  "vaddr": "0x1117c",
  "source": "script",
  "nodes": [
    {"index": 0, "address": "0x1117c", "mnemonic": "lw", "operands": "a0,-28(s0)"},
    {"index": 1, "address": "0x11180", "mnemonic": "srliw", "operands": "a1,a0,31"},
    ...
  ],
  "edges": [
    {"src": 0, "dst": 1, "register": "a0"},
    ...
  ]
}
```

## How to Verify

### Step 1: Parse the Basic Block

Extract each instruction's mnemonic and operands. Build your own mental model of which registers each instruction reads (source) and writes (destination).

### Step 2: Build the Expected DFG

Walk instructions in order. Track `last_writer[reg] = instruction_index`. For each instruction, for each source register, there should be an edge from `last_writer[reg]` to the current instruction. After processing sources, update `last_writer` for destination registers.

**RISC-V register flow rules** (RV64I):

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
| Pseudo: j, ret, nop, ecall, ebreak, fence | - | - |
| Pseudo: mv | rd | rs1 |
| Pseudo: li | rd | - |
| Pseudo: la, auipc, lui | rd | - |
| Pseudo: not, neg, negw, seqz, snez, sltz, sgtz | rd | rs1 |

**Key rules:**
- `zero` / `x0` is a constant: it is never truly written, so no edge should come from a write to zero
- Pseudo-instructions like `j`, `ret`, `nop` have no register effects
- `mv rd, rs1` copies: dst=rd, src=rs1
- `li rd, imm` loads immediate: dst=rd, src=none (zero is a constant, not a dependency)
- For stores `sw rs2, offset(rs1)`: rs2 is the data source, rs1 is the base address source
- For loads `lw rd, offset(rs1)`: rd is the destination, rs1 is the base address source
- `jal rd, offset` writes rd (return address), reads nothing
- `jalr rd, rs1, offset` writes rd, reads rs1

### Step 3: Compare Against Provided DFG

For each edge in the provided DFG, verify:
- The source instruction actually writes the register named in the edge
- The destination instruction actually reads that register
- There is no closer writer of the same register between them (only the latest writer should have the edge)

Check for missing edges: are there any register dependencies that exist but are not represented as edges?

### Step 4: Output Result

**Respond with ONLY this JSON object:**

```json
{"verdict": "pass", "issues": []}
```

Or if issues found:

```json
{"verdict": "fail", "issues": [{"type": "extra_edge", "msg": "Edge 0->2 on register 's0': insn 0 (addi sp,sp,-32) does not write s0"}, {"type": "missing_edge", "msg": "Missing edge 1->3 on register 'a1': insn 1 writes a1, insn 3 reads a1 with no closer writer"}]}
```

**Issue types:**
- `extra_edge`: An edge exists in the DFG but the source doesn't write the register, or the destination doesn't read it, or there's a closer writer in between
- `missing_edge`: A RAW dependency exists (instruction X writes reg, instruction Y reads it with no closer writer) but no edge represents it
- `wrong_register`: The edge exists for the right instructions but names the wrong register

## Important Constraints

- Do NOT modify any files. You are read-only.
- Do NOT output anything other than the JSON result.
- If you are unsure, prefer "pass" — the script's result is authoritative, this check is advisory.
- If the DFG is for an unsupported ISA extension (e.g., V-vector instructions), you can still verify the edges based on general RISC-V conventions, but be more lenient.
