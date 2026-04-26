---
name: fusion-scheme
description: Generate a fusion encoding scheme for a ranked candidate. Trigger when generating fusion instruction encoding proposals.
---

# Fusion Scheme Generation

## Purpose

Given a ranked fusion candidate from Feature 2 output, generate a complete fusion
encoding scheme document with encoding layout, instruction semantics, and constraint
compliance. The scheme is suitable for research documentation and Phase 3 simulation.

## Input

Provide these to the skill invocation:

1. **Candidate JSON**: A single entry from `output/fusion_candidates.json` containing:
   - `pattern`: instruction sequence with opcodes and chain_registers
   - `frequency`: BBV-weighted execution count
   - `score`: weighted score with breakdown
   - `hardware`: feasibility status and constraint notes

2. **ISA descriptor context**: The skill will read `tools/dfg/isadesc/rv64*.py` for
   encoding reference.

## Output

A Markdown document following the fixed structure below.

## Process (5 steps)

### Step 1: Parse the candidate

Extract:
- Instruction opcodes (e.g., `["fadd.s", "fmul.s"]`)
- Register class (`integer`, `float`, `vector`)
- RAW dependency chain (which output feeds which input)
- Score breakdown (frequency, tightness, hardware)

### Step 2: Identify encoding space

Scan the ISA descriptors to find an unused encoding slot:
- Check `tools/dfg/isadesc/rv64i.py` for integer opcodes
- Check `tools/dfg/isadesc/rv64f.py` for float opcodes
- Check `tools/dfg/isadesc/rv64v.py` for vector opcodes

Look for gaps in opcode + funct3 + funct7 combinations.

### Step 3: Draft the encoding

Assign:
- `opcode`: Choose from register-class-compatible regions
- `funct3`: Sub-operation variant
- `funct7`: Further variant (R-type instructions)
- `rd`, `rs1`, `rs2`: Map to pattern's source/destination roles

Ensure the encoding preserves original semantics (same computation).

### Step 4: Generate the Markdown scheme

Fill the template below with all sections.

### Step 5: Self-check

Include a "Validation Log" section documenting your encoding space search.
Note any near-conflicts or partial field usage.

## Revision Protocol

After generation, the user will run the validator CLI:

```
python -m tools.fusion validate --opcode N --funct3 M --funct7 K --reg-class X
```

If validation fails:
1. Read the `conflicts` and `suggested_alternatives` from validator output
2. Choose an alternative encoding slot from suggestions
3. Regenerate the scheme with new encoding
4. Repeat until pass (max 3 attempts)

If still failing after 3 attempts, output scheme with "UNVERIFIED" badge.

## Encoding Space Guidance

### Integer patterns

Candidate opcode regions:
- `0x0B`: custom-0 extension space (generally unused)
- `0x33`: OP (densely populated with funct3/funct7 combinations)
- `0x3B`: OP-32 (RV64-specific operations)

Look for unused funct3 values (0-7) on chosen opcode.

### Float patterns

Candidate opcode regions:
- `0x43`: MADD/QMADD (R4-type fused multiply-add)
- `0x47`: MSUB/QMSUB
- `0x4B`: NMSUB/QNMSUB
- `0x4F`: NMADD/QNMADD
- `0x53`: OP-FP (standard FP operations)

### Vector patterns

Candidate opcode regions:
- `0x57`: OP-V (V-extension operations)

V-extension has many funct3/funct7 combinations. Look for gaps in:
- funct3: OPIVI (0x0), OPIVV (0x1), OPIVX (0x2), OPFVF (0x4), OPFVV (0x5)
- funct7: varies per operation class

## Markdown Template

```markdown
# Fusion Scheme: <MNEMONIC>

## Candidate Summary
- Source pattern: <INSTRUCTION_SEQUENCE>
- Frequency: <COUNT>
- Score: <TOTAL> (freq=<FREQ>, tight=<TIGHT>, hw=<HW>)
- Register class: <CLASS>

## Encoding Layout

| Field   | Bits  | Value | Description         |
|---------|-------|-------|---------------------|
| opcode  | [6-7] | 0x??  | <OPCODE_REGION>     |
| funct3  | [3]   | 0x??  | <SUB_OP>            |
| funct7  | [7]   | 0x??  | <VARIANT>           |
| rd      | [5]   | dst   | Destination         |
| rs1     | [5]   | src1  | Source operand 1    |
| rs2     | [5]   | src2  | Source operand 2    |

### Encoding Justification
<WHY_THIS_ENCODING_SPACE_WAS_CHOSEN>

## Instruction Semantics

### Assembly Syntax
`<MNEMONIC> rd, rs1, rs2  # <DESCRIPTION>`

### Operation
rd = rs1 <OP> rs2  // <SEMANTIC_DESCRIPTION>

### Register Flow
- Reads: rs1, rs2
- Writes: rd
- Preserves: <ANY_IMPLICITLY_PRESERVED>

### Latency Estimate
- Original: <N> cycles (<BREAKDOWN>)
- Fused: <M> cycles (estimated)
- Saving: <N-M> cycles per occurrence

## Constraint Compliance

| Constraint           | Status | Notes                  |
|---------------------|--------|------------------------|
| Encoding space       | PASS/FAIL | <VALIDATOR_CHECK>   |
| Register class       | PASS   | <CLASS_MATCH>          |
| Operand count        | PASS   | Within 2src+1dst       |
| No config write      | PASS   | <FOR_V:_NO_VSETVLI>    |
| No load/store        | PASS   | Memory-free pattern    |

## Validation Log
- <TIMESTAMP> PASS/FAIL - <VALIDATOR_OUTPUT_OR_SELF_CHECK>
```

## Example Invocation

User provides candidate JSON excerpt:

```json
{
  "pattern": {
    "opcodes": ["fadd.s", "fmul.s"],
    "register_class": "float",
    "chain_registers": [["frd", "frs1"]]
  },
  "total_frequency": 1234567,
  "score": 0.8923,
  "hardware": {"status": "feasible"}
}
```

Skill generates scheme for `fadd.s` -> `fmul.s` fusion (FMADD.S-like pattern).

## Constraints

- **JSON-only validation feedback**: Validator CLI output is JSON; parse it precisely.
- **No speculative extensions**: Use only existing ISA descriptor context.
- **Prefer "pass" on uncertainty**: If encoding space is ambiguous, prefer available slots.