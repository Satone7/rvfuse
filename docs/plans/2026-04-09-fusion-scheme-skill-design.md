# Fusion Scheme Specification Agent Skill — Design Document

**Date**: 2026-04-09
**Feature**: Phase 2 Feature 3
**Status**: Approved
**Predecessor**: Feature 2 (Scoring & Constraints) — Completed

---

## Context

Phase 2 Feature 2 delivered a ranked fusion candidate list with hardware feasibility analysis.
Feature 3 takes a ranked candidate and generates a complete fusion encoding scheme —
encoding layout, instruction semantics, and verification constraints — ready for Phase 3 simulation.

**Key context from prior work:**
- V-extension is fully integrated in DFG layer (375+ instructions, `VECTOR_KIND` registers)
- V-extension constraints exist in `constraints.py` (config write, load/store detection)
- Agent infrastructure exists: `tools/fusion/agent.py` with Claude CLI integration
- ISA descriptors (`tools/dfg/isadesc/rv64*.py`) carry full encoding metadata

---

## Scope Adjustments

The original roadmap specified Feature 3 as a scalar-only scheme generator. Given:
1. V-extension was fully integrated during Phase 2 development
2. V-extension is high priority across the project
3. Feature 2 already handles V-pattern scoring through the same constraint pipeline

**Adjusted scope: Dual-track** — full fusion schemes for both scalar (I/F/M) and vector (V) patterns,
using the same scheme format. V-specific details (SEW/LMUL tuning) remain deferred to Phase 4.

---

## Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Architecture | Standalone agent skill (`fusion-scheme/SKILL.md`) | Clean separation from fusion CLI pipeline; interactive single-candidate focus |
| Output format | Markdown primary | Human-readable for research documentation; Phase 3 can add JSON export later |
| Batch mode | Interactive only (no batch CLI) | Skill invocation is per-candidate; user picks candidates interactively |
| Validation | Agent-validator loop (generate → validate → feedback → revise) | Deterministic encoding conflict checking; feedback-driven revision |
| Validator scope | Lightweight (~100-150 lines) | Checks opcode/funct conflicts + register class; reuses ISARegistry |

---

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `.claude/skills/fusion-scheme/SKILL.md` | Structured prompt for scheme generation + revision |
| `tools/fusion/scheme_validator.py` | Deterministic validator: encoding conflict checking |
| `tools/fusion/tests/test_scheme_validator.py` | Unit tests for validator |

**Minimal change to existing code:**
- Add `validate` subcommand to `tools/fusion/__main__.py` (thin CLI wrapper for validator)

### Data Flow

```text
User invokes skill with candidate info
       │
       ▼
SKILL.md prompt instructs Claude:
  1. Parse candidate (pattern + score + constraints)
  2. Scan ISA descriptors for encoding space
  3. Draft encoding (opcode + funct3 + funct7 + rd/rs1/rs2)
  4. Generate Markdown scheme
  5. Self-check (include validation log section)
       │
       ▼
User runs validator CLI:
  python -m tools.fusion validate --opcode N --funct3 M --funct7 K --reg-class X
       │
       ├─ PASS → Scheme accepted
       │
       └─ FAIL → User pastes feedback to Claude → Claude revises → Re-validate
```

---

## Scheme Output Format (Markdown)

Each generated scheme follows this fixed structure:

```markdown
# Fusion Scheme: <mnemonic>

## Candidate Summary
- Source pattern: [instruction sequence with RAW deps]
- Frequency: [BBV-weighted count]
- Score: [weighted score + breakdown]
- Register class: integer | float | vector

## Encoding Layout

| Field   | Bits  | Value | Description         |
|---------|-------|-------|---------------------|
| opcode  | [6-7] | 0x??  | Custom opcode space |
| funct3  | [3]   | 0x??  | Sub-operation       |
| funct7  | [7]   | 0x??  | Variant             |
| rd      | [5]   | dst   | Destination         |
| rs1     | [5]   | src1  | Source operand 1    |
| rs2     | [5]   | src2  | Source operand 2    |

### Encoding Justification
[Why this encoding space was chosen — unused opcode/funct combinations]

## Instruction Semantics

### Assembly Syntax
`<mnemonic> rd, rs1, rs2  # <description>`

### Operation
```
rd = rs1 <op> rs2  // Semantic description
```

### Register Flow
- Reads: rs1, rs2
- Writes: rd
- Preserves: [any implicitly preserved registers]

### Latency Estimate
- Original: [N] cycles (sum of constituent instructions)
- Fused: [M] cycles (estimated)
- Saving: [N-M] cycles per occurrence

## Constraint Compliance

| Constraint           | Status | Notes                  |
|---------------------|--------|------------------------|
| Encoding space       | ✅/❌  | [Conflict check result]|
| Register class       | ✅     | [Class compatibility]  |
| Operand count        | ✅     | [Within 2src+1dst]     |
| No config write      | ✅     | [For V: no vsetvli]   |
| No load/store        | ✅     | [Memory-free pattern]  |

## Validation Log
- [Timestamp] PASS/FAIL — [validator output]
```

**Same format for scalar and vector patterns.** Vector patterns use `v*` register names; the constraint checklist covers V-specific checks naturally.

---

## Skill Prompt Structure (SKILL.md)

### Frontmatter

```yaml
name: fusion-scheme
description: Generate a fusion encoding scheme for a ranked candidate. Trigger when generating fusion instruction encoding proposals.
```

### Body

```markdown
# Fusion Scheme Generation

## Input
- Candidate JSON from Feature 2 output: pattern + frequency + score + constraints
- RISC-V ISA descriptor context (read from tools/dfg/isadesc/)

## Output
- Markdown scheme document (fixed structure — see template)

## Process (5 steps)

1. **Parse the candidate**: Extract instruction sequence, register class, RAW dependencies, score breakdown

2. **Identify encoding space**: Scan the opcode/funct landscape in the ISA descriptors. Find an unused encoding slot compatible with the pattern's register class

3. **Draft the encoding**: Assign opcode, funct3, funct7 fields. Map rs1/rs2/rd to constituent instruction operands. Ensure encoding preserves original semantics

4. **Generate the scheme**: Fill the Markdown template with encoding layout, semantics, latency estimate, constraint checklist

5. **Self-check**: Verify that your encoding doesn't conflict with any existing RISC-V instruction. Include a "Validation Log" section with findings

## Revision Protocol

If validation fails, feedback will indicate:
- Which field conflicts (opcode X already used by Y)
- Suggested alternative encoding spaces

On revision: try an alternate encoding slot and regenerate the scheme.

Recommend max 3 revision attempts. After that, output scheme with "UNVERIFIED" badge.

## Encoding Space Guidance

For register class compatibility:
- Integer: opcodes 0x33, 0x3B, 0x0B (custom extension space)
- Float: opcodes 0x43, 0x47, 0x4B, 0x53 (existing FP space has gaps)
- Vector: opcode 0x57 (V-extension has many unused funct3/funct7 combinations)

Check `tools/dfg/isadesc/rv64*.py` for existing instruction encodings.
```

---

## Scheme Validator (`scheme_validator.py`)

### Interface

```python
@dataclass
class ValidationResult:
    passed: bool
    conflicts: list[str]           # e.g., "opcode 0x37 conflicts with LUI"
    warnings: list[str]            # e.g., "funct3=0x3 is partially used"
    suggested_alternatives: list[str]  # e.g., "opcode 0x0B funct3=0x2"

def validate_encoding(
    opcode: int,
    funct3: int | None,
    funct7: int | None,
    reg_class: str,               # "integer" | "float" | "vector"
) -> ValidationResult
```

### Check Logic

1. **Opcode conflict**: Query ISARegistry for any instruction with matching opcode + funct3 + funct7. Flag as conflict if found.

2. **Register class mismatch**: If proposed encoding uses an opcode region reserved for different register class (e.g., FP opcode for integer pattern), flag as conflict.

3. **Partial field usage**: If funct3 or funct7 alone would conflict (without full combination), emit warning (not failure).

### ISA Registry Integration

Reuses existing `ISARegistry` from `tools/dfg/isadesc/`:
- Load I, F, M, V extensions
- Query all registered instructions' `encoding` fields for conflict detection

### Unit Tests

| Test Case | Expected Result |
|-----------|-----------------|
| opcode 0x33 + funct3 0x0 (conflicts with ADD) | FAIL |
| Unused encoding combination | PASS |
| FP opcode for integer pattern | FAIL |
| opcode 0x57 + unused funct3/funct7 (V) | PASS |
| opcode 0x33 + funct3 0x1 but unique funct7 | Warning only |

---

## Agent-Validator Loop

### Interactive Flow (Manual)

1. Claude generates scheme → user reads Markdown output
2. User extracts encoding fields and runs validator CLI:
   ```
   python -m tools.fusion validate --opcode 0x0B --funct3 2 --funct7 0 --reg-class integer
   ```
3. If FAIL: user pastes validator output → Claude revises scheme
4. If PASS: scheme accepted

### CLI Addition

New `validate` subcommand in `tools/fusion/__main__.py`:

```
python -m tools.fusion validate --opcode N --funct3 M --funct7 K --reg-class X
```

Arguments:
- `--opcode`: Hex opcode value (required)
- `--funct3`: Hex funct3 value (optional, for R/I/S/B-type)
- `--funct7`: Hex funct7 value (optional, for R-type)
- `--reg-class`: Register class (`integer`, `float`, `vector`) — used for class-mismatch check
- `--isa`: ISA extensions to load (default: `I,F,M,V`)

Output: JSON `ValidationResult` printed to stdout.

---

## Out of Scope

- Actual hardware implementation or RTL
- Cycle-accurate simulation (Phase 3)
- Automated encoding allocation (agent proposes, human validates)
- SEW/LMUL-specific V-extension fields (Phase 4)
- Batch processing of multiple candidates

---

## Acceptance Criteria

1. For a top-ranked YOLO candidate (scalar or vector), the skill produces a Markdown scheme with valid encoding layout that passes validator check

2. Scheme Markdown is human-readable and suitable for research documentation

3. Validator correctly identifies encoding conflicts with existing RISC-V instructions (opcode/funct overlap)

4. Validator suggests alternative encoding spaces on failure

5. `validate` CLI subcommand integrates cleanly with existing `tools/fusion/__main__.py`

6. Unit test coverage >= 80% for `scheme_validator.py`

---

## File Summary

| File | Type | Lines (est.) |
|------|------|--------------|
| `.claude/skills/fusion-scheme/SKILL.md` | Skill prompt | ~80 |
| `tools/fusion/scheme_validator.py` | Python module | ~100-150 |
| `tools/fusion/tests/test_scheme_validator.py` | Unit tests | ~120 |
| `tools/fusion/__main__.py` (modify) | CLI addition | ~20 |

**Total new code: ~300-350 lines** (excluding skill prompt)

---

## Dependencies

- `tools/dfg/isadesc/` (ISA descriptors with encoding metadata)
- `tools/dfg/instruction.py` (ISARegistry, InstructionFormat)
- Claude CLI (existing agent infrastructure pattern)

---

## Risks

| Risk | Mitigation |
|------|------------|
| Agent proposes invalid encoding | Validator catches before user acceptance |
| Encoding space exhaustion for dense ISAs | Validator suggests alternatives; multiple revision attempts |
| V-extension encoding complexity | Start with simple V patterns; defer SEW/LMUL to Phase 4 |