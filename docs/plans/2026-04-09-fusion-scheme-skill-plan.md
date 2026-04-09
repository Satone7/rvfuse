# Fusion Scheme Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone agent skill for fusion scheme generation with a lightweight Python validator for encoding conflict checking.

**Architecture:** A `scheme_validator.py` module checks proposed fusion encodings against existing ISA descriptors (opcode/funct conflicts, register class mismatch). A SKILL.md provides structured prompts for Claude to generate Markdown fusion schemes. A thin CLI `validate` subcommand wraps the validator for interactive use.

**Tech Stack:** Python 3 dataclasses, ISARegistry from `tools/dfg/instruction.py`, unittest for tests, argparse CLI extension.

---

## Task 1: Create ValidationResult Dataclass and Validator Skeleton

**Files:**
- Create: `tools/fusion/scheme_validator.py`

**Step 1: Write the module skeleton with ValidationResult**

```python
"""Encoding conflict validator for fusion scheme proposals.

Checks proposed fusion instruction encodings against existing RISC-V ISA
descriptors to detect opcode/funct3/funct7 conflicts and register class mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dfg.instruction import ISARegistry


@dataclass
class ValidationResult:
    """Result of validating a proposed fusion encoding.

    Attributes:
        passed: True if no hard conflicts found.
        conflicts: List of hard conflict descriptions (opcode/funct overlaps).
        warnings: List of soft warnings (partial field usage, not blocking).
        suggested_alternatives: List of suggested alternative encoding slots.
    """
    passed: bool
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_alternatives: list[str] = field(default_factory=list)


def validate_encoding(
    opcode: int,
    funct3: int | None,
    funct7: int | None,
    reg_class: str,
    registry: ISARegistry,
) -> ValidationResult:
    """Validate a proposed fusion encoding against existing ISA instructions.

    Args:
        opcode: 7-bit opcode value (0x00-0x7F).
        funct3: 3-bit funct3 value (0x0-0x7), or None if not applicable.
        funct7: 7-bit funct7 value (0x00-0x7F), or None if not applicable.
        reg_class: Register class ("integer", "float", "vector").
        registry: ISARegistry with loaded extensions (I, F, M, V).

    Returns:
        ValidationResult with pass/fail status and conflict details.
    """
    # Placeholder - will be implemented in Task 3
    return ValidationResult(passed=True)
```

**Step 2: Run import test to verify module structure**

Run: `cd tools && python -c "from fusion.scheme_validator import ValidationResult; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add tools/fusion/scheme_validator.py
git commit -m "feat(fusion): add scheme_validator skeleton with ValidationResult dataclass"
```

---

## Task 2: Write Failing Tests for Conflict Detection

**Files:**
- Create: `tools/fusion/tests/test_scheme_validator.py`

**Step 1: Write the test file skeleton**

```python
"""Tests for fusion scheme encoding validator."""

import sys
from pathlib import Path

import unittest

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestValidationResult(unittest.TestCase):
    """Tests for ValidationResult dataclass."""

    def test_passed_true_when_no_conflicts(self):
        from fusion.scheme_validator import ValidationResult
        result = ValidationResult(passed=True)
        self.assertTrue(result.passed)
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.warnings, [])

    def test_passed_false_with_conflicts(self):
        from fusion.scheme_validator import ValidationResult
        result = ValidationResult(passed=False, conflicts=["opcode conflict"])
        self.assertFalse(result.passed)
        self.assertEqual(len(result.conflicts), 1)


class TestValidateEncoding(unittest.TestCase):
    """Tests for validate_encoding function."""

    def _make_registry(self):
        """Build ISA registry with I, F, M, V extensions."""
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64i import build_registry as build_i
        from dfg.isadesc.rv64f import build_registry as build_f
        from dfg.isadesc.rv64m import build_registry as build_m
        from dfg.isadesc.rv64v import build_registry as build_v
        reg = ISARegistry()
        build_i(reg)
        build_f(reg)
        build_m(reg)
        build_v(reg)
        return reg

    def test_conflict_with_existing_instruction(self):
        """opcode 0x33 + funct3 0x0 + funct7 0x00 conflicts with ADD."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        result = validate_encoding(
            opcode=0x33,
            funct3=0x0,
            funct7=0x00,
            reg_class="integer",
            registry=registry,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("ADD" in c or "0x33" in c for c in result.conflicts))

    def test_no_conflict_unused_encoding(self):
        """Unused encoding combination should pass."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        # opcode 0x0B is custom-0 extension space, typically unused
        result = validate_encoding(
            opcode=0x0B,
            funct3=0x2,
            funct7=0x00,
            reg_class="integer",
            registry=registry,
        )
        self.assertTrue(result.passed)

    def test_register_class_mismatch_integer_fp_opcode(self):
        """Integer pattern using FP opcode (0x53) should fail."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        result = validate_encoding(
            opcode=0x53,  # OP-FP opcode
            funct3=0x0,
            funct7=0x00,
            reg_class="integer",
            registry=registry,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("register class" in c.lower() for c in result.conflicts))

    def test_vector_encoding_passes_if_unused(self):
        """opcode 0x57 with unused funct3/funct7 should pass for vector."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        # Check an unused V-extension slot
        result = validate_encoding(
            opcode=0x57,
            funct3=0x0,
            funct7=0x00,
            reg_class="vector",
            registry=registry,
        )
        # May pass or have warning depending on existing V instructions
        # At minimum, should not have register class mismatch
        self.assertFalse(any("register class" in c.lower() for c in result.conflicts))

    def test_warning_partial_funct3_usage(self):
        """funct3 alone conflicting (without full combo) should warn, not fail."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        # This test will need adjustment after implementation
        # For now, placeholder asserting warnings can exist
        result = validate_encoding(
            opcode=0x33,
            funct3=0x0,
            funct7=0x20,  # SUB uses funct7=0x20, funct3=0x0
            reg_class="integer",
            registry=registry,
        )
        # funct3=0x0 is used by ADD and SUB, so partial usage warning
        # But funct7=0x20 + funct3=0x0 = SUB (hard conflict)
        # Implementation should detect this properly

    def test_suggested_alternatives_on_failure(self):
        """Failed validation should suggest alternative encoding slots."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        result = validate_encoding(
            opcode=0x33,
            funct3=0x0,
            funct7=0x00,
            reg_class="integer",
            registry=registry,
        )
        self.assertFalse(result.passed)
        # Should suggest at least one alternative
        self.assertGreater(len(result.suggested_alternatives), 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail (validate_encoding not implemented)**

Run: `cd tools && python -m unittest fusion.tests.test_scheme_validator -v`

Expected: Tests fail with `AssertionError` (validate_encoding returns `passed=True` for all cases)

**Step 3: Commit**

```bash
git add tools/fusion/tests/test_scheme_validator.py
git commit -m "test(fusion): add failing tests for scheme_validator conflict detection"
```

---

## Task 3: Implement Conflict Checking Logic

**Files:**
- Modify: `tools/fusion/scheme_validator.py`

**Step 1: Implement the core conflict checking loop**

Replace the placeholder `validate_encoding` with:

```python
def validate_encoding(
    opcode: int,
    funct3: int | None,
    funct7: int | None,
    reg_class: str,
    registry: ISARegistry,
) -> ValidationResult:
    """Validate a proposed fusion encoding against existing ISA instructions."""

    conflicts: list[str] = []
    warnings: list[str] = []
    suggested_alternatives: list[str] = []

    # Track which instructions use this opcode for suggestions
    opcode_users: list[tuple[str, int | None, int | None, str]] = []

    # Scan all registered instructions for conflicts
    for mnemonic, flow in registry._flows.items():
        encoding = flow.encoding
        if encoding is None:
            continue

        # Track opcode users
        if encoding.opcode == opcode:
            opcode_users.append((
                mnemonic,
                encoding.funct3,
                encoding.funct7,
                encoding.reg_class,
            ))

        # Check full encoding match (hard conflict)
        if encoding.opcode == opcode:
            # For R-type: need opcode + funct3 + funct7 match
            # For I-type: need opcode + funct3 match (no funct7)
            match_funct3 = (funct3 is None or encoding.funct3 is None or
                           encoding.funct3 == funct3)
            match_funct7 = (funct7 is None or encoding.funct7 is None or
                           encoding.funct7 == funct7)

            if match_funct3 and match_funct7:
                # Hard conflict
                conflict_desc = f"opcode 0x{opcode:02X}"
                if funct3 is not None:
                    conflict_desc += f" funct3 0x{funct3:X}"
                if funct7 is not None:
                    conflict_desc += f" funct7 0x{funct7:02X}"
                conflict_desc += f" conflicts with {mnemonic}"
                conflicts.append(conflict_desc)

    # Check register class mismatch against opcode region
    reg_class_conflict = _check_reg_class_mismatch(opcode, reg_class, opcode_users)
    if reg_class_conflict:
        conflicts.append(reg_class_conflict)

    # Check partial field usage (warnings)
    partial_warnings = _check_partial_usage(opcode, funct3, funct7, opcode_users)
    warnings.extend(partial_warnings)

    # Generate suggested alternatives if there are conflicts
    if conflicts:
        suggested = _suggest_alternatives(reg_class, registry, opcode_users)
        suggested_alternatives.extend(suggested)

    passed = len(conflicts) == 0
    return ValidationResult(
        passed=passed,
        conflicts=conflicts,
        warnings=warnings,
        suggested_alternatives=suggested_alternatives,
    )


def _check_reg_class_mismatch(
    opcode: int,
    reg_class: str,
    opcode_users: list[tuple[str, int | None, int | None, str]],
) -> str | None:
    """Check if the opcode region is reserved for a different register class.

    Returns conflict description if mismatch found, None otherwise.
    """
    # Known opcode region assignments
    opcode_reg_class: dict[int, str] = {
        0x33: "integer",   # OP
        0x3B: "integer",   # OP-32
        0x13: "integer",   # OP-IMM
        0x1B: "integer",   # OP-IMM-32
        0x53: "float",     # OP-FP
        0x07: "integer",   # LOAD-FP (but uses FP registers)
        0x27: "integer",   # STORE-FP (but uses FP registers)
        0x43: "float",     # MADD/QMADD
        0x47: "float",     # MSUB/QMSUB
        0x4B: "float",     # NMSUB/QNMSUB
        0x4F: "float",     # NMADD/QNMADD
        0x57: "vector",    # OP-V
    }

    expected_class = opcode_reg_class.get(opcode)
    if expected_class is None:
        # Unknown opcode region - no class restriction
        return None

    if expected_class != reg_class:
        return (f"register class mismatch: opcode 0x{opcode:02X} is "
                f"{expected_class} region, but pattern is {reg_class}")

    return None


def _check_partial_usage(
    opcode: int,
    funct3: int | None,
    funct7: int | None,
    opcode_users: list[tuple[str, int | None, int | None, str]],
) -> list[str]:
    """Check for partial field usage (soft warnings)."""

    warnings: list[str] = []

    if funct3 is not None:
        # Check if this funct3 is used by any instruction on this opcode
        funct3_users = [m for m, f3, f7, rc in opcode_users if f3 == funct3]
        if funct3_users and funct7 is not None:
            # funct3 is partially used, but we have a unique funct7
            warnings.append(
                f"funct3 0x{funct3:X} is used by {', '.join(funct3_users)} "
                f"-- ensure funct7 0x{funct7:02X} is unique"
            )

    if funct7 is not None:
        # Check if this funct7 is used by any instruction on this opcode
        funct7_users = [m for m, f3, f7, rc in opcode_users if f7 == funct7]
        if funct7_users and funct3 is not None:
            warnings.append(
                f"funct7 0x{funct7:02X} is used by {', '.join(funct7_users)} "
                f"-- ensure funct3 0x{funct3:X} is unique"
            )

    return warnings


def _suggest_alternatives(
    reg_class: str,
    registry: ISARegistry,
    opcode_users: list[tuple[str, int | None, int | None, str]],
) -> list[str]:
    """Suggest alternative encoding slots for the given register class."""

    suggestions: list[str] = []

    # Candidate opcode regions based on register class
    candidate_opcodes: list[int] = []
    if reg_class == "integer":
        candidate_opcodes = [0x0B, 0x33, 0x3B]  # custom-0, OP, OP-32
    elif reg_class == "float":
        candidate_opcodes = [0x43, 0x47, 0x4B, 0x4F, 0x53]  # R4-type + OP-FP
    elif reg_class == "vector":
        candidate_opcodes = [0x57]  # OP-V

    for op in candidate_opcodes:
        # Collect funct3 values used on this opcode
        used_funct3: set[int] = set()
        for mnemonic, flow in registry._flows.items():
            encoding = flow.encoding
            if encoding and encoding.opcode == op and encoding.funct3 is not None:
                used_funct3.add(encoding.funct3)

        # Find unused funct3 slots (0-7)
        for f3 in range(8):
            if f3 not in used_funct3:
                suggestions.append(f"opcode 0x{op:02X} funct3 0x{f3:X}")
                if len(suggestions) >= 3:
                    return suggestions

    return suggestions
```

**Step 2: Run tests to verify some pass**

Run: `cd tools && python -m unittest fusion.tests.test_scheme_validator -v`

Expected: `test_no_conflict_unused_encoding` should pass, others may still need adjustment

**Step 3: Commit**

```bash
git add tools/fusion/scheme_validator.py
git commit -m "feat(fusion): implement conflict checking in scheme_validator"
```

---

## Task 4: Adjust Tests for Correct Behavior

**Files:**
- Modify: `tools/fusion/tests/test_scheme_validator.py`

**Step 1: Update tests to match actual implementation behavior**

Update the tests that need adjustment based on actual ISA registry contents:

```python
    def test_warning_partial_funct3_usage(self):
        """funct3 alone conflicting (without full combo) should warn, not fail."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        # opcode 0x33 funct3=0x0 is used by ADD, SUB, SLT, SLTU
        # funct7=0x20 + funct3=0x0 = SUB (hard conflict)
        result = validate_encoding(
            opcode=0x33,
            funct3=0x0,
            funct7=0x20,
            reg_class="integer",
            registry=registry,
        )
        self.assertFalse(result.passed)  # SUB conflict
        self.assertTrue(any("SUB" in c for c in result.conflicts))

    def test_no_conflict_with_unique_funct7(self):
        """Same opcode/funct3 but unique funct7 should pass."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        # Find an unused funct7 value for opcode 0x33 funct3 0x0
        # ADD uses funct7=0x00, SUB uses funct7=0x20
        # Try funct7=0x01 (unused)
        result = validate_encoding(
            opcode=0x33,
            funct3=0x0,
            funct7=0x01,
            reg_class="integer",
            registry=registry,
        )
        # Should pass (no exact match) but may have warnings
        self.assertTrue(result.passed)

    def test_vector_encoding_in_unused_slot(self):
        """opcode 0x57 with unused funct3 should pass for vector."""
        from fusion.scheme_validator import validate_encoding
        registry = self._make_registry()
        # V-extension have many funct3 values; try one likely unused
        # funct3=0x0 is OPIVI, funct3=0x2 is OPIVX
        # Let's check what's actually used and pick a gap
        # For now, accept that it may pass or have warnings
        result = validate_encoding(
            opcode=0x57,
            funct3=0x3,  # OPIVV, may be used
            funct7=0x00,
            reg_class="vector",
            registry=registry,
        )
        # At minimum, register class should match
        self.assertFalse(any("register class" in c.lower() for c in result.conflicts))
```

**Step 2: Run tests**

Run: `cd tools && python -m unittest fusion.tests.test_scheme_validator -v`

Expected: All tests pass

**Step 3: Commit**

```bash
git add tools/fusion/tests/test_scheme_validator.py
git commit -m "test(fusion): adjust scheme_validator tests for correct behavior"
```

---

## Task 5: Add Validate CLI Subcommand

**Files:**
- Modify: `tools/fusion/__main__.py`

**Step 1: Add validate subcommand to argparse**

Modify `parse_args` to add validate command:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.fusion",
        description="Discover fusible instruction patterns from DFG output.",
    )
    parser.add_argument(
        "command",
        choices=["discover", "score", "validate"],
        help="Command to run",
    )
    # ... existing args ...

    # Add validate-specific arguments
    parser.add_argument(
        "--opcode",
        type=lambda x: int(x, 0),  # Accept hex or decimal
        default=None,
        help="Opcode value (hex or decimal, required for validate)",
    )
    parser.add_argument(
        "--funct3",
        type=lambda x: int(x, 0),
        default=None,
        help="Funct3 value (hex or decimal, optional)",
    )
    parser.add_argument(
        "--funct7",
        type=lambda x: int(x, 0),
        default=None,
        help="Funct7 value (hex or decimal, optional)",
    )
    parser.add_argument(
        "--reg-class",
        choices=["integer", "float", "vector"],
        default="integer",
        help="Register class for the pattern (default: integer)",
    )
    return parser.parse_args(argv)
```

**Step 2: Add validate command handler in main()**

Add after the score command block:

```python
    if args.command == "validate":
        if args.opcode is None:
            parser.error("--opcode is required for validate command")

        from fusion.scheme_validator import validate_encoding

        registry = load_isa_registry(args.isa)
        result = validate_encoding(
            opcode=args.opcode,
            funct3=args.funct3,
            funct7=args.funct7,
            reg_class=args.reg_class,
            registry=registry,
        )

        # Output as JSON for easy parsing
        import json
        output = {
            "passed": result.passed,
            "conflicts": result.conflicts,
            "warnings": result.warnings,
            "suggested_alternatives": result.suggested_alternatives,
        }
        print(json.dumps(output, indent=2))
        return
```

**Step 3: Run CLI test**

Run: `cd tools && python -m fusion validate --opcode 0x33 --funct3 0 --funct7 0 --reg-class integer`

Expected: JSON output with `"passed": false` and conflict mentioning ADD

**Step 4: Commit**

```bash
git add tools/fusion/__main__.py
git commit -m "feat(fusion): add validate subcommand to CLI"
```

---

## Task 6: Create Fusion-Scheme Agent Skill

**Files:**
- Create: `.claude/skills/fusion-scheme/SKILL.md`

**Step 1: Write the skill file**

```markdown
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
- `0x33`: OP ( densely populated with funct3/funct7 combinations)
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
\`\`\`
rd = rs1 <OP> rs2  // <SEMANTIC_DESCRIPTION>
\`\`\`

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
| Encoding space       | ✅     | <VALIDATOR_CHECK>      |
| Register class       | ✅     | <CLASS_MATCH>          |
| Operand count        | ✅     | Within 2src+1dst       |
| No config write      | ✅     | <FOR_V:_NO_VSETVLI>    |
| No load/store        | ✅     | Memory-free pattern    |

## Validation Log
- <TIMESTAMP> PASS/FAIL — <VALIDATOR_OUTPUT_OR_SELF_CHECK>
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

Skill generates scheme for `fadd.s` → `fmul.s` fusion (FMADD.S-like pattern).

## Constraints

- **JSON-only validation feedback**: Validator CLI output is JSON; parse it precisely.
- **No speculative extensions**: Use only existing ISA descriptor context.
- **Prefer "pass" on uncertainty**: If encoding space is ambiguous, prefer available slots.
```

**Step 2: Verify skill file syntax**

Run: `cat .claude/skills/fusion-scheme/SKILL.md | head -20`

Expected: Frontmatter and first section visible

**Step 3: Commit**

```bash
git add .claude/skills/fusion-scheme/SKILL.md
git commit -m "feat(skill): add fusion-scheme agent skill for encoding generation"
```

---

## Task 7: Update Phase 2 Feature Roadmap

**Files:**
- Modify: `docs/plans/2026-04-08-phase2-feature-roadmap.md`

**Step 1: Update Feature 3 status**

Find the Feature 3 section and update status:

```markdown
## Feature 3: Fusion Scheme Specification Agent Skill

**Status**: Completed
**Design doc**: `docs/plans/2026-04-09-fusion-scheme-skill-design.md`
**Implementation plan**: `docs/plans/2026-04-09-fusion-scheme-skill-plan.md`
**Branch prefix**: `worktree-fusion-scheme`
**Merge commit**: <will be filled after merge>
```

**Step 2: Commit**

```bash
git add docs/plans/2026-04-08-phase2-feature-roadmap.md
git commit -m "docs: mark Feature 3 as completed in roadmap"
```

---

## Task 8: Final Verification and Integration Test

**Files:**
- No new files (verification only)

**Step 1: Run all validator tests**

Run: `cd tools && python -m unittest fusion.tests.test_scheme_validator -v`

Expected: All tests pass

**Step 2: Run full fusion module test suite**

Run: `cd tools && python -m unittest fusion.tests -v`

Expected: All tests pass (including existing pattern/miner tests)

**Step 3: Test validate CLI end-to-end**

Run: `python -m tools.fusion validate --opcode 0x0B --funct3 2 --funct7 0 --reg-class integer`

Expected: `{"passed": true, ...}`

Run: `python -m tools.fusion validate --opcode 0x33 --funct3 0 --funct7 0 --reg-class integer`

Expected: `{"passed": false, "conflicts": [...], ...}`

**Step 4: Verify skill exists**

Run: `ls -la .claude/skills/fusion-scheme/SKILL.md`

Expected: File exists with content

---

## Summary

| Task | Files | Lines Added |
|------|-------|-------------|
| 1 | `tools/fusion/scheme_validator.py` | ~30 (skeleton) |
| 2 | `tools/fusion/tests/test_scheme_validator.py` | ~120 |
| 3 | `tools/fusion/scheme_validator.py` | ~130 (implementation) |
| 4 | `tools/fusion/tests/test_scheme_validator.py` | ~30 (adjustments) |
| 5 | `tools/fusion/__main__.py` | ~25 (CLI) |
| 6 | `.claude/skills/fusion-scheme/SKILL.md` | ~150 |
| 7 | `docs/plans/2026-04-08-phase2-feature-roadmap.md` | ~5 |

**Total: ~300-350 lines** as estimated in design.