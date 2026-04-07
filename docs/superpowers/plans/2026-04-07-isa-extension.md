# DFG ISA Extension (F+M) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the DFG tool to generate data-flow graphs for F (single-precision float) and M (integer multiply/divide) RISC-V instructions by using `llvm-tblgen --dump-json` to auto-generate accurate ISA descriptor files.

**Architecture:** Extend `_extract_registers()` in `instruction.py` to recognize float registers via a pluggable `RegisterKind` system. Write a generator script (`gen_isadesc.py`) that reads `llvm-tblgen` JSON output and produces `rv64f.py` and `rv64m.py`. Register these new extensions in `__main__.py`. Existing `rv64i.py` is retained for pseudo-instructions not defined in TableGen.

**Tech Stack:** Python 3, `llvm-tblgen` (built from Xuantie LLVM submodule), RISC-V TableGen `.td` files

---

## File Map

| File | Responsibility |
|------|---------------|
| `tools/dfg/instruction.py` | Data models + `_extract_registers()`. Modify to add `RegisterKind`, extend `RegisterFlow`, and update register extraction for multi-kind support. |
| `tools/dfg/isadesc/rv64i.py` | RV64I base + pseudo-instructions (existing, no change). |
| `tools/dfg/isadesc/rv64f.py` | F extension RegisterFlow definitions (new — generated then committed). |
| `tools/dfg/isadesc/rv64m.py` | M extension RegisterFlow definitions (new — generated then committed). |
| `tools/dfg/gen_isadesc.py` | Generator script: reads `llvm-tblgen` JSON, outputs Python ISA descriptor files. |
| `tools/dfg/__main__.py` | CLI entry point. Modify to register F and M modules in `_ISA_MODULES`. |
| `tools/dfg/tests/test_instruction.py` | Existing tests. Add float-register extraction tests. |
| `tools/dfg/tests/test_dfg.py` | Existing tests. Add F-extension DFG tests. |
| `.gitignore` | Add `tools/dfg/riscv_instrs.json` (intermediate build artifact). |

---

### Task 1: Add RegisterKind and extend RegisterFlow in instruction.py

**Files:**
- Modify: `tools/dfg/instruction.py`
- Test: `tools/dfg/tests/test_instruction.py`

- [ ] **Step 1: Add the RegisterKind dataclass**

Add after the `RegisterFlow` dataclass (before `_extract_registers`):

```python
@dataclass
class RegisterKind:
    """Describes a register name space (integer, float, vector, ...).

    Attributes:
        name: Kind identifier, e.g. "integer", "float", "vector".
        pattern: Compiled regex matching register names in this space.
        position_prefix: Prefix for position names extracted from operands.
            E.g. "f" means operands map to "frd", "frs1", "frs2", "frs3".
    """

    name: str
    pattern: re.Pattern
    position_prefix: str
```

- [ ] **Step 2: Add `kinds` field to RegisterFlow**

Modify the existing `RegisterFlow` dataclass to include an optional `kinds` dict:

```python
@dataclass
class RegisterFlow:
    """Describes which register positions are dst/src for an instruction.

    Positions use RISC-V spec names (rd, rs1, rs2, rs3, etc.) which are
    resolved to actual register names from the operand string via resolve().
    When kinds is empty, all operands default to "integer".
    """

    dst_regs: list[str]
    src_regs: list[str]
    kinds: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 3: Add builtin register kinds**

Add after the `_extract_registers` function:

```python
# Builtin register kinds
INTEGER_KIND = RegisterKind(
    name="integer",
    pattern=re.compile(
        r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|[sf]\d+|jt?\d*)$"
    ),
    position_prefix="",
)

FLOAT_KIND = RegisterKind(
    name="float",
    pattern=re.compile(
        r"^(f\d+|ft\d+|fs\d+|fa\d+|fv\d+)$"
    ),
    position_prefix="f",
)

_BUILTIN_KINDS: list[RegisterKind] = [INTEGER_KIND, FLOAT_KIND]

_ROUNDING_MODES = frozenset({"dyn", "rne", "rtz", "rdn", "rup", "rmm"})
```

- [ ] **Step 4: Rewrite `_extract_registers()` to support multi-kind**

Replace the existing `_extract_registers` function with a version that:
1. Accepts an optional `kinds` list of `RegisterKind` (defaults to `_BUILTIN_KINDS`)
2. Recognizes float register names in addition to integer
3. Skips rounding mode tokens (`dyn`, `rne`, `rtz`, `rdn`, `rup`, `rmm`)
4. Maps matched registers to position names with the kind's prefix

```python
def _extract_registers(
    operands: str,
    kinds: list[RegisterKind] | None = None,
) -> dict[str, tuple[str, str]]:
    """Extract register names from an operand string.

    Handles formats:
      - "rd,rs1,rs2" -> {"rd": ("a0", "integer"), "rs1": ("a1", "integer"), ...}
      - "rs2,offset(rs1)" -> {"rs2": ("a0", "integer"), "rs1": ("s0", "integer")}
      - "fmadd.s dyn,ft2,fa4,ft0,ft2" -> {"frd": ("ft2","float"), "frs1": ("fa4","float"), ...}
      - "flw fa4,0(a6)" -> {"frd": ("fa4","float"), "rs1": ("a6","integer")}
    Returns a mapping from position name to (register_name, kind_name).
    """
    if kinds is None:
        kinds = _BUILTIN_KINDS
    regs: dict[str, tuple[str, str]] = {}
    parts = [p.strip() for p in operands.split(",")]
    if not parts:
        return regs

    # Detect memory format: "rs2, offset(rs1)" or "rd, offset(rs1)"
    if len(parts) == 2 and "(" in parts[1]:
        first = parts[0].strip()
        match = re.match(r"(-?\d+)\((\w+)\)", parts[1].strip())
        if match:
            offset_reg_name = match.group(2)
            offset_reg_kind = _match_kind(offset_reg_name, kinds)
            if offset_reg_kind:
                regs[offset_reg_kind.position_prefix + "rs1"] = (
                    offset_reg_name,
                    offset_reg_kind.name,
                )
            # Both rd and rs2 use the first operand — RegisterFlow.resolve()
            # selects only the positions each instruction specifies.
            first_kind = _match_kind(first, kinds)
            if first_kind:
                regs[first_kind.position_prefix + "rd"] = (first, first_kind.name)
                regs[first_kind.position_prefix + "rs2"] = (first, first_kind.name)
        return regs

    # Standard format: "rd, rs1, rs2, rs3" or "rd, rs1, imm"
    position_names = ["rd", "rs1", "rs2", "rs3"]
    pos_idx = 0
    for token in parts:
        token = token.strip()
        # Skip immediates
        if re.match(r"^-?\d+$", token) or re.match(r"^0x[0-9a-fA-F]+$", token):
            continue
        # Skip rounding modes
        if token in _ROUNDING_MODES:
            continue
        # Try matching against registered kinds
        matched_kind = _match_kind(token, kinds)
        if matched_kind is None:
            continue
        if pos_idx >= len(position_names):
            break
        pos_name = matched_kind.position_prefix + position_names[pos_idx]
        regs[pos_name] = (token, matched_kind.name)
        pos_idx += 1
    return regs


def _match_kind(
    token: str,
    kinds: list[RegisterKind],
) -> RegisterKind | None:
    """Find the first RegisterKind whose pattern matches *token*."""
    for kind in kinds:
        if kind.pattern.match(token):
            return kind
    return None
```

- [ ] **Step 5: Update RegisterFlow.resolve() to use new _extract_registers**

Replace the existing `resolve` method in `RegisterFlow`:

```python
    def resolve(self, operands: str) -> ResolvedFlow:
        """Map positional names to actual register names from the operand string."""
        regs = _extract_registers(operands, kinds=_BUILTIN_KINDS)
        dst = [regs[p][0] for p in self.dst_regs if p in regs]
        src = [regs[p][0] for p in self.src_regs if p in regs]
        return ResolvedFlow(dst_regs=dst, src_regs=src)
```

- [ ] **Step 6: Write failing tests for float register extraction**

Add to `tools/dfg/tests/test_instruction.py` inside a new test class:

```python
class TestRegisterKind(unittest.TestCase):
    """Tests for the RegisterKind system and float register extraction."""

    def test_float_kind_matches_f_reg(self):
        from dfg.instruction import FLOAT_KIND
        self.assertIsNotNone(FLOAT_KIND.pattern.match("f0"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("f31"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("ft0"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("fa7"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("fs11"))

    def test_float_kind_rejects_int_reg(self):
        from dfg.instruction import FLOAT_KIND
        self.assertIsNone(FLOAT_KIND.pattern.match("a0"))
        self.assertIsNone(FLOAT_KIND.pattern.match("t0"))

    def test_integer_kind_matches_abi_names(self):
        from dfg.instruction import INTEGER_KIND
        self.assertIsNotNone(INTEGER_KIND.pattern.match("a0"))
        self.assertIsNotNone(INTEGER_KIND.pattern.match("zero"))
        self.assertIsNotNone(INTEGER_KIND.pattern.match("ra"))


class TestExtractRegistersFloat(unittest.TestCase):
    """Tests for _extract_registers with float register kinds."""

    def test_fadd_s_two_float_regs(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("fa5,fa0,fa4")
        self.assertEqual(regs["frd"][0], "fa5")
        self.assertEqual(regs["frd"][1], "float")
        self.assertEqual(regs["frs1"][0], "fa0")
        self.assertEqual(regs["frs1"][1], "float")
        self.assertEqual(regs["frs2"][0], "fa4")
        self.assertEqual(regs["frs2"][1], "float")

    def test_fmadd_s_with_rounding_mode(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("dyn,ft2,fa4,ft0,ft2")
        self.assertEqual(regs["frd"][0], "ft2")
        self.assertEqual(regs["frs1"][0], "fa4")
        self.assertEqual(regs["frs2"][0], "ft0")
        self.assertEqual(regs["frs3"][0], "ft2")

    def test_flw_float_dst_int_base(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("fa4,0(a6)")
        self.assertEqual(regs["frd"][0], "fa4")
        self.assertEqual(regs["frd"][1], "float")
        self.assertEqual(regs["rs1"][0], "a6")
        self.assertEqual(regs["rs1"][1], "integer")

    def test_fsw_float_src_int_base(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("ft2,0(a2)")
        self.assertEqual(regs["frs2"][0], "ft2")
        self.assertEqual(regs["frs2"][1], "float")
        self.assertEqual(regs["rs1"][0], "a2")
        self.assertEqual(regs["rs1"][1], "integer")

    def test_integer_registers_still_work(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("a0,a1,a2")
        self.assertEqual(regs["rd"][0], "a0")
        self.assertEqual(regs["rd"][1], "integer")
        self.assertEqual(regs["rs1"][0], "a1")
        self.assertEqual(regs["rs2"][0], "a2")

    def test_memory_format_int_still_works(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("ra,24(sp)")
        self.assertEqual(regs["rs2"][0], "ra")
        self.assertEqual(regs["rs1"][0], "sp")

    def test_rounding_mode_rne_skipped(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("rne,fa5,fa0,fa4")
        self.assertEqual(regs["frd"][0], "fa5")
        self.assertEqual(regs["frs1"][0], "fa0")
        self.assertEqual(regs["frs2"][0], "fa4")
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest dfg/tests/test_instruction.py::TestRegisterKind -v`
Expected: FAIL — `ImportError: cannot import name 'FLOAT_KIND' from 'dfg.instruction'`

- [ ] **Step 8: Implement the code changes from Steps 1-5**

Apply all changes to `tools/dfg/instruction.py`.

- [ ] **Step 9: Run all instruction tests to verify they pass**

Run: `cd tools && python3 -m pytest dfg/tests/test_instruction.py -v`
Expected: All tests PASS (both old I-extension tests and new float tests).

- [ ] **Step 10: Run DFG tests to verify no regression**

Run: `cd tools && python3 -m pytest dfg/tests/test_dfg.py -v`
Expected: All tests PASS.

- [ ] **Step 11: Commit**

```bash
git add tools/dfg/instruction.py tools/dfg/tests/test_instruction.py
git commit -m "feat(dfg): add RegisterKind system and float register extraction"
```

---

### Task 2: Build llvm-tblgen and extract RISC-V instruction JSON

**Files:**
- Create (gitignored): `tools/dfg/riscv_instrs.json`
- Modify: `.gitignore`

- [ ] **Step 1: Add riscv_instrs.json to .gitignore**

Append to `.gitignore`:

```
# llvm-tblgen generated instruction JSON (intermediate build artifact)
tools/dfg/riscv_instrs.json
```

- [ ] **Step 2: Initialize and build llvm-tblgen from Xuantie LLVM submodule**

```bash
cd /home/claude/wsp/cx/rvfuse-2/.claude/worktrees/dfg-v1.3-addisa
git submodule update --init third_party/llvm-project
cd third_party/llvm-project
mkdir -p build && cd build
cmake -DLLVM_TARGETS_TO_BUILD=RISCV \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="" \
      ../llvm
make llvm-tblgen -j$(nproc)
```

Expected: `bin/llvm-tblgen` binary produced.

- [ ] **Step 3: Extract RISC-V instruction definitions as JSON**

```bash
./bin/llvm-tblgen \
  -I ../llvm/include \
  -I ../llvm/lib/Target/RISCV \
  ../llvm/lib/Target/RISCV/RISCV.td \
  --dump-json \
  > /home/claude/wsp/cx/rvfuse-2/.claude/worktrees/dfg-v1.3-addisa/tools/dfg/riscv_instrs.json
```

Expected: JSON file created (likely 50-200MB).

- [ ] **Step 4: Verify the JSON contains expected instructions**

```bash
python3 -c "
import json, sys
with open('/home/claude/wsp/cx/rvfuse-2/.claude/worktrees/dfg-v1.3-addisa/tools/dfg/riscv_instrs.json') as f:
    data = json.load(f)
# Check it's a dict of records
print(f'Total records: {len(data)}')
# Look for specific F-extension instructions
for name in ['FMADD_S', 'FLW', 'FSW', 'FADD_S', 'FMUL_S', 'FEQ_S']:
    if name in data:
        rec = data[name]
        print(f'{name}: OutOperandList={rec.get(\"!superclasses\", [])[:3]}')
    else:
        print(f'{name}: NOT FOUND')
"
```

Expected: All 6 instructions found in the JSON.

- [ ] **Step 5: Commit .gitignore change**

```bash
git add .gitignore
git commit -m "chore: gitignore llvm-tblgen intermediate JSON artifact"
```

---

### Task 3: Write the gen_isadesc.py generator script

**Files:**
- Create: `tools/dfg/gen_isadesc.py`

This script reads the `llvm-tblgen --dump-json` output and generates Python ISA descriptor files. The script must be written before we can generate `rv64f.py` and `rv64m.py`, so we write it first and test it against the JSON from Task 2.

**Important:** The exact JSON schema of `--dump-json` must be inspected from the actual output before writing the script. The steps below provide the expected structure based on the LLVM docs, but the implementor must verify against the real JSON.

- [ ] **Step 1: Inspect the actual JSON structure for key F/M instructions**

```bash
python3 -c "
import json
with open('tools/dfg/riscv_instrs.json') as f:
    data = json.load(f)
# Print full records for FMADD_S, FLW, MUL to understand the schema
for name in ['FMADD_S', 'FLW', 'FSW', 'FADD_S', 'FEQ_S', 'MUL', 'DIV']:
    if name in data:
        import pprint
        print(f'=== {name} ===')
        pprint.pprint(data[name], width=200, compact=False)
        print()
"
```

This output will reveal the exact JSON field names. Key fields to look for:
- The superclass list (how to identify Instruction records)
- `OutOperandList` / `InOperandList` format
- `AsmString` format
- `Predicates` format

- [ ] **Step 2: Write gen_isadesc.py**

The script structure (adjust field names based on Step 1 findings):

```python
#!/usr/bin/env python3
"""Generate DFG ISA descriptor Python files from llvm-tblgen --dump-json output.

Usage:
    python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext F -o tools/dfg/isadesc/rv64f.py
    python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext M -o tools/dfg/isadesc/rv64m.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys


# Mapping from LLVM register class names to DFG register kind position prefixes.
# Empty string prefix = default (integer) register positions: rd, rs1, rs2, rs3.
# "f" prefix = float register positions: frd, frs1, frs2, frs3.
REG_CLASS_TO_PREFIX: dict[str, str] = {
    "GPR": "",
    "GPRNoX0": "",
    "GPRMem": "",
    "FPR32": "f",
    "FPR64": "f",
    "FPR16": "f",
}

# Operand names that are NOT data registers (skip them).
SKIP_OPERANDS = frozenset({
    "frm",          # rounding mode
    "imm12",        # 12-bit immediate
})

# Predicate strings that indicate an instruction belongs to an extension.
PREDICATE_TO_EXT: dict[str, str] = {
    "HasStdExtF": "F",
    "HasStdExtD": "D",
    "HasStdExtM": "M",
    "HasStdExtA": "A",
    "HasStdExtV": "V",
}

# Mapping from LLVM instruction def names to QEMU disassembly mnemonics.
# Most follow a simple pattern: FMADD_S -> fmadd.s, but some need explicit mapping.
def _llvm_name_to_mnemonic(llvm_name: str) -> str:
    """Convert LLVM TableGen def name to QEMU disassembly mnemonic.

    Examples:
        FMADD_S -> fmadd.s
        FLW -> flw
        MUL -> mul
    """
    # Common pattern: remove underscores, lowercase
    mnemonic = llvm_name.lower().replace("_", ".")
    # Handle cases like "FMADD_S" -> "fmadd.s" (already correct with above)
    return mnemonic


def _parse_dag_operands(dag_str: str) -> list[tuple[str, str, str]]:
    """Parse an operand DAG string from TableGen JSON.

    Input format: "(outs FPR32:$rd)" or "(ins FPR32:$rs1, GPR:$rs2, frmarg:$frm)"
    Returns list of (reg_class, operand_name, direction) tuples.
    direction is "out" or "in".
    """
    # Remove wrapping "(outs " and ")" or "(ins " and ")"
    dag_str = dag_str.strip()
    direction = "out"
    if dag_str.startswith("(ins "):
        direction = "in"
    elif dag_str.startswith("(outs "):
        direction = "out"
    # Strip parens
    inner = dag_str.strip("()")
    inner = inner.removeprefix("ins ").removeprefix("outs ")

    results: list[tuple[str, str, str]] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        # Format: "RegClass:$name"
        match = re.match(r"(\w+):\$(\w+)", part)
        if match:
            reg_class = match.group(1)
            op_name = match.group(2)
            results.append((reg_class, op_name, direction))
    return results


def _operand_to_position(
    op_name: str,
    prefix: str,
    direction: str,
) -> str | None:
    """Map a TableGen operand name to a DFG position name.

    Returns None if the operand should be skipped (not a data register).
    """
    if op_name in SKIP_OPERANDS:
        return None
    # Map standard RISC-V operand names to position names
    if op_name == "rd":
        return prefix + "rd"
    elif op_name == "rs1":
        return prefix + "rs1"
    elif op_name == "rs2":
        return prefix + "rs2"
    elif op_name == "rs3":
        return prefix + "rs3"
    # Unknown operand name — skip
    return None


def _get_predicates(record: dict) -> list[str]:
    """Extract predicate names from a TableGen record."""
    preds = record.get("Predicates", [])
    # JSON format may be a list of strings or list of dicts
    result = []
    for p in preds:
        if isinstance(p, str):
            result.append(p)
        elif isinstance(p, dict) and "def" in p:
            result.append(p["def"])
    return result


def _belongs_to_extension(predicates: list[str], target_ext: str) -> bool:
    """Check if the instruction's predicates indicate it belongs to target_ext."""
    for pred in predicates:
        if PREDICATE_TO_EXT.get(pred) == target_ext:
            return True
    return False


def extract_instructions(
    data: dict,
    target_ext: str,
) -> list[dict]:
    """Extract all Instruction records belonging to target_ext.

    Returns list of dicts with keys:
        mnemonic: str (QEMU disassembly mnemonic, e.g. "fmadd.s")
        dst_regs: list[str] (position names, e.g. ["frd"])
        src_regs: list[str] (position names, e.g. ["frs1", "frs2"])
        dst_kind: str (register kind for dst, e.g. "float")
        src_kind: str (register kind for first src)
    """
    results = []
    for record_name, record in data.items():
        # Filter: must be an Instruction subclass
        superclasses = record.get("!superclasses", [])
        is_instruction = any(
            sc.get("def") == "Instruction" if isinstance(sc, dict) else sc == "Instruction"
            for sc in superclasses
        )
        if not is_instruction:
            continue

        predicates = _get_predicates(record)
        if not _belongs_to_extension(predicates, target_ext):
            continue

        # Parse operands
        out_dag = record.get("OutOperandList", "")
        in_dag = record.get("InOperandList", "")

        dst_regs: list[str] = []
        src_regs: list[str] = []
        dst_kind = ""
        src_kind = ""

        out_ops = _parse_dag_operands(out_dag) if out_dag else []
        in_ops = _parse_dag_operands(in_dag) if in_dag else []

        for reg_class, op_name, direction in out_ops:
            prefix = REG_CLASS_TO_PREFIX.get(reg_class, "")
            pos = _operand_to_position(op_name, prefix, direction)
            if pos and direction == "out":
                dst_regs.append(pos)
                kind = "float" if prefix == "f" else "integer"
                if not dst_kind:
                    dst_kind = kind

        for reg_class, op_name, direction in in_ops:
            prefix = REG_CLASS_TO_PREFIX.get(reg_class, "")
            pos = _operand_to_position(op_name, prefix, direction)
            if pos and direction == "in":
                src_regs.append(pos)
                kind = "float" if prefix == "f" else "integer"
                if not src_kind:
                    src_kind = kind

        if not dst_regs and not src_regs:
            continue  # Skip instructions with no register operands

        mnemonic = _llvm_name_to_mnemonic(record_name)
        results.append({
            "mnemonic": mnemonic,
            "llvm_name": record_name,
            "dst_regs": dst_regs,
            "src_regs": src_regs,
            "dst_kind": dst_kind,
            "src_kind": src_kind,
        })

    return results


def generate_python(instructions: list[dict], ext_name: str) -> str:
    """Generate Python source code for an ISA descriptor module."""
    lines = [
        f'"""RV64{ext_name} extension register flow definitions (auto-generated).',
        f'"""',
        "",
        "from __future__ import annotations",
        "",
        "from dfg.instruction import ISARegistry, RegisterFlow",
        "",
        "",
    ]

    for insn in instructions:
        mnem = insn["mnemonic"]
        dst = insn["dst_regs"]
        src = insn["src_regs"]
        lines.append(
            f'("{mnem}", RegisterFlow({dst}, {src})),'
        )

    lines.append("")
    lines.append("")
    lines.append(f"ALL_RV64{ext_name}: list[tuple[str, RegisterFlow]] = [")
    for insn in instructions:
        mnem = insn["mnemonic"]
        dst = insn["dst_regs"]
        src = insn["src_regs"]
        lines.append(
            f'    ("{mnem}", RegisterFlow({dst}, {src})),'
        )
    lines.append("]")
    lines.append("")
    lines.append("")
    lines.append(f"def build_registry(registry: ISARegistry) -> None:")
    lines.append(f'    """Register all RV64{ext_name} instructions into the given ISA registry."""')
    lines.append(f'    registry.load_extension("{ext_name}", ALL_RV64{ext_name})')
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate DFG ISA descriptor from llvm-tblgen JSON.",
    )
    parser.add_argument("json_path", type=str, help="Path to llvm-tblgen --dump-json output")
    parser.add_argument("--ext", required=True, type=str, help="Target extension: F, M, D, A")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output Python file path")
    args = parser.parse_args(argv)

    with open(args.json_path) as f:
        data = json.load(f)

    instructions = extract_instructions(data, args.ext)
    print(f"Found {len(instructions)} instructions for extension {args.ext}")

    source = generate_python(instructions, args.ext)
    with open(args.output, "w") as f:
        f.write(source)

    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the generator for F extension**

```bash
python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext F -o tools/dfg/isadesc/rv64f.py
```

Expected: Prints `Found N instructions for extension F` where N > 0, and generates `rv64f.py`.

- [ ] **Step 4: Run the generator for M extension**

```bash
python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext M -o tools/dfg/isadesc/rv64m.py
```

Expected: Prints `Found N instructions for extension M` where N > 0, and generates `rv64m.py`.

- [ ] **Step 5: Verify generated output contains expected instructions**

```bash
python3 -c "
from dfg.isadesc.rv64f import build_registry
from dfg.instruction import ISARegistry
reg = ISARegistry()
build_registry(reg)
for mnem in ['flw', 'fsw', 'fmadd.s', 'fadd.s', 'fmul.s', 'feq.s', 'flt.s', 'fmv.s.x', 'fmv.x.s', 'fdiv.s']:
    print(f'{mnem}: {reg.is_known(mnem)}')
"
```

Expected: All listed mnemonics print `True`.

```bash
python3 -c "
from dfg.isadesc.rv64m import build_registry
from dfg.instruction import ISARegistry
reg = ISARegistry()
build_registry(reg)
for mnem in ['mul', 'mulh', 'mulhu', 'mulw', 'div', 'divu', 'rem', 'remu']:
    print(f'{mnem}: {reg.is_known(mnem)}')
"
```

Expected: All listed mnemonics print `True`.

> **NOTE:** If any expected mnemonics are missing, the generator's `_llvm_name_to_mnemonic()` function or predicate filtering needs adjustment. Inspect the actual JSON to fix the mapping.

- [ ] **Step 6: Commit**

```bash
git add tools/dfg/gen_isadesc.py tools/dfg/isadesc/rv64f.py tools/dfg/isadesc/rv64m.py
git commit -m "feat(dfg): add gen_isadesc.py and generated F+M ISA descriptors"
```

---

### Task 4: Register F and M extensions in __main__.py

**Files:**
- Modify: `tools/dfg/__main__.py:35-37`

- [ ] **Step 1: Add F and M to _ISA_MODULES**

In `tools/dfg/__main__.py`, change the `_ISA_MODULES` dict from:

```python
_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
}
```

to:

```python
_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),
    "M": ("dfg.isadesc.rv64m", "build_registry"),
}
```

- [ ] **Step 2: Verify the CLI loads extensions correctly**

```bash
cd tools && python3 -m dfg --help
```

Expected: No errors. The `--isa` help text should still say `default: I`.

```bash
cd tools && python3 -m dfg --isa I,F,M --disas /dev/null 2>&1 | head -5
```

Expected: Logs show `Loaded ISA extension: I`, `Loaded ISA extension: F`, `Loaded ISA extension: M`.

- [ ] **Step 3: Commit**

```bash
git add tools/dfg/__main__.py
git commit -m "feat(dfg): register F and M ISA extensions in CLI"
```

---

### Task 5: Add F-extension DFG builder tests

**Files:**
- Modify: `tools/dfg/tests/test_dfg.py`

- [ ] **Step 1: Add F-extension DFG test cases**

Append to `tools/dfg/tests/test_dfg.py`:

```python
def _make_im_registry() -> ISARegistry:
    """Build a registry with I + M + F extensions."""
    reg = ISARegistry()
    from dfg.isadesc.rv64i import build_registry as build_i
    from dfg.isadesc.rv64m import build_registry as build_m
    from dfg.isadesc.rv64f import build_registry as build_f
    build_i(reg)
    build_m(reg)
    build_f(reg)
    return reg


class TestFExtensionDfg(unittest.TestCase):
    """DFG builder tests for F-extension instructions."""

    def test_fmadd_s_chain(self):
        """fmadd.s ft2,fa4,ft0,ft2 writes ft2; next fmadd reads ft2."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "fmadd.s", "dyn,ft2,fa4,ft0,ft2", ""),
            Instruction(0x1004, "fadd.s", "dyn,fa5,ft2,fa0", ""),
        ])
        dfg = build_dfg(bb, _make_im_registry())
        ft2_edges = [e for e in dfg.edges if e.register == "ft2"]
        self.assertEqual(len(ft2_edges), 1)
        self.assertEqual(ft2_edges[0].src_index, 0)
        self.assertEqual(ft2_edges[0].dst_index, 1)

    def test_flw_fadd_chain(self):
        """flw fa4,0(a6) writes fa4; fadd.s reads fa4."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "flw", "fa4,0(a6)", ""),
            Instruction(0x1004, "fadd.s", "dyn,fa5,fa4,fa0", ""),
        ])
        dfg = build_dfg(bb, _make_im_registry())
        fa4_edges = [e for e in dfg.edges if e.register == "fa4"]
        self.assertEqual(len(fa4_edges), 1)
        self.assertEqual(fa4_edges[0].src_index, 0)
        self.assertEqual(fa4_edges[0].dst_index, 1)

    def test_fsw_reads_float_reg(self):
        """fmul.s writes ft2; fsw stores ft2."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "fmul.s", "dyn,ft2,ft2,fa0", ""),
            Instruction(0x1004, "fsw", "ft2,0(a2)", ""),
        ])
        dfg = build_dfg(bb, _make_im_registry())
        ft2_edges = [e for e in dfg.edges if e.register == "ft2"]
        self.assertEqual(len(ft2_edges), 1)
        self.assertEqual(ft2_edges[0].src_index, 0)
        self.assertEqual(ft2_edges[0].dst_index, 1)

    def test_mixed_int_float_bb(self):
        """A BB mixing integer and float instructions."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "flw", "fa4,0(a6)", ""),
            Instruction(0x1004, "fmadd.s", "dyn,ft2,fa4,ft0,ft2", ""),
            Instruction(0x1008, "addi", "a6,a6,8", ""),
            Instruction(0x100c, "fsw", "ft2,0(a2)", ""),
        ])
        dfg = build_dfg(bb, _make_im_registry())
        # fa4: flw -> fmadd
        fa4_edges = [e for e in dfg.edges if e.register == "fa4"]
        self.assertEqual(len(fa4_edges), 1)
        self.assertEqual(fa4_edges[0].src_index, 0)
        self.assertEqual(fa4_edges[0].dst_index, 1)
        # ft2: fmadd writes, fsw reads
        ft2_edges = [e for e in dfg.edges if e.register == "ft2"]
        self.assertEqual(len(ft2_edges), 1)
        self.assertEqual(ft2_edges[0].src_index, 1)
        self.assertEqual(ft2_edges[0].dst_index, 3)
        # a6: addi writes — no reader in this BB
        self.assertEqual(len([e for e in dfg.edges if e.register == "a6"]), 0)

    def test_yolo_hot_bb31041_partial(self):
        """Partial test of the YOLO11n hottest BB (31041) — first 6 instructions."""
        bb = BasicBlock(bb_id=31041, vaddr=0x7eeef161ecac, instructions=[
            Instruction(0x7eeef161ecac, "flw", "fa4,0(a6)", ""),
            Instruction(0x7eeef161ecb0, "flw", "fa5,0(a7)", ""),
            Instruction(0x7eeef161ecb4, "flw", "ft3,12(a5)", ""),
            Instruction(0x7eeef161ecb8, "flw", "ft0,0(a5)", ""),
            Instruction(0x7eeef161ecbc, "flw", "fa1,4(a5)", ""),
            Instruction(0x7eeef161ecc0, "flw", "fa2,8(a5)", ""),
        ])
        dfg = build_dfg(bb, _make_im_registry())
        # All 6 instructions should be known (no unknown mnemonics)
        self.assertEqual(len(dfg.nodes), 6)
        # a5 is read by 4 loads (ft3, ft0, fa1, fa2 at offsets 12,0,4,8)
        a5_edges = [e for e in dfg.edges if e.register == "a5"]
        self.assertEqual(len(a5_edges), 4)
```

- [ ] **Step 2: Run all tests**

```bash
cd tools && python3 -m pytest dfg/tests/test_dfg.py -v
```

Expected: All tests PASS, including the new F-extension tests.

- [ ] **Step 3: Commit**

```bash
git add tools/dfg/tests/test_dfg.py
git commit -m "test(dfg): add F-extension DFG builder tests"
```

---

### Task 6: Integration test — run DFG on YOLO11n hot blocks

**Files:** None (CLI test only)

This is the end-to-end validation: run DFG generation with `--isa I,F,M` on the YOLO11n `.disas` file with the hotspot report and verify that the hot blocks now produce DFGs.

- [ ] **Step 1: Run DFG generation on YOLO11n hot blocks**

```bash
cd /home/claude/wsp/cx/rvfuse-2/.claude/worktrees/dfg-v1.3-addisa
rm -rf output/dfg
python3 tools/dfg/__main__.py \
  --disas output/yolo.bbv.disas \
  --isa I,F,M \
  --report output/hotspot.json \
  --coverage 80 \
  --output-dir output/dfg \
  --no-agent \
  --verbose
```

Expected output includes:
- `Loaded ISA extension: I (79 mnemonics)`
- `Loaded ISA extension: F (N mnemonics)`
- `Loaded ISA extension: M (N mnemonics)`
- `Report filter (coverage=80): 4/4 BBs matched`
- `BB 31041: N nodes, N edges (source=script)`
- `BB 31174: N nodes, N edges (source=script)`
- `BB 31287: N nodes, N edges (source=script)`
- `BB 31288: N nodes, N edges (source=script)`

- [ ] **Step 2: Verify DFG files were generated**

```bash
ls -la output/dfg/dot/ | head -20
```

Expected: At least 4 `.dot` files (one per hot BB).

```bash
cat output/dfg/summary.json | python3 -m json.tool
```

Expected:
```json
{
  "script_generated": 4,
  "agent_generated": 0,
  "unsupported_instructions": [],
  "png_generated": 4,
  ...
}
```

- [ ] **Step 3: Inspect one DFG to verify correctness**

```bash
head -40 output/dfg/dot/bb_31041.dot
```

Expected: DOT graph with float register edges (fa4, ft2, fa5, etc.) showing data dependencies between flw/fmadd.s instructions.

- [ ] **Step 4: Check for remaining unsupported instructions**

The `summary.json` `unsupported_instructions` field should be empty for the 80% coverage blocks. If there are remaining unsupported mnemonics (e.g. pseudo-instructions like `fmv.s` without rounding mode, or CSR instructions), note them for future work but do NOT fix in this task.

> **Stop condition:** If `script_generated == 4` and `unsupported_instructions == []`, this task is complete.

- [ ] **Step 5: Commit (if any fixes were needed)**

Only commit if Task 6 required fixes to pass. Otherwise, no commit needed for this task.

---

## Self-Review Checklist

**Spec coverage:**
- [x] Design: "RegisterKind system" → Task 1
- [x] Design: "llvm-tblgen build" → Task 2
- [x] Design: "gen_isadesc.py generator" → Task 3
- [x] Design: "Register F and M in __main__.py" → Task 4
- [x] Design: "Unit tests for _extract_registers" → Task 1 Step 6
- [x] Design: "F-extension DFG tests" → Task 5
- [x] Design: "Integration test on YOLO11n" → Task 6
- [x] Design: ".gitignore for riscv_instrs.json" → Task 2 Step 1
- [x] Design: "V extension future-proofing" → Handled by RegisterKind system (pluggable kinds)

**Placeholder scan:**
- No TBD/TODO found. All code blocks contain actual code.
- gen_isadesc.py has a NOTE about inspecting real JSON first (Step 1 of Task 3) — this is intentional since the exact JSON field names depend on the llvm-tblgen version.

**Type consistency:**
- `RegisterKind` defined once in Task 1, used consistently in Tasks 1, 3, 5
- `RegisterFlow` `kinds` field added in Task 1, used by `resolve()` in same task
- `_extract_registers` returns `dict[str, tuple[str, str]]` (position_name -> (reg_name, kind_name)) — consistent across all usages
- Extension module names `rv64f.py`, `rv64m.py` consistent with existing `rv64i.py` pattern
