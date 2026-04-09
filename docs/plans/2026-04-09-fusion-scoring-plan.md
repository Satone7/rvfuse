# Fusion Scoring & Constraint Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a hardware constraint checker and weighted scoring function that ranks fusion candidates from Feature 1's pattern catalog by frequency, dependency tightness, and encoding feasibility.

**Architecture:** Extend `RegisterFlow` with an `InstructionFormat` dataclass carrying opcode/format/field metadata. Build `ConstraintChecker` that validates encoding space budgets, register class compatibility, and operand limits. Build `Scorer` that computes a weighted sum of log-normalized frequency, RAW edge density, and hardware feasibility. Wire both into a new `score` CLI subcommand on the existing `python -m tools.fusion` entry point.

**Tech Stack:** Python 3 (dataclasses, stdlib only), existing `tools/dfg/instruction.py` ISARegistry, `unittest.TestCase` for tests, `gen_isadesc.py` for auto-generating encoding metadata.

**Design doc:** `docs/plans/2026-04-09-fusion-scoring-design.md`

**Priority:** V-extension encoding support is highest priority.

---

## Key File Reference

| File | Role |
|------|------|
| `tools/dfg/instruction.py:29-47` | `RegisterFlow` dataclass (to be extended) |
| `tools/dfg/instruction.py:289-313` | `ISARegistry` class |
| `tools/dfg/gen_isadesc.py` | ISA descriptor generator (to be extended) |
| `tools/dfg/isadesc/rv64v.py` | 376 V-extension instructions (auto-generated) |
| `tools/dfg/isadesc/rv64f.py` | 41 F-extension instructions (auto-generated) |
| `tools/dfg/isadesc/rv64m.py` | 14 M-extension instructions (auto-generated) |
| `tools/dfg/isadesc/rv64i.py` | 77 I-extension instructions (manual) |
| `tools/fusion/__main__.py` | CLI entry point (add `score` subcommand) |
| `tools/fusion/miner.py` | Feature 1 pattern mining (output format reference) |
| `tools/fusion/pattern.py` | Pattern dataclass (input to scoring) |
| `tools/fusion/tests/test_miner.py` | Test style reference (unittest.TestCase) |

## Encoding Metadata Strategy

The LLVM tablegen JSON (`tools/dfg/riscv_instrs.json`) stores instruction encoding in an `Inst` array (32 elements, LSB-first: `inst[i]` = bit `i`). Key fields extractable:

- **opcode** (bits 6:0): From separate `Opcode` field. Reverse the bit list to get value.
- **funct3** (bits 14:12): Fixed bits when not a variable encoding operand.
- **funct7** (bits 31:25): Fixed bits when not a variable encoding operand.
- **has_rd/rs1/rs2**: Determined by whether `var(rd/rs1/rs2)` appears in the Inst array.
- **mayLoad/mayStore**: Direct fields on the tablegen entry.
- **has_rs3**: Present for R4-type (fmadd/fnmadd). Check for `var(rs3)` at indices 27-31.

For instructions where funct3/funct7 are encoding variables (e.g., `FADD_S` has `var(funct3)`), these fields will be `None` in `InstructionFormat`. The constraint model treats this as "needs further analysis" (Feature 3).

---

## Task 1: Add `InstructionFormat` dataclass to `instruction.py`

**Files:**
- Modify: `tools/dfg/instruction.py:29` (before `RegisterFlow`)

**Step 1: Add `InstructionFormat` dataclass**

Insert before the `RegisterFlow` class (line 29 in current file):

```python
@dataclass
class InstructionFormat:
    """RISC-V instruction encoding layout metadata.

    Attributes:
        format_type: RISC-V instruction format ("R", "I", "S", "B", "U", "J", "R4", "V").
        opcode: 7-bit opcode value (e.g., 0x33 for OP, 0x53 for OP-FP, 0x57 for OP-V).
        funct3: 3-bit funct3 value, or None if variable.
        funct7: 7-bit funct7 value, or None if variable.
        has_rd: Whether the rd/destination field is present.
        has_rs1: Whether the rs1 field is present.
        has_rs2: Whether the rs2 field is present.
        has_rs3: Whether the rs3 field is present (R4-type only).
        has_imm: Whether an immediate field is present.
        imm_bits: Immediate field width in bits (0 if no imm).
        may_load: Whether the instruction accesses memory (load).
        may_store: Whether the instruction accesses memory (store).
        reg_class: Register class ("integer", "float", "vector").
    """

    format_type: str
    opcode: int
    funct3: int | None = None
    funct7: int | None = None
    has_rd: bool = True
    has_rs1: bool = True
    has_rs2: bool = True
    has_rs3: bool = False
    has_imm: bool = False
    imm_bits: int = 0
    may_load: bool = False
    may_store: bool = False
    reg_class: str = "integer"
```

**Step 2: Add `encoding` field to `RegisterFlow`**

In the `RegisterFlow` class (line 29), add one field after `config_regs`:

```python
    encoding: InstructionFormat | None = None
```

The full class becomes:

```python
@dataclass
class RegisterFlow:
    """Describes which register positions are dst/src for an instruction.

    Positions use RISC-V spec names (rd, rs1, rs2, rs3, etc.) which are
    resolved to actual register names from the operand string via resolve().
    """

    dst_regs: list[str]
    src_regs: list[str]
    config_regs: list[str] = field(default_factory=list)
    encoding: InstructionFormat | None = None

    def resolve(self, operands: str) -> ResolvedFlow:
        """Map positional names to actual register names from the operand string."""
        regs = _extract_registers(operands, kinds=_BUILTIN_KINDS)
        dst = [regs[p][0] for p in self.dst_regs if p in regs]
        src = [regs[p][0] for p in self.src_regs if p in regs]
        cfg = _resolve_config_regs(self.config_regs)
        return ResolvedFlow(dst_regs=dst, src_regs=src, config_regs=cfg)
```

**Step 3: Add `get_encoding` convenience method to `ISARegistry`**

After `is_known` method (line 313):

```python
    def get_encoding(self, mnemonic: str) -> InstructionFormat | None:
        """Get encoding format for a mnemonic, or None if unknown or not set."""
        flow = self._flows.get(mnemonic)
        if flow is None:
            return None
        return flow.encoding
```

**Step 4: Run existing tests to verify no regressions**

Run: `cd tools && python -m pytest dfg/tests/ -v`
Expected: All existing tests PASS (encoding field is optional, None default)

**Step 5: Run fusion tests**

Run: `cd tools && python -m pytest fusion/tests/ -v`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add tools/dfg/instruction.py
git commit -m "feat(dfg): add InstructionFormat dataclass and encoding field to RegisterFlow"
```

---

## Task 2: Extend `gen_isadesc.py` to extract encoding metadata

**Files:**
- Modify: `tools/dfg/gen_isadesc.py`

**Step 1: Add encoding extraction function**

Add after the `_extract_flow` function (after line 197):

```python
# ---------------------------------------------------------------------------
# Encoding extraction
# ---------------------------------------------------------------------------

def _extract_encoding(entry: dict, reg_class: str) -> InstructionFormat:
    """Extract InstructionFormat from a tablegen instruction entry.

    The Inst array is 32 elements, LSB-first: inst[i] = bit i of the
    32-bit instruction word.

    RISC-V field layout (bit indices):
      [6:0]   opcode
      [11:7]  rd
      [14:12] funct3
      [19:15] rs1
      [24:20] rs2
      [31:25] funct7
    """
    from dfg.instruction import InstructionFormat

    inst = entry.get("Inst", [])
    if len(inst) < 32:
        return InstructionFormat(format_type="R", opcode=0, reg_class=reg_class)

    def bv(el):
        return el if isinstance(el, int) else None

    def extract_field(start: int, width: int) -> int | None:
        bits = [bv(inst[start + i]) for i in range(width)]
        if any(b is None for b in bits):
            return None
        return sum(b * (1 << i) for i, b in enumerate(bits))

    def has_var_field(start: int, width: int) -> bool:
        return any(bv(inst[start + i]) is None for i in range(width))

    # Opcode from dedicated field (verified LSB-first)
    opcode_bits = entry.get("Opcode", [0] * 7)
    opcode = int("".join(str(b) for b in reversed(opcode_bits)), 2)

    # Fixed fields
    funct3 = extract_field(12, 3)
    funct7 = extract_field(25, 7)

    # Register field presence
    has_rd = has_var_field(7, 5)
    has_rs1 = has_var_field(15, 5)
    has_rs2 = has_var_field(20, 5)
    has_rs3 = has_var_field(27, 5)

    # Memory access
    may_load = bool(entry.get("mayLoad", 0))
    may_store = bool(entry.get("mayStore", 0))

    # Determine format type and immediate
    has_imm = not has_rs2 and not has_rs3
    imm_bits = 12 if has_imm and not may_load and not may_store else 0
    if may_store and not has_rd:
        imm_bits = 12
        has_imm = True
    if may_load and has_rd:
        imm_bits = 12
        has_imm = True

    if has_rs3:
        format_type = "R4"
    elif may_store:
        format_type = "S"
    elif may_load:
        format_type = "I"
    elif has_imm:
        format_type = "I"
    else:
        format_type = "R"

    return InstructionFormat(
        format_type=format_type,
        opcode=opcode,
        funct3=funct3,
        funct7=funct7,
        has_rd=has_rd,
        has_rs1=has_rs1,
        has_rs2=has_rs2,
        has_rs3=has_rs3,
        has_imm=has_imm,
        imm_bits=imm_bits,
        may_load=may_load,
        may_store=may_store,
        reg_class=reg_class,
    )
```

**Step 2: Add reg_class mapping**

Add a new mapping dict after `EXTENSION_PREDICATES` (after line 125):

```python
# Maps --ext flag to the register class used by most instructions in that extension.
EXTENSION_REG_CLASS: dict[str, str] = {
    "F": "float",
    "M": "integer",
    "V": "vector",
}
```

**Step 3: Update `_generate_module` to include encoding**

Replace the `_generate_module` function (lines 225-262) with:

```python
def _generate_module(
    ext: str,
    entries: list[tuple[str, list[str], list[str], InstructionFormat | None]],
) -> str:
    """Generate the Python source for an ISA descriptor module."""
    from dfg.instruction import InstructionFormat

    ext_lower = ext.lower()
    var_name = f"ALL_RV64{ext}"

    lines: list[str] = []
    lines.append(f'"""RV64{ext} extension instruction register flow definitions."""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow")
    lines.append("")
    lines.append("")

    # Emit the instruction list
    lines.append(f"# RV64{ext} instructions (auto-generated by gen_isadesc.py)")
    lines.append(f"{var_name}: list[tuple[str, RegisterFlow]] = [")

    for mnemonic, dst, src, encoding in entries:
        dst_repr = repr(dst)
        src_repr = repr(src)
        if encoding is not None:
            enc_repr = (
                f"InstructionFormat("
                f"format_type={encoding.format_type!r}, "
                f"opcode=0x{encoding.opcode:02x}, "
                f"funct3={encoding.funct3!r}, "
                f"funct7={encoding.funct7!r}, "
                f"has_rd={encoding.has_rd!r}, "
                f"has_rs1={encoding.has_rs1!r}, "
                f"has_rs2={encoding.has_rs2!r}, "
                f"has_rs3={encoding.has_rs3!r}, "
                f"has_imm={encoding.has_imm!r}, "
                f"imm_bits={encoding.imm_bits!r}, "
                f"may_load={encoding.may_load!r}, "
                f"may_store={encoding.may_store!r}, "
                f"reg_class={encoding.reg_class!r}"
                f")"
            )
            lines.append(f'    ("{mnemonic}", RegisterFlow({dst_repr}, {src_repr}, encoding={enc_repr})),')
        else:
            lines.append(f'    ("{mnemonic}", RegisterFlow({dst_repr}, {src_repr})),')

    lines.append("]")
    lines.append("")
    lines.append("")

    # Emit build_registry
    lines.append("")
    lines.append("def build_registry(registry: ISARegistry) -> None:")
    lines.append(
        f'    """Register all RV64{ext} instructions into the given ISA registry."""'
    )
    lines.append(f'    registry.load_extension("{ext}", {var_name})')
    lines.append("")
    lines.append("")

    return "\n".join(lines)
```

**Step 4: Update `main` to extract and pass encoding**

Replace the loop in `main` (lines 302-318) with:

```python
    reg_class = EXTENSION_REG_CLASS.get(args.ext, "integer")

    # Filter and extract instructions for the given extension
    entries: list[tuple[str, list[str], list[str], InstructionFormat | None]] = []
    for name, entry in sorted(data.items()):
        if not isinstance(entry, dict):
            continue
        if not _should_include(name):
            continue
        if not _has_extension(entry, args.ext):
            continue

        mnemonic = llvm_name_to_mnemonic(name)
        dst, src = _extract_flow(entry)

        # Skip instructions with no register flow at all (pure system)
        if not dst and not src:
            continue

        encoding = _extract_encoding(entry, reg_class)
        entries.append((mnemonic, dst, src, encoding))
```

Also add the import at the top of `main()`:

```python
    from dfg.instruction import InstructionFormat
```

**Step 5: Run generator for V-extension (highest priority)**

Run: `cd tools && python dfg/gen_isadesc.py dfg/riscv_instrs.json --ext V -o dfg/isadesc/rv64v.py`
Expected: "Generated ... with N instructions" (N should be ~376)

**Step 6: Run generator for F, M extensions**

Run: `cd tools && python dfg/gen_isadesc.py dfg/riscv_instrs.json --ext F -o dfg/isadesc/rv64f.py`
Expected: "Generated ... with N instructions" (N ~41)

Run: `cd tools && python dfg/gen_isadesc.py dfg/riscv_instrs.json --ext M -o dfg/isadesc/rv64m.py`
Expected: "Generated ... with N instructions" (N ~14)

**Step 7: Run all existing tests**

Run: `cd tools && python -m pytest dfg/tests/ fusion/tests/ -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add tools/dfg/gen_isadesc.py tools/dfg/isadesc/rv64v.py tools/dfg/isadesc/rv64f.py tools/dfg/isadesc/rv64m.py
git commit -m "feat(dfg): extend gen_isadesc to generate InstructionFormat encoding metadata"
```

---

## Task 3: Add encoding metadata to manually-written `rv64i.py`

**Files:**
- Modify: `tools/dfg/isadesc/rv64i.py`

Since `rv64i.py` is manually written (not auto-generated), encoding metadata must be added by hand. The approach: add a helper `_ef()` (encoding format) function at the top, then wrap each RegisterFlow call.

**Step 1: Add import and helper function**

Update the imports at the top:

```python
from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow
```

Add after imports:

```python
def _ef(
    format_type: str,
    opcode: int,
    funct3: int | None = None,
    funct7: int | None = None,
    *,
    has_rd: bool = True,
    has_rs1: bool = True,
    has_rs2: bool = True,
    has_imm: bool = False,
    imm_bits: int = 0,
    may_load: bool = False,
    may_store: bool = False,
) -> InstructionFormat:
    """Shorthand for creating an integer-class InstructionFormat."""
    return InstructionFormat(
        format_type=format_type,
        opcode=opcode,
        funct3=funct3,
        funct7=funct7,
        has_rd=has_rd,
        has_rs1=has_rs1,
        has_rs2=has_rs2,
        has_imm=has_imm,
        imm_bits=imm_bits,
        may_load=may_load,
        may_store=may_store,
        reg_class="integer",
    )
```

**Step 2: Add encoding to R_TYPE (add, sub, etc.)**

R-type: opcode=0x33, funct3 varies, funct7=0x00 (except sub/sra=0x20)

Replace the R_TYPE list with:

```python
R_TYPE: list[tuple[str, RegisterFlow]] = [
    ("add",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x0, 0x00))),
    ("sub",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x0, 0x20))),
    ("and",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x7, 0x00))),
    ("or",   RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x6, 0x00))),
    ("xor",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x4, 0x00))),
    ("sll",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x1, 0x00))),
    ("srl",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x5, 0x00))),
    ("sra",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x5, 0x20))),
    ("slt",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x2, 0x00))),
    ("sltu", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x3, 0x00))),
]
```

**Step 3: Add encoding to I_TYPE_IMM, I_TYPE_SHIFT, W_R_TYPE, W_I_TYPE**

```python
I_TYPE_IMM: list[tuple[str, RegisterFlow]] = [
    ("addi",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x0, has_imm=True, imm_bits=12))),
    ("andi",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x7, has_imm=True, imm_bits=12))),
    ("ori",   RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x6, has_imm=True, imm_bits=12))),
    ("xori",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x4, has_imm=True, imm_bits=12))),
    ("slti",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_imm=True, imm_bits=12))),
    ("sltiu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x3, has_imm=True, imm_bits=12))),
]

I_TYPE_SHIFT: list[tuple[str, RegisterFlow]] = [
    ("slli", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x1, 0x00, has_imm=True, imm_bits=5))),
    ("srli", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x5, 0x00, has_imm=True, imm_bits=5))),
    ("srai", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x5, 0x20, has_imm=True, imm_bits=5))),
]

W_R_TYPE: list[tuple[str, RegisterFlow]] = [
    ("addw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x0, 0x00))),
    ("subw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x0, 0x20))),
    ("sllw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x1, 0x00))),
    ("srlw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x5, 0x00))),
    ("sraw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x5, 0x20))),
]

W_I_TYPE: list[tuple[str, RegisterFlow]] = [
    ("addiw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x0, has_imm=True, imm_bits=12))),
    ("slliw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x1, 0x00, has_imm=True, imm_bits=5))),
    ("srliw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x5, 0x00, has_imm=True, imm_bits=5))),
    ("sraiw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x5, 0x20, has_imm=True, imm_bits=5))),
]
```

**Step 4: Add encoding to LOADS and STORES**

```python
LOADS: list[tuple[str, RegisterFlow]] = [
    ("lb",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x0, may_load=True, has_imm=True, imm_bits=12))),
    ("lbu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x4, may_load=True, has_imm=True, imm_bits=12))),
    ("lh",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x1, may_load=True, has_imm=True, imm_bits=12))),
    ("lhu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x5, may_load=True, has_imm=True, imm_bits=12))),
    ("lw",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x2, may_load=True, has_imm=True, imm_bits=12))),
    ("lwu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x6, may_load=True, has_imm=True, imm_bits=12))),
    ("ld",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x3, may_load=True, has_imm=True, imm_bits=12))),
]

STORES: list[tuple[str, RegisterFlow]] = [
    ("sb", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x0, may_store=True, has_rd=False, has_imm=True, imm_bits=12))),
    ("sh", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x1, may_store=True, has_rd=False, has_imm=True, imm_bits=12))),
    ("sw", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x2, may_store=True, has_rd=False, has_imm=True, imm_bits=12))),
    ("sd", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x3, may_store=True, has_rd=False, has_imm=True, imm_bits=12))),
]
```

**Step 5: Add encoding to BRANCHES, JUMPS, PSEUDO, UPPER_IMM, SYSTEM**

```python
BRANCHES_2REG: list[tuple[str, RegisterFlow]] = [
    ("beq",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x0, has_rd=False, has_imm=True, imm_bits=12))),
    ("bne",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x1, has_rd=False, has_imm=True, imm_bits=12))),
    ("blt",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x4, has_rd=False, has_imm=True, imm_bits=12))),
    ("bge",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x5, has_rd=False, has_imm=True, imm_bits=12))),
    ("bltu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x6, has_rd=False, has_imm=True, imm_bits=12))),
    ("bgeu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x7, has_rd=False, has_imm=True, imm_bits=12))),
]

BRANCHES_1REG: list[tuple[str, RegisterFlow]] = [
    ("beqz",  RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x0, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("bnez",  RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x1, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("bgtz",  RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x4, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("blez",  RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x5, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("bgt",   RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x6, has_rd=False, has_imm=True, imm_bits=12))),
    ("ble",   RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x7, has_rd=False, has_imm=True, imm_bits=12))),
    ("bgtu",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x7, has_rd=False, has_imm=True, imm_bits=12))),
    ("bleu",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x6, has_rd=False, has_imm=True, imm_bits=12))),
]

JUMPS: list[tuple[str, RegisterFlow]] = [
    ("jal",  RegisterFlow(["rd"], [], encoding=_ef("J", 0x6f, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("jalr", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x67, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
]

PSEUDO: list[tuple[str, RegisterFlow]] = [
    ("j",    RegisterFlow([], [], encoding=_ef("J", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("ret",  RegisterFlow([], [], encoding=_ef("I", 0x67, 0x0, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("nop",  RegisterFlow([], [], encoding=_ef("I", 0x13, 0x0, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("mv",   RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
    ("li",   RegisterFlow(["rd"], [], encoding=_ef("I", 0x13, 0x0, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("la",   RegisterFlow(["rd"], [], encoding=_ef("U", 0x17, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("not",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x4, has_rs2=False, has_imm=True, imm_bits=12))),
    ("neg",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("R", 0x33, 0x0, 0x20))),
    ("negw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("R", 0x3b, 0x0, 0x20))),
    ("seqz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x3, has_rs2=False, has_imm=True, imm_bits=12))),
    ("snez", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x7, has_rs2=False, has_imm=True, imm_bits=12))),
    ("sltz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_rs2=False, has_imm=True, imm_bits=12))),
    ("sgtz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_rs2=False, has_imm=True, imm_bits=12))),
    ("call", RegisterFlow([], [], encoding=_ef("U", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("tail", RegisterFlow([], [], encoding=_ef("U", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
]

UPPER_IMM: list[tuple[str, RegisterFlow]] = [
    ("auipc", RegisterFlow(["rd"], [], encoding=_ef("U", 0x17, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("lui",   RegisterFlow(["rd"], [], encoding=_ef("U", 0x37, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
]

SYSTEM: list[tuple[str, RegisterFlow]] = [
    ("ecall",   RegisterFlow([], [], encoding=_ef("I", 0x73, 0x0, has_rd=False, has_rs1=False, has_rs2=False))),
    ("ebreak",  RegisterFlow([], [], encoding=_ef("I", 0x73, 0x0, has_rd=False, has_rs1=False, has_rs2=False))),
    ("fence",   RegisterFlow([], [], encoding=_ef("I", 0x0f, 0x0, has_rd=False, has_rs2=False, has_imm=True, imm_bits=4))),
    ("fence.i", RegisterFlow([], [], encoding=_ef("I", 0x0f, 0x1, has_rd=False, has_rs1=False, has_rs2=False))),
]
```

**Step 6: Run all tests**

Run: `cd tools && python -m pytest dfg/tests/ fusion/tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add tools/dfg/isadesc/rv64i.py
git commit -m "feat(dfg): add InstructionFormat encoding to rv64i ISA descriptor"
```

---

## Task 4: Build `Verdict` dataclass and `ConstraintChecker` (tests first)

**Files:**
- Create: `tools/fusion/constraints.py`
- Create: `tools/fusion/tests/test_constraints.py`

**Step 1: Write failing tests for ConstraintChecker**

Create `tools/fusion/tests/test_constraints.py`:

```python
"""Unit tests for the hardware constraint model."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow
from fusion.constraints import ConstraintChecker, Verdict


def _make_registry() -> ISARegistry:
    """Build a registry with a few instructions that have encoding metadata."""
    registry = ISARegistry()
    instructions = [
        # Float R-type: fadd, fmul, fsub
        ("fadd.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x00, reg_class="float"))),
        ("fsub.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x08, reg_class="float"))),
        ("fmul.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x1, 0x00, reg_class="float"))),
        ("fdiv.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x0c, reg_class="float"))),
        ("fmadd.s", RegisterFlow(["frd"], ["frs1", "frs2", "frs3"],
            encoding=InstructionFormat("R4", 0x43, reg_class="float", has_rs3=True))),
        # Float load/store
        ("flw", RegisterFlow(["frd"], ["rs1"],
            encoding=InstructionFormat("I", 0x07, 0x2, may_load=True, has_rs2=False, reg_class="float"))),
        ("fsw", RegisterFlow([], ["frs2", "rs1"],
            encoding=InstructionFormat("S", 0x27, 0x2, may_store=True, has_rd=False, reg_class="float"))),
        # Integer R-type
        ("add", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x00, reg_class="integer"))),
        ("mul", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x01, reg_class="integer"))),
        # V-extension
        ("vadd.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x00, reg_class="vector"))),
        ("vadd.vx", RegisterFlow(["vrd"], ["vrs2", "rs1"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x04, reg_class="vector"))),
        ("vmul.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x2, 0x02, reg_class="vector"))),
        ("vadd.vi", RegisterFlow(["vrd"], ["vrs2"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x03, has_rs1=False, reg_class="vector"))),
        ("vsetvli", RegisterFlow(["rd"], ["rs1"],
            encoding=InstructionFormat("V", 0x57, 0x7, 0x00, has_rs2=False,
                                      has_imm=True, imm_bits=12, reg_class="vector"))),
        # Integer with imm
        ("addi", RegisterFlow(["rd"], ["rs1"],
            encoding=InstructionFormat("I", 0x13, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
    ]
    for mnemonic, flow in instructions:
        registry.register(mnemonic, flow)
    return registry


class TestVerdict(unittest.TestCase):
    """Test Verdict dataclass."""

    def test_feasible_verdict(self):
        v = Verdict(status="feasible", reasons=[], violations=[])
        self.assertEqual(v.status, "feasible")
        self.assertFalse(v.violations)

    def test_infeasible_verdict(self):
        v = Verdict(status="infeasible",
                    reasons=["load instruction cannot fuse"],
                    violations=["no_load_store"])
        self.assertEqual(v.status, "infeasible")
        self.assertIn("no_load_store", v.violations)


class TestConstraintCheckerFeasible(unittest.TestCase):
    """Test patterns that should be feasible."""

    def setUp(self):
        self.checker = ConstraintChecker(_make_registry())

    def test_float_add_mul_chain(self):
        """fadd.s → fmul.s: same opcode, 2 unique src (frs1, frs2 reused via chain), 1 dst."""
        pattern = {
            "opcodes": ["fadd.s", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_int_add_mul_chain(self):
        """add → mul: same opcode group, straightforward fusion."""
        pattern = {
            "opcodes": ["add", "mul"],
            "register_class": "integer",
            "chain_registers": [["rd", "rs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_vext_vadd_vmul_chain(self):
        """vadd.vv → vmul.vv: same V opcode, vector class."""
        pattern = {
            "opcodes": ["vadd.vv", "vmul.vv"],
            "register_class": "vector",
            "chain_registers": [["vrd", "vrs2"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_float_3_instruction_chain(self):
        """fadd.s → fmul.s → fsub.s: 3-instr chain, same opcode."""
        pattern = {
            "opcodes": ["fadd.s", "fmul.s", "fsub.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"], ["frd", "frs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")


class TestConstraintCheckerInfeasible(unittest.TestCase):
    """Test patterns that should be infeasible."""

    def setUp(self):
        self.checker = ConstraintChecker(_make_registry())

    def test_load_in_chain(self):
        """flw → fmul.s: load instruction cannot fuse."""
        pattern = {
            "opcodes": ["flw", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("no_load_store", verdict.violations)

    def test_store_in_chain(self):
        """fmul.s → fsw: store instruction cannot fuse."""
        pattern = {
            "opcodes": ["fmul.s", "fsw"],
            "register_class": "float",
            "chain_registers": [["frd", "frs2"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("no_load_store", verdict.violations)

    def test_cross_register_class(self):
        """add → fadd.s: cross-register-class is infeasible (defensive check)."""
        pattern = {
            "opcodes": ["add", "fadd.s"],
            "register_class": "integer",  # mixed
            "chain_registers": [["rd", "frs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("register_class_mismatch", verdict.violations)

    def test_config_register_write(self):
        """vsetvli → vadd.vv: config register write breaks pipeline."""
        pattern = {
            "opcodes": ["vsetvli", "vadd.vv"],
            "register_class": "vector",
            "chain_registers": [["rd", "rs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("no_config_write", verdict.violations)

    def test_unknown_instruction(self):
        """Pattern with unknown mnemonic is infeasible."""
        pattern = {
            "opcodes": ["fadd.s", "unknown_op"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("unknown_instruction", verdict.violations)


class TestConstraintCheckerConstrained(unittest.TestCase):
    """Test patterns that should be constrained (not hard-rejected)."""

    def setUp(self):
        self.checker = ConstraintChecker(_make_registry())

    def test_immediate_in_chain(self):
        """addi → add: immediate field present, constrained but not infeasible."""
        pattern = {
            "opcodes": ["addi", "add"],
            "register_class": "integer",
            "chain_registers": [["rd", "rs1"]],
        }
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "constrained")
        self.assertIn("has_immediate", verdict.violations)

    def test_different_opcode_same_regclass(self):
        """vadd.vv → vsetvli: different opcode groups, constrained."""
        # Note: vsetvli also has config write, so it's actually infeasible.
        # Test with a different pair instead.
        pattern = {
            "opcodes": ["vadd.vi", "vadd.vx"],
            "register_class": "vector",
            "chain_registers": [["vrd", "vrs2"]],
        }
        verdict = self.checker.check(pattern)
        # Same opcode 0x57, so should be feasible
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_missing_encoding_metadata(self):
        """Instruction without encoding metadata is constrained."""
        registry = ISARegistry()
        registry.register("fadd.s", RegisterFlow(["frd"], ["frs1", "frs2"]))
        registry.register("fmul.s", RegisterFlow(["frd"], ["frs1", "frs2"]))
        checker = ConstraintChecker(registry)
        pattern = {
            "opcodes": ["fadd.s", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
        }
        verdict = checker.check(pattern)
        self.assertEqual(verdict.status, "constrained")
        self.assertIn("missing_encoding", verdict.violations)


class TestOperandCounting(unittest.TestCase):
    """Test unique source/destination register counting."""

    def setUp(self):
        self.checker = ConstraintChecker(_make_registry())

    def test_fmadd_creates_3_unique_sources(self):
        """fmadd.s has 3 source registers (frs1, frs2, frs3)."""
        pattern = {
            "opcodes": ["fmadd.s"],
            "register_class": "float",
            "chain_registers": [],
        }
        # Single-instruction pattern: 3 sources, 1 dst — at the limit
        # This tests that counting works, not the verdict
        enc = self.checker._registry.get_encoding("fmadd.s")
        self.assertIsNotNone(enc)
        src_count = int(enc.has_rs1) + int(enc.has_rs2) + int(enc.has_rs3)
        self.assertEqual(src_count, 3)
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_constraints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion.constraints'`

**Step 3: Implement ConstraintChecker**

Create `tools/fusion/constraints.py`:

```python
"""Hardware constraint model for fusion candidate evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from dfg.instruction import ISARegistry

logger = logging.getLogger("fusion.constraints")


@dataclass
class Verdict:
    """Feasibility verdict for a fusion pattern.

    Attributes:
        status: One of "feasible", "constrained", or "infeasible".
        reasons: Human-readable rationale for the verdict.
        violations: Names of specific constraints violated.
    """

    status: Literal["feasible", "constrained", "infeasible"]
    reasons: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


# Constraints that make a pattern hard-infeasible (cannot fuse in any scenario)
_HARD_VIOLATIONS = frozenset({
    "no_load_store",
    "register_class_mismatch",
    "no_config_write",
    "unknown_instruction",
    "too_many_destinations",
    "too_many_sources",
})

# Constraints that make a pattern constrained (might be feasible with effort)
_SOFT_VIOLATIONS = frozenset({
    "has_immediate",
    "missing_encoding",
    "different_opcode",
})


class ConstraintChecker:
    """Checks hardware feasibility of fusion patterns.

    Args:
        registry: ISA registry with encoding metadata on RegisterFlow entries.
    """

    def __init__(self, registry: ISARegistry) -> None:
        self._registry = registry

    def check(self, pattern: dict) -> Verdict:
        """Evaluate a fusion pattern against hardware constraints.

        Args:
            pattern: Dict with keys "opcodes", "register_class", "chain_registers".

        Returns:
            A Verdict with status, reasons, and violation names.
        """
        violations: list[str] = []
        reasons: list[str] = []

        opcodes = pattern["opcodes"]
        reg_class = pattern.get("register_class", "unknown")

        # Look up encoding for each instruction
        encodings = []
        for opcode in opcodes:
            flow = self._registry.get_flow(opcode)
            if flow is None:
                violations.append("unknown_instruction")
                reasons.append(f"Instruction '{opcode}' not found in ISA registry")
                encodings.append(None)
                continue
            enc = flow.encoding
            if enc is None:
                violations.append("missing_encoding")
                reasons.append(f"Instruction '{opcode}' has no encoding metadata")
                encodings.append(None)
                continue
            encodings.append(enc)

        # If any instruction is unknown, immediately infeasible
        if "unknown_instruction" in violations:
            return Verdict(status="infeasible", reasons=reasons, violations=violations)

        valid_encodings = [e for e in encodings if e is not None]
        if not valid_encodings:
            return Verdict(status="constrained", reasons=reasons, violations=violations)

        # Check register class compatibility
        for enc in valid_encodings:
            if enc.reg_class != reg_class:
                violations.append("register_class_mismatch")
                reasons.append(
                    f"Register class mismatch: pattern says '{reg_class}' "
                    f"but '{opcodes[valid_encodings.index(enc)]}' is '{enc.reg_class}'"
                )
                break

        # Check for load/store
        for i, enc in enumerate(valid_encodings):
            if enc.may_load or enc.may_store:
                violations.append("no_load_store")
                reasons.append(
                    f"Instruction '{opcodes[i]}' is a load/store "
                    f"(load={enc.may_load}, store={enc.may_store}) "
                    f"— memory access cannot be fused"
                )
                break

        # Check for config register writes
        for i, opcode in enumerate(opcodes):
            flow = self._registry.get_flow(opcode)
            if flow and flow.config_regs:
                violations.append("no_config_write")
                reasons.append(
                    f"Instruction '{opcode}' writes config registers "
                    f"({flow.config_regs}) — pipeline state change"
                )
                break

        # Count unique sources and destinations across the chain
        total_src_fields = 0
        total_dst_fields = 0
        has_any_imm = False
        opcodes_seen: set[int] = set()

        for enc in valid_encodings:
            opcodes_seen.add(enc.opcode)
            total_src_fields += int(enc.has_rs1) + int(enc.has_rs2) + int(enc.has_rs3)
            total_dst_fields += int(enc.has_rd)
            if enc.has_imm:
                has_any_imm = True

        # Operand count limits (hard)
        if total_dst_fields > 1:
            violations.append("too_many_destinations")
            reasons.append(
                f"Chain has {total_dst_fields} destination registers "
                f"(max 1 for 32-bit encoding)"
            )
        if total_src_fields > 3:
            violations.append("too_many_sources")
            reasons.append(
                f"Chain has {total_src_fields} source register fields "
                f"(max 3 for rs1/rs2/rs3)"
            )

        # Opcode compatibility (soft)
        if len(opcodes_seen) > 1:
            violations.append("different_opcode")
            reasons.append(
                f"Instructions use different opcodes: "
                f"{', '.join(f'0x{o:02x}' for o in sorted(opcodes_seen))}"
            )

        # Immediate presence (soft)
        if has_any_imm:
            violations.append("has_immediate")
            reasons.append(
                "Chain contains instructions with immediates — "
                "fused encoding may not have space for all immediates"
            )

        # Determine verdict
        has_hard = any(v in _HARD_VIOLATIONS for v in violations)
        has_soft = any(v in _SOFT_VIOLATIONS for v in violations)

        if has_hard:
            status = "infeasible"
        elif has_soft:
            status = "constrained"
        else:
            status = "feasible"

        if status == "feasible":
            reasons.append("Pattern passes all hardware constraint checks")

        return Verdict(status=status, reasons=reasons, violations=violations)
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_constraints.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tools/fusion/constraints.py tools/fusion/tests/test_constraints.py
git commit -m "feat(fusion): add ConstraintChecker and Verdict with hardware constraint model"
```

---

## Task 5: Build `Scorer` module (tests first)

**Files:**
- Create: `tools/fusion/scorer.py`
- Create: `tools/fusion/tests/test_scorer.py`

**Step 1: Write failing tests for Scorer**

Create `tools/fusion/tests/test_scorer.py`:

```python
"""Unit tests for the fusion scoring function."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow
from fusion.constraints import ConstraintChecker, Verdict
from fusion.scorer import Scorer, DEFAULT_WEIGHTS


def _make_registry() -> ISARegistry:
    """Minimal registry with encoding for scorer tests."""
    registry = ISARegistry()
    instructions = [
        ("fadd.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x00, reg_class="float"))),
        ("fmul.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x1, 0x00, reg_class="float"))),
        ("fsub.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x08, reg_class="float"))),
        ("vadd.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x00, reg_class="vector"))),
        ("vmul.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x2, 0x02, reg_class="vector"))),
        ("add", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x00, reg_class="integer"))),
        ("mul", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x01, reg_class="integer"))),
        ("flw", RegisterFlow(["frd"], ["rs1"],
            encoding=InstructionFormat("I", 0x07, 0x2, may_load=True, has_rs2=False, reg_class="float"))),
    ]
    for mnemonic, flow in instructions:
        registry.register(mnemonic, flow)
    return registry


class TestFreqScore(unittest.TestCase):
    """Test frequency normalization."""

    def setUp(self):
        self.scorer = Scorer(_make_registry(), max_frequency=100000)

    def test_max_frequency_scores_one(self):
        score = self.scorer._freq_score(100000)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_zero_frequency_scores_zero(self):
        score = self.scorer._freq_score(0)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_half_frequency_is_less_than_one(self):
        score = self.scorer._freq_score(50000)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_log_normalization_compresses_range(self):
        """Log normalization should compress large range."""
        s_low = self.scorer._freq_score(1000)
        s_high = self.scorer._freq_score(100000)
        # With log, 1000/100000 = 1% but log-normalized should be > 0.5
        self.assertGreater(s_low, 0.5)


class TestTightScore(unittest.TestCase):
    """Test dependency tightness scoring."""

    def setUp(self):
        self.scorer = Scorer(_make_registry())

    def test_full_density_single_chain(self):
        """2-instr chain with 1 RAW edge: density = 1/1 = 1.0."""
        score = self.scorer._tight_score(
            chain_registers=[["frd", "frs1"]], length=2
        )
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_no_edges_zero_density(self):
        """Chain with no RAW edges: density = 0."""
        score = self.scorer._tight_score(
            chain_registers=[], length=2
        )
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_three_instr_chain_boost(self):
        """3-instr chain with 2 RAW edges gets chain_factor=1.2."""
        score = self.scorer._tight_score(
            chain_registers=[["frd", "frs1"], ["frd", "frs1"]], length=3
        )
        # density = 2/2 = 1.0, chain_factor = 1.2, score = 1.0 * 1.2 = 1.2 (capped at 1.0)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_two_instr_no_chain_factor(self):
        """2-instr chain: chain_factor = 1.0."""
        score = self.scorer._tight_score(
            chain_registers=[["rd", "rs1"]], length=2
        )
        self.assertAlmostEqual(score, 1.0, places=5)


class TestHwScore(unittest.TestCase):
    """Test hardware feasibility score."""

    def test_feasible_scores_one(self):
        score = Scorer._hw_score(Verdict(status="feasible", reasons=[], violations=[]))
        self.assertAlmostEqual(score, 1.0)

    def test_constrained_scores_half(self):
        score = Scorer._hw_score(Verdict(status="constrained", reasons=["imm"], violations=["has_immediate"]))
        self.assertAlmostEqual(score, 0.5)

    def test_infeasible_scores_zero(self):
        score = Scorer._hw_score(Verdict(status="infeasible", reasons=["load"], violations=["no_load_store"]))
        self.assertAlmostEqual(score, 0.0)


class TestScorePattern(unittest.TestCase):
    """Test the full score_pattern method."""

    def setUp(self):
        self.scorer = Scorer(_make_registry(), max_frequency=100000)

    def test_feasible_pattern_has_positive_score(self):
        pattern = {
            "opcodes": ["fadd.s", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
            "total_frequency": 50000,
            "occurrence_count": 10,
        }
        result = self.scorer.score_pattern(pattern)
        self.assertGreater(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertIn("score_breakdown", result)
        self.assertIn("tightness", result)
        self.assertIn("hardware", result)

    def test_infeasible_pattern_gets_zero_hw(self):
        pattern = {
            "opcodes": ["flw", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
            "total_frequency": 50000,
            "occurrence_count": 10,
        }
        result = self.scorer.score_pattern(pattern)
        self.assertAlmostEqual(result["score_breakdown"]["hw_score"], 0.0)

    def test_high_frequency_ranks_higher(self):
        pattern_low = {
            "opcodes": ["fadd.s", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
            "total_frequency": 1000,
            "occurrence_count": 1,
        }
        pattern_high = {
            "opcodes": ["fadd.s", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
            "total_frequency": 90000,
            "occurrence_count": 100,
        }
        r_low = self.scorer.score_pattern(pattern_low)
        r_high = self.scorer.score_pattern(pattern_high)
        self.assertGreater(r_high["score"], r_low["score"])

    def test_custom_weights(self):
        scorer = Scorer(_make_registry(), max_frequency=100000,
                        weights={"frequency": 1.0, "tightness": 0.0, "hardware": 0.0})
        pattern = {
            "opcodes": ["fadd.s", "fmul.s"],
            "register_class": "float",
            "chain_registers": [["frd", "frs1"]],
            "total_frequency": 50000,
            "occurrence_count": 10,
        }
        result = scorer.score_pattern(pattern)
        # With only frequency weight, score = freq_score
        expected = scorer._freq_score(50000)
        self.assertAlmostEqual(result["score"], expected, places=5)


class TestScorePatterns(unittest.TestCase):
    """Test batch scoring of patterns."""

    def setUp(self):
        self.scorer = Scorer(_make_registry(), max_frequency=100000)

    def test_batch_returns_sorted_results(self):
        patterns = [
            {"opcodes": ["add", "mul"], "register_class": "integer",
             "chain_registers": [["rd", "rs1"]], "total_frequency": 1000, "occurrence_count": 1},
            {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
             "chain_registers": [["frd", "frs1"]], "total_frequency": 90000, "occurrence_count": 100},
            {"opcodes": ["fadd.s", "fsub.s"], "register_class": "float",
             "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 50},
        ]
        results = self.scorer.score_patterns(patterns)
        # Should be sorted by score descending
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_n_filtering(self):
        patterns = [
            {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
             "chain_registers": [["frd", "frs1"]], "total_frequency": 90000, "occurrence_count": 100},
            {"opcodes": ["fadd.s", "fsub.s"], "register_class": "float",
             "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 50},
            {"opcodes": ["add", "mul"], "register_class": "integer",
             "chain_registers": [["rd", "rs1"]], "total_frequency": 1000, "occurrence_count": 1},
        ]
        results = self.scorer.score_patterns(patterns, top=2)
        self.assertEqual(len(results), 2)

    def test_min_score_filtering(self):
        patterns = [
            {"opcodes": ["flw", "fmul.s"], "register_class": "float",
             "chain_registers": [["frd", "frs1"]], "total_frequency": 90000, "occurrence_count": 100},
            {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
             "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 50},
        ]
        results = self.scorer.score_patterns(patterns, min_score=0.1)
        # Infeasible pattern should be filtered out (score near 0)
        statuses = [r["hardware"]["status"] for r in results]
        self.assertNotIn("infeasible", statuses)
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion.scorer'`

**Step 3: Implement Scorer**

Create `tools/fusion/scorer.py`:

```python
"""Fusion candidate scoring function."""

from __future__ import annotations

import math
import logging
from pathlib import Path

from dfg.instruction import ISARegistry

from fusion.constraints import ConstraintChecker, Verdict

logger = logging.getLogger("fusion.scorer")

DEFAULT_WEIGHTS = {"frequency": 0.4, "tightness": 0.3, "hardware": 0.3}

# Chain length multiplier for dependency tightness
_CHAIN_FACTOR = {2: 1.0, 3: 1.2}


class Scorer:
    """Score and rank fusion patterns.

    Args:
        registry: ISA registry with encoding metadata.
        max_frequency: Maximum frequency across all patterns (for normalization).
        weights: Dict with "frequency", "tightness", "hardware" weight values.
    """

    def __init__(
        self,
        registry: ISARegistry,
        max_frequency: int,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._checker = ConstraintChecker(registry)
        self._max_freq = max(max_frequency, 1)
        self._weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)

    def _freq_score(self, frequency: int) -> float:
        """Log-normalize execution frequency to [0, 1]."""
        if frequency <= 0:
            return 0.0
        return math.log(1 + frequency) / math.log(1 + self._max_freq)

    @staticmethod
    def _tight_score(chain_registers: list, length: int) -> float:
        """Compute dependency tightness score.

        Args:
            chain_registers: List of (dst_role, src_role) pairs for each
                consecutive instruction pair in the chain.
            length: Number of instructions in the chain.
        """
        n_pairs = length - 1
        if n_pairs <= 0:
            return 0.0
        raw_density = len(chain_registers) / n_pairs
        chain_factor = _CHAIN_FACTOR.get(length, 1.0)
        return min(raw_density * chain_factor, 1.0)

    @staticmethod
    def _hw_score(verdict: Verdict) -> float:
        """Map verdict status to a numeric score."""
        if verdict.status == "feasible":
            return 1.0
        if verdict.status == "constrained":
            return 0.5
        return 0.0  # infeasible

    def score_pattern(self, pattern: dict) -> dict:
        """Score a single fusion pattern.

        Args:
            pattern: Dict from Feature 1 catalog with keys "opcodes",
                "register_class", "chain_registers", "total_frequency",
                "occurrence_count".

        Returns:
            Dict with "score", "score_breakdown", "tightness", "hardware",
            and original pattern info.
        """
        frequency = pattern.get("total_frequency", 0)
        chain_regs = pattern.get("chain_registers", [])
        length = pattern.get("length", len(pattern.get("opcodes", [])))

        # Sub-scores
        freq_s = self._freq_score(frequency)
        tight_s = self._tight_score(chain_regs, length)
        verdict = self._checker.check(pattern)
        hw_s = self._hw_score(verdict)

        # Weighted sum
        w = self._weights
        score = w["frequency"] * freq_s + w["tightness"] * tight_s + w["hardware"] * hw_s
        score = min(score, 1.0)

        return {
            "pattern": {
                "opcodes": pattern["opcodes"],
                "register_class": pattern.get("register_class"),
                "chain_registers": chain_regs,
            },
            "input_frequency": frequency,
            "input_occurrence_count": pattern.get("occurrence_count", 0),
            "tightness": {
                "raw_density": len(chain_regs) / max(length - 1, 1),
                "chain_factor": _CHAIN_FACTOR.get(length, 1.0),
                "score": tight_s,
            },
            "hardware": {
                "status": verdict.status,
                "reasons": verdict.reasons,
                "violations": verdict.violations,
            },
            "score": round(score, 6),
            "score_breakdown": {
                "freq_score": round(freq_s, 6),
                "tight_score": round(tight_s, 6),
                "hw_score": round(hw_s, 6),
            },
        }

    def score_patterns(
        self,
        patterns: list[dict],
        top: int | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Score and rank a list of patterns.

        Args:
            patterns: List of pattern dicts from Feature 1 catalog.
            top: Return only top N results (by score descending).
            min_score: Filter out patterns below this score.

        Returns:
            List of scored candidate dicts, sorted by score descending.
        """
        scored = [self.score_pattern(p) for p in patterns]

        # Filter by min_score
        if min_score > 0:
            scored = [s for s in scored if s["score"] >= min_score]

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Add rank
        for i, s in enumerate(scored):
            s["rank"] = i + 1

        if top is not None:
            scored = scored[:top]

        return scored


def score(
    catalog_path: Path,
    registry: ISARegistry,
    output_path: Path,
    top: int | None = None,
    min_score: float = 0.0,
    weights: dict[str, float] | None = None,
    feasibility_only: bool = False,
) -> list[dict]:
    """Run the full scoring pipeline: load catalog, score, write output.

    Args:
        catalog_path: Path to Feature 1 pattern catalog JSON.
        registry: ISA registry with encoding metadata.
        output_path: Path to write ranked candidates JSON.
        top: Max candidates to output.
        min_score: Minimum score threshold.
        weights: Custom scoring weights.
        feasibility_only: Only check constraints, skip scoring.

    Returns:
        List of scored/checked candidates.
    """
    import json
    from datetime import datetime, timezone

    catalog = json.loads(catalog_path.read_text())
    patterns = catalog.get("patterns", [])

    if not patterns:
        logger.warning("No patterns found in catalog")
        return []

    max_freq = max(p.get("total_frequency", 0) for p in patterns)

    if feasibility_only:
        # Only check constraints, don't compute scores
        checker = ConstraintChecker(registry)
        results = []
        for p in patterns:
            verdict = checker.check(p)
            results.append({
                "pattern": {
                    "opcodes": p["opcodes"],
                    "register_class": p.get("register_class"),
                    "chain_registers": p.get("chain_registers", []),
                },
                "input_frequency": p.get("total_frequency", 0),
                "input_occurrence_count": p.get("occurrence_count", 0),
                "hardware": {
                    "status": verdict.status,
                    "reasons": verdict.reasons,
                    "violations": verdict.violations,
                },
                "score": 0.0 if verdict.status == "infeasible" else (
                    0.5 if verdict.status == "constrained" else 1.0
                ),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        if top is not None:
            results = results[:top]
    else:
        scorer = Scorer(registry, max_freq, weights)
        results = scorer.score_patterns(patterns, top=top, min_score=min_score)

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_pattern_count": len(patterns),
        "candidate_count": len(results),
        "candidates": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    logger.info(
        "Scored %d patterns -> %d candidates (top=%s, min_score=%s)",
        len(patterns), len(results), top, min_score,
    )
    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_scorer.py -v`
Expected: All tests PASS

**Step 5: Run all tests together**

Run: `cd tools && python -m pytest fusion/tests/ -v`
Expected: All tests PASS (constraints + scorer + existing miner/pattern tests)

**Step 6: Commit**

```bash
git add tools/fusion/scorer.py tools/fusion/tests/test_scorer.py
git commit -m "feat(fusion): add weighted sum scorer with freq/tightness/hw sub-scores"
```

---

## Task 6: Add `score` subcommand to CLI

**Files:**
- Modify: `tools/fusion/__main__.py`

**Step 1: Add `V` to ISA modules mapping**

In `__main__.py`, update `_ISA_MODULES` (line 23):

```python
_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),
    "M": ("dfg.isadesc.rv64m", "build_registry"),
    "V": ("dfg.isadesc.rv64v", "build_registry"),
}
```

**Step 2: Update `parse_args` to accept `score` command and new flags**

Update the `choices` in the `command` argument and add new arguments after `--top`:

```python
    parser.add_argument(
        "command",
        choices=["discover", "score"],
        help="Command to run",
    )
```

Add these arguments after `--top`:

```python
    # Score-specific arguments
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Path to Feature 1 pattern catalog JSON (required for score)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold (default: 0.0)",
    )
    parser.add_argument(
        "--feasibility-only",
        action="store_true",
        default=False,
        help="Only check constraints, skip scoring",
    )
    parser.add_argument(
        "--weight-freq",
        type=float,
        default=None,
        dest="weight_freq",
        help="Weight for frequency score (default: 0.4)",
    )
    parser.add_argument(
        "--weight-tight",
        type=float,
        default=None,
        dest="weight_tight",
        help="Weight for tightness score (default: 0.3)",
    )
    parser.add_argument(
        "--weight-hw",
        type=float,
        default=None,
        dest="weight_hw",
        help="Weight for hardware score (default: 0.3)",
    )
```

Update the `--isa` default:

```python
    parser.add_argument(
        "--isa",
        default="I,F,M,V",
        help="Comma-separated ISA extensions (default: I,F,M,V)",
    )
```

**Step 3: Add `score` command handling in `main`**

Add before the text summary section in `main()`:

```python
    if args.command == "score":
        if not args.catalog:
            parser.error("--catalog is required for score command")
        if not args.output:
            parser.error("--output is required for score command")

        from fusion.scorer import score as run_score

        # Build custom weights if any overrides provided
        weights = None
        if args.weight_freq is not None or args.weight_tight is not None or args.weight_hw is not None:
            weights = {
                "frequency": args.weight_freq if args.weight_freq is not None else 0.4,
                "tightness": args.weight_tight if args.weight_tight is not None else 0.3,
                "hardware": args.weight_hw if args.weight_hw is not None else 0.3,
            }

        candidates = run_score(
            catalog_path=args.catalog,
            registry=registry,
            output_path=args.output,
            top=args.top,
            min_score=args.min_score,
            weights=weights,
            feasibility_only=args.feasibility_only,
        )

        # Text summary
        feasible = sum(1 for c in candidates if c.get("hardware", {}).get("status") == "feasible")
        constrained = sum(1 for c in candidates if c.get("hardware", {}).get("status") == "constrained")
        infeasible = sum(1 for c in candidates if c.get("hardware", {}).get("status") == "infeasible")

        print(f"\nFusion Candidate Scoring Results")
        print(f"  Candidates: {len(candidates)} (feasible={feasible}, constrained={constrained}, infeasible={infeasible})")
        if candidates:
            top = candidates[0]
            print(f"  Top: {' → '.join(top['pattern']['opcodes'])} "
                  f"(score={top['score']:.4f}, {top['hardware']['status']})")
        print(f"  Output: {args.output}")
        return
```

**Step 4: Run existing `discover` tests to verify no regression**

Run: `cd tools && python -m pytest fusion/tests/test_miner.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tools/fusion/__main__.py
git commit -m "feat(fusion): add score subcommand to CLI with filtering and weight options"
```

---

## Task 7: Integration test and roadmap update

**Files:**
- Create: `tools/fusion/tests/test_integration.py`
- Modify: `docs/plans/2026-04-08-phase2-feature-roadmap.md`

**Step 1: Write end-to-end integration test**

Create `tools/fusion/tests/test_integration.py`:

```python
"""Integration tests for the scoring pipeline."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow
from dfg.isadesc.rv64i import build_registry as build_i
from dfg.isadesc.rv64f import build_registry as build_f
from dfg.isadesc.rv64m import build_registry as build_m
from dfg.isadesc.rv64v import build_registry as build_v
from fusion.scorer import score as run_score


def _full_registry() -> ISARegistry:
    """Build a registry with all four extensions."""
    reg = ISARegistry()
    build_i(reg)
    build_f(reg)
    build_m(reg)
    build_v(reg)
    return reg


class TestScoringPipeline(unittest.TestCase):
    """End-to-end test: catalog JSON -> scored candidates JSON."""

    def setUp(self):
        self.registry = _full_registry()

        # Create a realistic pattern catalog
        self.catalog = {
            "generated": "2026-04-09T00:00:00Z",
            "source_df_count": 5,
            "pattern_count": 4,
            "patterns": [
                {
                    "opcodes": ["fadd.s", "fmul.s"],
                    "register_class": "float",
                    "length": 2,
                    "occurrence_count": 42,
                    "total_frequency": 150000,
                    "chain_registers": [["frd", "frs1"]],
                    "source_bbs": ["0x1000", "0x2000"],
                    "rank": 1,
                },
                {
                    "opcodes": ["vadd.vv", "vmul.vv"],
                    "register_class": "vector",
                    "length": 2,
                    "occurrence_count": 30,
                    "total_frequency": 100000,
                    "chain_registers": [["vrd", "vrs2"]],
                    "source_bbs": ["0x3000"],
                    "rank": 2,
                },
                {
                    "opcodes": ["add", "mul"],
                    "register_class": "integer",
                    "length": 2,
                    "occurrence_count": 15,
                    "total_frequency": 50000,
                    "chain_registers": [["rd", "rs1"]],
                    "source_bbs": ["0x4000"],
                    "rank": 3,
                },
                {
                    "opcodes": ["flw", "fmul.s"],
                    "register_class": "float",
                    "length": 2,
                    "occurrence_count": 8,
                    "total_frequency": 20000,
                    "chain_registers": [["frd", "frs1"]],
                    "source_bbs": ["0x5000"],
                    "rank": 4,
                },
            ],
        }

    def test_full_pipeline(self):
        """Score a catalog and verify output JSON structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(
                catalog_path=catalog_path,
                registry=self.registry,
                output_path=output_path,
            )

            # Verify output file was written
            self.assertTrue(output_path.exists())
            output_data = json.loads(output_path.read_text())
            self.assertIn("candidates", output_data)
            self.assertEqual(output_data["candidate_count"], 4)

            # Verify all candidates have required fields
            for c in output_data["candidates"]:
                self.assertIn("score", c)
                self.assertIn("score_breakdown", c)
                self.assertIn("hardware", c)
                self.assertIn("status", c["hardware"])
                self.assertIn("pattern", c)

    def test_feasibility_only_mode(self):
        """Feasibility-only mode skips scoring but checks constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(
                catalog_path=catalog_path,
                registry=self.registry,
                output_path=output_path,
                feasibility_only=True,
            )

            # flw pattern should be infeasible
            statuses = [r["hardware"]["status"] for r in results]
            self.assertIn("infeasible", statuses)

    def test_min_score_filters_infeasible(self):
        """min_score > 0 should filter out infeasible patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(
                catalog_path=catalog_path,
                registry=self.registry,
                output_path=output_path,
                min_score=0.1,
            )

            for c in results:
                self.assertGreaterEqual(c["score"], 0.1)

    def test_top_n_limits_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(
                catalog_path=catalog_path,
                registry=self.registry,
                output_path=output_path,
                top=2,
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["rank"], 1)
            self.assertEqual(results[1]["rank"], 2)

    def test_vext_pattern_gets_encoding(self):
        """V-extension patterns should have encoding metadata available."""
        enc = self.registry.get_encoding("vadd.vv")
        self.assertIsNotNone(enc)
        self.assertEqual(enc.opcode, 0x57)
        self.assertEqual(enc.reg_class, "vector")


class TestEncodingMetadataAvailable(unittest.TestCase):
    """Verify encoding metadata is populated across all extensions."""

    def setUp(self):
        self.registry = _full_registry()

    def test_float_instructions_have_encoding(self):
        for mn in ["fadd.s", "fsub.s", "fmul.s", "fdiv.s", "flw", "fsw", "fmadd.s"]:
            enc = self.registry.get_encoding(mn)
            self.assertIsNotNone(enc, f"Missing encoding for {mn}")
            self.assertEqual(enc.reg_class, "float")

    def test_integer_instructions_have_encoding(self):
        for mn in ["add", "sub", "mul", "addi", "lw", "sw"]:
            enc = self.registry.get_encoding(mn)
            self.assertIsNotNone(enc, f"Missing encoding for {mn}")

    def test_vector_instructions_have_encoding(self):
        for mn in ["vadd.vv", "vadd.vx", "vmul.vv", "vadd.vi", "vsetvli"]:
            enc = self.registry.get_encoding(mn)
            self.assertIsNotNone(enc, f"Missing encoding for {mn}")
            self.assertEqual(enc.reg_class, "vector")

    def test_load_store_flagged_correctly(self):
        self.assertTrue(self.registry.get_encoding("lw").may_load)
        self.assertTrue(self.registry.get_encoding("sw").may_store)
        self.assertFalse(self.registry.get_encoding("add").may_load)
```

**Step 2: Run integration tests**

Run: `cd tools && python -m pytest fusion/tests/test_integration.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `cd tools && python -m pytest dfg/tests/ fusion/tests/ -v`
Expected: All tests PASS

**Step 4: Update feature roadmap status**

In `docs/plans/2026-04-08-phase2-feature-roadmap.md`, update Feature 2 status:

Change:
```
**Status**: Not started
**Design doc**: TBD (`docs/plans/2026-04-XX-fusion-scoring-design.md`)
```
To:
```
**Status**: Completed
**Design doc**: `docs/plans/2026-04-09-fusion-scoring-design.md`
**Implementation plan**: `docs/plans/2026-04-09-fusion-scoring-plan.md`
```

**Step 5: Commit**

```bash
git add tools/fusion/tests/test_integration.py docs/plans/2026-04-08-phase2-feature-roadmap.md
git commit -m "test(fusion): add integration tests for scoring pipeline and update roadmap"
```

---

## Summary

| Task | Component | Files | Est. Steps |
|------|-----------|-------|------------|
| 1 | `InstructionFormat` dataclass | `instruction.py` | 6 |
| 2 | Extend `gen_isadesc.py` | `gen_isadesc.py`, `rv64v/f/m.py` | 8 |
| 3 | Manual `rv64i.py` encoding | `rv64i.py` | 7 |
| 4 | `ConstraintChecker` + tests | `constraints.py`, `test_constraints.py` | 5 |
| 5 | `Scorer` + tests | `scorer.py`, `test_scorer.py` | 6 |
| 6 | CLI `score` subcommand | `__main__.py` | 5 |
| 7 | Integration tests + roadmap | `test_integration.py`, roadmap | 5 |
| **Total** | | | **42 steps** |
