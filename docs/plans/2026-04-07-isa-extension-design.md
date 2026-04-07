# Design: Extend DFG ISA Support via llvm-tblgen Code Generation

**Date**: 2026-04-07
**Status**: Approved
**Scope**: F + M extensions (D and A deferred; V extension future-proofing)

## Problem

The DFG tool (`tools/dfg/`) currently supports only RV64I base integer instructions
(79 mnemonics). Running DFG generation against the YOLO11n workload produces **zero
DFGs** because all 4 hot basic blocks (80.6% of execution) consist entirely of
floating-point instructions from `libonnxruntime.so`.

Instruction frequency from `output/yolo.bbv.disas` (314,407 lines, 35,715 BBs):

| Extension | Key Mnemonics       | Total Occurrences |
|-----------|---------------------|-------------------|
| F (single) | flw, fsw, fmadd.s, fmul.s, fmv.s.x | ~2,862 |
| D (double) | fcvt.d.*, fdiv.d, fmul.d, fld, fsd | ~275 |
| M (multiply) | mul, mulw, mulh, mulhu, div, rem | ~667 |
| A (atomic) | lr.w, sc.w, amoswap.*, amoadd.* | ~30 |

The `_extract_registers()` function in `instruction.py` only recognizes integer
register names (`x0-x31` / ABI names), so even adding RegisterFlow entries for
float instructions would fail at operand parsing.

## Decision

Use `llvm-tblgen --dump-json` to generate DFG ISA descriptor files from the
Xuantie LLVM RISC-V backend `.td` files. This ensures every instruction's
register operand mapping is accurate and authoritative.

**First phase**: F + M extensions.
**Deferred**: D, A, V extensions.

## Architecture

```
third_party/llvm-project (Xuantie LLVM submodule)
  llvm/lib/Target/RISCV/*.td
        │
        ▼  llvm-tblgen --dump-json  (build once)
  tools/dfg/riscv_instrs.json  (all expanded instruction records)
        │
        ▼  gen_isadesc.py  (Python generator script)
  tools/dfg/isadesc/rv64f.py   (F extension — generated)
  tools/dfg/isadesc/rv64m.py   (M extension — generated)
```

Existing `isadesc/rv64i.py` is retained — it covers pseudo-instructions
(`mv`, `li`, `call`, `ret`, `j`, `nop`, etc.) that `llvm-tblgen` does not define.

## Component Changes

### 1. Build llvm-tblgen (one-time setup)

```bash
git submodule update --init third_party/llvm-project
cd third_party/llvm-project && mkdir -p build && cd build
cmake -DLLVM_TARGETS_TO_BUILD=RISCV -DCMAKE_BUILD_TYPE=Release ../llvm
make llvm-tblgen -j$(nproc)
```

### 2. Extract instruction JSON

```bash
./bin/llvm-tblgen \
  -I ../llvm/include -I ../llvm/lib/Target/RISCV \
  ../llvm/lib/Target/RISCV/RISCV.td \
  --dump-json > tools/dfg/riscv_instrs.json
```

Each `Instruction` record in the JSON contains:
- `OutOperandList`: destination register DAG, e.g. `(outs FPR32:$rd)`
- `InOperandList`: source operand DAG, e.g. `(ins FPR32:$rs1, FPR32:$rs2, frmarg:$frm)`
- `Predicates`: ISA extension predicates (`HasStdExtF`, `HasStdExtM`, etc.)
- `AsmString`: assembly syntax string

### 3. Generator script: `tools/dfg/gen_isadesc.py`

Reads the JSON, extracts and transforms:

| JSON Field | Extraction | DFG Mapping |
|------------|-----------|-------------|
| `OutOperandList` `GPR:$rd` | dst, integer | `RegisterFlow(dst=["rd"], ...)` |
| `InOperandList` `FPR32:$rs1` | src, float | `RegisterFlow(src=["rs1"], ...)` |
| `frmarg:$frm` | rounding mode | Skip (not a register) |
| `simm12_lo:$imm12` | immediate | Skip (not a register) |
| `Predicates` `HasStdExtF` | Belongs to F ext | Group under `"F"` |

The generator produces a Python file with the same `build_registry(registry)` /
`load_extension()` interface as the existing `rv64i.py`.

**Instruction filtering**:
- Only include records that are subclasses of `Instruction`
- Filter by `Predicates` for the target extension
- Skip instructions where all operands are immediates (e.g. `fence`)
- Map instruction names to QEMU disassembly mnemonics (e.g. `FMADD_S` -> `fmadd.s`)

**Register class to DFG kind mapping**:

| LLVM Register Class | DFG Register Kind |
|---------------------|-------------------|
| `GPR`, `GPRNoX0` | `integer` |
| `FPR32` | `float` |
| `FPR64` | `float` (same physical registers, wider) |
| `GPRMem` | `integer` (base address) |
| `frmarg` | *(skip)* |

### 4. Extend `instruction.py` — register type system

**`RegisterFlow`** gains an optional `kinds` dict mapping each operand to its
register kind:

```python
@dataclass
class RegisterFlow:
    dst_regs: list[str]
    src_regs: list[str]
    kinds: dict[str, str] = field(default_factory=dict)
    # e.g. {"frd": "float", "frs1": "float", "rs1": "integer"}
```

When `kinds` is empty, all operands default to `"integer"` (backward compatible).

**`ISARegistry`** gains `register_kind()` for registering new register types:

```python
class RegisterKind:
    name: str            # "integer", "float", "vector"
    pattern: re.Pattern  # regex matching register names
    position_prefix: str # prefix in position_names: "f" -> "frd", "frs1"
```

**`_extract_registers()`** is modified to:
1. Iterate all registered `RegisterKind` patterns
2. Match each token in the operand string against kinds
3. Map matches to position names using the kind's prefix
4. Skip non-register tokens (immediates, rounding modes: `dyn`, `rne`, `rtz`, `rdn`, `rup`, `rmm`)

**Float register name patterns** (RISC-V ABI):
- `f0`-`f31` (numeric names)
- `ft0`-`ft7`, `fs0`-`fs11`, `fa0`-`fa7`, `ft8`-`ft11`

### 5. Update `__main__.py` — extension registry

```python
_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),   # new
    "M": ("dfg.isadesc.rv64m", "build_registry"),   # new
}
```

The `--isa` flag now accepts `"I,F,M"` (comma-separated).

### 6. Special format handling

| Format | Example | Handling |
|--------|---------|----------|
| 4-operand FMA | `fmadd.s dyn,ft2,fa4,ft0,ft2` | First token is rounding mode (`dyn`), skip it; remaining 4 tokens map to `frd=ft2, frs1=fa4, frs2=ft0, frs3=ft2` |
| FP compare -> GPR | `feq.s a5,fa0,fa0` | dst=`a5`(integer), src=`fa0,fa0`(float) — auto-detected from generated RegisterFlow |
| Mixed store | `fsw ft2,0(a2)` | src=`ft2`(float), base=`a2`(integer) — existing memory format handling extended |
| Move cross-type | `fmv.s.x fa6,zero` | dst=`fa6`(float), src=`zero`(integer) |

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `tools/dfg/instruction.py` | Modify | Add `RegisterKind`, extend `RegisterFlow.kinds`, update `_extract_registers()` for multi-kind support |
| `tools/dfg/__main__.py` | Modify | Register F and M modules in `_ISA_MODULES` |
| `tools/dfg/gen_isadesc.py` | Create | Generator script: JSON -> Python RegisterFlow code |
| `tools/dfg/isadesc/rv64f.py` | Create | Generated F extension (or committed snapshot) |
| `tools/dfg/isadesc/rv64m.py` | Create | Generated M extension (or committed snapshot) |
| `tools/dfg/isadesc/rv64i.py` | No change | Retained for pseudo-instructions |
| `tools/dfg/riscv_instrs.json` | Create (gitignore) | Intermediate artifact, not committed |
| `docs/plans/2026-04-07-isa-extension-design.md` | Create | This document |

## Testing Strategy

1. **Unit tests for `_extract_registers()`**: Add test cases for float registers, mixed operands, rounding mode tokens, and 4-operand FMA format
2. **Integration test**: Run DFG generation on `output/yolo.bbv.disas` with `--isa I,F,M --report output/hotspot.json --coverage 80` and verify all 4 hot BBs produce valid DFGs (not zero)
3. **Regression**: Existing I-extension tests must still pass

## V Extension Future-Proofing

When V extension is needed:
1. LLVM `.td` files already contain full V instruction definitions
2. `gen_isadesc.py` recognizes new register class `VR` and operands `vd/vs1/vs2/vm`
3. `ISARegistry.register_kind("vector", ...)` handles `v0`-`v31` names
4. No changes to the generation pipeline or DFG builder

## Risks

| Risk | Mitigation |
|------|------------|
| Xuantie LLVM `.td` may differ from upstream | Pin to a known-good commit; validate generated output against disassembly |
| `llvm-tblgen` JSON format may be large | Stream-parse; filter early by class and predicates |
| QEMU disassembly mnemonics may differ from LLVM names | Generator includes a name mapping table; verify against actual `.disas` output |
| `_extract_registers` changes break existing I-extension parsing | Preserve exact backward compatibility when `kinds` is empty |
