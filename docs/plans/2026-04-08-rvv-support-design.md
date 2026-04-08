# RVV 1.0 Instruction Set Support for DFG Engine

**Date**: 2026-04-08
**Status**: Approved
**Depends on**: Phase 1 (DFG engine with I/F/M support)
**Blocks**: Phase 2 F1 (Pattern Mining) — fusion candidates need vector instruction coverage

## Overview

Add RISC-V Vector Extension 1.0 (RVV) support to the `tools/dfg` engine, enabling
Data Flow Graph generation from basic blocks containing vector instructions. The
implementation follows the same auto-generation pattern used for I/F/M extensions,
extended with vector-specific concepts: LMUL-aware register grouping, mask register
modeling, and vector CSR tracking.

## Scope

- **Full RVV 1.0 instruction coverage**: All vector instructions present in
  `riscv_instrs.json` (arithmetic, load/store, reduction, mask, permutation,
  convert, widening/narrowing, config)
- **Vector register modeling with configuration attributes**: LMUL, SEW, VL
- **v0 dual-role modeling**: Separate MASK_KIND from VECTOR_KIND
- **Vector CSR tracking**: vl, vtype, vstart, vxrm, vxsat as DFG nodes
- **LMUL register grouping**: Expand logical vector registers to physical
  register groups in the DFG

## Baseline Assessment: riscv_instrs.json RVV Coverage

| Metric | Count |
|--------|-------|
| Total instruction entries | 67,252 |
| V-prefixed entries | 873 |
| PseudoV entries | 10,490 |

**Coverage**: Comprehensive. All major RVV 1.0 instruction categories are present:
arithmetic (vadd/vsub/vmul/vdiv/vand/vor/vxor), load/store (VLE/VSE/VLSE/VSSE/
fault-tolerant/ordered), config (VSETVLI/VSETVL/VSETIVLI), mask (VMAND/VMOR/VMXOR),
widening (VWADD/VWSUB/VWMUL), narrowing (VNCLIPU/VNCLIP), reduction (VRED*),
permutation (VSLIDE/VRGATHER/VCOMPRESS).

**Gap**: No RVV CSR register definitions in the JSON — handled via manual injection
in `gen_isadesc.py`.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Coverage scope | Full RVV 1.0 | Complete fusion candidate discovery |
| Generation method | Auto-generate via gen_isadesc.py | Consistent with I/F/M pipeline |
| Vector register model | With config attributes (LMUL/SEW) | Enables accurate register grouping |
| v0 dual role | Separate MASK_KIND | Track mask vs data flows distinctly |
| CSR tracking | Track vl/vtype/vstart/vxrm/vxsat | Config instructions affect all subsequent ops |
| Register grouping | Expand LMUL groups in DFG | Accurate physical register dependencies |
| Mnemonic mapping | Separate task (own design doc) | Requires LLVM source analysis |

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `tools/dfg/instruction.py` | Modify | Add VECTOR_KIND, MASK_KIND, CSR_VEC_KIND; config_regs field on RegisterFlow |
| `tools/dfg/gen_isadesc.py` | Modify | Add V extension predicate, vector register class mappings, remove V exclusion |
| `tools/dfg/isadesc/rv64v.py` | **New** | Auto-generated RVV register flow descriptor (~800-1000 instructions) |
| `tools/dfg/__main__.py` | Modify | Add V to _ISA_MODULES |
| `tools/dfg/parser.py` | Modify | Add _annotate_vector_config() for LMUL/SEW tracking |
| `tools/dfg/dfg.py` | Modify | Add _expand_lmul() and _expand_mask_operand() post-processing |
| `tools/dfg/tests/` | Modify | Vector instruction tests |

## Section 1: RegisterKind Extension (instruction.py)

Three new RegisterKind instances added alongside INTEGER_KIND and FLOAT_KIND:

### VECTOR_KIND — Vector data registers v1–v31

v0 is excluded (it has its own MASK_KIND entry).

```
name: "vector"
pattern: ^v([1-9]|[1-2]\d|3[0-1])$
position_prefix: "v"
```

### MASK_KIND — v0 used as mask register

```
name: "mask"
pattern: ^v0$
position_prefix: "v"
```

### CSR_VEC_KIND — Vector CSR registers

```
name: "csr_vec"
pattern: ^(vl|vtype|vstart|vxrm|vxsat)$
position_prefix: "c"
```

### Config Attributes on RegisterFlow

Add optional field `config_regs: list[str]` to `RegisterFlow`:

```python
@dataclass
class RegisterFlow:
    dst_regs: list[str]
    src_regs: list[str]
    config_regs: list[str] = field(default_factory=list)  # NEW
```

Example — VSETVLI writes both vl and vtype CSRs:

```python
RegisterFlow(dst=["rd", "cvl", "cvtype"], src=["rs1"])
```

### LMUL-Aware Resolve Post-Processing

Add `_expand_grouping(resolved: ResolvedFlow, lmul: int) -> ResolvedFlow`
as a post-processing step after `RegisterFlow.resolve()`:

- LMUL=1: `v4` → `["v4"]`
- LMUL=2: `v4` → `["v4", "v5"]`
- LMUL=4: `v4` → `["v4", "v5", "v6", "v7"]`
- LMUL=8: `v4` → `["v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"]`

LMUL is obtained by tracking the most recent VSETVLI/VSETVL in the BB.
Default at BB entry: LMUL=1.

`_BUILTIN_KINDS` updated to `[INTEGER_KIND, FLOAT_KIND, VECTOR_KIND, MASK_KIND, CSR_VEC_KIND]`.

## Section 2: gen_isadesc.py Modifications

### Extension Predicate

```python
EXTENSION_PREDICATES: dict[str, set[str]] = {
    "F": {"HasStdExtF"},
    "M": {"HasStdExtM"},
    "V": {"HasStdExtV"},   # NEW
}
```

### Vector Register Class Mappings

```python
REG_CLASS_TO_PREFIX: dict[str, str] = {
    # ... existing GPR/FPR entries ...
    "VR":       "v",    # Generic vector register class
    "VRM1":     "v",    # LMUL=1
    "VRM2":     "v",    # LMUL=2
    "VRM4":     "v",    # LMUL=4
    "VRM8":     "v",    # LMUL=8
    "VRN":      "v",    # Vector register class (no LMUL dependency)
}
```

### Vector Operand Position Mappings

```python
OPERAND_TO_POSITION: dict[str, str] = {
    # ... existing entries ...
    "vd": "rd",      # already exists
    "vs1": "rs1",    # already exists
    "vs2": "rs2",    # already exists
    "vs3": "rs3",    # NEW — ternary ops (vfmacc, vmacc)
    "vl": "rd",      # VSETVLI vl output
    "vtype": "rd",   # VSETVLI vtype output
    "vstart": "rd",  # vstart CSR
    "vxrm": "rd",    # vxrm CSR
    "vxsat": "rd",   # vxsat CSR
}
```

### Remove V Instruction Exclusion

Delete the V-prefix exclusion logic in `_should_include()` (lines 204–207):

```python
# REMOVE THIS BLOCK:
if name.startswith("V") and not name.startswith(("VFMV_F_S", "VFMV_S_F")):
    return False
```

### CSR Injection for Config Instructions

VSETVLI/VSETVL output vl and vtype as side effects that may not be represented
as standard register-class operands in llvm-tblgen. Add `_inject_vec_csr(entry, dst, src)`
to detect VSET* instructions and manually inject `cvl` and `cvtype` into dst.

### Mnemonic Overrides

RVV LLVM def names diverge from QEMU disassembly mnemonics significantly
(e.g., `VADD_VV` → `vadd.vv`, `VLE8_V` → `vle8.v`). The mapping table is
expected to contain 500–800 entries. This is handled as a **separate task** with
its own design doc (see Section 8 below).

## Section 3: LMUL Tracking in parser.py

Add a BB-level post-processing phase `_annotate_vector_config(blocks)` called
after `parse_disas()`.

### VectorConfig Data Class

```python
@dataclass
class VectorConfig:
    vlen: int            # Static — from vlen CSR at BB entry
    sew: int             # 8, 16, 32, or 64
    lmul: int            # 1, 2, 4, or 8
    vl: int | None       # Actual vector length
    tail_policy: str     # "undisturbed" or "agnostic"
    mask_policy: str     # "undisturbed" or "agnostic"
    change_points: list[tuple[int, VectorConfig]]  # (insn_index, config) for mid-BB changes
```

### BasicBlock Extension

```python
@dataclass
class BasicBlock:
    bb_id: int
    vaddr: int
    instructions: list[Instruction] = field(default_factory=list)
    vec_config: VectorConfig | None = None  # NEW — None if no vector instructions
```

### Parsing Logic

1. Scan each BB's instruction sequence for VSETVLI/VSETVL/VSETIVLI
2. Parse SEW/LMUL from QEMU comment annotations (e.g., `# e8,m2,tu,mu`)
   or from instruction operands (zimm11 encoding)
3. Store the parsed config into `bb.vec_config`
4. For BBs with multiple VSETVLI, record each configuration change point
   as `(instruction_index, VectorConfig)` in `change_points`
5. BBs without vector instructions get `vec_config = None` (zero overhead)

## Section 4: rv64v.py Auto-Generation Specification

Auto-generated file following the same format as `rv64f.py` / `rv64m.py`.

### Expected Scale

~800–1000 instructions (all non-Pseudo V-prefixed entries with HasStdExtV predicate).

### Instruction Categories and Flow Patterns

| Category | Example | dst_regs | src_regs |
|----------|---------|----------|----------|
| Arithmetic VV | vadd.vv, vsub.vv | vrd | vrs1, vrs2 |
| Arithmetic VX | vadd.vx, vsub.vx | vrd | vrs1, rs1 |
| Arithmetic VI | vadd.vi | vrd | vrs1 |
| Ternary | vfmacc.vv, vmacc.vv | vrd | vrs1, vrs2, vrs3 |
| Load | vle8.v, vlw.v | vrd | rs1 |
| Store | vse8.v, vsw.v | — | vrs2, rs1 |
| Strided Load | vlse8.v | vrd | rs1, rs2 |
| Strided Store | vsse8.v | — | vrs2, rs1, rs2 |
| Reduce | vredsum.vs | vrd | vrs1, vrs2 |
| Mask Logical | vmand.mm, vmor.mm | vrd (mask) | vrs1, vrs2 (mask) |
| Mask Compare | vmslt.vv | vrd (mask) | vrs1, vrs2 |
| Permute | vslideup.vi, vrgather.vv | vrd | vrs1 [, vrs2] |
| Config | vsetvli, vsetvl | rd, cvl, cvtype | rs1 |
| Move | vmv.v.v, vfmv.f.s | vrd | vrs1 |

### Mask Operand Handling

QEMU disassembles masked vector instructions with a trailing `v0.t` operand:
```
vadd.vv v1, v2, v3, v0.t
```

The position list needs a `mask` position name. `_extract_registers()` gains
logic to recognize `v0.t` format and map it to `vmask` → `v0` (mask_kind).

### PseudoV Exclusion

Existing `_should_include()` `Pseudo` prefix filter already covers all
`PseudoV*` entries. No additional work needed.

## Section 5: DFG Construction with Vector Support (dfg.py)

### Modified build_dfg Flow

```
parse_disas()
  → _annotate_vector_config(blocks)
  → for each instruction:
      registry.get_flow(mnemonic)
      → resolve()
      → _expand_lmul(resolved, vec_config)      # NEW
      → _expand_mask_operand(resolved)            # NEW
      → RAW edge detection (with expanded registers)
```

### _expand_lmul()

Takes a `ResolvedFlow` and the current `VectorConfig`, expands vector register
names to physical register lists:

| LMUL | v4 → | Physical registers |
|------|-------|-------------------|
| 1 | ["v4"] | v4 |
| 2 | ["v4", "v5"] | v4, v5 |
| 4 | ["v4", "v5", "v6", "v7"] | v4–v7 |
| 8 | ["v4".."v11"] | v4–v11 |

Expanded registers participate independently in RAW dependency detection.
Example: `vadd v2, v4, v8` with LMUL=2 produces edges:
`v4→v2`, `v5→v2`, `v8→v2`, `v9→v2` (writing v2/v3, reading v4/v5/v8/v9).

### _expand_mask_operand()

Parses `v0.t` mask operand in the resolved flow, distinguishing it from
vector data usage of v0. `DFGEdge.register` appends a kind suffix:
`"v0(mask)"` vs `"v0(vector)"`.

### Mid-BB Config Changes

When `vec_config.change_points` is non-empty, the DFG builder selects the
appropriate LMUL for each instruction based on its position relative to the
nearest preceding VSETVLI.

## Section 6: __main__.py Changes

Add V extension to the ISA module registry:

```python
_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),
    "M": ("dfg.isadesc.rv64m", "build_registry"),
    "V": ("dfg.isadesc.rv64v", "build_registry"),  # NEW
}
```

Usage: `python -m tools.dfg --disas file.disas --isa I,F,M,V`

## Section 7: Testing Strategy

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `test_instruction.py` | VECTOR_KIND/MASK_KIND/CSR_VEC_KIND regex matching; position prefixes; `_expand_lmul()` for all LMUL values; `_expand_mask_operand()` v0.t parsing |
| `test_parser.py` | Vector instruction .disas parsing; `_annotate_vector_config()` SEW/LMUL extraction from QEMU comments; mid-BB VSETVLI change tracking |
| `test_dfg.py` | Full DFG build for vector BB (VSETVLI + vadd.vv + vse8.v); LMUL-expanded edges; CSR tracking edges; mask edges |

### Test Fixture

`tools/dfg/tests/sample_vector_disas.txt`:

```
BB 1 (vaddr: 0x1000, 5 insns):
  0x1000: vsetvli a0, a1, e32,m2
  0x1004: vle32.v v2, (a2)
  0x1008: vadd.vv v4, v2, v6
  0x100c: vse32.v v4, (a3)
  0x1010: vsetvli zero, zero, e8,m1
```

### Integration Test

Run DFG generation on existing YOLO workload .disas with `--isa I,F,M,V`.
Verify no regression to agent fallback for previously-supported instructions.

## Section 8: Mnemonic Mapping Sub-Task (Separate Session)

### Problem

RVV LLVM tablegen def names and QEMU disassembly mnemonics follow different
conventions. The default transformation (`lowercase + underscore → dot`) does
not produce correct QEMU mnemonics for most vector instructions.

### Deliverable

A separate design doc: `docs/plans/2026-04-08-rvv-mnemonic-mapping-design.md`

### Scope of That Task

- **Input**: All V-predicate instruction LLVM def names from `riscv_instrs.json`
- **Output**: `MNEMONIC_OVERRIDES` dictionary (or inline mapping in rv64v.py)
- **Validation**: Sample real QEMU disassembly output, compare against mapping results
- **Estimated scale**: 500–800 mapping rules
- **Method**: Analyze LLVM RISC-V backend source in `third_party/llvm-project/`
  to understand naming conventions and derive systematic transformation rules
  where possible, with manual overrides for exceptions
