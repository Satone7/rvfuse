# Feature 2: Fusion Candidate Scoring & Hardware Constraint Model

**Date**: 2026-04-09
**Status**: Approved
**Predecessor**: Feature 1 (Pattern Mining) — Merged
**Branch prefix**: `worktree-fusion-scoring`

## Goal

Score and rank fusion patterns by combining execution frequency with hardware
feasibility. Encode RISC-V hardware constraints that determine which patterns can
realistically be fused in silicon.

**Priority**: V-extension support is the highest priority for this feature.

## Input

- Pattern catalog JSON (Feature 1 output: `output/fusion_patterns.json`)
- ISA encoding specification via extended `isadesc/` modules

## Output

- Ranked candidate list (`output/fusion_candidates.json`)
- Hardware constraint model module (`tools/fusion/constraints.py`)
- Scoring module (`tools/fusion/scorer.py`)

---

## Design Decisions

### D1: Encoding space + operand constraints (not full pipeline model)

Focus on 32-bit encoding space budget (opcode/funct3/funct7/rs1/rs2/rd field
allocation) and operand constraints (register class compatibility, source/dest
count limits, immediate field sizes). Pipeline-level modeling (execution units,
structural hazards, latency) is deferred to Phase 3 simulation.

**Why**: Minimum viable model that still produces meaningful feasibility verdicts.
Extensible for future phases without premature complexity.

### D2: Weighted sum scoring function

```
score = w_freq * freq_score + w_tight * tight_score + w_hw * hw_score
```

Each sub-score normalized to [0, 1]. Default weights: frequency=0.4,
tightness=0.3, hardware=0.3. User-overridable via CLI flags.

**Why**: More interpretable than multiplicative model. Independent tuning of
each dimension. Hardware score acts as a soft filter rather than a hard cutoff
(with `--feasibility-only` for hard filtering).

### D3: Extend `RegisterFlow` with encoding metadata

Add `InstructionFormat` dataclass as an optional field on `RegisterFlow`. This
co-locates encoding info with register flow in `ISARegistry`, providing a single
source of truth. Feature 3 (Scheme Specification) will also consume this data.

**Why**: Avoids maintaining parallel data structures. The `None` default keeps
backward compatibility — DFG engine code is unaffected.

---

## Architecture

### InstructionFormat dataclass

New dataclass in `tools/dfg/instruction.py`:

```python
@dataclass
class InstructionFormat:
    """RISC-V instruction encoding layout."""
    format_type: str        # "R", "I", "S", "B", "U", "J", "R4"
    opcode: int             # 7-bit opcode value
    funct3: int | None      # 3-bit funct3 (None if N/A)
    funct7: int | None      # 7-bit funct7 (None if N/A)
    has_rs1: bool = True
    has_rs2: bool = True
    has_rs3: bool = False   # R4-type (fmadd, fnmadd, etc.)
    has_rd: bool = True
    has_imm: bool = False
    imm_bits: int = 0       # Immediate field width (0 if no imm)
    reg_class: str = "integer"  # "integer", "float", "vector"
```

### RegisterFlow extension

Add optional `encoding` field to existing `RegisterFlow`:

```python
@dataclass
class RegisterFlow:
    dst_regs: list[str]
    src_regs: list[str]
    config_regs: list[str] = field(default_factory=list)
    encoding: InstructionFormat | None = None  # NEW
```

### isadesc updates

Each isadesc module adds `InstructionFormat` alongside existing `RegisterFlow`.
Priority order: V > F > M > I.

Example for `rv64v.py`:
```python
("vadd.vv", RegisterFlow(
    ['vrd'], ['vrs2', 'vrs1'],
    encoding=InstructionFormat(
        format_type="V", opcode=0x57, funct3=0x0, funct7=0x00,
        reg_class="vector",
    ),
)),
```

### ISA extensions to update

| Extension | File | Priority |
|-----------|------|----------|
| V | `isadesc/rv64v.py` | Highest |
| F | `isadesc/rv64f.py` | High |
| M | `isadesc/rv64m.py` | Medium |
| I | `isadesc/rv64i.py` | Medium |

---

## Hardware Constraint Model

### ConstraintChecker class (`tools/fusion/constraints.py`)

Receives `ISARegistry`. For each pattern, performs:

1. **Encoding space budget** — verify fused instruction fits in 32-bit encoding
   - Total unique source registers across chain <= 3 (rs1, rs2, rs3 fields)
   - Total unique destination registers across chain <= 1 (rd field)
   - All instructions share the same opcode (can be differentiated by funct fields)

2. **Register class compatibility** — all instructions in pattern must share
   the same `reg_class` (defensive check; already guaranteed by Feature 1)

3. **Operand count limits** —
   - Unique source register count <= 3
   - Unique destination register count <= 1
   - If any instruction has an immediate, fused instruction immediate field check

4. **Structural constraints** —
   - No load/store instructions in chain (memory access cannot fuse)
   - No control flow (already guaranteed by Feature 1)
   - No CSR/config register writes (vector config changes break pipeline)

### Verdict

```python
@dataclass
class Verdict:
    status: Literal["feasible", "constrained", "infeasible"]
    reasons: list[str]       # Human-readable rationale
    violations: list[str]    # Specific constraint names violated
```

- `feasible`: No violations
- `constrained`: Minor issues (e.g., funct field conflict, borderline operand count)
- `infeasible`: Hard violations (cross-register-class, load/store, >3 unique sources)

---

## Scoring Function

### Sub-scores

| Component | Formula | Description |
|-----------|---------|-------------|
| `freq_score` | `log(1 + freq) / log(1 + max_freq)` | Log-normalized execution frequency |
| `tight_score` | `raw_density * chain_factor` | RAW edge density × chain length factor |
| `hw_score` | feasible=1.0, constrained=0.5, infeasible=0.0 | Hardware feasibility score |

### Dependency tightness metrics

- `raw_density`: number of RAW edges / max possible RAW edges in the chain
  (pairs count for 2-instr chains, triples for 3-instr chains)
- `chain_factor`: 1.0 for 2-instruction chains, 1.2 for 3-instruction chains
  (longer chains are more valuable fusion targets)

### Default weights

```python
DEFAULT_WEIGHTS = {"frequency": 0.4, "tightness": 0.3, "hardware": 0.3}
```

### Output schema

```json
{
  "pattern": {
    "opcodes": ["fadd.s", "fmul.s"],
    "register_class": "float",
    "chain_registers": [["frd", "frs1"]]
  },
  "input_frequency": 150000,
  "input_occurrence_count": 42,
  "tightness": {
    "raw_density": 0.8,
    "chain_factor": 1.0,
    "score": 0.8
  },
  "hardware": {
    "status": "feasible",
    "reasons": [],
    "violations": []
  },
  "score": 0.72,
  "score_breakdown": {
    "freq_score": 0.65,
    "tight_score": 0.80,
    "hw_score": 1.00
  }
}
```

---

## CLI

Extend existing `python -m tools.fusion` with `score` subcommand:

```bash
python -m tools.fusion score \
  --catalog output/fusion_patterns.json \
  --isa I,F,M,V \
  --output output/fusion_candidates.json \
  --top 20 \
  --min-score 0.3 \
  --feasibility-only \
  --weight-freq 0.5 --weight-tight 0.25 --weight-hw 0.25
```

| Flag | Description |
|------|-------------|
| `--catalog` | Path to Feature 1 pattern catalog JSON (required) |
| `--isa` | ISA extensions to load (default: `I,F,M,V`) |
| `--output` | Output path for ranked candidates JSON (required) |
| `--top` | Max candidates to output (default: 20) |
| `--min-score` | Minimum score threshold (default: 0.0) |
| `--feasibility-only` | Only check constraints, skip scoring |
| `--weight-freq/tight/hw` | Custom scoring weights |
| `--verbose` | Enable debug logging |

---

## File Changes

### New files

| File | Description |
|------|-------------|
| `tools/fusion/scorer.py` | Scoring function and sub-score calculations |
| `tools/fusion/constraints.py` | Hardware constraint model and Verdict |
| `tools/fusion/tests/test_constraints.py` | Constraint checker unit tests |
| `tools/fusion/tests/test_scorer.py` | Scoring function unit tests |

### Modified files

| File | Change |
|------|--------|
| `tools/dfg/instruction.py` | Add `InstructionFormat` dataclass; add `encoding` field to `RegisterFlow` |
| `tools/dfg/isadesc/rv64v.py` | Add `InstructionFormat` to V-extension instructions |
| `tools/dfg/isadesc/rv64f.py` | Add `InstructionFormat` to F-extension instructions |
| `tools/dfg/isadesc/rv64m.py` | Add `InstructionFormat` to M-extension instructions |
| `tools/dfg/isadesc/rv64i.py` | Add `InstructionFormat` to I-extension instructions |
| `tools/fusion/__main__.py` | Add `score` subcommand and wiring |

### Unchanged files

- `tools/dfg/parser.py`, `tools/dfg/dfg.py`, `tools/dfg/output.py` — core DFG engine
- `tools/fusion/miner.py`, `tools/fusion/pattern.py`, `tools/fusion/agent.py` — Feature 1 code

---

## Acceptance Criteria

1. Every candidate has a feasibility verdict: `feasible`, `constrained`, or
   `infeasible` with written rationale
2. Score breakdown is transparent (individual component scores visible)
3. Known infeasible patterns (cross-register-class, load/store, >3 source regs)
   are correctly rejected
4. Top candidates pass manual review against RISC-V unprivileged ISA spec
5. Constraint model covers I, F, M, and V extensions
6. Unit test coverage >= 80% for new code
7. V-extension constraint checks work correctly (highest priority)

---

## Out of Scope

- Actual instruction encoding generation (Feature 3)
- Cycle-accurate simulation (Phase 3)
- Vector LMUL/SEW-specific constraint tuning (Phase 4)
