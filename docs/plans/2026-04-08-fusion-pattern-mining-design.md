# F1: Fusion Pattern Mining Engine — Design

**Date**: 2026-04-08
**Status**: Approved
**Phase**: 2 — Fusion Candidate Discovery and Design
**Predecessor**: Phase 1 (DFG generation engine, BBV profiling)
**Successor**: F2 — Scoring & Hardware Constraint Model

---

## Context

Phase 1 delivered a DFG generation engine (`tools/dfg/`) that produces per-basic-block
dependency graphs in JSON format, and a BBV profiling pipeline (`analyze_bbv.py`) that
provides per-BB execution frequency.

F1 builds on this foundation to discover recurring instruction subgraphs suitable for
hardware fusion. The design uses an **Agent + Miner hybrid** architecture:

- **Miner** (bottom layer): deterministic pattern extraction, normalization,
  aggregation, and frequency ranking — pure computation, no judgment.
- **Agent** (top layer): calls the Miner, analyzes the results, and produces
  recommendations on which patterns have the highest fusion value.

This hybrid approach suits the exploratory nature of Phase 2 research: the Miner
provides reliable, reproducible statistics, while the Agent applies analytical
judgment that a deterministic pipeline cannot.

---

## Pattern Model

### Fusible Pattern Definition

A fusible pattern is a consecutive instruction sequence of length 2 or 3 where:

1. Every adjacent pair has at least one RAW dependency edge
2. All instructions belong to the same register class (all integer or all float)
3. No branch, jump, or call instructions appear in the sequence

### Normalization: Opcode + Register-Role Template

Patterns are normalized for cross-BB aggregation using opcode mnemonics and register
role positions (not concrete register names).

**Pattern template key** records:
- `opcodes`: ordered list of mnemonics (e.g., `["fadd.s", "fmul.s"]`)
- `register_class`: `"integer"` or `"float"`
- `chain_registers`: for each consecutive pair, which operand positions carry the
  RAW dependency (e.g., `[["frd", "frs1"]]` means the `frd` output of instruction 1
  feeds into the `frs1` input of instruction 2)
- `length`: 2 or 3

Two concrete instruction sequences map to the same pattern template if they have
identical opcode sequences and the RAW dependencies flow through the same role
positions, regardless of the actual register names used.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Subgraph scope | Strictly linear chains only | Real fusion hardware fuses adjacent instructions; chains are the natural model |
| Register class constraint | Same class only (int/int, float/float) | Cross-class fusion is extremely unlikely in RISC-V hardware |
| Normalization | Opcode + register-role template | Captures structural similarity while abstracting away concrete register names |

---

## Architecture

### Module Structure

```text
tools/fusion/
├── __init__.py
├── __main__.py       # CLI entry point: python -m tools.fusion discover
├── pattern.py        # Pattern dataclass, normalization logic, dedup key
├── miner.py          # Deterministic pipeline: enumerate, normalize, aggregate, rank
├── agent.py          # Agent layer: calls miner, analyzes results, produces report
└── tests/
    ├── __init__.py
    ├── test_pattern.py   # Normalization, template key, edge cases
    ├── test_miner.py     # End-to-end mining, aggregation, ranking
    └── fixtures/         # Small DFG JSON fixtures for testing
```

### Data Flow

```text
DFG JSON files ──┐
                 ├──→ Miner ──→ Pattern statistics JSON ──→ Agent ──→ Discovery report
BBV hotspot JSON ─┘         (deterministic)                  (analytical)
```

### Miner Layer

The Miner is a pure-computation module with no AI or judgment. Pipeline stages:

1. **Load**: Read all DFG JSON files from `--dfg-dir`. Read BBV hotspot report from
   `--report`. Import `ISARegistry` from `tools/dfg/instruction.py` for register
   role resolution.
2. **Enumerate**: For each DFG (per-BB), walk the instruction sequence and extract
   all valid linear chains of length 2-3:
   - For consecutive pair (i, i+1): check if a RAW edge exists. If yes, form
     a length-2 chain.
   - For valid length-2 chain ending at (i+1): check if (i+1, i+2) also has a RAW
     edge. If yes, extend to length-3.
   - Skip chains where instructions have different register classes or contain
     control flow mnemonics.
3. **Normalize**: Convert each concrete chain to a pattern template by resolving
   operand roles via `ISARegistry.resolve()` and abstracting register names to
   role positions.
4. **Aggregate**: Group identical templates across all BBs. For each group, compute
   `occurrence_count` (number of BBs) and `total_frequency` (sum of BBV execution
   counts).
5. **Rank**: Sort patterns by `total_frequency` descending.
6. **Output**: Write pattern statistics to JSON file.

### Agent Layer

The Agent provides analytical judgment on top of Miner output:

- Reuses the `dfg/agent.py` Claude CLI subprocess pattern (`--model`, `--no-agent`)
- **Agent mode** (default): Invokes Claude CLI with a prompt containing:
  - DFG context (top BB summaries)
  - Miner pattern statistics (top patterns with frequencies)
  - Requests: analyze which patterns are most promising for fusion, explain why,
    flag any interesting patterns the Miner may have missed
- **No-agent mode** (`--no-agent`): Runs Miner only, outputs statistical JSON
  without analysis

---

## CLI Interface

```bash
python -m tools.fusion discover \
  --dfg-dir output/dfg/json/ \
  --report output/yolo.bbv.hotspot.json \
  --output output/fusion_patterns.json \
  [--top 20] \
  [--no-agent] \
  [--model sonnet]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dfg-dir` | required | Directory containing DFG JSON files |
| `--report` | required | BBV hotspot JSON (per-BB execution frequency) |
| `--output` | required | Output path for pattern catalog JSON |
| `--top` | 20 | Only include top N patterns in output |
| `--no-agent` | false | Skip agent analysis, run miner only |
| `--model` | sonnet | Claude model for agent analysis |

---

## Output Schemas

### Miner Output (JSON)

```json
{
  "generated": "2026-04-08T12:00:00Z",
  "source_df_count": 42,
  "pattern_count": 15,
  "patterns": [
    {
      "rank": 1,
      "opcodes": ["fadd.s", "fmul.s"],
      "register_class": "float",
      "length": 2,
      "occurrence_count": 8,
      "total_frequency": 150000,
      "chain_registers": [["frd", "frs1"]],
      "source_bbs": ["0x1234", "0x5678", "0xabcd"]
    }
  ]
}
```

### Agent Output (appended to JSON + text summary to stdout)

The Agent appends an `analysis` section to the JSON:

```json
{
  "patterns": [...],
  "analysis": {
    "top_recommendations": [
      {
        "pattern_rank": 1,
        "recommendation": "Strong candidate",
        "rationale": "High frequency (150K) in hot BBs, pure float pipeline, tight RAW dependency chain",
        "notes": "fadd+fmul is a classic FP fusion opportunity seen in other architectures"
      }
    ],
    "missed_patterns": [],
    "summary": "Top 3 float patterns in F-extension hot BBs dominate frequency. Integer patterns are more diverse but lower frequency."
  }
}
```

---

## Testing Strategy

### Unit Tests

- `test_pattern.py`:
  - Template key generation (same ops + same roles = same key)
  - Cross-register-class chains rejected
  - Control flow instructions rejected
  - Length-2 vs length-3 chain extraction
  - Edge cases: empty DFG, single-instruction BB, no RAW edges

- `test_miner.py`:
  - End-to-end mining with fixture DFG JSON files
  - Aggregation: identical patterns from different BBs merge correctly
  - Frequency weighting: BBV counts are summed correctly
  - Ranking: sorted by total_frequency descending
  - `--top` filtering

### Test Fixtures

Small, hand-crafted DFG JSON files in `tools/fusion/tests/fixtures/`:
- `float_chain_2.json`: simple 2-instruction float chain
- `float_chain_3.json`: 3-instruction float chain
- `mixed_class.json`: mixed integer/float (should be filtered)
- `no_raw.json`: instructions with no RAW edges (empty result)
- `branch_in_chain.json`: chain containing a branch (filtered)

### Coverage Target

>= 80% for new code (per project ground rules).

---

## Dependencies

| Internal dependency | Purpose |
|--------------------|---------|
| `tools/dfg/instruction.py` → `ISARegistry` | Resolve operand register roles for normalization |
| `tools/dfg/agent.py` → subprocess patterns | Reuse Claude CLI invocation patterns for Agent layer |

No new external dependencies. The Miner uses only stdlib + internal imports.

---

## Out of Scope

- Hardware constraint checking (F2)
- Instruction encoding proposals (F3)
- Modifying DFG engine core (parser, builder, output)
- Cycle-accurate simulation (Phase 3)

---

## Acceptance Criteria

1. Running against YOLO workload DFGs produces a non-empty pattern catalog
2. Patterns are ranked by BBV-weighted frequency (descending)
3. Top patterns correspond to the F-extension instruction sequences in the hot BBs
   identified in Phase 1
4. Agent analysis provides actionable recommendations (not just raw statistics)
5. `--no-agent` mode produces reproducible statistical output without AI
6. Unit test coverage >= 80% for new code
7. CLI integrates with existing `profile_to_dfg.sh` pipeline
