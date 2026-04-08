# Phase 2 Feature Roadmap: Fusion Candidate Discovery & Design

**Date**: 2026-04-08
**Status**: Planning
**Predecessor**: Phase 1 (Setup + Profiling + DFG Generation) — Completed

---

## Context

Phase 1 delivered a fully automated pipeline: QEMU BBV profiling, hotspot analysis
(`analyze_bbv.py`), DFG generation (`tools/dfg/` with I/F/M ISA extensions), and
end-to-end orchestration (`setup.sh`, `profile_to_dfg.sh`).

Phase 2 builds on top of this foundation to **discover and rank fusible instruction
patterns** from real workload DFGs. The goal is to produce a prioritized list of
instruction combination candidates with hardware feasibility analysis, ready for
simulation in Phase 3.

## Feature Dependency Chain

```text
Feature 1: Pattern Mining  ──→  Feature 2: Scoring & Constraints  ──→  Feature 3: Scheme Specification
    (algorithms)                    (hardware model)                    (agent skill)
```

Each feature is self-contained: it has its own design doc, implementation plan, test
suite, and merge commit. Features must be completed in order — each one consumes the
previous feature's output.

---

## Feature 1: Fusion Pattern Mining Engine

**Status**: Not started
**Design doc**: TBD (`docs/plans/2026-04-XX-fusion-pattern-mining-design.md`)
**Branch prefix**: `worktree-fusion-mining`

### Goal

Scan DFG JSON output to discover recurring instruction subgraphs with RAW data
dependencies across hot basic blocks. Produce a ranked pattern catalog weighted by
execution frequency.

### Input

- DFG JSON files from `tools/dfg/` (per-BB dependency graphs)
- BBV hotspot report JSON from `analyze_bbv.py --json-output`

### Output

- Fusion pattern catalog (`output/fusion_patterns.json`)
  - Each entry: instruction template, RAW dependency chain, occurrence count,
    BBV-weighted frequency, source BB addresses
- Text summary report for human review

### Scope

- Define "fusible pattern" rules: 2-3 instruction sequences with RAW edges,
  same register kind (integer/float), no control flow between them
- Subgraph enumeration: extract all valid pair/triple patterns per BB
- Cross-BB aggregation: merge identical patterns, sum frequencies weighted by
  BBV execution count
- Output module: JSON catalog + text summary
- CLI: `python -m tools.fusion mine --dfg <dir> --report <json> [--top N]`
- Unit tests: pattern extraction, aggregation, frequency weighting, edge cases

### Out of Scope

- Hardware constraint checking (Feature 2)
- Instruction encoding proposals (Feature 3)
- Modifying the DFG engine core (parser, builder, output)

### Acceptance Criteria

1. Running against YOLO workload DFGs produces a non-empty pattern catalog
2. Patterns are ranked by BBV-weighted frequency (descending)
3. Top patterns correspond to the 4 hot BBs identified in Phase 1 (F-extension
   instruction sequences in `libonnxruntime.so`)
4. Unit test coverage >= 80% for new code
5. CLI integrates with existing `profile_to_dfg.sh` pipeline (or new orchestration
   script)

---

## Feature 2: Fusion Candidate Scoring & Hardware Constraint Model

**Status**: Not started
**Design doc**: TBD (`docs/plans/2026-04-XX-fusion-scoring-design.md`)
**Branch prefix**: `worktree-fusion-scoring`

### Goal

Score and rank fusion patterns by combining execution frequency with hardware
feasibility. Encode RISC-V hardware constraints that determine which patterns can
realistically be fused in silicon.

### Input

- Pattern catalog (Feature 1 output)
- RISC-V ISA encoding specification (32-bit instruction format)

### Output

- Ranked candidate list (`output/fusion_candidates.json`)
  - Each entry: pattern, score breakdown (frequency, dependency tightness,
    hardware feasibility), constraint notes, feasibility verdict
- Hardware constraint model module (`tools/fusion/constraints.py`)

### Scope

- Hardware constraint model:
  - **Encoding space budget**: verify candidate fits within 32-bit instruction
    encoding (opcode, funct3, funct7, rs1/rs2/rd fields)
  - **Pipeline compatibility**: same execution unit class, no structural hazards
  - **Operand constraints**: register class compatibility (integer/float),
    immediate field sizes, source/destination count limits
- Scoring function: `score = frequency × dependency_tightness × hardware_feasibility`
- Dependency tightness metric: RAW edge density, chain length, register reuse
- CLI filtering: `--min-score`, `--min-frequency`, `--max-operands`, `--feasibility-only`
- Unit tests: constraint validation for known feasible/infeasible patterns, scoring
  edge cases, encoding space verification

### Out of Scope

- Actual instruction encoding generation (Feature 3)
- Cycle-accurate simulation (Phase 3)
- D/A/V extension-specific constraints (Phase 4)

### Acceptance Criteria

1. Every candidate has a feasibility verdict: `feasible`, `constrained`, or `infeasible`
   with written rationale
2. Score breakdown is transparent (individual component scores visible)
3. Known infeasible patterns (e.g., cross-register-class, >3 operands) are correctly
   rejected
4. Top candidates pass manual review against RISC-V unprivileged ISA spec
5. Constraint model is extensible for future extensions (D, A, V)
6. Unit test coverage >= 80% for new code

---

## Feature 3: Fusion Scheme Specification Agent Skill

**Status**: Not started
**Design doc**: TBD (`docs/plans/2026-04-XX-fusion-scheme-skill-design.md`)
**Branch prefix**: `worktree-fusion-scheme`

### Goal

Agent skill that takes a ranked candidate and generates a complete fusion scheme —
encoding layout, instruction semantics, and verification constraints — ready for
simulation implementation in Phase 2.

### Input

- Ranked candidate (Feature 2 output): pattern + frequency + constraint notes
- RISC-V ISA specification context (fed to agent via prompt)

### Output

- Fusion scheme documents (`output/fusion_schemes/<candidate_id>.json` and `.md`)
  - Encoding layout: opcode, funct3, funct7, rs1/rs2/rd field assignments
  - Fused instruction mnemonic and assembly syntax
  - Read/write register semantics (preserved from original instructions)
  - Latency model estimate
  - Constraint compliance checklist
- Batch mode: generate schemes for top-N candidates

### Scope

- Design scheme document format (JSON schema + Markdown template)
- Agent skill prompt engineering: feed pattern + constraints, request encoding
  proposal with validation
- Constraint-aware validation: verify proposed encoding doesn't conflict with
  existing RISC-V ISA opcodes/funct fields
- Integration with existing `agent.py` dispatcher (reuse `--model`, `--no-agent`
  patterns)
- Batch CLI: `python -m tools.fusion scheme --candidates <json> --top 5`
- Output to `output/fusion_schemes/`

### Out of Scope

- Actual hardware implementation or RTL
- Cycle-accurate simulation (Phase 3)
- Automated encoding allocation (agent proposes, human validates)

### Acceptance Criteria

1. For the top-ranked YOLO candidate, the agent produces a scheme with a valid
   encoding layout that passes manual cross-check against RISC-V ISA spec
2. Scheme JSON conforms to defined schema (machine-parseable for Phase 2 consumption)
3. Markdown scheme is human-readable and suitable for research documentation
4. `--no-agent` mode produces a scheme template (placeholder encoding) without
   calling Claude
5. Batch mode correctly processes multiple candidates
6. Unit test coverage >= 80% for non-agent code (validation, formatting, schema)

---

## Cross-Feature Concerns

### New Module: `tools/fusion/`

Features 1-3 share a common Python package under `tools/fusion/`:

```text
tools/fusion/
├── __init__.py
├── __main__.py          # CLI entry point (mine, score, scheme subcommands)
├── miner.py             # Feature 1: pattern extraction & aggregation
├── scorer.py            # Feature 2: scoring function
├── constraints.py       # Feature 2: hardware constraint model
├── scheme.py            # Feature 3: scheme generation & validation
├── agent.py             # Feature 3: agent dispatcher (extends dfg/agent.py patterns)
└── tests/               # Unit tests for all features
```

### Integration Points with Existing Code

| Existing Component | Integration |
|--------------------|-------------|
| `tools/dfg/` (DFG JSON output) | Feature 1 reads `dfg/json/*.json` |
| `tools/analyze_bbv.py` (hotspot JSON) | Feature 1 reads hotspot report for BBV weighting |
| `tools/dfg/agent.py` (agent patterns) | Feature 3 reuses `--model`, `--no-agent` patterns |
| `profile_to_dfg.sh` (orchestration) | New `profile_to_fusion.sh` orchestrates full pipeline |
| `setup.sh` (Step 7 report) | Extend report to include fusion candidate summary |

### Technical Debt Items Addressed

| TD ID | Description | Feature |
|--------|-------------|---------|
| TD-001 | Fusion candidate search algorithm undefined | Feature 1 |
| TD-002 | Hardware constraint modeling undefined | Feature 2 |

### Downstream Impact on Phase 3

Phase 3 (Simulation and Benefit Quantification) will consume:
- Ranked candidate list (Feature 2 output) as simulation targets
- Fusion scheme documents (Feature 3 output) as instruction behavior specs
- Pattern frequency data for BBV-weighted benefit calculation

---

## How to Pick Up a Feature

1. Read this roadmap for the feature's scope and acceptance criteria
2. Check if a design doc exists in `docs/plans/` — if not, create one using the
   Superpowers workflow (`/writing-plans` skill)
3. Follow the standard development workflow: design → plan → implement → verify → merge
4. After merge, update the feature's status in this document from "Not started" to
   "Completed" with the merge commit hash
