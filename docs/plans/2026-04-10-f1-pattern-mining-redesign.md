# F1 Pattern Mining Redesign

**Status**: Approved
**Date**: 2026-04-10
**Supersedes**: `docs/plans/2026-04-08-fusion-pattern-mining-design.md`

## Problem

The current F1 miner incorrectly assumes micro-architectural micro-op fusion: it only
scans adjacent instruction pairs `(i, i+1)` for RAW dependency edges. The actual
project goal is **instruction set extension design** — discovering data-flow subgraphs
in DFGs where multiple instructions could be replaced by a single new custom instruction,
implemented in hardware and compiler.

Real DFG data (YOLO matrix-multiply loop bodies, ~33 instructions per BB) contains
exclusively long-range edges (e.g., `flw[0] → fmadd.s[8]`) with zero adjacent RAW
pairs. The current miner produces 0 patterns from these blocks.

## Requirements (from brainstorming)

| Decision | Choice |
|----------|--------|
| Subgraph size | No hard limit; F2 constraints naturally filter oversized patterns |
| Subgraph shape | General connected subgraph mining, no shape priority |
| Output schema | Completely new schema for both F1 and F2 |
| Instruction order | Topological sort; same layer unordered |

## Design

### 1. Subgraph Enumeration Algorithm

**Seed + BFS expansion** over DFG nodes and edges.

1. For each node `v` in the DFG, create initial subgraph `S = {v}`
2. BFS expand: for current subgraph `S`, try adding any neighbor node `u` that has
   a direct RAW edge to/from any node in `S`. Accept if `|S| + 1 ≤ MAX_NODES`.
3. Pruning rules:
   - Subgraph must be **connected** (via DFG edges)
   - `|S| ≤ MAX_NODES` (default 8)
   - Control flow instructions are NOT excluded — F2 constraint checker decides
     feasibility
4. Enumerates all connected subgraphs up to `MAX_NODES` nodes. No adjacency
   requirement — edges may span arbitrary distances in the original instruction stream.

### 2. Pattern Normalization and Deduplication

Subgraphs from different BBs that share the same structure must be aggregated.

**Normalization steps:**

1. Extract the subgraph's edge set with concrete register names
2. Replace register names with operand role names (via ISARegistry)
   - e.g., `flw ft4 → fmadd.s ft4` becomes `flw.frd → fmadd.s.frs3`
3. Topological sort instructions by dependency layer; within each layer, sort
   opcodes lexicographically for stability
   - Layer 0: `{flw, flw}` (no predecessors)
   - Layer 1: `{fmadd.s}` (depends on layer 0)
4. Generate pattern key: `(topology, normalized_edge_roles, register_class)`
   - e.g., `(["flw","flw","fmadd.s"], [("frd","frs3"),("frd","frs2")], "float")`

**Deduplication:** Same key across BBs merges into one pattern; accumulate
`occurrence_count` and `total_frequency` (looked up from BBV hotspot report).

### 3. Output Schema (`fusion_patterns.json`)

```json
{
  "generated": "2026-04-10T...",
  "source_df_count": 2,
  "pattern_count": 5,
  "patterns": [
    {
      "rank": 1,
      "opcodes": ["flw", "flw", "fmadd.s"],
      "register_class": "float",
      "size": 3,
      "topology": [
        ["flw", "flw"],
        ["fmadd.s"]
      ],
      "edges": [
        {
          "src_layer": 0, "src_opcode": "flw",
          "dst_layer": 1, "dst_opcode": "fmadd.s",
          "src_role": "frd", "dst_role": "frs3"
        },
        {
          "src_layer": 0, "src_opcode": "flw",
          "dst_layer": 1, "dst_opcode": "fmadd.s",
          "src_role": "frd", "dst_role": "frs2"
        }
      ],
      "occurrence_count": 4,
      "total_frequency": 800000,
      "source_bbs": ["0x7d765747d1ea", "0x7d765747d988"],
      "examples": [
        {
          "bb_id": 84469,
          "vaddr": "0x7d765747d1ea",
          "instructions": [
            {"index": 0, "mnemonic": "flw", "operands": "ft4,0(a5)"},
            {"index": 4, "mnemonic": "flw", "operands": "fa6,0(s0)"},
            {"index": 8, "mnemonic": "fmadd.s", "operands": "dyn,fa5,fa6,ft4,fa5"}
          ]
        }
      ]
    }
  ],
  "rejected_combinations": { ... }
}
```

**Key changes from old schema:**
- `chain_registers` (linear chain role pairs) → `edges` (general edge list with topology info)
- New `topology` field (topological layer structure)
- New `examples` field (concrete instances per source BB for F3 reference)
- Retained `rejected_combinations` for diagnostics

### 4. F2 Adaptation

**ConstraintChecker:**
- Input changes from linear chain to subgraph pattern
- Internal vs external register derivation from `edges` list:
  - Registers on edges → internal (passed within the new instruction)
  - Operands not on edges → external (I/O ports of the new instruction)
- New constraint options (disabled by default):
  - `no_control_flow`: subgraph contains control flow → infeasible
  - `max_nodes`: subgraph node count exceeds limit → infeasible
  - `must_be_single_consumer`: subgraph must have exactly one root node
- Existing constraints (`encoding_32bit`, `operand_format`, `too_many_sources`, etc.)
  preserved, adapted to new input format

**Scorer:**
- `tight_score`: changed from "edge density / max edges" to "edge count / node count"
  ratio, measuring data-flow compactness of the subgraph
- `freq_score` and `hw_score` logic unchanged

**`fusion_candidates.json`** output structure preserved; pattern section uses new schema.

### 5. Module Structure

```
tools/fusion/
├── subgraph.py           # NEW: DFG subgraph enumeration (seed + expand + prune)
├── pattern.py            # REWRITE: SubgraphPattern model + normalization
├── miner.py              # REWRITE: orchestration (load → enumerate → normalize → aggregate → write)
├── constraints.py        # MODIFY: adapt to subgraph input, new constraint items
├── scorer.py             # MODIFY: adapt to subgraph input, tight_score adjustment
├── __main__.py           # MODIFY: CLI args (--max-nodes, etc.)
├── agent.py              # UNCHANGED
├── scheme_validator.py   # UNCHANGED
└── tests/
    ├── test_subgraph.py      # NEW: subgraph enumeration tests
    ├── test_pattern.py       # NEW: normalization + dedup tests
    ├── test_miner.py         # REWRITE: end-to-end mining tests
    ├── test_integration.py   # MODIFY: scoring pipeline tests
    └── fixtures/             # EXTEND: add non-adjacent edge DFG fixtures
```

| Module | Responsibility |
|--------|---------------|
| `subgraph.py` | Pure algorithm: given DFG nodes + edges, enumerate all valid connected subgraphs |
| `pattern.py` | Data model + normalization: subgraph → SubgraphPattern template, dedup key generation |
| `miner.py` | Orchestration: load DFGs → subgraph enumeration → pattern normalization → cross-BB aggregation → write JSON |
