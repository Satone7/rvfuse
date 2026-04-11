# F1 Subgraph Mining Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the fusion pattern miner to discover ISA extension candidates by enumerating connected DFG subgraphs (not just adjacent instruction pairs), normalizing them into templates, and aggregating across basic blocks.

**Architecture:** Seed-based BFS enumeration grows connected subgraphs from each DFG node up to `MAX_NODES` (default 8). Each subgraph is normalized to a `SubgraphPattern` via topological layering + register role abstraction. Patterns with identical structure are deduplicated and ranked by BBV-weighted frequency. F2's constraint checker and scorer are adapted to consume the new subgraph-based schema.

**Tech Stack:** Python 3.10+, stdlib only (json, dataclasses, collections), existing `dfg.instruction.ISARegistry` for register flow resolution.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `tools/fusion/subgraph.py` | Create | Connected subgraph enumeration algorithm |
| `tools/fusion/pattern.py` | Rewrite | `SubgraphPattern` model + topological normalization + role abstraction |
| `tools/fusion/miner.py` | Rewrite | Pipeline: load DFGs → enumerate → normalize → aggregate → write JSON |
| `tools/fusion/constraints.py` | Modify | Adapt `check()` to consume subgraph edges instead of linear chain_registers |
| `tools/fusion/scorer.py` | Modify | Adapt `tight_score` to subgraph density; consume new pattern schema |
| `tools/fusion/__main__.py` | Modify | Add `--max-nodes` CLI arg; remove `--no-agent` gate on pattern output |
| `tools/fusion/tests/test_subgraph.py` | Create | Unit tests for subgraph enumeration |
| `tools/fusion/tests/test_pattern.py` | Create | Unit tests for subgraph normalization |
| `tools/fusion/tests/test_miner.py` | Rewrite | Unit tests for rewritten miner pipeline |
| `tools/fusion/tests/test_integration.py` | Modify | Update F2 integration tests for new schema |
| `tools/fusion/tests/fixtures/fan_in_2to1.json` | Create | Test fixture: two flw → one fmadd.s |
| `tools/fusion/tests/fixtures/fan_out_1to2.json` | Create | Test fixture: one flw → two fmadd.s |

---

### Task 1: Create subgraph.py — Connected Subgraph Enumeration

**Files:**
- Create: `tools/fusion/subgraph.py`
- Test: `tools/fusion/tests/test_subgraph.py`

- [ ] **Step 1: Write failing tests for subgraph enumeration**

Create `tools/fusion/tests/test_subgraph.py`:

```python
"""Tests for DFG connected subgraph enumeration."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FIXTURES = Path(__file__).parent / "fixtures"


class TestEnumerateSubgraphs(unittest.TestCase):
    """enumerate_subgraphs() produces all connected subgraphs up to max_nodes."""

    def test_single_node_dfg(self):
        """A DFG with one node produces one subgraph of size 1."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = {"nodes": [{"index": 0, "mnemonic": "addi", "operands": "a0,a0,1"}], "edges": []}
        result = enumerate_subgraphs(dfg, max_nodes=8)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], frozenset({0}))

    def test_two_nodes_with_edge(self):
        """Two connected nodes produce subgraphs: {0}, {1}, {0,1}."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
                {"index": 1, "mnemonic": "fmul.s", "operands": "ft3,ft2,fa2"},
            ],
            "edges": [{"src": 0, "dst": 1, "register": "ft2"}],
        }
        result = enumerate_subgraphs(dfg, max_nodes=8)
        node_sets = [frozenset(s) for s in result]
        self.assertIn(frozenset({0}), node_sets)
        self.assertIn(frozenset({1}), node_sets)
        self.assertIn(frozenset({0, 1}), node_sets)
        self.assertEqual(len(result), 3)

    def test_two_nodes_no_edge(self):
        """Two disconnected nodes produce two single-node subgraphs."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "flw", "operands": "ft2,0(a0)"},
                {"index": 1, "mnemonic": "flw", "operands": "ft3,4(a0)"},
            ],
            "edges": [],
        }
        result = enumerate_subgraphs(dfg, max_nodes=8)
        node_sets = [frozenset(s) for s in result]
        self.assertEqual(len(result), 2)
        self.assertIn(frozenset({0}), node_sets)
        self.assertIn(frozenset({1}), node_sets)

    def test_fan_in_2to1(self):
        """Two sources feeding one consumer: 7 subgraphs total."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "flw", "operands": "ft4,0(a5)"},
                {"index": 1, "mnemonic": "flw", "operands": "fa6,0(s0)"},
                {"index": 2, "mnemonic": "fmadd.s", "operands": "dyn,fa5,fa6,ft4,fa5"},
            ],
            "edges": [
                {"src": 0, "dst": 2, "register": "ft4"},
                {"src": 1, "dst": 2, "register": "fa6"},
            ],
        }
        result = enumerate_subgraphs(dfg, max_nodes=8)
        node_sets = [frozenset(s) for s in result]
        # Single nodes: {0}, {1}, {2}
        # Pairs: {0,2}, {1,2} (0-1 not connected)
        # Triple: {0,1,2}
        self.assertEqual(len(result), 6)
        self.assertIn(frozenset({0, 1, 2}), node_sets)

    def test_three_node_chain(self):
        """Linear chain 0→1→2: 6 subgraphs."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
                {"index": 1, "mnemonic": "fmul.s", "operands": "ft3,ft2,fa2"},
                {"index": 2, "mnemonic": "fsub.s", "operands": "ft4,ft3,fa3"},
            ],
            "edges": [
                {"src": 0, "dst": 1, "register": "ft2"},
                {"src": 1, "dst": 2, "register": "ft3"},
            ],
        }
        result = enumerate_subgraphs(dfg, max_nodes=8)
        self.assertEqual(len(result), 6)
        # {0}, {1}, {2}, {0,1}, {1,2}, {0,1,2}

    def test_max_nodes_limits_size(self):
        """Subgraphs larger than max_nodes are excluded."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "flw", "operands": "ft4,0(a5)"},
                {"index": 1, "mnemonic": "flw", "operands": "fa6,0(s0)"},
                {"index": 2, "mnemonic": "fmadd.s", "operands": "dyn,fa5,fa6,ft4,fa5"},
            ],
            "edges": [
                {"src": 0, "dst": 2, "register": "ft4"},
                {"src": 1, "dst": 2, "register": "fa6"},
            ],
        }
        result = enumerate_subgraphs(dfg, max_nodes=2)
        for s in result:
            self.assertLessEqual(len(s), 2)

    def test_no_duplicate_subgraphs(self):
        """Each connected subgraph appears exactly once."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
                {"index": 1, "mnemonic": "fmul.s", "operands": "ft3,ft2,fa2"},
            ],
            "edges": [{"src": 0, "dst": 1, "register": "ft2"}],
        }
        result = enumerate_subgraphs(dfg, max_nodes=8)
        seen = []
        for s in result:
            fs = frozenset(s)
            self.assertNotIn(fs, seen, f"Duplicate subgraph: {fs}")
            seen.append(fs)

    def test_real_dfg_bb_84469(self):
        """Real 33-node DFG produces subgraphs of all sizes up to max."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        result = enumerate_subgraphs(dfg, max_nodes=8)
        sizes = sorted(len(s) for s in result)
        self.assertEqual(sizes[0], 1)
        self.assertLessEqual(sizes[-1], 8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_subgraph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fusion.subgraph'`

- [ ] **Step 3: Implement `enumerate_subgraphs` in `tools/fusion/subgraph.py`**

Create `tools/fusion/subgraph.py`:

```python
"""Connected subgraph enumeration from DFG node/edge data.

Given a DFG (nodes + edges), enumerates all connected subgraphs up to
a maximum node count. Uses seed-based expansion with canonical ordering
(seed = minimum node index) to guarantee each subgraph is produced once.
"""

from __future__ import annotations

from collections import defaultdict


def enumerate_subgraphs(
    dfg_data: dict,
    max_nodes: int = 8,
) -> list[frozenset[int]]:
    """Enumerate all connected subgraphs in a DFG.

    Args:
        dfg_data: DFG JSON dict with 'nodes' (list of {index, mnemonic, operands})
            and 'edges' (list of {src, dst, register}).
        max_nodes: Maximum number of nodes per subgraph (inclusive).

    Returns:
        List of frozensets, each containing node indices of a connected subgraph.
        Single-node subgraphs are always included. No duplicates.
    """
    nodes = dfg_data["nodes"]
    edges = dfg_data["edges"]

    if not nodes:
        return []

    # Build undirected adjacency (DFG edges are directional but connectivity
    # is undirected — both endpoints share the dependency relationship).
    adj: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        adj[e["src"]].add(e["dst"])
        adj[e["dst"]].add(e["src"])

    results: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()

    for seed in range(len(nodes)):
        # Only enumerate subgraphs where `seed` is the minimum index.
        # This guarantees each subgraph is generated exactly once.
        stack: list[frozenset[int]] = [frozenset([seed])]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            results.append(current)

            if len(current) >= max_nodes:
                continue

            # Collect all neighbors of the current subgraph that have
            # index > seed (to maintain canonical ordering).
            candidates: set[int] = set()
            for node in current:
                for nb in adj[node]:
                    if nb not in current and nb > seed:
                        candidates.add(nb)

            for nb in candidates:
                expanded = current | {nb}
                if expanded not in seen:
                    stack.append(expanded)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_subgraph.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Create fan_in_2to1.json fixture**

Create `tools/fusion/tests/fixtures/fan_in_2to1.json`:

```json
{
  "bb_id": 10,
  "vaddr": "0xA000",
  "source": "script",
  "nodes": [
    {"index": 0, "address": "0xA000", "mnemonic": "flw", "operands": "ft4,0(a5)"},
    {"index": 1, "address": "0xA004", "mnemonic": "flw", "operands": "fa6,0(s0)"},
    {"index": 2, "address": "0xA008", "mnemonic": "fmadd.s", "operands": "dyn,fa5,fa6,ft4,fa5"}
  ],
  "edges": [
    {"src": 0, "dst": 2, "register": "ft4"},
    {"src": 1, "dst": 2, "register": "fa6"}
  ]
}
```

- [ ] **Step 6: Commit**

```bash
git add tools/fusion/subgraph.py tools/fusion/tests/test_subgraph.py tools/fusion/tests/fixtures/fan_in_2to1.json
git commit -m "feat(fusion): add connected subgraph enumeration algorithm

Seed-based BFS expansion with canonical ordering (seed = minimum
node index) guarantees each connected subgraph is produced exactly once.
Capped at max_nodes (default 8) to control combinatorial growth.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Rewrite pattern.py — Subgraph Pattern Normalization

**Files:**
- Rewrite: `tools/fusion/pattern.py`
- Test: `tools/fusion/tests/test_pattern.py`
- Fixture: `tools/fusion/tests/fixtures/fan_in_2to1.json` (from Task 1)

- [ ] **Step 1: Write failing tests for subgraph normalization**

Create `tools/fusion/tests/test_pattern.py`:

```python
"""Tests for subgraph pattern normalization."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FIXTURES = Path(__file__).parent / "fixtures"


def _make_registry():
    from dfg.instruction import ISARegistry
    from dfg.isadesc.rv64f import build_registry as build_f
    from dfg.isadesc.rv64i import build_registry as build_i
    reg = ISARegistry()
    build_i(reg)
    build_f(reg)
    return reg


class TestNormalizeSubgraph(unittest.TestCase):
    """normalize_subgraph() converts a DFG subgraph into a SubgraphPattern."""

    def setUp(self):
        self.registry = _make_registry()

    def test_two_node_chain(self):
        """fadd.s → fmul.s chain: layer0=[fadd.s], layer1=[fmul.s]."""
        from fusion.pattern import normalize_subgraph
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
                {"index": 1, "mnemonic": "fmul.s", "operands": "ft3,ft2,fa2"},
            ],
            "edges": [{"src": 0, "dst": 1, "register": "ft2"}],
        }
        pattern = normalize_subgraph(frozenset({0, 1}), dfg, self.registry)
        self.assertEqual(pattern.size, 2)
        self.assertEqual(pattern.topology, [["fadd.s"], ["fmul.s"]])
        self.assertEqual(pattern.register_class, "float")
        self.assertEqual(len(pattern.edges), 1)
        self.assertEqual(pattern.edges[0].src_role, "frd")
        self.assertEqual(pattern.edges[0].dst_role, "frs1")
        self.assertEqual(pattern.edges[0].src_layer, 0)
        self.assertEqual(pattern.edges[0].dst_layer, 1)

    def test_fan_in_2to1(self):
        """Two flw → fmadd.s: layer0=[flw, flw], layer1=[fmadd.s], 2 edges."""
        from fusion.pattern import normalize_subgraph
        dfg = json.loads((FIXTURES / "fan_in_2to1.json").read_text())
        pattern = normalize_subgraph(frozenset({0, 1, 2}), dfg, self.registry)
        self.assertEqual(pattern.size, 3)
        self.assertEqual(pattern.topology, [["flw", "flw"], ["fmadd.s"]])
        self.assertEqual(pattern.register_class, "float")
        self.assertEqual(len(pattern.edges), 2)
        # Verify role pairs (order may vary due to set operations)
        role_pairs = sorted((e.src_role, e.dst_role) for e in pattern.edges)
        self.assertIn(("frd", "frs2"), role_pairs)
        self.assertIn(("frd", "frs3"), role_pairs)

    def test_same_layer_alphabetical(self):
        """Nodes in the same topological layer are sorted alphabetically."""
        from fusion.pattern import normalize_subgraph
        dfg = json.loads((FIXTURES / "fan_in_2to1.json").read_text())
        pattern = normalize_subgraph(frozenset({0, 1, 2}), dfg, self.registry)
        self.assertEqual(pattern.topology[0], ["flw", "flw"])

    def test_template_key_dedup(self):
        """Structurally identical subgraphs produce the same template_key."""
        from fusion.pattern import normalize_subgraph
        dfg_a = {
            "nodes": [
                {"index": 0, "mnemonic": "flw", "operands": "ft4,0(a5)"},
                {"index": 1, "mnemonic": "flw", "operands": "fa6,0(s0)"},
                {"index": 2, "mnemonic": "fmadd.s", "operands": "dyn,fa5,fa6,ft4,fa5"},
            ],
            "edges": [
                {"src": 0, "dst": 2, "register": "ft4"},
                {"src": 1, "dst": 2, "register": "fa6"},
            ],
        }
        dfg_b = {
            "nodes": [
                {"index": 0, "mnemonic": "flw", "operands": "ft0,8(a1)"},
                {"index": 1, "mnemonic": "flw", "operands": "ft1,16(a2)"},
                {"index": 2, "mnemonic": "fmadd.s", "operands": "dyn,ft2,ft1,ft0,ft2"},
            ],
            "edges": [
                {"src": 0, "dst": 2, "register": "ft0"},
                {"src": 1, "dst": 2, "register": "ft1"},
            ],
        }
        p_a = normalize_subgraph(frozenset({0, 1, 2}), dfg_a, self.registry)
        p_b = normalize_subgraph(frozenset({0, 1, 2}), dfg_b, self.registry)
        self.assertEqual(p_a.template_key, p_b.template_key)

    def test_single_node(self):
        """Single-node subgraph: one layer, no edges."""
        from fusion.pattern import normalize_subgraph
        dfg = {
            "nodes": [{"index": 0, "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"}],
            "edges": [],
        }
        pattern = normalize_subgraph(frozenset({0}), dfg, self.registry)
        self.assertEqual(pattern.size, 1)
        self.assertEqual(pattern.topology, [["fadd.s"]])
        self.assertEqual(len(pattern.edges), 0)

    def test_unknown_mnemonic_raises(self):
        """Unknown mnemonic in subgraph raises ValueError."""
        from fusion.pattern import normalize_subgraph
        dfg = {
            "nodes": [
                {"index": 0, "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
                {"index": 1, "mnemonic": "unknown_op", "operands": "x1,x2,x3"},
            ],
            "edges": [{"src": 0, "dst": 1, "register": "ft2"}],
        }
        with self.assertRaises(ValueError):
            normalize_subgraph(frozenset({0, 1}), dfg, self.registry)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_pattern.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_subgraph' from 'fusion.pattern'`

- [ ] **Step 3: Rewrite `tools/fusion/pattern.py`**

Replace the entire file with:

```python
"""Subgraph pattern model and normalization for fusion candidate discovery.

Converts a connected DFG subgraph (set of node indices) into a normalized
SubgraphPattern template:
  1. Extract edges within the subgraph
  2. Map concrete registers to operand roles via ISARegistry
  3. Compute topological layers (same-layer opcodes sorted alphabetically)
  4. Build a hashable template_key for cross-BB deduplication
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Register classification patterns (matching dfg/instruction.py)
_INTEGER_RE = re.compile(r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|s\d+|jt?\d*)$")
_FLOAT_RE = re.compile(r"^(f\d+|ft\d+|fs\d+|fa\d+|fv\d+)$")


def classify_register(reg_name: str) -> str | None:
    """Return 'integer', 'float', or None for a register name."""
    if _INTEGER_RE.match(reg_name):
        return "integer"
    if _FLOAT_RE.match(reg_name):
        return "float"
    return None


def _classify_subgraph(
    node_indices: set[int],
    nodes: list[dict],
    registry,
) -> str:
    """Determine the dominant register class of a subgraph.

    Returns the register class of the first node that has a known class.
    Returns "unknown" if no node has a known class.
    """
    for idx in sorted(node_indices):
        mn = nodes[idx]["mnemonic"]
        flow = registry.get_flow(mn)
        if flow is None:
            continue
        resolved = flow.resolve(nodes[idx]["operands"])
        all_regs = resolved.dst_regs + resolved.src_regs
        if all_regs:
            cls = classify_register(all_regs[0])
            if cls:
                return cls
    return "unknown"


def _topological_layers(
    node_indices: set[int],
    intra_edges: list[dict],
    nodes: list[dict],
) -> list[list[str]]:
    """Compute topological layers of the subgraph.

    Layer 0 = nodes with no incoming edges within the subgraph.
    Remove layer 0 nodes and repeat. Within each layer, sort opcodes
    alphabetically for deterministic ordering.
    """
    remaining = set(node_indices)
    # Build in-degree map (edges within subgraph only)
    in_degree: dict[int, int] = {n: 0 for n in remaining}
    fwd: dict[int, set[int]] = {n: set() for n in remaining}
    for e in intra_edges:
        if e["src"] in remaining and e["dst"] in remaining:
            in_degree[e["dst"]] = in_degree.get(e["dst"], 0) + 1
            fwd[e["src"]].add(e["dst"])

    layers: list[list[str]] = []
    while remaining:
        layer_nodes = sorted(n for n in remaining if in_degree.get(n, 0) == 0)
        if not layer_nodes:
            # Cycle (shouldn't happen in RAW DFGs) — break remaining as one layer
            layer_nodes = sorted(remaining)
        layer_opcodes = sorted(nodes[n]["mnemonic"] for n in layer_nodes)
        layers.append(layer_opcodes)
        for n in layer_nodes:
            remaining.discard(n)
            for nb in fwd.get(n, set()):
                in_degree[nb] -= 1

    return layers


def _compute_node_layer(
    node_indices: set[int],
    intra_edges: list[dict],
) -> dict[int, int]:
    """Map each node index to its topological layer number."""
    remaining = set(node_indices)
    in_degree: dict[int, int] = {n: 0 for n in remaining}
    fwd: dict[int, set[int]] = {n: set() for n in remaining}
    for e in intra_edges:
        if e["src"] in remaining and e["dst"] in remaining:
            in_degree[e["dst"]] = in_degree.get(e["dst"], 0) + 1
            fwd[e["src"]].add(e["dst"])

    layer_map: dict[int, int] = {}
    layer_num = 0
    while remaining:
        layer_nodes = sorted(n for n in remaining if in_degree.get(n, 0) == 0)
        if not layer_nodes:
            layer_nodes = sorted(remaining)
        for n in layer_nodes:
            layer_map[n] = layer_num
            remaining.discard(n)
            for nb in fwd.get(n, set()):
                in_degree[nb] -= 1
        layer_num += 1

    return layer_map


def _find_role(
    reg_name: str,
    resolved_regs: list[str],
    role_names: list[str],
) -> str | None:
    """Find the role name corresponding to a concrete register name."""
    role_map = dict(zip(resolved_regs, role_names))
    return role_map.get(reg_name)


def _normalize_edges(
    intra_edges: list[dict],
    node_indices: set[int],
    nodes: list[dict],
    layer_map: dict[int, int],
    registry,
) -> list[NormalizedEdge]:
    """Convert concrete DFG edges to role-abstracted NormalizedEdge list."""
    result: list[NormalizedEdge] = []
    for e in intra_edges:
        if e["src"] not in node_indices or e["dst"] not in node_indices:
            continue
        reg_name = e["register"]
        src_mn = nodes[e["src"]]["mnemonic"]
        dst_mn = nodes[e["dst"]]["mnemonic"]
        src_ops = nodes[e["src"]]["operands"]
        dst_ops = nodes[e["dst"]]["operands"]

        src_flow = registry.get_flow(src_mn)
        dst_flow = registry.get_flow(dst_mn)
        if src_flow is None or dst_flow is None:
            continue

        src_resolved = src_flow.resolve(src_ops)
        dst_resolved = dst_flow.resolve(dst_ops)

        src_role = _find_role(reg_name, src_resolved.dst_regs, src_flow.dst_regs)
        dst_role = _find_role(reg_name, dst_resolved.src_regs, dst_flow.src_regs)

        if src_role and dst_role:
            result.append(NormalizedEdge(
                src_layer=layer_map[e["src"]],
                src_opcode=src_mn,
                dst_layer=layer_map[e["dst"]],
                dst_opcode=dst_mn,
                src_role=src_role,
                dst_role=dst_role,
            ))
    return result


@dataclass
class NormalizedEdge:
    """A role-abstracted edge between two instructions in the pattern."""

    src_layer: int
    src_opcode: str
    dst_layer: int
    dst_opcode: str
    src_role: str
    dst_role: str


@dataclass
class SubgraphPattern:
    """A normalized fusible subgraph template.

    Attributes:
        topology: Topological layers of opcodes.
            e.g. [["flw", "flw"], ["fmadd.s"]]
        edges: Role-abstracted edges between instructions.
        register_class: "integer", "float", or "unknown".
        size: Number of instructions in the subgraph.
    """

    topology: list[list[str]]
    edges: list[NormalizedEdge]
    register_class: str
    size: int

    @property
    def template_key(self) -> tuple:
        """Hashable key for grouping identical patterns across BBs."""
        topo_tuple = tuple(tuple(layer) for layer in self.topology)
        edge_tuple = tuple(
            (e.src_layer, e.src_opcode, e.dst_layer, e.dst_opcode, e.src_role, e.dst_role)
            for e in sorted(self.edges, key=lambda e: (e.src_layer, e.dst_layer, e.src_role, e.dst_role))
        )
        return (topo_tuple, self.register_class, edge_tuple)


def normalize_subgraph(
    node_indices: frozenset[int] | set[int],
    dfg_data: dict,
    registry,
) -> SubgraphPattern:
    """Normalize a DFG subgraph into a SubgraphPattern template.

    Args:
        node_indices: Set of DFG node indices forming the subgraph.
        dfg_data: DFG JSON dict with 'nodes' and 'edges'.
        registry: ISA registry for register flow resolution.

    Returns:
        A SubgraphPattern with normalized topology, edges, and register class.

    Raises:
        ValueError: If any mnemonic is unknown in the registry.
    """
    node_set = set(node_indices)
    nodes = dfg_data["nodes"]
    all_edges = dfg_data["edges"]

    # Validate all mnemonics are known
    for idx in node_set:
        mn = nodes[idx]["mnemonic"]
        if registry.get_flow(mn) is None:
            raise ValueError(f"unknown instruction: {mn}")

    # Filter edges to those within the subgraph
    intra_edges = [
        e for e in all_edges
        if e["src"] in node_set and e["dst"] in node_set
    ]

    # Classify register class
    reg_class = _classify_subgraph(node_set, nodes, registry)

    # Compute topological layers and node-to-layer mapping
    layer_map = _compute_node_layer(node_set, intra_edges)
    layers = _topological_layers(node_set, intra_edges, nodes)

    # Normalize edges
    norm_edges = _normalize_edges(intra_edges, node_set, nodes, layer_map, registry)

    return SubgraphPattern(
        topology=layers,
        edges=norm_edges,
        register_class=reg_class,
        size=len(node_set),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_pattern.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/fusion/pattern.py tools/fusion/tests/test_pattern.py
git commit -m "refactor(fusion): rewrite pattern.py for subgraph normalization

SubgraphPattern replaces linear chain Pattern. Topological layering
with same-layer alphabetical sort. Role-abstracted NormalizedEdge
replaces linear chain_registers. Template key supports cross-BB
deduplication of arbitrary graph shapes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Rewrite miner.py — Subgraph-Based Pipeline

**Files:**
- Rewrite: `tools/fusion/miner.py`
- Rewrite: `tools/fusion/tests/test_miner.py`

- [ ] **Step 1: Write failing tests for the rewritten miner**

Replace `tools/fusion/tests/test_miner.py` with:

```python
"""Tests for the subgraph-based fusion pattern mining pipeline."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FIXTURES = Path(__file__).parent / "fixtures"


def _make_registry():
    from dfg.instruction import ISARegistry
    from dfg.isadesc.rv64f import build_registry as build_f
    from dfg.isadesc.rv64i import build_registry as build_i
    reg = ISARegistry()
    build_i(reg)
    build_f(reg)
    return reg


class TestMineOutput(unittest.TestCase):
    """End-to-end mine function: load -> enumerate -> normalize -> serialize."""

    def setUp(self):
        self.registry = _make_registry()

    def test_mine_produces_valid_json(self):
        """mine() writes a JSON file with the new subgraph schema."""
        from fusion.miner import mine
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patterns.json"
            mine(
                dfg_dir=FIXTURES,
                hotspot_path=FIXTURES / "hotspot.json",
                registry=self.registry,
                output_path=output_path,
                top=5,
                max_nodes=4,
            )
            data = json.loads(output_path.read_text())
            self.assertIn("generated", data)
            self.assertIn("source_df_count", data)
            self.assertIn("pattern_count", data)
            self.assertIn("patterns", data)
            self.assertIn("rejected_combinations", data)
            # Patterns should have subgraph fields
            if data["pattern_count"] > 0:
                p = data["patterns"][0]
                self.assertIn("topology", p)
                self.assertIn("edges", p)
                self.assertIn("register_class", p)
                self.assertIn("size", p)
                self.assertIn("rank", p)
                self.assertIn("examples", p)

    def test_mine_empty_directory(self):
        """mine() on a directory with no JSON files produces empty result."""
        from fusion.miner import mine
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patterns.json"
            patterns = mine(
                dfg_dir=Path(tmpdir),
                hotspot_path=FIXTURES / "hotspot.json",
                registry=self.registry,
                output_path=output_path,
            )
            self.assertEqual(len(patterns), 0)

    def test_mine_fan_in_produces_pattern(self):
        """fan_in_2to1 fixture produces at least one 3-node pattern."""
        from fusion.miner import mine
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patterns.json"
            mine(
                dfg_dir=FIXTURES,
                hotspot_path=FIXTURES / "hotspot.json",
                registry=self.registry,
                output_path=output_path,
                top=20,
                max_nodes=4,
                min_size=2,
            )
            data = json.loads(output_path.read_text())
            three_node = [p for p in data["patterns"] if p["size"] == 3]
            self.assertGreater(len(three_node), 0)

    def test_patterns_ranked_by_frequency(self):
        """Patterns are sorted by total_frequency descending."""
        from fusion.miner import mine
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patterns.json"
            mine(
                dfg_dir=FIXTURES,
                hotspot_path=FIXTURES / "hotspot.json",
                registry=self.registry,
                output_path=output_path,
                top=20,
                max_nodes=4,
                min_size=2,
            )
            data = json.loads(output_path.read_text())
            if len(data["patterns"]) >= 2:
                freqs = [p["total_frequency"] for p in data["patterns"]]
                self.assertEqual(freqs, sorted(freqs, reverse=True))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_miner.py -v`
Expected: FAIL — the rewritten mine() doesn't exist yet

- [ ] **Step 3: Rewrite `tools/fusion/miner.py`**

Replace the entire file with:

```python
"""Subgraph-based fusion pattern mining pipeline.

Enumerates connected DFG subgraphs, normalizes them to pattern templates,
aggregates across basic blocks by BBV frequency, and writes a ranked
pattern catalog JSON.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dfg.instruction import ISARegistry

from fusion.pattern import normalize_subgraph, SubgraphPattern
from fusion.subgraph import enumerate_subgraphs

logger = logging.getLogger("fusion")


def _build_bbv_map(hotspot: dict) -> dict[str, int]:
    """Build vaddr -> execution count mapping from hotspot JSON."""
    result: dict[str, int] = {}
    for block in hotspot.get("blocks", []):
        result[block["address"]] = block["count"]
    return result


def aggregate_patterns(
    dfg_list: list[dict],
    hotspot: dict,
    registry: ISARegistry,
    max_nodes: int = 8,
    min_size: int = 2,
    top: int | None = None,
) -> tuple[list[dict], int]:
    """Enumerate, normalize, aggregate, and rank patterns across multiple DFGs.

    Args:
        dfg_list: List of DFG JSON dicts.
        hotspot: BBV hotspot JSON with blocks[].address and blocks[].count.
        registry: ISA registry for register flow resolution.
        max_nodes: Maximum subgraph size to enumerate.
        min_size: Minimum subgraph size to include in output (default 2).
        top: Maximum number of patterns to return.

    Returns:
        A tuple of (patterns_list, total_subgraphs_enumerated).
    """
    bbv_map = _build_bbv_map(hotspot)
    groups: dict[tuple, dict] = {}
    total_subgraphs = 0

    for dfg_data in dfg_list:
        vaddr = dfg_data["vaddr"]
        bb_id = dfg_data.get("bb_id", 0)
        frequency = bbv_map.get(vaddr, 0)
        nodes = dfg_data["nodes"]

        subgraphs = enumerate_subgraphs(dfg_data, max_nodes=max_nodes)
        total_subgraphs += len(subgraphs)

        for sg in subgraphs:
            if len(sg) < min_size:
                continue

            try:
                pattern = normalize_subgraph(sg, dfg_data, registry)
            except ValueError:
                continue

            key = pattern.template_key
            if key not in groups:
                groups[key] = {
                    "pattern": pattern,
                    "occurrence_count": 0,
                    "total_frequency": 0,
                    "source_bbs": [],
                    "examples": [],
                }
            groups[key]["occurrence_count"] += 1
            groups[key]["total_frequency"] += frequency
            if vaddr not in groups[key]["source_bbs"]:
                groups[key]["source_bbs"].append(vaddr)
                # Store one concrete example per source BB
                example = {
                    "bb_id": bb_id,
                    "vaddr": vaddr,
                    "instructions": [
                        {"index": nodes[i]["index"], "mnemonic": nodes[i]["mnemonic"],
                         "operands": nodes[i]["operands"]}
                        for i in sorted(sg)
                    ],
                }
                groups[key]["examples"].append(example)

    results = []
    for key, group in groups.items():
        p = group["pattern"]
        results.append({
            "opcodes": [mn for layer in p.topology for mn in layer],
            "register_class": p.register_class,
            "size": p.size,
            "topology": p.topology,
            "edges": [
                {
                    "src_layer": e.src_layer,
                    "src_opcode": e.src_opcode,
                    "dst_layer": e.dst_layer,
                    "dst_opcode": e.dst_opcode,
                    "src_role": e.src_role,
                    "dst_role": e.dst_role,
                }
                for e in p.edges
            ],
            "occurrence_count": group["occurrence_count"],
            "total_frequency": group["total_frequency"],
            "source_bbs": group["source_bbs"],
            "examples": group["examples"],
        })

    results.sort(key=lambda x: x["total_frequency"], reverse=True)
    if top is not None:
        results = results[:top]
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results, total_subgraphs


def mine(
    dfg_dir: Path,
    hotspot_path: Path,
    registry: ISARegistry,
    output_path: Path,
    top: int | None = None,
    max_nodes: int = 8,
    min_size: int = 2,
) -> list[dict]:
    """Run the full mining pipeline: load DFGs, aggregate, write output."""
    dfg_files = sorted(dfg_dir.glob("*.json"))
    dfg_list = []
    for f in dfg_files:
        try:
            data = json.loads(f.read_text())
            if "nodes" in data and "edges" in data:
                dfg_list.append(data)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Skipping invalid DFG file: %s", f)

    hotspot = json.loads(hotspot_path.read_text())
    patterns, total_subgraphs = aggregate_patterns(
        dfg_list, hotspot, registry,
        max_nodes=max_nodes, min_size=min_size, top=top,
    )

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_df_count": len(dfg_list),
        "pattern_count": len(patterns),
        "patterns": patterns,
        "enumeration_stats": {
            "total_subgraphs": total_subgraphs,
            "max_nodes": max_nodes,
            "min_size": min_size,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    logger.info(
        "Mined %d patterns from %d DFGs (top=%s, max_nodes=%s, min_size=%s), "
        "%d total subgraphs enumerated",
        len(patterns), len(dfg_list), top, max_nodes, min_size, total_subgraphs,
    )
    return patterns
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_miner.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/fusion/miner.py tools/fusion/tests/test_miner.py
git commit -m "refactor(fusion): rewrite miner for subgraph-based pattern mining

Replaces adjacent-pair chain scanning with connected subgraph enumeration.
New output schema includes topology, normalized edges, and concrete
examples per source BB. Supports max_nodes and min_size parameters.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update __main__.py CLI

**Files:**
- Modify: `tools/fusion/__main__.py`

- [ ] **Step 1: Add `--max-nodes` and `--min-size` arguments, pass to mine()**

In `tools/fusion/__main__.py`, add these arguments after the `--top` argument (around line 112):

```python
parser.add_argument(
    "--max-nodes",
    type=int,
    default=8,
    dest="max_nodes",
    help="Maximum subgraph size to enumerate (default: 8)",
)
parser.add_argument(
    "--min-size",
    type=int,
    default=2,
    dest="min_size",
    help="Minimum subgraph size to include in output (default: 2)",
)
```

Then update the `mine()` call (around line 296) to pass the new args:

```python
patterns = mine(
    dfg_dir=args.dfg_dir,
    hotspot_path=args.report,
    registry=registry,
    output_path=args.output,
    top=args.top,
    max_nodes=args.max_nodes,
    min_size=args.min_size,
)
```

Also update the stdout summary to show subgraph stats:

```python
print(f"\nFusion Pattern Mining Results")
print(f"  Patterns found: {len(patterns)}")
if patterns:
    print(f"  Top pattern: {' + '.join(mn for layer in patterns[0]['topology'] for mn in layer)} "
          f"(frequency: {patterns[0]['total_frequency']:,}, size: {patterns[0]['size']})")
print(f"  Output: {args.output}")
```

- [ ] **Step 2: Run CLI to verify new args work**

Run:
```bash
cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && \
python3 -m tools.fusion discover \
  --dfg-dir tools/fusion/tests/fixtures \
  --report tools/fusion/tests/fixtures/hotspot.json \
  --output /tmp/test_patterns.json \
  --no-agent --top 5 --max-nodes 4 --min-size 2
```
Expected: Runs without error, prints summary including pattern count

- [ ] **Step 3: Commit**

```bash
git add tools/fusion/__main__.py
git commit -m "feat(fusion): add --max-nodes and --min-size CLI arguments

Allow controlling subgraph enumeration bounds from the command line.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Adapt constraints.py for Subgraph Input

**Files:**
- Modify: `tools/fusion/constraints.py`
- Test: `tools/fusion/tests/test_integration.py` (existing)

- [ ] **Step 1: Add `max_nodes` and `no_control_flow` constraints to ConstraintConfig**

In `tools/fusion/constraints.py`, add two new entries to `ALL_CONSTRAINTS` (after `missing_encoding`):

```python
"max_nodes":             ("hard", False, "子图节点数超过硬件可实现的融合上限"),
"no_control_flow":       ("hard", False, "子图包含控制流指令（分支/跳转）"),
```

- [ ] **Step 2: Add `max_nodes` check in `ConstraintChecker.check()`**

In `tools/fusion/constraints.py`, inside the `check()` method, before the final verdict section (before the `# --- Determine final verdict ---` comment), add:

```python
# --- Check subgraph size limit ---
if self.is_enabled("max_nodes"):
    num_nodes = len(opcodes)
    # Infer max from config or use a reasonable default
    max_allowed = self._config.enabled.get("_max_nodes_value", 4)
    if num_nodes > max_allowed:
        hard_violations.append("max_nodes")
        reasons.append(f"subgraph has {num_nodes} nodes (max {max_allowed})")
```

Note: the `_max_nodes_value` is a soft internal config key. For now, use the default of 4. A future improvement could make this configurable via the JSON config file.

- [ ] **Step 3: Add `no_control_flow` check**

In the same `check()` method, add after the `no_load_store` check:

```python
# --- Check for control flow instructions ---
if self.is_enabled("no_control_flow"):
    control_flow_mnemonics = frozenset({
        "beq", "bne", "blt", "bge", "bltu", "bgeu",
        "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
        "jal", "jalr", "j", "jr", "call", "ret",
        "bgtu", "blt", "bge",
    })
    for i, opcode in enumerate(opcodes):
        if opcode in control_flow_mnemonics:
            hard_violations.append("no_control_flow")
            reasons.append(f"{opcode} is a control flow instruction")
            break
```

- [ ] **Step 4: Run existing integration tests to verify no regressions**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_integration.py -v`
Expected: All tests PASS (existing tests don't enable the new constraints, so no behavior change)

- [ ] **Step 5: Commit**

```bash
git add tools/fusion/constraints.py
git commit -m "feat(fusion): add max_nodes and no_control_flow constraints

New hard constraints for subgraph-based fusion candidates:
- max_nodes: rejects subgraphs exceeding the fusion instruction size limit
- no_control_flow: rejects patterns containing branch/jump instructions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Adapt scorer.py for Subgraph Input

**Files:**
- Modify: `tools/fusion/scorer.py`
- Test: `tools/fusion/tests/test_integration.py` (existing)

- [ ] **Step 1: Update `Scorer._tight_score` for subgraph density**

In `tools/fusion/scorer.py`, replace the `_tight_score` method:

```python
def _tight_score(
    self,
    edges: list[dict],
    size: int,
) -> float:
    """Data-dependency density in [0, 1].

    For subgraph patterns: density = num_edges / (size * (size - 1) / 2),
    i.e., what fraction of all possible directed edges exist. A bonus
    chain_factor is applied for larger subgraphs.
    """
    if size <= 1:
        return 0.0
    max_possible = size * (size - 1) / 2
    raw_density = len(edges) / max_possible if max_possible > 0 else 0.0
    factor = _CHAIN_FACTOR.get(size, _CHAIN_FACTOR.get(max(_CHAIN_FACTOR), 1.0))
    return min(raw_density * factor, 1.0)
```

- [ ] **Step 2: Update `Scorer.score_pattern` to use new pattern schema**

Replace the body of `score_pattern` to read from the subgraph schema:

```python
def score_pattern(self, pattern: dict[str, Any]) -> dict[str, Any]:
    """Score a single fusion candidate pattern."""
    freq = pattern.get("total_frequency", 0)
    occurrences = pattern.get("occurrence_count", 0)
    edges = pattern.get("edges", [])
    size = pattern.get("size", len(pattern.get("opcodes", [])))

    # Sub-scores
    freq_score = self._freq_score(freq)
    tight_score = self._tight_score(edges, size)

    verdict = self._checker.check(pattern)
    hw_score = self._hw_score(verdict)

    w = self._weights
    if hw_score == 0.0:
        final_score = 0.0
    else:
        final_score = (
            w.get("frequency", 0.0) * freq_score
            + w.get("tightness", 0.0) * tight_score
            + w.get("hardware", 0.0) * hw_score
        )

    return {
        "pattern": {
            "opcodes": pattern["opcodes"],
            "register_class": pattern.get("register_class"),
            "topology": pattern.get("topology"),
            "edges": edges,
            "size": size,
        },
        "input_frequency": freq,
        "input_occurrence_count": occurrences,
        "tightness": {
            "edge_count": len(edges),
            "max_possible_edges": size * (size - 1) // 2 if size > 1 else 0,
            "raw_density": len(edges) / (size * (size - 1) / 2) if size > 1 else 0.0,
            "chain_factor": _CHAIN_FACTOR.get(size, _CHAIN_FACTOR.get(max(_CHAIN_FACTOR), 1.0)),
            "score": tight_score,
        },
        "hardware": {
            "status": verdict.status,
            "reasons": verdict.reasons,
            "violations": verdict.violations,
        },
        "score": final_score,
        "score_breakdown": {
            "freq_score": freq_score,
            "tight_score": tight_score,
            "hw_score": hw_score,
        },
    }
```

- [ ] **Step 3: Update `score()` CLI helper to handle `examples` in patterns**

In the `score()` function (the CLI helper around line 238), the `patterns` variable reads from `catalog.get("patterns", [])`. No changes needed there since the new schema is a superset. Just verify the output JSON includes the new `examples` field by updating the output construction (around line 280):

After `"candidates": results`, the output dict is fine as-is. The `pattern` key in each scored result already includes the new fields from `score_pattern`.

- [ ] **Step 4: Run integration tests**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/test_integration.py -v`
Expected: All tests PASS (note: `test_integer_instructions_have_encoding` may still fail for `mul` — this is a pre-existing issue, not related to this change)

- [ ] **Step 5: Commit**

```bash
git add tools/fusion/scorer.py
git commit -m "refactor(fusion): adapt scorer for subgraph-based patterns

tight_score now uses subgraph edge density instead of linear chain
density. score_pattern reads topology, edges, and size from the new
schema. Output includes subgraph structure in scored results.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: End-to-End Integration on Real Data

**Files:**
- Run against: `output/dfg/json/*.json` + `output/hotspot.json`

- [ ] **Step 1: Run discover on real DFG data**

Run:
```bash
cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && \
python3 -m tools.fusion discover \
  --dfg-dir output/dfg/json \
  --report output/hotspot.json \
  --output output/fusion_patterns.json \
  --no-agent --top 20 --max-nodes 8 --min-size 2
```
Expected: Outputs pattern count > 0 (previously was 0), prints summary

- [ ] **Step 2: Run score on discovered patterns**

Run:
```bash
cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && \
python3 -m tools.fusion score \
  --catalog output/fusion_patterns.json \
  --output output/fusion_candidates.json \
  --top 20
```
Expected: Outputs candidate count > 0

- [ ] **Step 3: Verify output schema**

Run:
```bash
cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && \
python3 -c "
import json
data = json.load(open('output/fusion_patterns.json'))
print(f'Patterns: {data[\"pattern_count\"]}')
print(f'Source DFGs: {data[\"source_df_count\"]}')
print(f'Enumeration stats: {data[\"enumeration_stats\"]}')
if data['patterns']:
    p = data['patterns'][0]
    print(f'Top pattern: size={p[\"size\"]}, rank={p[\"rank\"]}')
    print(f'  topology: {p[\"topology\"]}')
    print(f'  edges: {len(p[\"edges\"])}')
    print(f'  frequency: {p[\"total_frequency\"]}')
    print(f'  examples: {len(p[\"examples\"])}')
"
```
Expected: Shows non-zero pattern count with topology, edges, and examples

- [ ] **Step 4: Commit output files**

```bash
git add output/fusion_patterns.json output/fusion_candidates.json
git commit -m "data: update fusion output with subgraph-based mining results

Real DFG data (bb_84469, bb_84777) now produces fusion candidates
instead of 0 patterns. flw+flw+fmadd.s fan-in patterns are the top
candidates by frequency.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Run Full Test Suite

- [ ] **Step 1: Run all fusion tests**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/fusion/tests/ -v`
Expected: All tests in `test_subgraph.py`, `test_pattern.py`, `test_miner.py` PASS. `test_integration.py` may have 1 pre-existing failure (`mul` encoding).

- [ ] **Step 2: Run DFG engine tests to verify no regressions**

Run: `cd /home/pren/wsp/rvfuse/.claude/worktrees/fusion-v1.2 && python -m pytest tools/dfg/tests/ -v`
Expected: All DFG tests PASS (fusion changes don't touch DFG engine)

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "test(fusion): fix any issues found during full test run

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
