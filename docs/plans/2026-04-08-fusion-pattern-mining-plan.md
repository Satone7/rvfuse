# F1: Fusion Pattern Mining — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an Agent + Miner hybrid system that discovers fusible instruction patterns from DFG JSON output and produces ranked fusion candidates with agent analysis.

**Architecture:** Miner is a deterministic pipeline (load DFG JSON → enumerate linear chains → normalize to opcode+role templates → aggregate by frequency → rank). Agent calls Claude CLI subprocess to analyze Miner output and produce recommendations. Both live under `tools/fusion/`, importing `ISARegistry` from `tools/dfg/instruction.py` for register role resolution.

**Tech Stack:** Python 3 stdlib (json, argparse, subprocess, unittest), existing `tools/dfg/instruction.py` (ISARegistry, RegisterFlow, ResolvedFlow, Instruction, BasicBlock), existing `tools/dfg/agent.py` (subprocess pattern).

**Design doc:** `docs/plans/2026-04-08-fusion-pattern-mining-design.md`

**Key reference files (read these first):**
- `tools/dfg/instruction.py` — ISARegistry, RegisterFlow, ResolvedFlow, _extract_registers, RegisterKind, Instruction, BasicBlock
- `tools/dfg/dfg.py` — DFGNode, DFGEdge, DFG, build_dfg
- `tools/dfg/output.py` — dfg_to_json (DFG JSON schema: bb_id, vaddr, nodes[].index/address/mnemonic/operands, edges[].src/dst/register)
- `tools/dfg/agent.py` — AgentDispatcher._invoke_claude (Claude CLI subprocess pattern)
- `tools/dfg/__main__.py` — load_isa_registry, _ISA_MODULES
- `tools/analyze_bbv.py:392-418` — BBV hotspot JSON schema (blocks[].address/count/bb_id)

---

### Task 1: Create package skeleton and test fixtures

**Files:**
- Create: `tools/fusion/__init__.py`
- Create: `tools/fusion/tests/__init__.py`
- Create: `tools/fusion/tests/fixtures/float_chain_2.json`
- Create: `tools/fusion/tests/fixtures/float_chain_3.json`
- Create: `tools/fusion/tests/fixtures/mixed_class.json`
- Create: `tools/fusion/tests/fixtures/no_raw.json`
- Create: `tools/fusion/tests/fixtures/branch_in_chain.json`
- Create: `tools/fusion/tests/fixtures/hotspot.json`

**Step 1: Create package init files**

```python
# tools/fusion/__init__.py
"""Fusion candidate discovery tools for RVFuse."""
```

```python
# tools/fusion/tests/__init__.py
```

**Step 2: Create float_chain_2.json fixture**

A 3-instruction BB where instructions 0→1 have a RAW edge (ft2 flows), and instructions 1→2 have no edge (different registers). This produces exactly 1 length-2 chain: `[fadd.s, fmul.s]`.

```json
{
  "bb_id": 1,
  "vaddr": "0x1000",
  "source": "script",
  "nodes": [
    {"index": 0, "address": "0x1000", "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
    {"index": 1, "address": "0x1004", "mnemonic": "fmul.s", "operands": "ft3,ft2,fa2"},
    {"index": 2, "address": "0x1008", "mnemonic": "fsw", "operands": "ft3,0(a0)"}
  ],
  "edges": [
    {"src": 0, "dst": 1, "register": "ft2"}
  ]
}
```

**Step 3: Create float_chain_3.json fixture**

A 4-instruction BB where instructions 0→1→2 all have RAW edges (ft2→ft2→ft3 chain). Produces 1 length-3 chain: `[fadd.s, fmul.s, fsub.s]` and 2 length-2 chains: `[fadd.s, fmul.s]`, `[fmul.s, fsub.s]`.

```json
{
  "bb_id": 2,
  "vaddr": "0x2000",
  "source": "script",
  "nodes": [
    {"index": 0, "address": "0x2000", "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
    {"index": 1, "address": "0x2004", "mnemonic": "fmul.s", "operands": "ft2,ft2,fa2"},
    {"index": 2, "address": "0x2008", "mnemonic": "fsub.s", "operands": "ft3,ft2,fa3"},
    {"index": 3, "address": "0x200c", "mnemonic": "fsw", "operands": "ft3,0(a0)"}
  ],
  "edges": [
    {"src": 0, "dst": 1, "register": "ft2"},
    {"src": 1, "dst": 2, "register": "ft2"}
  ]
}
```

**Step 4: Create mixed_class.json fixture**

Integer instruction followed by float instruction with a RAW edge on a0 (but different register classes). Should produce 0 patterns because cross-class chains are filtered.

```json
{
  "bb_id": 3,
  "vaddr": "0x3000",
  "source": "script",
  "nodes": [
    {"index": 0, "address": "0x3000", "mnemonic": "addi", "operands": "a0,a0,1"},
    {"index": 1, "address": "0x3004", "mnemonic": "fmv.w.x", "operands": "ft2,a0"}
  ],
  "edges": [
    {"src": 0, "dst": 1, "register": "a0"}
  ]
}
```

**Step 5: Create no_raw.json fixture**

Two float instructions with no RAW edge between them. Should produce 0 patterns.

```json
{
  "bb_id": 4,
  "vaddr": "0x4000",
  "source": "script",
  "nodes": [
    {"index": 0, "address": "0x4000", "mnemonic": "fadd.s", "operands": "ft0,fa0,fa1"},
    {"index": 1, "address": "0x4004", "mnemonic": "fmul.s", "operands": "ft1,fa2,fa3"}
  ],
  "edges": []
}
```

**Step 6: Create branch_in_chain.json fixture**

A float chain with a branch in the middle. The branch (beqz) should break the chain.

```json
{
  "bb_id": 5,
  "vaddr": "0x5000",
  "source": "script",
  "nodes": [
    {"index": 0, "address": "0x5000", "mnemonic": "fadd.s", "operands": "ft2,fa0,fa1"},
    {"index": 1, "address": "0x5004", "mnemonic": "beqz", "operands": "a0,16"},
    {"index": 2, "address": "0x5008", "mnemonic": "fmul.s", "operands": "ft3,ft2,fa2"}
  ],
  "edges": [
    {"src": 0, "dst": 2, "register": "ft2"}
  ]
}
```

Note: edge 0→2 skips over the branch at index 1. The chain [fadd.s, beqz] has no RAW edge between indices 0 and 1 (the edge is 0→2, not 0→1). The chain [beqz, fmul.s] has no RAW edge between 1 and 2. So this produces 0 patterns even without explicit branch filtering — but the design says to also filter branches, so tests should verify both behaviors.

**Step 7: Create hotspot.json fixture**

BBV hotspot report with frequencies matching the fixture BBs.

```json
{
  "total_blocks": 3,
  "total_executions": 350000,
  "blocks": [
    {"rank": 1, "address": "0x1000", "count": 200000, "pct": 57.14, "cumulative_pct": 57.14, "location": "libonnxruntime.so+0x1000", "bb_id": 1},
    {"rank": 2, "address": "0x2000", "count": 100000, "pct": 28.57, "cumulative_pct": 85.71, "location": "libonnxruntime.so+0x2000", "bb_id": 2},
    {"rank": 3, "address": "0x3000", "count": 50000, "pct": 14.29, "cumulative_pct": 100.0, "location": "libonnxruntime.so+0x3000", "bb_id": 3}
  ]
}
```

**Step 8: Commit**

```bash
git add tools/fusion/__init__.py tools/fusion/tests/__init__.py tools/fusion/tests/fixtures/
git commit -m "feat(fusion): add package skeleton and test fixtures"
```

---

### Task 2: Pattern dataclass and template key

**Files:**
- Create: `tools/fusion/pattern.py`
- Create: `tools/fusion/tests/test_pattern.py`

**Context:** A `Pattern` represents a normalized fusible instruction template. The `template_key` is a hashable tuple used to group identical patterns across BBs. It consists of `(opcodes_tuple, register_class, chain_registers_tuple)` where `chain_registers` encodes which role positions carry the RAW dependency for each consecutive pair.

For example, `fadd.s ft2,fa0,fa1 → fmul.s ft3,ft2,fa2` normalizes to:
- opcodes: `("fadd.s", "fmul.s")`
- register_class: `"float"`
- chain_registers: `(("frd", "frs1"),)` — meaning the `frd` output of fadd.s feeds into the `frs1` input of fmul.s

**Step 1: Write the failing tests**

```python
# tools/fusion/tests/test_pattern.py
"""Tests for the Pattern model — normalization and template keys."""

import sys
from pathlib import Path
import unittest

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestPatternTemplateKey(unittest.TestCase):
    """Same opcode sequence + same role positions = same template key."""

    def test_identical_chains_same_key(self):
        """Two concrete instances of the same abstract pattern share a key."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        p2 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        self.assertEqual(p1.template_key, p2.template_key)

    def test_different_opcodes_different_key(self):
        """Different opcode sequences produce different keys."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        p2 = Pattern(
            opcodes=["fsub.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        self.assertNotEqual(p1.template_key, p2.template_key)

    def test_different_register_class_different_key(self):
        """Same opcodes but different register class produce different keys."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["add", "sub"],
            register_class="integer",
            chain_registers=[["rd", "rs1"]],
        )
        p2 = Pattern(
            opcodes=["add", "sub"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        self.assertNotEqual(p1.template_key, p2.template_key)

    def test_different_chain_roles_different_key(self):
        """Same opcodes but dependency flows through different role positions."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        p2 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs2"]],
        )
        self.assertNotEqual(p1.template_key, p2.template_key)

    def test_length_3_chain_key(self):
        """Length-3 patterns have two chain_register entries."""
        from fusion.pattern import Pattern

        p = Pattern(
            opcodes=["fadd.s", "fmul.s", "fsub.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"], ["frd", "frs1"]],
        )
        # Key should capture all three opcodes and both chain links
        self.assertIn(("fadd.s", "fmul.s", "fsub.s"), p.template_key)
        self.assertEqual(len(p.template_key[2]), 2)  # two chain links

    def test_key_is_hashable(self):
        """Template key must be usable as a dict key."""
        from fusion.pattern import Pattern

        p = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        d = {p.template_key: 42}
        self.assertEqual(d[p.template_key], 42)
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_pattern.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion.pattern'`

**Step 3: Write minimal implementation**

```python
# tools/fusion/pattern.py
"""Fusible instruction pattern model and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pattern:
    """A normalized fusible instruction template.

    Attributes:
        opcodes: Ordered list of instruction mnemonics in the chain.
        register_class: "integer" or "float".
        chain_registers: For each consecutive pair, a list of (dst_role, src_role)
            tuples describing which operand positions carry the RAW dependency.
            E.g. [["frd", "frs1"]] means the frd output of instruction i
            feeds into the frs1 input of instruction i+1.
    """

    opcodes: list[str]
    register_class: str
    chain_registers: list[list[str]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.opcodes)

    @property
    def template_key(self) -> tuple:
        """Hashable key for grouping identical patterns across BBs.

        Converts lists to tuples so the key is hashable (usable in dicts/sets).
        """
        chain_tuple = tuple(tuple(pair) for pair in self.chain_registers)
        return (tuple(self.opcodes), self.register_class, chain_tuple)
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_pattern.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add tools/fusion/pattern.py tools/fusion/tests/test_pattern.py
git commit -m "feat(fusion): add Pattern dataclass with template key"
```

---

### Task 3: Register class detection and control flow filtering

**Files:**
- Modify: `tools/fusion/pattern.py`
- Modify: `tools/fusion/tests/test_pattern.py`

**Context:** The miner needs two utility functions:
1. `classify_register(reg_name) -> str | None` — returns `"integer"` or `"float"` or `None`. Reuses the same regex patterns as `tools/dfg/instruction.py` (INTEGER_KIND pattern and FLOAT_KIND pattern).
2. `is_control_flow(mnemonic) -> bool` — returns True for branches/jumps/calls. Use a set of common RISC-V control flow mnemonics.

**Step 1: Write the failing tests**

Append to `tools/fusion/tests/test_pattern.py`:

```python
class TestRegisterClassification(unittest.TestCase):

    def test_integer_registers(self):
        from fusion.pattern import classify_register
        for name in ("a0", "t1", "s2", "ra", "sp", "zero", "x5", "tp", "gp"):
            self.assertEqual(classify_register(name), "integer")

    def test_float_registers(self):
        from fusion.pattern import classify_register
        for name in ("ft0", "ft2", "fa0", "fa4", "fs0", "f5", "fv0"):
            self.assertEqual(classify_register(name), "float")

    def test_unknown_registers(self):
        from fusion.pattern import classify_register
        self.assertIsNone(classify_register("v0"))
        self.assertIsNone(classify_register("unknown_reg"))
        self.assertIsNone(classify_register(""))


class TestControlFlowDetection(unittest.TestCase):

    def test_branches(self):
        from fusion.pattern import is_control_flow
        for mn in ("beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"):
            self.assertTrue(is_control_flow(mn))

    def test_jumps_and_calls(self):
        from fusion.pattern import is_control_flow
        for mn in ("jal", "jalr", "call", "ret", "j", "jr"):
            self.assertTrue(is_control_flow(mn))

    def test_non_control_flow(self):
        from fusion.pattern import is_control_flow
        for mn in ("add", "addi", "fadd.s", "fmul.s", "ld", "sd", "lw", "fsw"):
            self.assertFalse(is_control_flow(mn))
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_pattern.py -v`
Expected: FAIL with `ImportError: cannot import name 'classify_register'`

**Step 3: Add implementation to pattern.py**

Append to `tools/fusion/pattern.py`:

```python
import re


# ── Register classification (matches dfg/instruction.py patterns) ──

_INTEGER_RE = re.compile(r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|s\d+|jt?\d*)$")
_FLOAT_RE = re.compile(r"^(f\d+|ft\d+|fs\d+|fa\d+|fv\d+)$")


def classify_register(name: str) -> str | None:
    """Return 'integer' or 'float' for a register name, or None if unknown."""
    if _INTEGER_RE.match(name):
        return "integer"
    if _FLOAT_RE.match(name):
        return "float"
    return None


# ── Control flow detection ──

_CONTROL_FLOW_MNEMONICS = frozenset({
    # Branches
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
    # Jumps
    "jal", "jalr", "j", "jr",
    # Calls/returns (pseudo)
    "call", "ret",
})


def is_control_flow(mnemonic: str) -> bool:
    """Return True if the mnemonic is a branch, jump, or call."""
    return mnemonic in _CONTROL_FLOW_MNEMONICS
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_pattern.py -v`
Expected: All tests PASS (6 pattern key tests + 3 register tests + 3 control flow tests)

**Step 5: Commit**

```bash
git add tools/fusion/pattern.py tools/fusion/tests/test_pattern.py
git commit -m "feat(fusion): add register classification and control flow detection"
```

---

### Task 4: Pattern normalization from concrete chain + ISARegistry

**Files:**
- Modify: `tools/fusion/pattern.py`
- Modify: `tools/fusion/tests/test_pattern.py`

**Context:** Given a concrete instruction chain (list of `(mnemonic, operands)` tuples) and the RAW edges between them, we need to produce a `Pattern` with normalized `chain_registers`. This uses `ISARegistry` from `tools/dfg/instruction.py` to resolve operand strings to role positions.

The algorithm for a chain of length N:
1. For each consecutive pair (i, i+1), find RAW edges where `src == i` and `dst == i+1`.
2. For each such edge, determine which role position wrote that register (from instruction i's `dst_regs`) and which role position read it (from instruction i+1's `src_regs`).
3. Record as `(dst_role, src_role)` pair.

**Step 1: Write the failing tests**

Append to `tools/fusion/tests/test_pattern.py`:

```python
class TestNormalizeChain(unittest.TestCase):
    """Test normalization of concrete chains to Pattern templates."""

    def _make_registry(self):
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64f import build_registry as build_f
        from dfg.isadesc.rv64i import build_registry as build_i
        reg = ISARegistry()
        build_i(reg)
        build_f(reg)
        return reg

    def test_float_pair_chain(self):
        """fadd.s ft2,fa0,fa1 → fmul.s ft3,ft2,fa2 normalizes to chain_registers=[['frd','frs1']]."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("fadd.s", "ft2,fa0,fa1"),
            ("fmul.s", "ft3,ft2,fa2"),
        ]
        edges_between = [{"src": 0, "dst": 1, "register": "ft2"}]
        pattern = normalize_chain(chain, edges_between, registry)
        self.assertEqual(pattern.opcodes, ["fadd.s", "fmul.s"])
        self.assertEqual(pattern.register_class, "float")
        self.assertEqual(pattern.chain_registers, [["frd", "frs1"]])

    def test_integer_pair_chain(self):
        """addi a0,sp,16 → add a1,a0,a2 normalizes to chain_registers=[['rd','rs1']]."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("addi", "a0,sp,16"),
            ("add", "a1,a0,a2"),
        ]
        edges_between = [{"src": 0, "dst": 1, "register": "a0"}]
        pattern = normalize_chain(chain, edges_between, registry)
        self.assertEqual(pattern.opcodes, ["addi", "add"])
        self.assertEqual(pattern.register_class, "integer")
        self.assertEqual(pattern.chain_registers, [["rd", "rs1"]])

    def test_length_3_chain(self):
        """Three-instruction chain with two RAW links."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("fadd.s", "ft2,fa0,fa1"),
            ("fmul.s", "ft2,ft2,fa2"),
            ("fsub.s", "ft3,ft2,fa3"),
        ]
        edges_between = [
            {"src": 0, "dst": 1, "register": "ft2"},
            {"src": 1, "dst": 2, "register": "ft2"},
        ]
        pattern = normalize_chain(chain, edges_between, registry)
        self.assertEqual(pattern.opcodes, ["fadd.s", "fmul.s", "fsub.s"])
        self.assertEqual(len(pattern.chain_registers), 2)

    def test_unknown_mnemonic_raises(self):
        """Unknown instruction mnemonic in the chain raises ValueError."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("fadd.s", "ft2,fa0,fa1"),
            ("unknown_op", "ft3,ft2,fa2"),
        ]
        with self.assertRaises(ValueError):
            normalize_chain(chain, [{"src": 0, "dst": 1, "register": "ft2"}], registry)
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_pattern.py::TestNormalizeChain -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_chain'`

**Step 3: Add normalize_chain implementation**

Append to `tools/fusion/pattern.py`:

```python
from dfg.instruction import ISARegistry


def normalize_chain(
    chain: list[tuple[str, str]],
    edges_between: list[dict],
    registry: ISARegistry,
) -> Pattern:
    """Normalize a concrete instruction chain to a Pattern template.

    Args:
        chain: List of (mnemonic, operands) tuples.
        edges_between: RAW edges within the chain, each with 'src', 'dst', 'register'.
            src/dst are 0-based indices relative to the chain start.
        registry: ISA registry for resolving operand role positions.

    Returns:
        A Pattern with normalized opcodes and chain_registers.

    Raises:
        ValueError: If any mnemonic is unknown in the registry.
    """
    opcodes = [mn for mn, _ in chain]

    # Determine register class from the first instruction's registers
    first_flow = registry.get_flow(chain[0][0])
    if first_flow is None:
        raise ValueError(f"Unknown mnemonic: {chain[0][0]}")
    first_resolved = first_flow.resolve(chain[0][1])
    all_regs = first_resolved.dst_regs + first_resolved.src_regs
    reg_class = classify_register(all_regs[0]) if all_regs else None

    # Build chain_registers: for each consecutive pair, find which roles carry RAW
    chain_regs: list[list[str]] = []
    for pair_idx in range(len(chain) - 1):
        # Find edges between instruction pair_idx and pair_idx+1
        pair_edges = [
            e for e in edges_between
            if e["src"] == pair_idx and e["dst"] == pair_idx + 1
        ]
        roles_for_pair: list[list[str]] = []
        for edge in pair_edges:
            reg_name = edge["register"]
            # Find dst role in instruction pair_idx
            src_flow = registry.get_flow(chain[pair_idx][0])
            if src_flow is None:
                raise ValueError(f"Unknown mnemonic: {chain[pair_idx][0]}")
            src_resolved = src_flow.resolve(chain[pair_idx][1])
            dst_role = _find_role(reg_name, src_resolved.dst_regs, src_flow.dst_regs)

            # Find src role in instruction pair_idx+1
            dst_flow = registry.get_flow(chain[pair_idx + 1][0])
            if dst_flow is None:
                raise ValueError(f"Unknown mnemonic: {chain[pair_idx + 1][0]}")
            dst_resolved = dst_flow.resolve(chain[pair_idx + 1][1])
            src_role = _find_role(reg_name, dst_resolved.src_regs, dst_flow.src_regs)

            if dst_role and src_role:
                roles_for_pair.append([dst_role, src_role])
        chain_regs.extend(roles_for_pair)

    return Pattern(
        opcodes=opcodes,
        register_class=reg_class or "unknown",
        chain_registers=chain_regs,
    )


def _find_role(
    reg_name: str,
    resolved_regs: list[str],
    role_names: list[str],
) -> str | None:
    """Find the role name (e.g. 'frd') corresponding to a concrete register name.

    Args:
        reg_name: Concrete register name (e.g. 'ft2').
        resolved_regs: Resolved register names from flow.resolve() (e.g. ['ft2']).
        role_names: Position names from RegisterFlow (e.g. ['frd']).

    Returns:
        The role name if found, None otherwise.
    """
    for i, resolved in enumerate(resolved_regs):
        if resolved == reg_name and i < len(role_names):
            return role_names[i]
    return None
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_pattern.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tools/fusion/pattern.py tools/fusion/tests/test_pattern.py
git commit -m "feat(fusion): add normalize_chain with ISARegistry role resolution"
```

---

### Task 5: Chain enumeration — extract linear RAW chains from a single DFG

**Files:**
- Create: `tools/fusion/miner.py`
- Create: `tools/fusion/tests/test_miner.py`

**Context:** The enumerator walks a single DFG's node/edge lists and extracts all valid linear chains of length 2-3. For each consecutive pair of instructions (i, i+1), it checks if a RAW edge exists between them. If yes, it forms a length-2 chain. If the next pair (i+1, i+2) also has a RAW edge, it extends to length-3. Chains are filtered: all instructions must be same register class and none can be control flow.

**Step 1: Write the failing tests**

```python
# tools/fusion/tests/test_miner.py
"""Tests for the Miner pipeline."""

import json
import sys
from pathlib import Path
import unittest

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


class TestEnumerateChains(unittest.TestCase):
    """Extract linear RAW chains from a single DFG JSON."""

    def test_float_chain_2_produces_one_chain(self):
        """float_chain_2.json has one RAW edge (0→1), producing 1 length-2 chain."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        chains = enumerate_chains(dfg, registry)
        self.assertEqual(len(chains), 1)
        self.assertEqual(len(chains[0]), 2)

    def test_float_chain_3_produces_three_chains(self):
        """float_chain_3.json has edges 0→1 and 1→2.
        Produces: 1 length-3 chain (0,1,2) + 2 length-2 chains (0,1) and (1,2)."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_3.json").read_text())
        chains = enumerate_chains(dfg, registry)
        # Expect 3 chains total: (0,1,2), (0,1), (1,2)
        self.assertEqual(len(chains), 3)
        lengths = sorted(len(c) for c in chains)
        self.assertEqual(lengths, [2, 2, 3])

    def test_mixed_class_produces_no_chains(self):
        """Cross-class chain should be filtered out."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "mixed_class.json").read_text())
        chains = enumerate_chains(dfg, registry)
        self.assertEqual(len(chains), 0)

    def test_no_raw_produces_no_chains(self):
        """No RAW edges means no chains."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "no_raw.json").read_text())
        chains = enumerate_chains(dfg, registry)
        self.assertEqual(len(chains), 0)

    def test_chain_contains_mnemonic_operands_tuples(self):
        """Each chain entry should be a (mnemonic, operands) tuple."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        chains = enumerate_chains(dfg, registry)
        chain = chains[0]
        self.assertEqual(chain[0], ("fadd.s", "ft2,fa0,fa1"))
        self.assertEqual(chain[1], ("fmul.s", "ft3,ft2,fa2"))
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_miner.py::TestEnumerateChains -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion.miner'`

**Step 3: Write enumerate_chains implementation**

```python
# tools/fusion/miner.py
"""Deterministic fusion pattern mining pipeline."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dfg.instruction import ISARegistry

from fusion.pattern import (
    Pattern,
    classify_register,
    is_control_flow,
    normalize_chain,
)

logger = logging.getLogger("fusion")


def enumerate_chains(
    dfg_data: dict,
    registry: ISARegistry,
) -> list[list[tuple[str, str]]]:
    """Extract all valid linear RAW chains from a single DFG.

    Args:
        dfg_data: Parsed DFG JSON dict with 'nodes' and 'edges'.
        registry: ISA registry for register classification.

    Returns:
        List of chains. Each chain is a list of (mnemonic, operands) tuples.
    """
    nodes = dfg_data["nodes"]
    edges = dfg_data["edges"]

    if len(nodes) < 2:
        return []

    # Build adjacency: edge_map[(src_idx, dst_idx)] = [register, ...]
    edge_map: dict[tuple[int, int], list[str]] = defaultdict(list)
    for e in edges:
        edge_map[(e["src"], e["dst"])].append(e["register"])

    # Determine register class for each node
    def node_reg_class(idx: int) -> str | None:
        mn = nodes[idx]["mnemonic"]
        flow = registry.get_flow(mn)
        if flow is None:
            return None
        resolved = flow.resolve(nodes[idx]["operands"])
        all_regs = resolved.dst_regs + resolved.src_regs
        return classify_register(all_regs[0]) if all_regs else None

    results: list[list[tuple[str, str]]] = []

    def _valid_pair(i: int, j: int) -> bool:
        """Check if instructions i and j can form a valid adjacent pair."""
        mn_i, mn_j = nodes[i]["mnemonic"], nodes[j]["mnemonic"]
        if is_control_flow(mn_i) or is_control_flow(mn_j):
            return False
        rc_i, rc_j = node_reg_class(i), node_reg_class(j)
        if rc_i is None or rc_j is None or rc_i != rc_j:
            return False
        return True

    # Scan for length-2 and length-3 chains
    for i in range(len(nodes) - 1):
        if not _valid_pair(i, i + 1):
            continue
        if (i, i + 1) not in edge_map:
            continue

        # Length-2 chain
        chain_2 = [
            (nodes[i]["mnemonic"], nodes[i]["operands"]),
            (nodes[i + 1]["mnemonic"], nodes[i + 1]["operands"]),
        ]
        results.append(chain_2)

        # Try to extend to length-3
        if i + 2 < len(nodes) and _valid_pair(i + 1, i + 2):
            if (i + 1, i + 2) in edge_map:
                chain_3 = chain_2 + [
                    (nodes[i + 2]["mnemonic"], nodes[i + 2]["operands"]),
                ]
                results.append(chain_3)

    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_miner.py::TestEnumerateChains -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tools/fusion/miner.py tools/fusion/tests/test_miner.py
git commit -m "feat(fusion): add chain enumeration from DFG JSON"
```

---

### Task 6: Aggregation and frequency weighting

**Files:**
- Modify: `tools/fusion/miner.py`
- Modify: `tools/fusion/tests/test_miner.py`

**Context:** The aggregator takes all enumerated chains across multiple DFGs, normalizes each to a Pattern, groups identical patterns by template_key, and sums BBV-weighted frequencies. It reads the hotspot JSON to get per-BB execution counts.

The BBV hotspot JSON uses `address` field (hex string like "0x1000") to identify BBs. The DFG JSON also has a `vaddr` field. We match them by address to look up frequency.

**Step 1: Write the failing tests**

Append to `tools/fusion/tests/test_miner.py`:

```python
class TestAggregatePatterns(unittest.TestCase):
    """Aggregate normalized patterns across multiple DFGs with BBV weighting."""

    def test_single_dfg_aggregation(self):
        """One DFG, one chain → one aggregated pattern."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        hotspot = json.loads((FIXTURES / "hotspot.json").read_text())
        patterns = aggregate_patterns([dfg], hotspot, registry)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["opcodes"], ["fadd.s", "fmul.s"])
        # BB at 0x1000 has count=200000 in hotspot
        self.assertEqual(patterns[0]["total_frequency"], 200000)
        self.assertEqual(patterns[0]["occurrence_count"], 1)

    def test_cross_bb_aggregation(self):
        """Two BBs with the same pattern → merged, frequencies summed."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        # Both fixtures have fadd.s→fmul.s chain
        dfg1 = json.loads((FIXTURES / "float_chain_2.json").read_text())
        dfg2 = json.loads((FIXTURES / "float_chain_3.json").read_text())
        hotspot = json.loads((FIXTURES / "hotspot.json").read_text())
        patterns = aggregate_patterns([dfg1, dfg2], hotspot, registry)
        # fadd.s→fmul.s appears in both BBs
        fadd_fmul = [p for p in patterns if p["opcodes"] == ["fadd.s", "fmul.s"]]
        self.assertEqual(len(fadd_fmul), 1)
        self.assertEqual(fadd_fmul[0]["occurrence_count"], 2)
        # 0x1000: 200000, 0x2000: 100000 → total 300000
        self.assertEqual(fadd_fmul[0]["total_frequency"], 300000)

    def test_bb_not_in_hotspot_gets_zero_frequency(self):
        """BB without hotspot entry gets frequency=0."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        # Hotspot without the matching address
        hotspot = {"total_blocks": 0, "blocks": []}
        patterns = aggregate_patterns([dfg], hotspot, registry)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["total_frequency"], 0)

    def test_ranked_by_frequency_descending(self):
        """Patterns are sorted by total_frequency descending."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        dfg1 = json.loads((FIXTURES / "float_chain_2.json").read_text())
        dfg2 = json.loads((FIXTURES / "float_chain_3.json").read_text())
        hotspot = json.loads((FIXTURES / "hotspot.json").read_text())
        patterns = aggregate_patterns([dfg1, dfg2], hotspot, registry)
        frequencies = [p["total_frequency"] for p in patterns]
        self.assertEqual(frequencies, sorted(frequencies, reverse=True))

    def test_top_n_filtering(self):
        """--top limits the number of returned patterns."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        dfg1 = json.loads((FIXTURES / "float_chain_2.json").read_text())
        dfg2 = json.loads((FIXTURES / "float_chain_3.json").read_text())
        hotspot = json.loads((FIXTURES / "hotspot.json").read_text())
        patterns = aggregate_patterns([dfg1, dfg2], hotspot, registry, top=1)
        self.assertEqual(len(patterns), 1)
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_miner.py::TestAggregatePatterns -v`
Expected: FAIL with `ImportError: cannot import name 'aggregate_patterns'`

**Step 3: Add aggregate_patterns implementation**

Append to `tools/fusion/miner.py`:

```python
def _build_bbv_map(hotspot: dict) -> dict[str, int]:
    """Build vaddr → execution count mapping from hotspot JSON."""
    result: dict[str, int] = {}
    for block in hotspot.get("blocks", []):
        result[block["address"]] = block["count"]
    return result


def aggregate_patterns(
    dfg_list: list[dict],
    hotspot: dict,
    registry: ISARegistry,
    top: int | None = None,
) -> list[dict]:
    """Enumerate, normalize, aggregate, and rank patterns across multiple DFGs.

    Args:
        dfg_list: List of parsed DFG JSON dicts.
        hotspot: Parsed BBV hotspot JSON dict.
        registry: ISA registry for role resolution.
        top: If set, return only the top N patterns.

    Returns:
        List of pattern dicts sorted by total_frequency descending.
    """
    bbv_map = _build_bbv_map(hotspot)

    # Group by template key
    groups: dict[tuple, dict] = {}

    for dfg_data in dfg_list:
        vaddr = dfg_data["vaddr"]
        frequency = bbv_map.get(vaddr, 0)

        chains = enumerate_chains(dfg_data, registry)

        for chain in chains:
            # Build edge list for this chain
            node_indices = list(range(len(chain)))
            all_edges = dfg_data["edges"]
            chain_edges = [
                {"src": src, "dst": dst, "register": reg}
                for e in all_edges
                if (src := e["src"]) in node_indices
                and (dst := e["dst"]) in node_indices
                and dst == src + 1  # only consecutive edges
            ]

            try:
                pattern = normalize_chain(chain, chain_edges, registry)
            except ValueError:
                continue

            key = pattern.template_key
            if key not in groups:
                groups[key] = {
                    "pattern": pattern,
                    "occurrence_count": 0,
                    "total_frequency": 0,
                    "source_bbs": [],
                }

            groups[key]["occurrence_count"] += 1
            groups[key]["total_frequency"] += frequency
            if vaddr not in groups[key]["source_bbs"]:
                groups[key]["source_bbs"].append(vaddr)

    # Build sorted result
    results = []
    for key, group in groups.items():
        p = group["pattern"]
        results.append({
            "opcodes": p.opcodes,
            "register_class": p.register_class,
            "length": p.length,
            "occurrence_count": group["occurrence_count"],
            "total_frequency": group["total_frequency"],
            "chain_registers": p.chain_registers,
            "source_bbs": group["source_bbs"],
        })

    results.sort(key=lambda x: x["total_frequency"], reverse=True)

    if top is not None:
        results = results[:top]

    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_miner.py::TestAggregatePatterns -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tools/fusion/miner.py tools/fusion/tests/test_miner.py
git commit -m "feat(fusion): add pattern aggregation with BBV frequency weighting"
```

---

### Task 7: Miner output serialization

**Files:**
- Modify: `tools/fusion/miner.py`
- Modify: `tools/fusion/tests/test_miner.py`

**Context:** The miner's `mine()` function ties everything together: loads DFG JSON files from a directory, reads the hotspot report, runs aggregation, and writes the output JSON.

**Step 1: Write the failing tests**

Append to `tools/fusion/tests/test_miner.py`:

```python
class TestMineOutput(unittest.TestCase):
    """End-to-end mine function: load → aggregate → serialize."""

    def test_mine_produces_valid_json(self):
        """mine() writes a JSON file with correct structure."""
        import tempfile
        from fusion.miner import mine
        registry = _make_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patterns.json"
            patterns = mine(
                dfg_dir=FIXTURES,
                hotspot_path=FIXTURES / "hotspot.json",
                registry=registry,
                output_path=output_path,
                top=5,
            )
            self.assertGreater(len(patterns), 0)
            # Verify output file
            data = json.loads(output_path.read_text())
            self.assertIn("generated", data)
            self.assertIn("source_df_count", data)
            self.assertIn("pattern_count", data)
            self.assertIn("patterns", data)
            self.assertEqual(data["patterns"][0]["rank"], 1)

    def test_mine_empty_directory(self):
        """mine() on a directory with no JSON files produces empty result."""
        import tempfile
        from fusion.miner import mine
        registry = _make_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patterns.json"
            patterns = mine(
                dfg_dir=Path(tmpdir),
                hotspot_path=FIXTURES / "hotspot.json",
                registry=registry,
                output_path=output_path,
            )
            self.assertEqual(len(patterns), 0)
            data = json.loads(output_path.read_text())
            self.assertEqual(data["pattern_count"], 0)
```

**Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest fusion/tests/test_miner.py::TestMineOutput -v`
Expected: FAIL with `ImportError: cannot import name 'mine'`

**Step 3: Add mine() function to miner.py**

Append to `tools/fusion/miner.py`:

```python
def mine(
    dfg_dir: Path,
    hotspot_path: Path,
    registry: ISARegistry,
    output_path: Path,
    top: int | None = None,
) -> list[dict]:
    """Run the full mining pipeline: load DFGs, aggregate, write output.

    Args:
        dfg_dir: Directory containing DFG JSON files.
        hotspot_path: Path to BBV hotspot JSON.
        registry: ISA registry for role resolution.
        output_path: Where to write the output JSON.
        top: If set, return only top N patterns.

    Returns:
        List of pattern dicts (same as aggregate_patterns).
    """
    # Load DFG JSON files
    dfg_files = sorted(dfg_dir.glob("*.json"))
    dfg_list = []
    for f in dfg_files:
        try:
            data = json.loads(f.read_text())
            if "nodes" in data and "edges" in data:
                dfg_list.append(data)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Skipping invalid DFG file: %s", f)

    # Load hotspot
    hotspot = json.loads(hotspot_path.read_text())

    # Aggregate
    patterns = aggregate_patterns(dfg_list, hotspot, registry, top=top)

    # Write output
    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_df_count": len(dfg_list),
        "pattern_count": len(patterns),
        "patterns": patterns,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    logger.info(
        "Mined %d patterns from %d DFGs (top=%s)",
        len(patterns), len(dfg_list), top,
    )

    return patterns
```

**Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest fusion/tests/test_miner.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add tools/fusion/miner.py tools/fusion/tests/test_miner.py
git commit -m "feat(fusion): add mine() pipeline with JSON output"
```

---

### Task 8: CLI entry point (`__main__.py`)

**Files:**
- Create: `tools/fusion/__main__.py`

**Context:** The CLI provides `python -m tools.fusion discover [options]`. It parses arguments, builds the ISA registry, runs the miner, and optionally invokes the agent.

Reference: `tools/dfg/__main__.py` for the ISA registry loading pattern (`_ISA_MODULES`, `load_isa_registry`).

**Step 1: Write __main__.py**

```python
# tools/fusion/__main__.py
"""CLI entry point for fusion pattern mining.

Usage:
    python -m tools.fusion discover --dfg-dir <dir> --report <json> --output <json>
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfg.instruction import ISARegistry

from fusion.miner import mine


_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),
    "M": ("dfg.isadesc.rv64m", "build_registry"),
}


def load_isa_registry(extensions: str) -> ISARegistry:
    """Build an ISA registry from comma-separated extension names."""
    registry = ISARegistry()
    for ext in extensions.split(","):
        ext = ext.strip().upper()
        if ext in _ISA_MODULES:
            module_path, func_name = _ISA_MODULES[ext]
            mod = importlib.import_module(module_path)
            builder = getattr(mod, func_name)
            builder(registry)
        else:
            logging.warning("Unknown ISA extension '%s' -- skipping", ext)
    return registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.fusion",
        description="Discover fusible instruction patterns from DFG output.",
    )
    parser.add_argument(
        "command",
        choices=["discover"],
        help="Command to run (currently: discover)",
    )
    parser.add_argument(
        "--dfg-dir",
        required=True,
        type=Path,
        help="Directory containing DFG JSON files",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Path to BBV hotspot JSON report",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for pattern catalog JSON",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top patterns to include (default: 20)",
    )
    parser.add_argument(
        "--isa",
        default="I,F,M",
        help="Comma-separated ISA extensions (default: I,F,M)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        default=False,
        help="Skip agent analysis, run miner only",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Claude model name for agent analysis",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    registry = load_isa_registry(args.isa)

    patterns = mine(
        dfg_dir=args.dfg_dir,
        hotspot_path=args.report,
        registry=registry,
        output_path=args.output,
        top=args.top,
    )

    # Agent analysis
    if not args.no_agent and patterns:
        from fusion.agent import run_agent
        run_agent(
            patterns=patterns,
            output_path=args.output,
            model=args.model,
        )

    # Text summary to stdout
    print(f"\nFusion Pattern Mining Results")
    print(f"  Patterns found: {len(patterns)}")
    if patterns:
        print(f"  Top pattern: {' → '.join(patterns[0]['opcodes'])} "
              f"(frequency: {patterns[0]['total_frequency']:,})")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Test CLI runs with --help**

Run: `cd tools && python -m fusion discover --help`
Expected: Usage text printed with all flags visible

**Step 3: Test CLI runs on fixtures in --no-agent mode**

Run: `cd tools && python -m fusion discover --dfg-dir fusion/tests/fixtures/ --report fusion/tests/fixtures/hotspot.json --output /tmp/test_patterns.json --no-agent -v`
Expected: Prints summary with patterns found > 0, and `/tmp/test_patterns.json` contains valid JSON

**Step 4: Commit**

```bash
git add tools/fusion/__main__.py
git commit -m "feat(fusion): add CLI entry point with discover command"
```

---

### Task 9: Agent layer

**Files:**
- Create: `tools/fusion/agent.py`

**Context:** The agent layer calls Claude CLI subprocess (same pattern as `tools/dfg/agent.py:_invoke_claude`) to analyze the miner's pattern output. It reads the pattern JSON, constructs a prompt, and appends the agent's analysis to the output JSON.

Since agent calls require the `claude` CLI and are non-deterministic, this module is not unit-tested in isolation. Integration testing happens in Task 10.

**Step 1: Write agent.py**

```python
# tools/fusion/agent.py
"""Agent layer for fusion pattern analysis.

Calls Claude CLI subprocess to analyze miner output and produce
recommendations on which patterns have the highest fusion value.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("fusion.agent")


def _build_analysis_prompt(patterns: list[dict]) -> str:
    """Build a prompt for Claude to analyze fusion patterns.

    Args:
        patterns: List of pattern dicts from the miner.

    Returns:
        Prompt string for Claude CLI.
    """
    # Include top 20 patterns for analysis (truncated for prompt size)
    top = patterns[:20]
    pattern_text = json.dumps(top, indent=2)

    prompt = (
        "fusion-discover: Analyze the following RISC-V instruction fusion "
        "pattern candidates extracted from real workload DFGs. For each "
        "pattern, evaluate its fusion potential based on:\n"
        "1. Frequency (higher = more impact from fusion)\n"
        "2. Dependency tightness (RAW chain density, register reuse)\n"
        "3. Hardware feasibility (same execution unit, operand count)\n\n"
        "Return JSON with keys:\n"
        "- 'top_recommendations': list of objects with 'pattern_rank' (int), "
        "'recommendation' (str: 'Strong candidate'/'Moderate'/'Weak'), "
        "'rationale' (str), 'notes' (str, optional)\n"
        "- 'missed_patterns': list of pattern descriptions the miner might "
        "have missed (empty list if none)\n"
        "- 'summary': str, 2-3 sentence overall assessment\n\n"
        f"Patterns (total: {len(patterns)}):\n{pattern_text}"
    )
    return prompt


def _invoke_claude(prompt: str, model: str | None = None) -> str | None:
    """Call Claude CLI subprocess. Returns stdout or None on failure."""
    cmd: list[str] = ["claude"]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--print", prompt])

    logger.debug("Agent command: %s", " ".join(cmd[:-1]) + " <prompt>")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        logger.warning("Claude CLI not found — skipping agent analysis")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out — skipping agent analysis")
        return None

    if result.returncode != 0:
        logger.warning("Claude CLI failed (rc=%d) — skipping agent analysis", result.returncode)
        return None

    return result.stdout.strip() or None


def run_agent(
    patterns: list[dict],
    output_path: Path,
    model: str | None = None,
) -> dict | None:
    """Run agent analysis on miner output and append to JSON file.

    Args:
        patterns: List of pattern dicts from the miner.
        output_path: Path to the miner output JSON (will be updated in-place).
        model: Optional Claude model name.

    Returns:
        Parsed analysis dict, or None if agent was unavailable.
    """
    prompt = _build_analysis_prompt(patterns)
    response = _invoke_claude(prompt, model=model)

    if response is None:
        logger.info("Agent analysis skipped (CLI unavailable or failed)")
        return None

    # Try to parse the response as JSON
    try:
        analysis = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Agent response was not valid JSON — storing as text summary")
        analysis = {
            "top_recommendations": [],
            "missed_patterns": [],
            "summary": response,
        }

    # Append analysis to output file
    output_data = json.loads(output_path.read_text())
    output_data["analysis"] = analysis
    output_path.write_text(json.dumps(output_data, indent=2) + "\n")

    logger.info("Agent analysis appended to %s", output_path)

    # Print summary to stdout
    if "summary" in analysis:
        print(f"\nAgent Analysis Summary:\n  {analysis['summary']}")

    return analysis
```

**Step 2: Test --no-agent mode still works (agent module importable but not called)**

Run: `cd tools && python -m fusion discover --dfg-dir fusion/tests/fixtures/ --report fusion/tests/fixtures/hotspot.json --output /tmp/test_no_agent.json --no-agent`
Expected: Runs successfully, no agent analysis section in output

**Step 3: Commit**

```bash
git add tools/fusion/agent.py
git commit -m "feat(fusion): add agent layer for pattern analysis"
```

---

### Task 10: Full test suite run and coverage check

**Files:**
- No new files — verify everything works end-to-end.

**Step 1: Run all fusion tests**

Run: `cd tools && python -m pytest fusion/tests/ -v`
Expected: All tests PASS (pattern tests + miner tests)

**Step 2: Run with coverage**

Run: `cd tools && python -m pytest fusion/tests/ -v --cov=fusion --cov-report=term-missing`
Expected: Coverage >= 80% for pattern.py and miner.py (agent.py excluded since it requires Claude CLI)

If coverage is below 80%, add targeted tests for uncovered lines.

**Step 3: Run existing DFG tests to ensure no regressions**

Run: `cd tools && python -m pytest dfg/tests/ -v`
Expected: All existing DFG tests still PASS (fusion code doesn't modify DFG engine)

**Step 4: End-to-end CLI smoke test**

Run: `cd tools && python -m fusion discover --dfg-dir fusion/tests/fixtures/ --report fusion/tests/fixtures/hotspot.json --output /tmp/e2e_patterns.json --no-agent -v`
Expected:
- Prints summary with patterns found
- `/tmp/e2e_patterns.json` contains valid JSON with `generated`, `source_df_count`, `pattern_count`, `patterns` fields
- Top pattern has `rank: 1` and `opcodes: ["fadd.s", "fmul.s"]`

**Step 5: Verify output JSON against design schema**

Run: `python3 -c "import json; d=json.load(open('/tmp/e2e_patterns.json')); assert 'patterns' in d; assert d['patterns'][0].get('rank')==1; print('Schema OK')"`
Expected: `Schema OK`

**Step 6: Commit (if any test adjustments were needed)**

```bash
git add -A
git commit -m "test(fusion): verify full test suite and coverage"
```

---

## Summary

| Task | Component | Tests | Commit message |
|------|-----------|-------|----------------|
| 1 | Package skeleton + fixtures | N/A | `feat(fusion): add package skeleton and test fixtures` |
| 2 | Pattern dataclass + template key | 6 | `feat(fusion): add Pattern dataclass with template key` |
| 3 | Register classification + control flow | 6 | `feat(fusion): add register classification and control flow detection` |
| 4 | normalize_chain + ISARegistry | 4 | `feat(fusion): add normalize_chain with ISARegistry role resolution` |
| 5 | enumerate_chains | 5 | `feat(fusion): add chain enumeration from DFG JSON` |
| 6 | aggregate_patterns + BBV weighting | 5 | `feat(fusion): add pattern aggregation with BBV frequency weighting` |
| 7 | mine() pipeline + JSON output | 2 | `feat(fusion): add mine() pipeline with JSON output` |
| 8 | CLI __main__.py | manual | `feat(fusion): add CLI entry point with discover command` |
| 9 | Agent layer | manual | `feat(fusion): add agent layer for pattern analysis` |
| 10 | Full test suite + coverage | all | `test(fusion): verify full test suite and coverage` |

**Total: 10 tasks, 28 automated tests, 9 commits**

