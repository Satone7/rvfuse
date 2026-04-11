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
        """fadd.s -> fmul.s chain: layer0=[fadd.s], layer1=[fmul.s]."""
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
        """Two flw -> fmadd.s: layer0=[flw, flw], layer1=[fmadd.s], 2 edges."""
        from fusion.pattern import normalize_subgraph
        dfg = json.loads((FIXTURES / "fan_in_2to1.json").read_text())
        pattern = normalize_subgraph(frozenset({0, 1, 2}), dfg, self.registry)
        self.assertEqual(pattern.size, 3)
        self.assertEqual(pattern.topology, [["flw", "flw"], ["fmadd.s"]])
        self.assertEqual(pattern.register_class, "float")
        self.assertEqual(len(pattern.edges), 2)
        # Verify role pairs (order may vary due to set operations)
        role_pairs = sorted((e.src_role, e.dst_role) for e in pattern.edges)
        self.assertIn(("frd", "frs1"), role_pairs)
        self.assertIn(("frd", "frs2"), role_pairs)

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
