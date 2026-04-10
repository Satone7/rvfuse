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
        """Two sources feeding one consumer: 6 subgraphs total."""
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
        """Linear chain 0->1->2: 6 subgraphs."""
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

    def test_real_dfg_fixture(self):
        """Real 3-node DFG from fixture produces expected subgraphs."""
        from fusion.subgraph import enumerate_subgraphs
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        result = enumerate_subgraphs(dfg, max_nodes=8)
        sizes = sorted(len(s) for s in result)
        self.assertEqual(sizes[0], 1)
        self.assertLessEqual(sizes[-1], 8)


if __name__ == "__main__":
    unittest.main()
