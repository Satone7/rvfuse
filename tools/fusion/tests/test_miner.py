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
        """float_chain_2.json has one RAW edge (0->1), producing 1 length-2 chain."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        chains = enumerate_chains(dfg, registry)
        self.assertEqual(len(chains), 1)
        self.assertEqual(len(chains[0]), 2)

    def test_float_chain_3_produces_three_chains(self):
        """float_chain_3.json has edges 0->1 and 1->2.
        Produces: 1 length-3 chain (0,1,2) + 2 length-2 chains (0,1) and (1,2)."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_3.json").read_text())
        chains = enumerate_chains(dfg, registry)
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
