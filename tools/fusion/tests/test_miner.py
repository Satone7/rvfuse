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
        start, chain = chains[0]
        self.assertEqual(start, 0)
        self.assertEqual(len(chain), 2)

    def test_float_chain_3_produces_three_chains(self):
        """float_chain_3.json has edges 0->1 and 1->2.
        Produces: 1 length-3 chain (0,1,2) + 2 length-2 chains (0,1) and (1,2)."""
        from fusion.miner import enumerate_chains
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_3.json").read_text())
        chains = enumerate_chains(dfg, registry)
        self.assertEqual(len(chains), 3)
        lengths = sorted(len(chain) for _, chain in chains)
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
        start, chain = chains[0]
        self.assertEqual(start, 0)
        self.assertEqual(chain[0], ("fadd.s", "ft2,fa0,fa1"))
        self.assertEqual(chain[1], ("fmul.s", "ft3,ft2,fa2"))


class TestAggregatePatterns(unittest.TestCase):
    """Aggregate normalized patterns across multiple DFGs with BBV weighting."""

    def test_single_dfg_aggregation(self):
        """One DFG, one chain -> one aggregated pattern."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
        hotspot = json.loads((FIXTURES / "hotspot.json").read_text())
        patterns = aggregate_patterns([dfg], hotspot, registry)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["opcodes"], ["fadd.s", "fmul.s"])
        self.assertEqual(patterns[0]["total_frequency"], 200000)
        self.assertEqual(patterns[0]["occurrence_count"], 1)

    def test_cross_bb_aggregation(self):
        """Two BBs with the same pattern -> merged, frequencies summed."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        dfg1 = json.loads((FIXTURES / "float_chain_2.json").read_text())
        dfg2 = json.loads((FIXTURES / "float_chain_3.json").read_text())
        hotspot = json.loads((FIXTURES / "hotspot.json").read_text())
        patterns = aggregate_patterns([dfg1, dfg2], hotspot, registry)
        fadd_fmul = [p for p in patterns if p["opcodes"] == ["fadd.s", "fmul.s"]]
        self.assertEqual(len(fadd_fmul), 1)
        self.assertEqual(fadd_fmul[0]["occurrence_count"], 2)
        self.assertEqual(fadd_fmul[0]["total_frequency"], 300000)

    def test_bb_not_in_hotspot_gets_zero_frequency(self):
        """BB without hotspot entry gets frequency=0."""
        from fusion.miner import aggregate_patterns
        registry = _make_registry()
        dfg = json.loads((FIXTURES / "float_chain_2.json").read_text())
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


class TestMineOutput(unittest.TestCase):
    """End-to-end mine function: load -> aggregate -> serialize."""

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
