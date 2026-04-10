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
            self.assertIn("enumeration_stats", data)
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
