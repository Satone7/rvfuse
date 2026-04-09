"""Integration tests for the scoring pipeline."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfg.instruction import ISARegistry
from dfg.isadesc.rv64i import build_registry as build_i
from dfg.isadesc.rv64f import build_registry as build_f
from dfg.isadesc.rv64m import build_registry as build_m
from dfg.isadesc.rv64v import build_registry as build_v
from fusion.scorer import score as run_score


def _full_registry() -> ISARegistry:
    reg = ISARegistry()
    build_i(reg)
    build_f(reg)
    build_m(reg)
    build_v(reg)
    return reg


class TestScoringPipeline(unittest.TestCase):
    def setUp(self):
        self.registry = _full_registry()
        self.catalog = {
            "generated": "2026-04-09T00:00:00Z",
            "source_df_count": 5,
            "pattern_count": 4,
            "patterns": [
                {
                    "opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                    "length": 2, "occurrence_count": 42, "total_frequency": 150000,
                    "chain_registers": [["frd", "frs1"]], "source_bbs": ["0x1000"], "rank": 1,
                },
                {
                    "opcodes": ["vadd.vv", "vmul.vv"], "register_class": "vector",
                    "length": 2, "occurrence_count": 30, "total_frequency": 100000,
                    "chain_registers": [["vrd", "vrs2"]], "source_bbs": ["0x3000"], "rank": 2,
                },
                {
                    "opcodes": ["add", "mul"], "register_class": "integer",
                    "length": 2, "occurrence_count": 15, "total_frequency": 50000,
                    "chain_registers": [["rd", "rs1"]], "source_bbs": ["0x4000"], "rank": 3,
                },
                {
                    "opcodes": ["flw", "fmul.s"], "register_class": "float",
                    "length": 2, "occurrence_count": 8, "total_frequency": 20000,
                    "chain_registers": [["frd", "frs1"]], "source_bbs": ["0x5000"], "rank": 4,
                },
            ],
        }

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(catalog_path=catalog_path, registry=self.registry, output_path=output_path)

            self.assertTrue(output_path.exists())
            output_data = json.loads(output_path.read_text())
            self.assertIn("candidates", output_data)
            self.assertEqual(output_data["candidate_count"], 4)
            for c in output_data["candidates"]:
                self.assertIn("score", c)
                self.assertIn("score_breakdown", c)
                self.assertIn("hardware", c)
                self.assertIn("status", c["hardware"])

    def test_feasibility_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(catalog_path=catalog_path, registry=self.registry,
                                output_path=output_path, feasibility_only=True)
            statuses = [r["hardware"]["status"] for r in results]
            self.assertIn("infeasible", statuses)

    def test_min_score_filters_infeasible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(catalog_path=catalog_path, registry=self.registry,
                                output_path=output_path, min_score=0.1)
            for c in results:
                self.assertGreaterEqual(c["score"], 0.1)

    def test_top_n_limits_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "patterns.json"
            output_path = Path(tmpdir) / "candidates.json"
            catalog_path.write_text(json.dumps(self.catalog))

            results = run_score(catalog_path=catalog_path, registry=self.registry,
                                output_path=output_path, top=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["score"], max(r["score"] for r in results))

    def test_vext_pattern_gets_encoding(self):
        enc = self.registry.get_encoding("vadd.vv")
        self.assertIsNotNone(enc)
        self.assertEqual(enc.opcode, 0x57)
        self.assertEqual(enc.reg_class, "vector")


class TestEncodingMetadataAvailable(unittest.TestCase):
    def setUp(self):
        self.registry = _full_registry()

    def test_float_instructions_have_encoding(self):
        for mn in ["fadd.s", "fsub.s", "fmul.s", "fdiv.s", "flw", "fsw", "fmadd.s"]:
            enc = self.registry.get_encoding(mn)
            self.assertIsNotNone(enc, f"Missing encoding for {mn}")
            self.assertEqual(enc.reg_class, "float")

    def test_integer_instructions_have_encoding(self):
        for mn in ["add", "sub", "mul", "addi", "lw", "sw"]:
            enc = self.registry.get_encoding(mn)
            self.assertIsNotNone(enc, f"Missing encoding for {mn}")

    def test_vector_instructions_have_encoding(self):
        for mn in ["vadd.vv", "vadd.vx", "vmul.vv", "vadd.vi", "vsetvli"]:
            enc = self.registry.get_encoding(mn)
            self.assertIsNotNone(enc, f"Missing encoding for {mn}")
            self.assertEqual(enc.reg_class, "vector")

    def test_load_store_flagged_correctly(self):
        self.assertTrue(self.registry.get_encoding("lw").may_load)
        self.assertTrue(self.registry.get_encoding("sw").may_store)
        self.assertFalse(self.registry.get_encoding("add").may_load)
