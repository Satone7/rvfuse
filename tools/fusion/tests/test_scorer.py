"""Unit tests for the fusion scoring function."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow
from fusion.constraints import ConstraintChecker, Verdict
from fusion.scorer import Scorer, DEFAULT_WEIGHTS


def _make_registry() -> ISARegistry:
    registry = ISARegistry()
    instructions = [
        ("fadd.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x00, reg_class="float"))),
        ("fmul.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x1, 0x00, reg_class="float"))),
        ("fsub.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x08, reg_class="float"))),
        ("vadd.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x00, reg_class="vector"))),
        ("vmul.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x2, 0x02, reg_class="vector"))),
        ("add", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x00, reg_class="integer"))),
        ("mul", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x01, reg_class="integer"))),
        ("flw", RegisterFlow(["frd"], ["rs1"],
            encoding=InstructionFormat("I", 0x07, 0x2, may_load=True, has_rs2=False, reg_class="float"))),
    ]
    for mnemonic, flow in instructions:
        registry.register(mnemonic, flow)
    return registry


class TestFreqScore(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(_make_registry(), max_frequency=100000)

    def test_max_frequency_scores_one(self):
        self.assertAlmostEqual(self.scorer._freq_score(100000), 1.0, places=5)

    def test_zero_frequency_scores_zero(self):
        self.assertAlmostEqual(self.scorer._freq_score(0), 0.0, places=5)

    def test_half_frequency_less_than_one(self):
        score = self.scorer._freq_score(50000)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_log_normalization_compresses_range(self):
        s_low = self.scorer._freq_score(1000)
        s_high = self.scorer._freq_score(100000)
        self.assertGreater(s_low, 0.5)


class TestTightScore(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(_make_registry())

    def test_full_density_single_chain(self):
        score = self.scorer._tight_score(chain_registers=[["frd", "frs1"]], length=2)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_no_edges_zero_density(self):
        score = self.scorer._tight_score(chain_registers=[], length=2)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_three_instr_chain_capped(self):
        score = self.scorer._tight_score(
            chain_registers=[["frd", "frs1"], ["frd", "frs1"]], length=3)
        self.assertAlmostEqual(score, 1.0, places=5)


class TestHwScore(unittest.TestCase):
    def test_feasible_scores_one(self):
        self.assertAlmostEqual(Scorer._hw_score(Verdict("feasible")), 1.0)

    def test_constrained_scores_half(self):
        self.assertAlmostEqual(Scorer._hw_score(Verdict("constrained")), 0.5)

    def test_infeasible_scores_zero(self):
        self.assertAlmostEqual(Scorer._hw_score(Verdict("infeasible")), 0.0)


class TestScorePattern(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(_make_registry(), max_frequency=100000)

    def test_feasible_has_positive_score(self):
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 10}
        result = self.scorer.score_pattern(pattern)
        self.assertGreater(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertIn("score_breakdown", result)
        self.assertIn("hardware", result)

    def test_infeasible_zero_hw(self):
        pattern = {"opcodes": ["flw", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 10}
        result = self.scorer.score_pattern(pattern)
        self.assertAlmostEqual(result["score_breakdown"]["hw_score"], 0.0)

    def test_high_frequency_ranks_higher(self):
        p_low = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                  "chain_registers": [["frd", "frs1"]], "total_frequency": 1000, "occurrence_count": 1}
        p_high = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]], "total_frequency": 90000, "occurrence_count": 100}
        self.assertGreater(self.scorer.score_pattern(p_high)["score"],
                          self.scorer.score_pattern(p_low)["score"])

    def test_custom_weights(self):
        scorer = Scorer(_make_registry(), max_frequency=100000,
                        weights={"frequency": 1.0, "tightness": 0.0, "hardware": 0.0})
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 10}
        result = scorer.score_pattern(pattern)
        self.assertAlmostEqual(result["score"], scorer._freq_score(50000), places=5)


class TestScorePatterns(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(_make_registry(), max_frequency=100000)

    def test_batch_sorted_descending(self):
        patterns = [
            {"opcodes": ["add", "mul"], "register_class": "integer", "chain_registers": [["rd", "rs1"]], "total_frequency": 1000, "occurrence_count": 1},
            {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]], "total_frequency": 90000, "occurrence_count": 100},
            {"opcodes": ["fadd.s", "fsub.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 50},
        ]
        results = self.scorer.score_patterns(patterns)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_n_filtering(self):
        patterns = [
            {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]], "total_frequency": 90000, "occurrence_count": 100},
            {"opcodes": ["fadd.s", "fsub.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 50},
            {"opcodes": ["add", "mul"], "register_class": "integer", "chain_registers": [["rd", "rs1"]], "total_frequency": 1000, "occurrence_count": 1},
        ]
        results = self.scorer.score_patterns(patterns, top=2)
        self.assertEqual(len(results), 2)

    def test_min_score_filters_infeasible(self):
        patterns = [
            {"opcodes": ["flw", "fmul.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]], "total_frequency": 90000, "occurrence_count": 100},
            {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]], "total_frequency": 50000, "occurrence_count": 50},
        ]
        results = self.scorer.score_patterns(patterns, min_score=0.1)
        statuses = [r["hardware"]["status"] for r in results]
        self.assertNotIn("infeasible", statuses)
