#!/usr/bin/env python3
"""Tests for output serialization."""

import json
import tempfile
import unittest
from pathlib import Path

from dfg.instruction import (
    BasicBlock, DFG, DFGEdge, DFGNode, Instruction,
)
from dfg.output import dfg_to_dot, dfg_to_json, write_summary


class TestDfgToDot(unittest.TestCase):
    def _make_dfg(self) -> DFG:
        bb = BasicBlock(bb_id=1, vaddr=0x111F4, instructions=[
            Instruction(0x111F4, "addi", "sp,sp,-32", ""),
            Instruction(0x111F6, "sd", "ra,24(sp)", ""),
        ])
        return DFG(
            bb=bb,
            nodes=[DFGNode(bb.instructions[0], 0), DFGNode(bb.instructions[1], 1)],
            edges=[DFGEdge(0, 1, "sp")],
            source="script",
        )

    def test_contains_digraph_header(self):
        dot = dfg_to_dot(self._make_dfg())
        self.assertIn("digraph DFG_BB1", dot)
        self.assertIn("shape=record", dot)

    def test_contains_node_labels(self):
        dot = dfg_to_dot(self._make_dfg())
        self.assertIn("addi", dot)
        self.assertIn("sp,sp,-32", dot)

    def test_contains_edge(self):
        dot = dfg_to_dot(self._make_dfg())
        self.assertIn("-> 1", dot)
        self.assertIn('"sp"', dot)

    def test_empty_dfg(self):
        bb = BasicBlock(bb_id=99, vaddr=0x1000, instructions=[])
        dfg = DFG(bb=bb, nodes=[], edges=[], source="script")
        dot = dfg_to_dot(dfg)
        self.assertIn("digraph DFG_BB99", dot)


class TestDfgToJson(unittest.TestCase):
    def test_basic_structure(self):
        bb = BasicBlock(bb_id=1, vaddr=0x111F4, instructions=[
            Instruction(0x111F4, "addi", "sp,sp,-32", ""),
            Instruction(0x111F6, "sd", "ra,24(sp)", ""),
        ])
        dfg = DFG(
            bb=bb,
            nodes=[DFGNode(bb.instructions[0], 0), DFGNode(bb.instructions[1], 1)],
            edges=[DFGEdge(0, 1, "sp")],
            source="script",
        )
        result = dfg_to_json(dfg)
        parsed = json.loads(result)
        self.assertEqual(parsed["bb_id"], 1)
        self.assertEqual(parsed["vaddr"], "0x111f4")
        self.assertEqual(parsed["source"], "script")
        self.assertEqual(len(parsed["nodes"]), 2)
        self.assertEqual(len(parsed["edges"]), 1)
        self.assertEqual(parsed["edges"][0]["register"], "sp")

    def test_node_fields(self):
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "sp,sp,-32", ""),
        ])
        dfg = DFG(bb=bb, nodes=[DFGNode(bb.instructions[0], 0)], edges=[], source="script")
        result = dfg_to_json(dfg)
        parsed = json.loads(result)
        node = parsed["nodes"][0]
        self.assertEqual(node["index"], 0)
        self.assertEqual(node["address"], "0x1000")
        self.assertEqual(node["mnemonic"], "addi")
        self.assertEqual(node["operands"], "sp,sp,-32")

    def test_edge_fields(self):
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "a0,zero,1", ""),
            Instruction(0x1004, "sw", "a0,0(s1)", ""),
        ])
        dfg = DFG(
            bb=bb,
            nodes=[DFGNode(bb.instructions[0], 0), DFGNode(bb.instructions[1], 1)],
            edges=[DFGEdge(0, 1, "a0")],
            source="script",
        )
        result = dfg_to_json(dfg)
        parsed = json.loads(result)
        edge = parsed["edges"][0]
        self.assertEqual(edge["src"], 0)
        self.assertEqual(edge["dst"], 1)
        self.assertEqual(edge["register"], "a0")


class TestWriteSummary(unittest.TestCase):
    def test_summary_structure(self):
        stats = {
            "input_file": "test.disas",
            "total_bbs": 3,
            "script_generated": 2,
            "agent_generated": 1,
            "agent_checked_pass": 2,
            "agent_checked_fail": 0,
            "isa_extensions_used": ["I"],
            "unsupported_instructions": ["vadd.vv"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            write_summary(stats, path)
            content = json.loads(path.read_text())
            self.assertEqual(content["total_bbs"], 3)
            self.assertEqual(content["agent_generated"], 1)
            self.assertIn("vadd.vv", content["unsupported_instructions"])


if __name__ == "__main__":
    unittest.main()
