#!/usr/bin/env python3
"""Tests for DFG builder."""

import unittest

from dfg.instruction import BasicBlock, DFGEdge, Instruction, ISARegistry
from dfg.isadesc.rv64i import build_registry
from dfg.dfg import build_dfg


def _make_registry() -> ISARegistry:
    reg = ISARegistry()
    build_registry(reg)
    return reg


class TestBuildDfg(unittest.TestCase):
    def test_simple_chain(self):
        """addi sp,sp,-32 -> sd ra,24(sp): sp flows from insn 0 to 1."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "sp,sp,-32", ""),
            Instruction(0x1004, "sd", "ra,24(sp)", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        self.assertEqual(len(dfg.nodes), 2)
        self.assertEqual(len(dfg.edges), 1)
        self.assertEqual(dfg.edges[0].register, "sp")
        self.assertEqual(dfg.edges[0].src_index, 0)
        self.assertEqual(dfg.edges[0].dst_index, 1)

    def test_multiple_deps(self):
        """addw a1,a1,a0 reads both a1 and a0 from prior instructions."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "lw", "a0,-28(s0)", ""),
            Instruction(0x1004, "lw", "a1,-24(s0)", ""),
            Instruction(0x1008, "addw", "a1,a1,a0", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        self.assertEqual(len(dfg.edges), 2)
        regs = {(e.src_index, e.dst_index): e.register for e in dfg.edges}
        self.assertEqual(regs[(0, 2)], "a0")
        self.assertEqual(regs[(1, 2)], "a1")

    def test_overwritten_register(self):
        """If a register is written twice, only the latest writer is used."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "a0,zero,1", ""),
            Instruction(0x1004, "addi", "a0,zero,2", ""),
            Instruction(0x1008, "sw", "a0,-20(s0)", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        a0_edges = [e for e in dfg.edges if e.register == "a0"]
        self.assertEqual(len(a0_edges), 1)
        self.assertEqual(a0_edges[0].src_index, 1)

    def test_no_dependencies(self):
        """Independent instructions produce no edges."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "a0,zero,1", ""),
            Instruction(0x1004, "addi", "a1,zero,2", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        self.assertEqual(len(dfg.edges), 0)

    def test_self_dependency(self):
        """addi sp,sp,-32: sp is both src and dst, but no self-edge."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "sp,sp,-32", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        self.assertEqual(len(dfg.edges), 0)

    def test_real_bb5(self):
        """Test against BB 5 from the YOLO demo (div/mod pattern)."""
        bb = BasicBlock(bb_id=5, vaddr=0x1117c, instructions=[
            Instruction(0x1117c, "lw", "a0,-28(s0)", ""),
            Instruction(0x11180, "srliw", "a1,a0,31", ""),
            Instruction(0x11184, "addw", "a1,a1,a0", ""),
            Instruction(0x11186, "andi", "a1,a1,-2", ""),
            Instruction(0x11188, "subw", "a0,a0,a1", ""),
            Instruction(0x1118a, "bnez", "a0,20", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        self.assertEqual(len(dfg.nodes), 6)
        edge_map = {(e.src_index, e.dst_index, e.register) for e in dfg.edges}
        self.assertIn((0, 1, "a0"), edge_map)
        self.assertIn((0, 4, "a0"), edge_map)

    def test_dfg_source_is_script(self):
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[Instruction(0x1000, "nop", "", "")])
        dfg = build_dfg(bb, _make_registry())
        self.assertEqual(dfg.source, "script")

    def test_source_preserved(self):
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[Instruction(0x1000, "nop", "", "")])
        dfg = build_dfg(bb, _make_registry(), source="agent")
        self.assertEqual(dfg.source, "agent")

    def test_zero_register_no_edge(self):
        """mv a0,zero: zero is constant, no edge from zero."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "mv", "a0,zero", ""),
            Instruction(0x1004, "sw", "a0,-20(s0)", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        zero_edges = [e for e in dfg.edges if e.register == "zero"]
        self.assertEqual(len(zero_edges), 0)
        a0_edges = [e for e in dfg.edges if e.register == "a0"]
        self.assertEqual(len(a0_edges), 1)


if __name__ == "__main__":
    unittest.main()
