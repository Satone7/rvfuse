#!/usr/bin/env python3
"""Integration tests for RVV support in the DFG engine."""

import unittest
from pathlib import Path

from dfg.instruction import ISARegistry, Instruction, BasicBlock, VectorConfig
from dfg.isadesc.rv64i import build_registry as build_i
from dfg.isadesc.rv64m import build_registry as build_m
from dfg.isadesc.rv64f import build_registry as build_f
from dfg.isadesc.rv64v import build_registry as build_v
from dfg.dfg import build_dfg


def _build_full_registry() -> ISARegistry:
    """Build registry with I + M + F + V extensions."""
    reg = ISARegistry()
    build_i(reg)
    build_m(reg)
    build_f(reg)
    build_v(reg)
    return reg


class TestFullRegistryLoad(unittest.TestCase):
    """Verify all four extensions load together without conflicts."""

    def test_no_mnemonic_collisions(self):
        """F/V should not have overlapping mnemonics."""
        reg_f = ISARegistry()
        build_f(reg_f)
        reg_v = ISARegistry()
        build_v(reg_v)
        collisions = set(reg_f._flows.keys()) & set(reg_v._flows.keys())
        # In LLVM 22, vfmv.f.s and vfmv.s.f have 'no pred' and appear in
        # neither extension, so there should be no collisions at all.
        self.assertEqual(collisions, set(), f"Unexpected collisions: {collisions}")

    def test_total_instruction_count(self):
        """V extension should add hundreds of instructions."""
        reg_i = ISARegistry()
        build_i(reg_i)
        reg_full = _build_full_registry()
        v_count = len(reg_full._flows) - len(reg_i._flows)
        self.assertGreater(v_count, 100)


class TestMixedIntFloatVectorBB(unittest.TestCase):
    """A BB mixing integer, float, and vector instructions."""

    def test_no_agent_fallback_for_known_instructions(self):
        """All I/F/M/V instructions should be recognized."""
        reg = _build_full_registry()
        known = ["addi", "fadd.s", "vadd.vv", "vsetvli", "vle32.v", "vse32.v", "fmul.s"]
        for mn in known:
            self.assertTrue(reg.is_known(mn), f"Expected '{mn}' to be known")

    def test_dfg_build_mixed_bb(self):
        """Build a DFG for a BB with int + float + vector instructions."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "a0, a1, 1", ""),
            Instruction(0x1004, "vsetvli", "zero, a0, e32,m1", ""),
            Instruction(0x1008, "vle32.v", "v2, (a2)", ""),
            Instruction(0x100c, "fmul.s", "dyn, ft0, fa0, fa1", ""),
            Instruction(0x1010, "vadd.vv", "v4, v2, v6", ""),
            Instruction(0x1014, "vse32.v", "v4, (a3)", ""),
        ])
        bb.vec_config = VectorConfig(
            vlen=128, sew=32, lmul=1, vl=None,
            tail_policy="undisturbed", mask_policy="undisturbed",
            change_points=[],
        )
        dfg = build_dfg(bb, _build_full_registry())
        self.assertEqual(len(dfg.nodes), 6)
        self.assertGreater(len(dfg.edges), 0)

    def test_existing_scalar_dfgs_unaffected(self):
        """Scalar-only DFGs should work exactly as before."""
        from dfg.isadesc.rv64i import build_registry as build_i_only
        from dfg.isadesc.rv64m import build_registry as build_m_only
        from dfg.isadesc.rv64f import build_registry as build_f_only
        reg = ISARegistry()
        build_i_only(reg)
        build_m_only(reg)
        build_f_only(reg)
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "sp,sp,-32", ""),
            Instruction(0x1004, "sd", "ra,24(sp)", ""),
        ])
        dfg = build_dfg(bb, reg)
        self.assertEqual(len(dfg.nodes), 2)
        self.assertEqual(len(dfg.edges), 1)
        self.assertEqual(dfg.edges[0].register, "sp")


class TestVectorEndToEndWithParser(unittest.TestCase):
    """End-to-end: parse fixture, annotate config, build DFG."""

    def test_sample_vector_disas(self):
        from dfg.parser import parse_disas, _annotate_vector_config
        fixture = Path(__file__).parent / "fixtures" / "sample_vector_disas.txt"
        blocks = parse_disas(fixture)
        self.assertEqual(len(blocks), 1)
        _annotate_vector_config(blocks)
        bb = blocks[0]
        self.assertIsNotNone(bb.vec_config)
        # First config is e32,m2
        self.assertEqual(bb.vec_config.sew, 32)
        self.assertEqual(bb.vec_config.lmul, 2)
        dfg = build_dfg(bb, _build_full_registry())
        self.assertEqual(len(dfg.nodes), 5)
        self.assertGreater(len(dfg.edges), 0)


if __name__ == "__main__":
    unittest.main()
