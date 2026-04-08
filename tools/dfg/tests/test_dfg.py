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

    def test_unknown_mnemonic_mid_block(self):
        """Edges still form around an unknown mnemonic: [known, unknown, known]."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "addi", "a0,zero,1", ""),
            Instruction(0x1004, "custom_op", "a0,a1", ""),
            Instruction(0x1008, "sw", "a0,-20(s0)", ""),
        ])
        dfg = build_dfg(bb, _make_registry())
        # custom_op is unknown, so no edges involve index 1.
        # But a0 written at index 0 is read at index 2 (via sw's implicit
        # dependency tracking). Since custom_op is skipped, the last-writer
        # for a0 remains index 0.
        a0_edges = [e for e in dfg.edges if e.register == "a0"]
        self.assertEqual(len(a0_edges), 1)
        self.assertEqual(a0_edges[0].src_index, 0)
        self.assertEqual(a0_edges[0].dst_index, 2)

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


def _make_imf_registry() -> ISARegistry:
    """Build a registry with I + M + F extensions."""
    reg = ISARegistry()
    from dfg.isadesc.rv64i import build_registry as build_i
    from dfg.isadesc.rv64m import build_registry as build_m
    from dfg.isadesc.rv64f import build_registry as build_f
    build_i(reg)
    build_m(reg)
    build_f(reg)
    return reg


class TestFExtensionDfg(unittest.TestCase):
    """DFG builder tests for F-extension instructions."""

    def test_fmadd_s_chain(self):
        """fmadd.s ft2,fa4,ft0,ft2 writes ft2; next fadd.s reads ft2."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "fmadd.s", "dyn,ft2,fa4,ft0,ft2", ""),
            Instruction(0x1004, "fadd.s", "dyn,fa5,ft2,fa0", ""),
        ])
        dfg = build_dfg(bb, _make_imf_registry())
        ft2_edges = [e for e in dfg.edges if e.register == "ft2"]
        self.assertEqual(len(ft2_edges), 1)
        self.assertEqual(ft2_edges[0].src_index, 0)
        self.assertEqual(ft2_edges[0].dst_index, 1)

    def test_flw_fadd_chain(self):
        """flw fa4,0(a6) writes fa4; fadd.s reads fa4."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "flw", "fa4,0(a6)", ""),
            Instruction(0x1004, "fadd.s", "dyn,fa5,fa4,fa0", ""),
        ])
        dfg = build_dfg(bb, _make_imf_registry())
        fa4_edges = [e for e in dfg.edges if e.register == "fa4"]
        self.assertEqual(len(fa4_edges), 1)
        self.assertEqual(fa4_edges[0].src_index, 0)
        self.assertEqual(fa4_edges[0].dst_index, 1)

    def test_fsw_reads_float_reg(self):
        """fmul.s writes ft2; fsw stores ft2."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "fmul.s", "dyn,ft2,ft2,fa0", ""),
            Instruction(0x1004, "fsw", "ft2,0(a2)", ""),
        ])
        dfg = build_dfg(bb, _make_imf_registry())
        ft2_edges = [e for e in dfg.edges if e.register == "ft2"]
        self.assertEqual(len(ft2_edges), 1)
        self.assertEqual(ft2_edges[0].src_index, 0)
        self.assertEqual(ft2_edges[0].dst_index, 1)

    def test_mixed_int_float_bb(self):
        """A BB mixing integer and float instructions."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "flw", "fa4,0(a6)", ""),
            Instruction(0x1004, "fmadd.s", "dyn,ft2,fa4,ft0,ft2", ""),
            Instruction(0x1008, "addi", "a6,a6,8", ""),
            Instruction(0x100c, "fsw", "ft2,0(a2)", ""),
        ])
        dfg = build_dfg(bb, _make_imf_registry())
        fa4_edges = [e for e in dfg.edges if e.register == "fa4"]
        self.assertEqual(len(fa4_edges), 1)
        self.assertEqual(fa4_edges[0].src_index, 0)
        self.assertEqual(fa4_edges[0].dst_index, 1)
        ft2_edges = [e for e in dfg.edges if e.register == "ft2"]
        self.assertEqual(len(ft2_edges), 1)
        self.assertEqual(ft2_edges[0].src_index, 1)
        self.assertEqual(ft2_edges[0].dst_index, 3)
        self.assertEqual(len([e for e in dfg.edges if e.register == "a6"]), 0)

    def test_yolo_hot_bb31041_partial(self):
        """Partial test of the YOLO11n hottest BB (31041) — first 6 instructions.

        All 6 instructions are flw loads. a5 is read as a base by 4 of them
        but never written within the BB, so there are no intra-BB edges for a5.
        The float registers (fa4, fa5, ft3, ft0, fa1, fa2) are each written
        exactly once and never re-read within this partial block, so there
        are no edges at all.
        """
        bb = BasicBlock(bb_id=31041, vaddr=0x7eeef161ecac, instructions=[
            Instruction(0x7eeef161ecac, "flw", "fa4,0(a6)", ""),
            Instruction(0x7eeef161ecb0, "flw", "fa5,0(a7)", ""),
            Instruction(0x7eeef161ecb4, "flw", "ft3,12(a5)", ""),
            Instruction(0x7eeef161ecb8, "flw", "ft0,0(a5)", ""),
            Instruction(0x7eeef161ecbc, "flw", "fa1,4(a5)", ""),
            Instruction(0x7eeef161ecc0, "flw", "fa2,8(a5)", ""),
        ])
        dfg = build_dfg(bb, _make_imf_registry())
        self.assertEqual(len(dfg.nodes), 6)
        # a5 is read but never written within this BB, so no edges.
        a5_edges = [e for e in dfg.edges if e.register == "a5"]
        self.assertEqual(len(a5_edges), 0)
        # No float register is both written and re-read, so no edges at all.
        self.assertEqual(len(dfg.edges), 0)


class TestVectorDfg(unittest.TestCase):
    """DFG builder tests for vector instructions with LMUL expansion."""

    def _make_v_registry(self) -> ISARegistry:
        from dfg.isadesc.rv64i import build_registry as build_i
        from dfg.isadesc.rv64v import build_registry as build_v
        reg = ISARegistry()
        build_i(reg)
        build_v(reg)
        return reg

    def test_vadd_vv_simple(self):
        """vadd.vv v4,v2,v6 — basic vector dependency, default LMUL=1."""
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "vle32.v", "v2, (a0)", ""),
            Instruction(0x1004, "vadd.vv", "v4, v2, v6", ""),
        ])
        bb.vec_config = None  # default LMUL=1
        dfg = build_dfg(bb, self._make_v_registry())
        v2_edges = [e for e in dfg.edges if e.register == "v2"]
        self.assertEqual(len(v2_edges), 1)
        self.assertEqual(v2_edges[0].src_index, 0)
        self.assertEqual(v2_edges[0].dst_index, 1)

    def test_vadd_vv_lmul2_expansion(self):
        """vadd.vv v4,v2,v6 with LMUL=2 — should expand to physical registers."""
        from dfg.instruction import VectorConfig
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "vle32.v", "v2, (a0)", ""),
            Instruction(0x1004, "vadd.vv", "v4, v2, v6", ""),
        ])
        bb.vec_config = VectorConfig(
            vlen=128, sew=32, lmul=2, vl=None,
            tail_policy="undisturbed", mask_policy="undisturbed",
            change_points=[],
        )
        dfg = build_dfg(bb, self._make_v_registry())
        edge_regs = [e.register for e in dfg.edges]
        # LMUL=2: vle32.v writes v2,v3; vadd.vv reads v2,v3,v6,v7
        # Only v2,v3 have intra-BB writers, so 2 edges
        self.assertIn("v2", edge_regs)
        self.assertIn("v3", edge_regs)
        self.assertEqual(len(dfg.edges), 2)

    def test_full_vector_bb(self):
        """Full vector BB from sample fixture: vsetvli + vle + vadd + vse."""
        from pathlib import Path
        from dfg.parser import parse_disas, _annotate_vector_config
        fixture = Path(__file__).parent / "fixtures" / "sample_vector_disas.txt"
        blocks = parse_disas(fixture)
        _annotate_vector_config(blocks)
        bb = blocks[0]
        dfg = build_dfg(bb, self._make_v_registry())
        self.assertEqual(len(dfg.nodes), 5)
        # Should have edges (at minimum: vle->vadd via v2, vadd->vse via v4)
        self.assertGreater(len(dfg.edges), 0)

    def test_vsetvli_writes_gpr(self):
        """vsetvli a0, a1, e32,m2 — should have edge from a1-writer to vsetvli."""
        from dfg.instruction import VectorConfig
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[
            Instruction(0x1000, "mv", "a1, a0", ""),
            Instruction(0x1004, "vsetvli", "a0, a1, e32,m2", ""),
            Instruction(0x1008, "vadd.vv", "v4, v2, v6", ""),
        ])
        bb.vec_config = VectorConfig(
            vlen=128, sew=32, lmul=2, vl=None,
            tail_policy="undisturbed", mask_policy="undisturbed",
            change_points=[],
        )
        dfg = build_dfg(bb, self._make_v_registry())
        self.assertEqual(len(dfg.nodes), 3)
        # mv writes a1; vsetvli reads a1 -> edge from 0 to 1
        a1_edges = [e for e in dfg.edges if e.register == "a1"]
        self.assertEqual(len(a1_edges), 1)
        self.assertEqual(a1_edges[0].src_index, 0)
        self.assertEqual(a1_edges[0].dst_index, 1)


if __name__ == "__main__":
    unittest.main()
