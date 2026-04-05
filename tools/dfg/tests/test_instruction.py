#!/usr/bin/env python3
"""Tests for instruction.py"""

import unittest
from dfg.instruction import (
    BasicBlock,
    DFG,
    DFGEdge,
    DFGNode,
    Instruction,
    RegisterFlow,
)


class TestInstruction(unittest.TestCase):
    def test_instruction_creation(self):
        insn = Instruction(
            address=0x111F4,
            mnemonic="addi",
            operands="sp,sp,-32",
            raw_line="  0x111f4: addi                    sp,sp,-32",
        )
        self.assertEqual(insn.address, 0x111F4)
        self.assertEqual(insn.mnemonic, "addi")
        self.assertEqual(insn.operands, "sp,sp,-32")

    def test_register_flow_creation(self):
        flow = RegisterFlow(dst_regs=["rd"], src_regs=["rs1", "rs2"])
        self.assertEqual(flow.dst_regs, ["rd"])
        self.assertEqual(flow.src_regs, ["rs1", "rs2"])

    def test_basic_block_creation(self):
        insns = [
            Instruction(0x111F4, "addi", "sp,sp,-32", ""),
            Instruction(0x111F6, "sd", "ra,24(sp)", ""),
        ]
        bb = BasicBlock(bb_id=1, vaddr=0x111F4, instructions=insns)
        self.assertEqual(bb.bb_id, 1)
        self.assertEqual(len(bb.instructions), 2)

    def test_dfg_node_creation(self):
        insn = Instruction(0x1000, "add", "a0,a1,a2", "")
        node = DFGNode(instruction=insn, index=0)
        self.assertEqual(node.index, 0)
        self.assertEqual(node.instruction.mnemonic, "add")

    def test_dfg_edge_creation(self):
        edge = DFGEdge(src_index=0, dst_index=2, register="a0")
        self.assertEqual(edge.src_index, 0)
        self.assertEqual(edge.dst_index, 2)
        self.assertEqual(edge.register, "a0")

    def test_dfg_creation(self):
        bb = BasicBlock(bb_id=1, vaddr=0x1000, instructions=[])
        dfg = DFG(bb=bb, nodes=[], edges=[], source="script")
        self.assertEqual(dfg.source, "script")


class TestISARegistry(unittest.TestCase):
    def test_registry_starts_empty(self):
        from dfg.instruction import ISARegistry
        reg = ISARegistry()
        self.assertIsNone(reg.get_flow("add"))

    def test_register_and_lookup(self):
        from dfg.instruction import ISARegistry
        reg = ISARegistry()
        reg.register("add", RegisterFlow(dst_regs=["rd"], src_regs=["rs1", "rs2"]))
        flow = reg.get_flow("add")
        self.assertIsNotNone(flow)
        self.assertEqual(flow.dst_regs, ["rd"])
        self.assertEqual(flow.src_regs, ["rs1", "rs2"])

    def test_known_mnemonics(self):
        from dfg.instruction import ISARegistry
        reg = ISARegistry()
        reg.register("add", RegisterFlow(dst_regs=["rd"], src_regs=["rs1", "rs2"]))
        self.assertTrue(reg.is_known("add"))
        self.assertFalse(reg.is_known("vadd.vv"))

    def test_load_extension(self):
        from dfg.instruction import ISARegistry
        reg = ISARegistry()
        reg.load_extension("add", [
            ("add", RegisterFlow(["rd"], ["rs1", "rs2"])),
            ("sub", RegisterFlow(["rd"], ["rs1", "rs2"])),
        ])
        self.assertTrue(reg.is_known("add"))
        self.assertTrue(reg.is_known("sub"))
        self.assertFalse(reg.is_known("mul"))

    def test_resolve_registers_add(self):
        """Test that register patterns get resolved to actual operand names."""
        from dfg.instruction import ISARegistry
        reg = ISARegistry()
        reg.register("add", RegisterFlow(dst_regs=["rd"], src_regs=["rs1", "rs2"]))
        flow = reg.get_flow("add")
        resolved = flow.resolve("a0,a1,a2")
        self.assertEqual(resolved.dst_regs, ["a0"])
        self.assertEqual(resolved.src_regs, ["a1", "a2"])

    def test_resolve_registers_sw(self):
        """Load/store: rs2 is src, offset(base) -> base is src."""
        from dfg.instruction import ISARegistry
        reg = ISARegistry()
        reg.register("sw", RegisterFlow(dst_regs=[], src_regs=["rs2", "rs1"]))
        flow = reg.get_flow("sw")
        resolved = flow.resolve("a0,-20(s0)")
        self.assertEqual(resolved.src_regs, ["a0", "s0"])

    def test_resolve_registers_addi(self):
        """Immediate: rd is dst, rs1 is src, immediate is ignored."""
        from dfg.instruction import ISARegistry
        reg = ISARegistry()
        reg.register("addi", RegisterFlow(dst_regs=["rd"], src_regs=["rs1"]))
        flow = reg.get_flow("addi")
        resolved = flow.resolve("sp,sp,-32")
        self.assertEqual(resolved.dst_regs, ["sp"])
        self.assertEqual(resolved.src_regs, ["sp"])


if __name__ == "__main__":
    unittest.main()
