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
    ResolvedFlow,
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


class TestRV64IInstructions(unittest.TestCase):
    """Tests for RV64I instruction register flow definitions."""

    def setUp(self) -> None:
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64i import build_registry

        self.registry = ISARegistry()
        build_registry(self.registry)

    def _resolve(self, mnemonic: str, operands: str) -> ResolvedFlow:
        flow = self.registry.get_flow(mnemonic)
        self.assertIsNotNone(flow, f"mnemonic '{mnemonic}' not registered")
        return flow.resolve(operands)  # type: ignore[union-attr]

    # --- R-type: rd = rs1 op rs2 ---
    def test_add(self) -> None:
        r = self._resolve("add", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_sub(self) -> None:
        r = self._resolve("sub", "t0,t1,t2")
        self.assertEqual(r.dst_regs, ["t0"])
        self.assertEqual(r.src_regs, ["t1", "t2"])

    def test_and(self) -> None:
        r = self._resolve("and", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_or(self) -> None:
        r = self._resolve("or", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_xor(self) -> None:
        r = self._resolve("xor", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_sll(self) -> None:
        r = self._resolve("sll", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_srl(self) -> None:
        r = self._resolve("srl", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_sra(self) -> None:
        r = self._resolve("sra", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_slt(self) -> None:
        r = self._resolve("slt", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_sltu(self) -> None:
        r = self._resolve("sltu", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    # --- I-type immediate: rd = rs1 op imm ---
    def test_addi(self) -> None:
        r = self._resolve("addi", "sp,sp,-32")
        self.assertEqual(r.dst_regs, ["sp"])
        self.assertEqual(r.src_regs, ["sp"])

    def test_andi(self) -> None:
        r = self._resolve("andi", "a0,a1,0xff")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_ori(self) -> None:
        r = self._resolve("ori", "a0,a0,0x1")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a0"])

    def test_xori(self) -> None:
        r = self._resolve("xori", "a0,a0,0x1")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a0"])

    def test_slti(self) -> None:
        r = self._resolve("slti", "a0,a1,10")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_sltiu(self) -> None:
        r = self._resolve("sltiu", "a0,a1,10")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    # --- I-type shift: rd = rs1 op shamt ---
    def test_slli(self) -> None:
        r = self._resolve("slli", "a0,a1,5")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_srli(self) -> None:
        r = self._resolve("srli", "a0,a1,5")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_srai(self) -> None:
        r = self._resolve("srai", "a0,a1,5")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    # --- W-suffix R-type ---
    def test_addw(self) -> None:
        r = self._resolve("addw", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    def test_subw(self) -> None:
        r = self._resolve("subw", "a0,a1,a2")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1", "a2"])

    # --- W-suffix I-type ---
    def test_addiw(self) -> None:
        r = self._resolve("addiw", "a0,a1,10")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_slliw(self) -> None:
        r = self._resolve("slliw", "a0,a1,5")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_srliw(self) -> None:
        r = self._resolve("srliw", "a0,a1,5")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_sraiw(self) -> None:
        r = self._resolve("sraiw", "a0,a1,5")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    # --- Loads: rd = mem[rs1+offset] ---
    def test_lw(self) -> None:
        r = self._resolve("lw", "a0,0(a1)")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_ld(self) -> None:
        r = self._resolve("ld", "a0,8(sp)")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["sp"])

    def test_lb(self) -> None:
        r = self._resolve("lb", "a0,0(a1)")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_lbu(self) -> None:
        r = self._resolve("lbu", "a0,0(a1)")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    # --- Stores: mem[rs1+offset] = rs2 ---
    def test_sw(self) -> None:
        r = self._resolve("sw", "a0,-20(s0)")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0", "s0"])

    def test_sd(self) -> None:
        r = self._resolve("sd", "ra,8(sp)")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["ra", "sp"])

    # --- Branches (2 reg) ---
    def test_beq(self) -> None:
        r = self._resolve("beq", "a0,a1,0x100")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0", "a1"])

    def test_bne(self) -> None:
        r = self._resolve("bne", "a0,a1,0x100")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0", "a1"])

    def test_blt(self) -> None:
        r = self._resolve("blt", "a0,a1,0x100")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0", "a1"])

    def test_bge(self) -> None:
        r = self._resolve("bge", "a0,a1,0x100")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0", "a1"])

    # --- Branches (1 reg / pseudo) ---
    def test_bgt(self) -> None:
        r = self._resolve("bgt", "a0,a1,0x100")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0", "a1"])

    def test_bnez(self) -> None:
        r = self._resolve("bnez", "a0,0x100")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0"])

    def test_beqz(self) -> None:
        r = self._resolve("beqz", "a0,0x100")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, ["a0"])

    # --- Jumps ---
    def test_jal(self) -> None:
        r = self._resolve("jal", "ra,0x1000")
        self.assertEqual(r.dst_regs, ["ra"])
        self.assertEqual(r.src_regs, [])

    def test_jalr(self) -> None:
        r = self._resolve("jalr", "ra,0(a1)")
        self.assertEqual(r.dst_regs, ["ra"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_j(self) -> None:
        r = self._resolve("j", "0x1000")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, [])

    def test_ret(self) -> None:
        r = self._resolve("ret", "")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, [])

    # --- Pseudo-instructions ---
    def test_mv(self) -> None:
        r = self._resolve("mv", "a0,a1")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, ["a1"])

    def test_li(self) -> None:
        r = self._resolve("li", "a0,42")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, [])

    def test_nop(self) -> None:
        r = self._resolve("nop", "")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, [])

    def test_ecall(self) -> None:
        r = self._resolve("ecall", "")
        self.assertEqual(r.dst_regs, [])
        self.assertEqual(r.src_regs, [])

    # --- Upper immediate ---
    def test_auipc(self) -> None:
        r = self._resolve("auipc", "a0,0x1000")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, [])

    def test_lui(self) -> None:
        r = self._resolve("lui", "a0,0x1000")
        self.assertEqual(r.dst_regs, ["a0"])
        self.assertEqual(r.src_regs, [])


class TestRegisterKind(unittest.TestCase):
    """Tests for the RegisterKind system and float register extraction."""

    def test_float_kind_matches_f_reg(self):
        from dfg.instruction import FLOAT_KIND
        self.assertIsNotNone(FLOAT_KIND.pattern.match("f0"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("f31"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("ft0"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("fa7"))
        self.assertIsNotNone(FLOAT_KIND.pattern.match("fs11"))

    def test_float_kind_rejects_int_reg(self):
        from dfg.instruction import FLOAT_KIND
        self.assertIsNone(FLOAT_KIND.pattern.match("a0"))
        self.assertIsNone(FLOAT_KIND.pattern.match("t0"))

    def test_integer_kind_matches_abi_names(self):
        from dfg.instruction import INTEGER_KIND
        self.assertIsNotNone(INTEGER_KIND.pattern.match("a0"))
        self.assertIsNotNone(INTEGER_KIND.pattern.match("zero"))
        self.assertIsNotNone(INTEGER_KIND.pattern.match("ra"))


class TestExtractRegistersFloat(unittest.TestCase):
    """Tests for _extract_registers with float register kinds."""

    def test_fadd_s_two_float_regs(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("fa5,fa0,fa4")
        self.assertEqual(regs["frd"][0], "fa5")
        self.assertEqual(regs["frd"][1], "float")
        self.assertEqual(regs["frs1"][0], "fa0")
        self.assertEqual(regs["frs1"][1], "float")
        self.assertEqual(regs["frs2"][0], "fa4")
        self.assertEqual(regs["frs2"][1], "float")

    def test_fmadd_s_with_rounding_mode(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("dyn,ft2,fa4,ft0,ft2")
        self.assertEqual(regs["frd"][0], "ft2")
        self.assertEqual(regs["frs1"][0], "fa4")
        self.assertEqual(regs["frs2"][0], "ft0")
        self.assertEqual(regs["frs3"][0], "ft2")

    def test_flw_float_dst_int_base(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("fa4,0(a6)")
        self.assertEqual(regs["frd"][0], "fa4")
        self.assertEqual(regs["frd"][1], "float")
        self.assertEqual(regs["rs1"][0], "a6")
        self.assertEqual(regs["rs1"][1], "integer")

    def test_fsw_float_src_int_base(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("ft2,0(a2)")
        self.assertEqual(regs["frs2"][0], "ft2")
        self.assertEqual(regs["frs2"][1], "float")
        self.assertEqual(regs["rs1"][0], "a2")
        self.assertEqual(regs["rs1"][1], "integer")

    def test_integer_registers_still_work(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("a0,a1,a2")
        self.assertEqual(regs["rd"][0], "a0")
        self.assertEqual(regs["rd"][1], "integer")
        self.assertEqual(regs["rs1"][0], "a1")
        self.assertEqual(regs["rs2"][0], "a2")

    def test_memory_format_int_still_works(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("ra,24(sp)")
        self.assertEqual(regs["rs2"][0], "ra")
        self.assertEqual(regs["rs1"][0], "sp")

    def test_rounding_mode_rne_skipped(self):
        from dfg.instruction import _extract_registers
        regs = _extract_registers("rne,fa5,fa0,fa4")
        self.assertEqual(regs["frd"][0], "fa5")
        self.assertEqual(regs["frs1"][0], "fa0")
        self.assertEqual(regs["frs2"][0], "fa4")


if __name__ == "__main__":
    unittest.main()
