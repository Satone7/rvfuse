"""Tests for the Pattern model -- normalization and template keys."""

import sys
from pathlib import Path

import unittest

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestPatternTemplateKey(unittest.TestCase):
    """Same opcode sequence + same role positions = same template key."""

    def test_identical_chains_same_key(self):
        """Two concrete instances of the same abstract pattern share a key."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        p2 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        self.assertEqual(p1.template_key, p2.template_key)

    def test_different_opcodes_different_key(self):
        """Different opcode sequences produce different keys."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        p2 = Pattern(
            opcodes=["fsub.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        self.assertNotEqual(p1.template_key, p2.template_key)

    def test_different_register_class_different_key(self):
        """Same opcodes but different register class produce different keys."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["add", "sub"],
            register_class="integer",
            chain_registers=[["rd", "rs1"]],
        )
        p2 = Pattern(
            opcodes=["add", "sub"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        self.assertNotEqual(p1.template_key, p2.template_key)

    def test_different_chain_roles_different_key(self):
        """Same opcodes but dependency flows through different role positions."""
        from fusion.pattern import Pattern

        p1 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        p2 = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs2"]],
        )
        self.assertNotEqual(p1.template_key, p2.template_key)

    def test_length_3_chain_key(self):
        """Length-3 patterns have two chain_register entries."""
        from fusion.pattern import Pattern

        p = Pattern(
            opcodes=["fadd.s", "fmul.s", "fsub.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"], ["frd", "frs1"]],
        )
        # Key should capture all three opcodes and both chain links
        self.assertIn(("fadd.s", "fmul.s", "fsub.s"), p.template_key)
        self.assertEqual(len(p.template_key[2]), 2)  # two chain links

    def test_key_is_hashable(self):
        """Template key must be usable as a dict key."""
        from fusion.pattern import Pattern

        p = Pattern(
            opcodes=["fadd.s", "fmul.s"],
            register_class="float",
            chain_registers=[["frd", "frs1"]],
        )
        d = {p.template_key: 42}
        self.assertEqual(d[p.template_key], 42)


class TestRegisterClassification(unittest.TestCase):

    def test_integer_registers(self):
        from fusion.pattern import classify_register
        for name in ("a0", "t1", "s2", "ra", "sp", "zero", "x5", "tp", "gp"):
            self.assertEqual(classify_register(name), "integer")

    def test_float_registers(self):
        from fusion.pattern import classify_register
        for name in ("ft0", "ft2", "fa0", "fa4", "fs0", "f5", "fv0"):
            self.assertEqual(classify_register(name), "float")

    def test_unknown_registers(self):
        from fusion.pattern import classify_register
        self.assertIsNone(classify_register("v0"))
        self.assertIsNone(classify_register("unknown_reg"))
        self.assertIsNone(classify_register(""))


class TestControlFlowDetection(unittest.TestCase):

    def test_branches(self):
        from fusion.pattern import is_control_flow
        for mn in ("beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"):
            self.assertTrue(is_control_flow(mn))

    def test_jumps_and_calls(self):
        from fusion.pattern import is_control_flow
        for mn in ("jal", "jalr", "call", "ret", "j", "jr"):
            self.assertTrue(is_control_flow(mn))

    def test_non_control_flow(self):
        from fusion.pattern import is_control_flow
        for mn in ("add", "addi", "fadd.s", "fmul.s", "ld", "sd", "lw", "fsw"):
            self.assertFalse(is_control_flow(mn))


class TestNormalizeChain(unittest.TestCase):
    """Test normalization of concrete chains to Pattern templates."""

    def _make_registry(self):
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64f import build_registry as build_f
        from dfg.isadesc.rv64i import build_registry as build_i
        reg = ISARegistry()
        build_i(reg)
        build_f(reg)
        return reg

    def test_float_pair_chain(self):
        """fadd.s ft2,fa0,fa1 -> fmul.s ft3,ft2,fa2 normalizes to chain_registers=[['frd','frs1']]."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("fadd.s", "ft2,fa0,fa1"),
            ("fmul.s", "ft3,ft2,fa2"),
        ]
        edges_between = [{"src": 0, "dst": 1, "register": "ft2"}]
        pattern = normalize_chain(chain, edges_between, registry)
        self.assertEqual(pattern.opcodes, ["fadd.s", "fmul.s"])
        self.assertEqual(pattern.register_class, "float")
        self.assertEqual(pattern.chain_registers, [["frd", "frs1"]])

    def test_integer_pair_chain(self):
        """addi a0,sp,16 -> add a1,a0,a2 normalizes to chain_registers=[['rd','rs1']]."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("addi", "a0,sp,16"),
            ("add", "a1,a0,a2"),
        ]
        edges_between = [{"src": 0, "dst": 1, "register": "a0"}]
        pattern = normalize_chain(chain, edges_between, registry)
        self.assertEqual(pattern.opcodes, ["addi", "add"])
        self.assertEqual(pattern.register_class, "integer")
        self.assertEqual(pattern.chain_registers, [["rd", "rs1"]])

    def test_length_3_chain(self):
        """Three-instruction chain with two RAW links."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("fadd.s", "ft2,fa0,fa1"),
            ("fmul.s", "ft2,ft2,fa2"),
            ("fsub.s", "ft3,ft2,fa3"),
        ]
        edges_between = [
            {"src": 0, "dst": 1, "register": "ft2"},
            {"src": 1, "dst": 2, "register": "ft2"},
        ]
        pattern = normalize_chain(chain, edges_between, registry)
        self.assertEqual(pattern.opcodes, ["fadd.s", "fmul.s", "fsub.s"])
        self.assertEqual(len(pattern.chain_registers), 2)

    def test_unknown_mnemonic_raises(self):
        """Unknown instruction mnemonic in the chain raises ValueError."""
        from fusion.pattern import normalize_chain
        registry = self._make_registry()
        chain = [
            ("fadd.s", "ft2,fa0,fa1"),
            ("unknown_op", "ft3,ft2,fa2"),
        ]
        with self.assertRaises(ValueError):
            normalize_chain(chain, [{"src": 0, "dst": 1, "register": "ft2"}], registry)
