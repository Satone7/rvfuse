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
