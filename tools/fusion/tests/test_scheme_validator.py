"""Tests for the scheme validator -- encoding conflict detection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow
from fusion.scheme_validator import ValidationResult, validate_encoding


def _make_registry() -> ISARegistry:
    """Build a registry with instructions that have encoding metadata."""
    registry = ISARegistry()
    instructions = [
        ("add", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x00, reg_class="integer"))),
        ("sub", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x20, reg_class="integer"))),
        ("mul", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x01, reg_class="integer"))),
        ("fadd.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x00, reg_class="float"))),
        ("fsub.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x08, reg_class="float"))),
        ("vadd.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x00, reg_class="vector"))),
        ("vmul.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x2, 0x02, reg_class="vector"))),
    ]
    for mnemonic, flow in instructions:
        registry.register(mnemonic, flow)
    return registry


class TestValidationResult(unittest.TestCase):

    def test_passed_true_when_no_conflicts(self):
        """ValidationResult with passed=True should have empty defaults."""
        result = ValidationResult(passed=True)
        self.assertTrue(result.passed)
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.suggested_alternatives, [])

    def test_passed_false_with_conflicts(self):
        """ValidationResult with passed=False should carry conflict details."""
        result = ValidationResult(passed=False, conflicts=["opcode conflict"])
        self.assertFalse(result.passed)
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(result.conflicts[0], "opcode conflict")


class TestValidateEncoding(unittest.TestCase):

    def setUp(self):
        self.registry = _make_registry()

    def test_conflict_with_existing_instruction(self):
        """opcode=0x33 funct3=0x0 funct7=0x00 conflicts with ADD."""
        result = validate_encoding(
            opcode=0x33, funct3=0x0, funct7=0x00,
            reg_class="integer", registry=self.registry,
        )
        self.assertFalse(result.passed, "Expected conflict with ADD instruction")
        self.assertTrue(
            any("ADD" in c or "0x33" in c for c in result.conflicts),
            f"Expected ADD or 0x33 in conflicts, got: {result.conflicts}",
        )

    def test_no_conflict_unused_encoding(self):
        """opcode=0x0B (custom-0) with unused funct3 should pass."""
        result = validate_encoding(
            opcode=0x0B, funct3=0x2, funct7=0x00,
            reg_class="integer", registry=self.registry,
        )
        self.assertTrue(result.passed, f"Expected pass for unused custom-0 encoding, got conflicts: {result.conflicts}")

    def test_register_class_mismatch_integer_fp_opcode(self):
        """opcode=0x53 (OP-FP) with reg_class='integer' should fail with class mismatch."""
        result = validate_encoding(
            opcode=0x53, funct3=0x0, funct7=0x00,
            reg_class="integer", registry=self.registry,
        )
        self.assertFalse(result.passed, "Expected register class mismatch for OP-FP with integer class")
        self.assertTrue(
            any("register class" in c for c in result.conflicts),
            f"Expected 'register class' in conflicts, got: {result.conflicts}",
        )

    def test_vector_encoding_no_class_mismatch(self):
        """opcode=0x57 (OP-V) with reg_class='vector' should not have class mismatch."""
        result = validate_encoding(
            opcode=0x57, funct3=0x4, funct7=0x40,
            reg_class="vector", registry=self.registry,
        )
        self.assertFalse(
            any("register class" in c for c in result.conflicts),
            f"Did not expect register class mismatch, got: {result.conflicts}",
        )

    def test_warning_partial_funct3_usage(self):
        """opcode=0x33 funct3=0x0 funct7=0x20 conflicts with SUB."""
        result = validate_encoding(
            opcode=0x33, funct3=0x0, funct7=0x20,
            reg_class="integer", registry=self.registry,
        )
        self.assertFalse(result.passed, "Expected conflict with SUB instruction")
        self.assertTrue(
            any("sub" in c.lower() for c in result.conflicts),
            f"Expected SUB in conflicts, got: {result.conflicts}",
        )

    def test_suggested_alternatives_on_failure(self):
        """When validation fails, suggested_alternatives should be provided."""
        result = validate_encoding(
            opcode=0x33, funct3=0x0, funct7=0x00,
            reg_class="integer", registry=self.registry,
        )
        self.assertFalse(result.passed)
        self.assertGreater(
            len(result.suggested_alternatives), 0,
            "Expected at least one suggested alternative on failure",
        )

    def test_no_conflict_with_unique_funct7(self):
        """Same opcode/funct3 but unique funct7 should pass."""
        result = validate_encoding(
            opcode=0x33, funct3=0x0, funct7=0x42,
            reg_class="integer", registry=self.registry,
        )
        self.assertTrue(result.passed, f"Expected pass with unique funct7, got conflicts: {result.conflicts}")
