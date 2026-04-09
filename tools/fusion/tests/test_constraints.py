"""Unit tests for the hardware constraint model."""

import sys
import tempfile
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow
from fusion.constraints import ConstraintChecker, Verdict, ConstraintConfig


def _make_registry() -> ISARegistry:
    """Build a registry with a few instructions that have encoding metadata."""
    registry = ISARegistry()
    instructions = [
        ("fadd.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x00, reg_class="float"))),
        ("fsub.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x08, reg_class="float"))),
        ("fmul.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x1, 0x00, reg_class="float"))),
        ("fdiv.s", RegisterFlow(["frd"], ["frs1", "frs2"],
            encoding=InstructionFormat("R", 0x53, 0x0, 0x0c, reg_class="float"))),
        ("fmadd.s", RegisterFlow(["frd"], ["frs1", "frs2", "frs3"],
            encoding=InstructionFormat("R4", 0x43, reg_class="float", has_rs3=True))),
        ("flw", RegisterFlow(["frd"], ["rs1"],
            encoding=InstructionFormat("I", 0x07, 0x2, may_load=True, has_rs2=False, reg_class="float"))),
        ("fsw", RegisterFlow([], ["frs2", "rs1"],
            encoding=InstructionFormat("S", 0x27, 0x2, may_store=True, has_rd=False, reg_class="float"))),
        ("add", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x00, reg_class="integer"))),
        ("mul", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x33, 0x0, 0x01, reg_class="integer"))),
        ("vadd.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x00, reg_class="vector"))),
        ("vadd.vx", RegisterFlow(["vrd"], ["vrs2", "rs1"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x04, reg_class="vector"))),
        ("vmul.vv", RegisterFlow(["vrd"], ["vrs2", "vrs1"],
            encoding=InstructionFormat("V", 0x57, 0x2, 0x02, reg_class="vector"))),
        ("vadd.vi", RegisterFlow(["vrd"], ["vrs2"],
            encoding=InstructionFormat("V", 0x57, 0x0, 0x03, has_rs1=False, reg_class="vector"))),
        ("vsetvli", RegisterFlow(["rd"], ["rs1"],
            encoding=InstructionFormat("V", 0x57, 0x7, 0x00, has_rs2=False,
                                      has_imm=True, imm_bits=12, reg_class="vector"))),
        ("addi", RegisterFlow(["rd"], ["rs1"],
            encoding=InstructionFormat("I", 0x13, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
    ]
    for mnemonic, flow in instructions:
        registry.register(mnemonic, flow)
    return registry


class TestVerdict(unittest.TestCase):
    def test_feasible_verdict(self):
        v = Verdict(status="feasible", reasons=[], violations=[])
        self.assertEqual(v.status, "feasible")
        self.assertFalse(v.violations)

    def test_infeasible_verdict(self):
        v = Verdict(status="infeasible", reasons=["load"], violations=["no_load_store"])
        self.assertEqual(v.status, "infeasible")
        self.assertIn("no_load_store", v.violations)


class TestConstraintCheckerFeasible(unittest.TestCase):
    def setUp(self):
        config = ConstraintConfig.defaults()
        # Enable all constraints for comprehensive feasible testing
        # Note: operand_format requires exact counts (3src+1dst or 2src+1dst+imm),
        # which these patterns don't satisfy, so disable it for these tests.
        for name in config.enabled:
            config.enabled[name] = True
        config.enabled["operand_format"] = False
        self.checker = ConstraintChecker(_make_registry(), config=config)

    def test_float_add_mul_chain(self):
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_int_add_mul_chain(self):
        pattern = {"opcodes": ["add", "mul"], "register_class": "integer", "chain_registers": [["rd", "rs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_vext_vadd_vmul_chain(self):
        pattern = {"opcodes": ["vadd.vv", "vmul.vv"], "register_class": "vector", "chain_registers": [["vrd", "vrs2"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_float_3_instruction_chain(self):
        pattern = {"opcodes": ["fadd.s", "fmul.s", "fsub.s"], "register_class": "float", "chain_registers": [["frd", "frs1"], ["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")


class TestConstraintCheckerInfeasible(unittest.TestCase):
    def setUp(self):
        config = ConstraintConfig.defaults()
        # Disable all constraints except old infeasible ones for isolated testing
        for name in config.enabled:
            config.enabled[name] = False
        config.enabled["no_load_store"] = True
        config.enabled["register_class_mismatch"] = True
        config.enabled["no_config_write"] = True
        config.enabled["unknown_instruction"] = True
        self.checker = ConstraintChecker(_make_registry(), config=config)

    def test_load_in_chain(self):
        pattern = {"opcodes": ["flw", "fmul.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("no_load_store", verdict.violations)

    def test_store_in_chain(self):
        pattern = {"opcodes": ["fmul.s", "fsw"], "register_class": "float", "chain_registers": [["frd", "frs2"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("no_load_store", verdict.violations)

    def test_cross_register_class(self):
        pattern = {"opcodes": ["add", "fadd.s"], "register_class": "integer", "chain_registers": [["rd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("register_class_mismatch", verdict.violations)

    def test_config_register_write(self):
        pattern = {"opcodes": ["vsetvli", "vadd.vv"], "register_class": "vector", "chain_registers": [["rd", "rs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("no_config_write", verdict.violations)

    def test_unknown_instruction(self):
        pattern = {"opcodes": ["fadd.s", "unknown_op"], "register_class": "float", "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("unknown_instruction", verdict.violations)


class TestConstraintCheckerConstrained(unittest.TestCase):
    def setUp(self):
        config = ConstraintConfig.defaults()
        # Disable all constraints except old soft ones for isolated testing
        for name in config.enabled:
            config.enabled[name] = False
        config.enabled["has_immediate"] = True
        config.enabled["missing_encoding"] = True
        self.checker = ConstraintChecker(_make_registry(), config=config)

    def test_immediate_in_chain(self):
        pattern = {"opcodes": ["addi", "add"], "register_class": "integer", "chain_registers": [["rd", "rs1"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "constrained")
        self.assertIn("has_immediate", verdict.violations)

    def test_same_opcode_vext_variants(self):
        pattern = {"opcodes": ["vadd.vi", "vadd.vx"], "register_class": "vector", "chain_registers": [["vrd", "vrs2"]]}
        verdict = self.checker.check(pattern)
        self.assertEqual(verdict.status, "feasible", f"Expected feasible, got: {verdict.reasons}")

    def test_missing_encoding_metadata(self):
        registry = ISARegistry()
        registry.register("fadd.s", RegisterFlow(["frd"], ["frs1", "frs2"]))
        registry.register("fmul.s", RegisterFlow(["frd"], ["frs1", "frs2"]))
        config = ConstraintConfig.defaults()
        for name in config.enabled:
            config.enabled[name] = False
        config.enabled["missing_encoding"] = True
        checker = ConstraintChecker(registry, config=config)
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float", "chain_registers": [["frd", "frs1"]]}
        verdict = checker.check(pattern)
        self.assertEqual(verdict.status, "constrained")
        self.assertIn("missing_encoding", verdict.violations)


class TestConstraintConfigDefaults(unittest.TestCase):
    def test_defaults_has_all_constraints(self):
        config = ConstraintConfig.defaults()
        self.assertEqual(len(config.enabled), 11)

    def test_defaults_new_constraints_enabled(self):
        config = ConstraintConfig.defaults()
        for name in ["encoding_32bit", "operand_format", "datatype_encoding_space"]:
            self.assertTrue(config.enabled[name], f"{name} should default enabled")

    def test_defaults_old_constraints_disabled(self):
        config = ConstraintConfig.defaults()
        for name in ["no_load_store", "register_class_mismatch", "no_config_write",
                     "unknown_instruction", "too_many_destinations", "too_many_sources",
                     "has_immediate", "missing_encoding"]:
            self.assertFalse(config.enabled[name], f"{name} should default disabled")

    def test_all_constraints_metadata_complete(self):
        for name, (category, default, desc) in ConstraintConfig.ALL_CONSTRAINTS.items():
            self.assertIn(category, ("hard", "soft"))
            self.assertIsInstance(default, bool)
            self.assertIsInstance(desc, str)
            self.assertTrue(len(desc) > 0)


class TestConstraintConfigFromFile(unittest.TestCase):
    def test_from_file_missing_returns_defaults(self):
        config = ConstraintConfig.from_file(Path("/nonexistent/path.json"))
        self.assertEqual(config.enabled, ConstraintConfig.defaults().enabled)

    def test_from_file_partial_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"constraints": {"no_load_store": True}}, f)
            f.flush()
            config = ConstraintConfig.from_file(Path(f.name))
        self.assertTrue(config.enabled["no_load_store"])
        self.assertFalse(config.enabled["register_class_mismatch"])
        self.assertTrue(config.enabled["encoding_32bit"])

    def test_from_file_invalid_json_returns_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            f.flush()
            config = ConstraintConfig.from_file(Path(f.name))
        self.assertEqual(config.enabled, ConstraintConfig.defaults().enabled)


class TestNewHardwareConstraints(unittest.TestCase):
    def setUp(self):
        self.registry = _make_registry()
        config = ConstraintConfig.defaults()
        # Disable all constraints except new ones for isolated testing
        for name in ConstraintConfig.ALL_CONSTRAINTS:
            config.enabled[name] = False
        config.enabled["encoding_32bit"] = True
        config.enabled["operand_format"] = True
        config.enabled["datatype_encoding_space"] = True
        self.checker = ConstraintChecker(self.registry, config=config)

    def test_encoding_32bit_passes_for_standard_instructions(self):
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertNotIn("encoding_32bit", verdict.violations)

    def test_encoding_32bit_detects_compressed(self):
        # Register a fake compressed instruction (opcode low bits = 0x00)
        registry = ISARegistry()
        registry.register("c.add", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x00, 0x0, 0x00, reg_class="integer")))
        config = ConstraintConfig.defaults()
        for name in config.enabled:
            config.enabled[name] = False
        config.enabled["encoding_32bit"] = True
        checker = ConstraintChecker(registry, config=config)
        pattern = {"opcodes": ["c.add"], "register_class": "integer",
                   "chain_registers": []}
        verdict = checker.check(pattern)
        self.assertIn("encoding_32bit", verdict.violations)

    def test_operand_format_passes_3src_1dst_no_imm(self):
        # fmadd.s has 3 sources + 1 destination, no immediate
        pattern = {"opcodes": ["fmadd.s"], "register_class": "float",
                   "chain_registers": []}
        verdict = self.checker.check(pattern)
        self.assertNotIn("operand_format", verdict.violations)

    def test_operand_format_mode_b_requires_exact_2src_1dst_imm(self):
        # Mode B requires EXACTLY 2 external sources + 1 external destination + immediate.
        # vsetvli has: rd (dst), rs1 (src), and imm (12-bit).
        # Without chaining: 1 external source (rs1), 1 external destination (rd), has imm.
        # This gives 1 src + 1 dst + imm - Mode B requires exactly 2 sources -> should fail.
        pattern = {"opcodes": ["vsetvli"], "register_class": "vector",
                   "chain_registers": []}
        verdict = self.checker.check(pattern)
        self.assertIn("operand_format", verdict.violations)

    def test_operand_format_passes_2src_imm_1dst(self):
        """Mode B: 2 external sources + immediate + 1 destination should pass."""
        # Create a custom instruction with exactly 2 sources + immediate + 1 destination
        registry = ISARegistry()
        # custom.ld: reads from rs1 and rs2, writes to rd, has immediate
        # This matches Mode B: 2 external sources + 1 dst + immediate
        registry.register("custom.ld", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("I", 0x13, 0x0, has_imm=True, reg_class="integer")))

        config = ConstraintConfig.defaults()
        for name in config.enabled:
            config.enabled[name] = False
        config.enabled["operand_format"] = True
        checker = ConstraintChecker(registry, config=config)

        pattern = {"opcodes": ["custom.ld"], "register_class": "integer",
                   "chain_registers": []}
        verdict = checker.check(pattern)
        # custom.ld: 2 src (rs1, rs2), 1 dst (rd), has imm -> Mode B satisfied
        self.assertNotIn("operand_format", verdict.violations)

    def test_operand_format_detects_mismatch(self):
        # fadd.s + fmul.s: each has 2 sources + 1 destination
        # Chain passes frd -> frs1, so external: frs2 from both = 2 sources
        # Mode A needs 3 sources, Mode B needs imm -> neither satisfied
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertIn("operand_format", verdict.violations)

    def test_datatype_encoding_space_single_type_ok(self):
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertNotIn("datatype_encoding_space", verdict.violations)
