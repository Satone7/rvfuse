#!/usr/bin/env python3
"""Tests for gen_isadesc.py V extension generation."""

import unittest


class TestVExtensionPredicate(unittest.TestCase):
    def test_vadd_vv_has_extension_v(self):
        from dfg.gen_isadesc import _has_extension
        entry = {
            "!superclasses": ["Instruction"],
            "Predicates": [{"def": "HasVInstructions"}],
        }
        self.assertTrue(_has_extension(entry, "V"))

    def test_add_does_not_have_extension_v(self):
        from dfg.gen_isadesc import _has_extension
        entry = {
            "!superclasses": ["Instruction"],
            "Predicates": [{"def": "HasStdExtM"}],
        }
        self.assertFalse(_has_extension(entry, "V"))

    def test_v_in_extension_predicates(self):
        from dfg.gen_isadesc import EXTENSION_PREDICATES
        self.assertIn("V", EXTENSION_PREDICATES)
        self.assertEqual(EXTENSION_PREDICATES["V"], {"HasVInstructions"})


class TestVShouldInclude(unittest.TestCase):
    def test_vadd_vv_included(self):
        from dfg.gen_isadesc import _should_include
        self.assertTrue(_should_include("VADD_VV"))

    def test_pseudo_excluded(self):
        from dfg.gen_isadesc import _should_include
        self.assertFalse(_should_include("PseudoVADD_VV"))

    def test_vsetvli_included(self):
        from dfg.gen_isadesc import _should_include
        self.assertTrue(_should_include("VSETVLI"))

    def test_vfmv_f_s_still_included(self):
        """VFMV_F_S was previously hard-coded as an exception."""
        from dfg.gen_isadesc import _should_include
        self.assertTrue(_should_include("VFMV_F_S"))


class TestVectorRegisterClasses(unittest.TestCase):
    def test_vr_class_has_prefix_v(self):
        from dfg.gen_isadesc import REG_CLASS_TO_PREFIX
        self.assertEqual(REG_CLASS_TO_PREFIX["VR"], "v")

    def test_vrm_classes_have_prefix_v(self):
        from dfg.gen_isadesc import REG_CLASS_TO_PREFIX
        for cls in ("VRM1", "VRM2", "VRM4", "VRM8", "VRN"):
            self.assertEqual(REG_CLASS_TO_PREFIX[cls], "v")


class TestVectorOperandPositions(unittest.TestCase):
    def test_vs3_maps_to_rs3(self):
        from dfg.gen_isadesc import OPERAND_TO_POSITION
        self.assertEqual(OPERAND_TO_POSITION["vs3"], "rs3")

    def test_vl_vtype_map_to_rd(self):
        from dfg.gen_isadesc import OPERAND_TO_POSITION
        self.assertEqual(OPERAND_TO_POSITION["vl"], "rd")
        self.assertEqual(OPERAND_TO_POSITION["vtype"], "rd")

    def test_vstart_vxrm_vxsat_map_to_rd(self):
        from dfg.gen_isadesc import OPERAND_TO_POSITION
        self.assertEqual(OPERAND_TO_POSITION["vstart"], "rd")
        self.assertEqual(OPERAND_TO_POSITION["vxrm"], "rd")
        self.assertEqual(OPERAND_TO_POSITION["vxsat"], "rd")


class TestDefaultRuleVInstructions(unittest.TestCase):
    """Verify default conversion produces correct QEMU mnemonics for V."""

    def test_vadd_vv_default_rule(self):
        from dfg.gen_isadesc import llvm_name_to_mnemonic
        self.assertEqual(llvm_name_to_mnemonic("VADD_VV"), "vadd.vv")

    def test_vle8_v_default_rule(self):
        from dfg.gen_isadesc import llvm_name_to_mnemonic
        self.assertEqual(llvm_name_to_mnemonic("VLE8_V"), "vle8.v")

    def test_vsetvli_default_rule(self):
        from dfg.gen_isadesc import llvm_name_to_mnemonic
        self.assertEqual(llvm_name_to_mnemonic("VSETVLI"), "vsetvli")

    def test_vredsum_vs_default_rule(self):
        from dfg.gen_isadesc import llvm_name_to_mnemonic
        self.assertEqual(llvm_name_to_mnemonic("VREDSUM_VS"), "vredsum.vs")


class TestIExtensionFiltering(unittest.TestCase):
    """Test _has_extension(entry, 'I') filtering logic."""

    def test_base_instruction_no_predicates_included(self):
        """A base instruction with no predicates should be included in I."""
        from dfg.gen_isadesc import _has_extension
        entry = {
            "!superclasses": ["Instruction"],
            "Predicates": [],
        }
        self.assertTrue(_has_extension(entry, "I"))

    def test_isrv64_only_included(self):
        """An instruction with only IsRV64 should be included in I."""
        from dfg.gen_isadesc import _has_extension
        entry = {
            "!superclasses": ["Instruction"],
            "Predicates": [{"def": "IsRV64"}],
        }
        self.assertTrue(_has_extension(entry, "I"))

    def test_hasstdextf_excluded(self):
        """An instruction with HasStdExtF should be excluded from I."""
        from dfg.gen_isadesc import _has_extension
        entry = {
            "!superclasses": ["Instruction"],
            "Predicates": [{"def": "HasStdExtF"}],
        }
        self.assertFalse(_has_extension(entry, "I"))

    def test_hasvinstructions_excluded(self):
        """An instruction with HasVInstructions should be excluded from I."""
        from dfg.gen_isadesc import _has_extension
        entry = {
            "!superclasses": ["Instruction"],
            "Predicates": [{"def": "HasVInstructions"}],
        }
        self.assertFalse(_has_extension(entry, "I"))


class TestMZmmulExtension(unittest.TestCase):
    """Test M_ZMMUL extension predicate."""

    def test_m_zmmul_in_extension_predicates(self):
        from dfg.gen_isadesc import EXTENSION_PREDICATES
        self.assertIn("M_ZMMUL", EXTENSION_PREDICATES)
        self.assertEqual(
            EXTENSION_PREDICATES["M_ZMMUL"], {"HasStdExtZmmul"}
        )


class TestAnonymousPrefixFiltering(unittest.TestCase):
    """Test anonymous prefix in SKIP_PREFIXES."""

    def test_anonymous_in_skip_prefixes(self):
        from dfg.gen_isadesc import SKIP_PREFIXES
        self.assertIn("anonymous", SKIP_PREFIXES)

    def test_anonymous_name_excluded(self):
        from dfg.gen_isadesc import _should_include
        self.assertFalse(_should_include("anonymous_12345"))

    def test_normal_name_included(self):
        from dfg.gen_isadesc import _should_include
        self.assertTrue(_should_include("ADD"))


class TestIExtensionInPredicates(unittest.TestCase):
    """Test that I extension is present in EXTENSION_PREDICATES."""

    def test_i_in_extension_predicates(self):
        from dfg.gen_isadesc import EXTENSION_PREDICATES
        self.assertIn("I", EXTENSION_PREDICATES)
        self.assertEqual(EXTENSION_PREDICATES["I"], set())


class TestRv64vModule(unittest.TestCase):
    def test_build_registry_loads(self):
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64v import build_registry
        reg = ISARegistry()
        build_registry(reg)
        self.assertGreater(len(reg._flows), 100)

    def test_key_instructions_registered(self):
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64v import build_registry
        reg = ISARegistry()
        build_registry(reg)
        self.assertTrue(reg.is_known("vsetvli"))
        self.assertTrue(any("vadd" in m for m in reg._flows))
        self.assertTrue(any("vle" in m for m in reg._flows))
        self.assertTrue(any("vse" in m for m in reg._flows))
        self.assertTrue(any("vred" in m for m in reg._flows))

    def test_vfmv_f_s_in_both_v_and_f(self):
        """VFMV_F_S and VFMV_S_F have 'no pred' in LLVM 22, so they are not
        matched by either HasVInstructions or HasStdExtF and no longer appear
        in either extension."""
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64v import build_registry
        from dfg.isadesc.rv64f import build_registry as build_f
        reg_v = ISARegistry()
        build_registry(reg_v)
        reg_f = ISARegistry()
        build_f(reg_f)
        # Neither extension contains these in LLVM 22
        self.assertFalse(reg_v.is_known("vfmv.f.s"))
        self.assertFalse(reg_v.is_known("vfmv.s.f"))
        self.assertFalse(reg_f.is_known("vfmv.f.s"))
        self.assertFalse(reg_f.is_known("vfmv.s.f"))


if __name__ == "__main__":
    unittest.main()
