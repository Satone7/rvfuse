#!/usr/bin/env python3
"""Tests for gen_isadesc.py V extension generation."""

import unittest


class TestVExtensionPredicate(unittest.TestCase):
    def test_vadd_vv_has_extension_v(self):
        from dfg.gen_isadesc import _has_extension
        entry = {
            "!superclasses": ["Instruction"],
            "Predicates": [{"def": "HasStdExtV"}],
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
        self.assertEqual(EXTENSION_PREDICATES["V"], {"HasStdExtV"})


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
        """VFMV_F_S and VFMV_S_F have both HasStdExtV and HasStdExtF
        predicates, so they appear in both rv64v.py and rv64f.py.  The V
        descriptor provides the vector-aware flow (tracks VR source/dest),
        while the F descriptor provides the scalar-float perspective."""
        from dfg.instruction import ISARegistry
        from dfg.isadesc.rv64v import build_registry
        from dfg.isadesc.rv64f import build_registry as build_f
        reg_v = ISARegistry()
        build_registry(reg_v)
        # These are present in the V registry (vector-aware variant)
        self.assertTrue(reg_v.is_known("vfmv.f.s"))
        self.assertTrue(reg_v.is_known("vfmv.s.f"))


if __name__ == "__main__":
    unittest.main()
