"""Tests for RVV mnemonic mapping.

Loads riscv_instrs.json, filters HasVInstructions + non-Pseudo + Instruction entries,
and verifies that all mnemonics are correctly derived.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import ClassVar
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_overrides_v import RVV_MNEMONIC_OVERRIDES  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
QEMU_MNEMONIC_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z0-9]+)*$")


def _default_rule(name: str) -> str:
    """Default mnemonic conversion: lowercase + underscore -> dot."""
    return name.lower().replace("_", ".")


def _resolve_mnemonic(name: str) -> str:
    """Resolve LLVM def name to QEMU mnemonic."""
    if name in RVV_MNEMONIC_OVERRIDES:
        return RVV_MNEMONIC_OVERRIDES[name]
    return _default_rule(name)


def _load_v_instructions(json_path: Path) -> list[tuple[str, str]]:
    """Load V-extension instructions and return (def_name, asm_string) pairs."""
    with open(json_path) as f:
        data = json.load(f)

    results: list[tuple[str, str]] = []
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        # Must be Instruction subclass
        superclasses = entry.get("!superclasses", [])
        if not isinstance(superclasses, list) or "Instruction" not in superclasses:
            continue
        # Must have HasVInstructions predicate
        preds = entry.get("Predicates", [])
        if not isinstance(preds, list):
            continue
        pred_names = {p if isinstance(p, str) else p.get("def", "") for p in preds}
        if "HasVInstructions" not in pred_names:
            continue
        # Must not be Pseudo
        is_pseudo = name.startswith("Pseudo")
        for p in preds:
            if isinstance(p, dict) and p.get("def", "") == "isPseudo":
                is_pseudo = True
            if isinstance(p, str) and p == "isPseudo":
                is_pseudo = True
        if is_pseudo:
            continue

        asm_str = entry.get("AsmString", "")
        results.append((name, asm_str))

    return results


class TestRvvMnemonics(TestCase):
    """Verify RVV mnemonic mapping against riscv_instrs.json."""

    json_path: ClassVar[Path] = Path(__file__).resolve().parents[1] / "riscv_instrs.json"
    instructions: ClassVar[list[tuple[str, str]]] = []

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.json_path.exists():
            raise FileNotFoundError(f"{cls.json_path} not found")
        cls.instructions = _load_v_instructions(cls.json_path)

    def test_all_instructions_have_resolved_mnemonic(self) -> None:
        """Every V-extension instruction should have a resolved mnemonic."""
        missing = [
            name for name, _ in self.instructions
            if not _resolve_mnemonic(name)
        ]
        if missing:
            self.fail(
                f"{len(missing)} instructions have no resolved mnemonic: "
                + ", ".join(missing[:20])
            )

    def test_mnemonics_match_asm_string(self) -> None:
        """Resolved mnemonic should match AsmString first token."""
        mismatches = []
        for name, asm_str in self.instructions:
            if not asm_str:
                continue
            expected = asm_str.split()[0]
            actual = _resolve_mnemonic(name)
            if expected != actual:
                mismatches.append(f"{name}: resolved={actual} expected={expected}")
        if mismatches:
            self.fail(
                f"{len(mismatches)} mnemonic mismatches:\n  "
                + "\n  ".join(mismatches[:20])
            )

    def test_mnemonics_match_qemu_format(self) -> None:
        """Mnemonics should be lowercase, dot-separated suffixes."""
        bad_format = [
            (name, _resolve_mnemonic(name))
            for name, _ in self.instructions
            if not QEMU_MNEMONIC_RE.match(_resolve_mnemonic(name))
        ]
        if bad_format:
            self.fail(
                f"{len(bad_format)} mnemonics don't match QEMU format:\n  "
                + "\n  ".join(f"{n}: {m}" for n, m in bad_format[:20])
            )

    def test_no_duplicate_mnemonics(self) -> None:
        """Each mnemonic should map to exactly one instruction."""
        seen: dict[str, list[str]] = {}
        for name, _ in self.instructions:
            mnemonic = _resolve_mnemonic(name)
            seen.setdefault(mnemonic, []).append(name)

        dupes = {m: names for m, names in seen.items() if len(names) > 1}
        if dupes:
            self.fail(
                f"{len(dupes)} duplicate mnemonics:\n  "
                + "\n  ".join(
                    f"{m}: {', '.join(n)}" for m, n in list(dupes.items())[:20]
                )
            )
