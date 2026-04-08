"""Verification script for RVV mnemonic mapping.

Loads riscv_instrs.json, filters HasStdExtV + non-Pseudo + Instruction entries,
and verifies that all mnemonics are correctly derived.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_overrides_v import RVV_MNEMONIC_OVERRIDES


def default_rule(name: str) -> str:
    """Default mnemonic conversion: lowercase + underscore -> dot."""
    return name.lower().replace("_", ".")


def resolve_mnemonic(name: str) -> str:
    """Resolve LLVM def name to QEMU mnemonic."""
    if name in RVV_MNEMONIC_OVERRIDES:
        return RVV_MNEMONIC_OVERRIDES[name]
    return default_rule(name)


def load_v_instructions(json_path: Path) -> list[tuple[str, str]]:
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
        # Must have HasStdExtV predicate
        preds = entry.get("Predicates", [])
        if not isinstance(preds, list):
            continue
        pred_names = {p if isinstance(p, str) else p.get("def", "") for p in preds}
        if "HasStdExtV" not in pred_names:
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


QEMU_MNEMONIC_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z0-9]+)*$")


def verify(json_path: Path) -> bool:
    """Run all verification checks. Returns True if all pass."""
    instructions = load_v_instructions(json_path)
    print(f"V-extension instructions found: {len(instructions)}")

    # Check 1: Coverage — every instruction has a resolved mnemonic
    missing = []
    for name, asm_str in instructions:
        mnemonic = resolve_mnemonic(name)
        if not mnemonic:
            missing.append(name)

    if missing:
        print(f"\nFAIL: {len(missing)} instructions have no resolved mnemonic:")
        for name in missing[:20]:
            print(f"  {name}")
        return False
    print(f"Check 1 PASS: All {len(instructions)} instructions have resolved mnemonics")

    # Check 2: Correctness — resolved mnemonic matches AsmString first token
    mismatches = []
    for name, asm_str in instructions:
        if not asm_str:
            continue
        expected = asm_str.split()[0]
        actual = resolve_mnemonic(name)
        if expected != actual:
            mismatches.append((name, expected, actual))

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} mnemonic mismatches:")
        for name, expected, actual in mismatches[:20]:
            print(f"  {name}: resolved={actual}  expected={expected}")
        return False
    print(f"Check 2 PASS: All mnemonics match AsmString")

    # Check 3: QEMU format — lowercase, dot-separated suffixes
    bad_format = []
    for name, _ in instructions:
        mnemonic = resolve_mnemonic(name)
        if not QEMU_MNEMONIC_RE.match(mnemonic):
            bad_format.append((name, mnemonic))

    if bad_format:
        print(f"\nFAIL: {len(bad_format)} mnemonics don't match QEMU format:")
        for name, mnemonic in bad_format[:20]:
            print(f"  {name}: {mnemonic}")
        return False
    print(f"Check 3 PASS: All mnemonics match QEMU format")

    # Check 4: No duplicate mnemonics
    seen: dict[str, list[str]] = {}
    for name, _ in instructions:
        mnemonic = resolve_mnemonic(name)
        seen.setdefault(mnemonic, []).append(name)

    dupes = {m: names for m, names in seen.items() if len(names) > 1}
    if dupes:
        print(f"\nFAIL: {len(dupes)} duplicate mnemonics:")
        for mnemonic, names in list(dupes.items())[:20]:
            print(f"  {mnemonic}: {', '.join(names)}")
        return False
    print(f"Check 4 PASS: No duplicate mnemonics")

    # Summary
    print(f"\nAll checks passed for {len(instructions)} V-extension instructions.")
    return True


def main() -> int:
    json_path = Path(__file__).resolve().parents[1] / "riscv_instrs.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        return 1

    ok = verify(json_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
