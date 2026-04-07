#!/usr/bin/env python3
"""Generate Python ISA descriptor files from llvm-tblgen JSON output.

Reads the RISC-V instruction JSON produced by llvm-tblgen and emits a Python
module that defines RegisterFlow entries for every instruction in a given
ISA extension (F, M, ...).

Usage:
    python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json \
        --ext F -o tools/dfg/isadesc/rv64f.py
    python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json \
        --ext M -o tools/dfg/isadesc/rv64m.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from textwrap import dedent


# ---------------------------------------------------------------------------
# Configuration: which LLVM tablegen names map to which QEMU mnemonics
# ---------------------------------------------------------------------------

# LLVM def name -> QEMU disassembly mnemonic (lowercase).
# Most names follow the simple pattern (lowercase, _ -> .), but some need
# explicit overrides because the LLVM naming diverges from the ISA spec.
MNEMONIC_OVERRIDES: dict[str, str] = {
    # F-extension moves: LLVM uses _W_/_X_W suffixes, QEMU uses .s/.x
    "FMV_W_X": "fmv.s.x",
    "FMV_X_W": "fmv.x.s",
    # F-extension loads/stores with register offsets (Zicsr-adjacent)
    "FLRW": "flrw",
    "FLURW": "flurw",
    "FSURW": "fsurw",
    "FSRW_TH": "fsrw.th",
    # Compressed F instructions -> their expanded mnemonic names
    "C_FLW": "c.flw",
    "C_FLWSP": "c.flwsp",
    "C_FSW": "c.fsw",
    "C_FSWSP": "c.fswsp",
}


def llvm_name_to_mnemonic(llvm_name: str) -> str:
    """Convert an LLVM tablegen def name to a QEMU disassembly mnemonic."""
    if llvm_name in MNEMONIC_OVERRIDES:
        return MNEMONIC_OVERRIDES[llvm_name]
    # Default: lowercase, replace underscores with dots
    return llvm_name.lower().replace("_", ".")


# ---------------------------------------------------------------------------
# Register class handling
# ---------------------------------------------------------------------------

# Mapping from LLVM register-class def names to the position prefix used by
# RegisterFlow.  Only register classes listed here produce position entries;
# everything else (immediates, rounding-mode args, etc.) is skipped.
REG_CLASS_TO_PREFIX: dict[str, str] = {
    "GPR": "",
    "GPRNoX0": "",
    "GPRMem": "",
    "FPR32": "f",
    "FPR64": "f",
    "FPR16": "f",
    # Compressed register classes
    "GPRC": "",
    "FPR32C": "f",
    "SP": "",
}

# Operand name -> canonical position name.  Operands not listed here are
# skipped (e.g. funct3, imm12, uimm2, uimm5, uimm7_lsb00, uimm8_lsb00,
# imm, sew, vl, vm, etc.).
OPERAND_TO_POSITION: dict[str, str] = {
    "rd": "rd",
    "rs1": "rs1",
    "rs2": "rs2",
    "rs3": "rs3",
    "dst": "rd",
    "lhs": "rs1",
    "rhs": "rs2",
    "truev": "rs1",
    "falsev": "rs2",
    "vd": "rd",
    "vs1": "rs1",
    "vs2": "rs2",
}


# ---------------------------------------------------------------------------
# Extension predicate matching
# ---------------------------------------------------------------------------

# Maps the --ext flag value to the set of tablegen predicates an instruction
# must satisfy.  All listed predicates must be present for an instruction to
# be included.
EXTENSION_PREDICATES: dict[str, set[str]] = {
    "F": {"HasStdExtF"},
    "M": {"HasStdExtM"},
    # Future extensions can be added here:
    # "D": {"HasStdExtD"},
    # "C": {"HasStdExtC"},
    # "A": {"HasStdExtA"},
}


def _has_extension(entry: dict, ext: str) -> bool:
    """Return True if *entry* is an instruction belonging to extension *ext*."""
    required = EXTENSION_PREDICATES.get(ext)
    if required is None:
        return False

    # Must be an Instruction subclass (not a random tablegen class)
    superclasses = entry.get("!superclasses", [])
    if not isinstance(superclasses, list) or "Instruction" not in superclasses:
        return False

    preds = entry.get("Predicates", [])
    if not isinstance(preds, list):
        return False

    pred_names = {p.get("def", "") for p in preds if isinstance(p, dict)}
    return required.issubset(pred_names)


# ---------------------------------------------------------------------------
# Operand extraction
# ---------------------------------------------------------------------------

def _extract_flow(entry: dict) -> tuple[list[str], list[str]]:
    """Extract (dst_positions, src_positions) from an instruction entry.

    Walks OutOperandList and InOperandList, maps register-class operands
    through REG_CLASS_TO_PREFIX and OPERAND_TO_POSITION, and returns the
    resulting position-name lists.
    """
    dst: list[str] = []
    src: list[str] = []

    # Out operands -> destination positions
    out_args = entry.get("OutOperandList", {}).get("args", [])
    if isinstance(out_args, list):
        for arg in out_args:
            if not isinstance(arg, list) or len(arg) < 2:
                continue
            reg_info, op_name = arg[0], arg[1]
            if not isinstance(reg_info, dict):
                continue
            reg_class = reg_info.get("def", "")
            prefix = REG_CLASS_TO_PREFIX.get(reg_class)
            if prefix is None:
                continue
            pos_name = OPERAND_TO_POSITION.get(op_name)
            if pos_name is None:
                continue
            dst.append(prefix + pos_name)

    # In operands -> source positions
    in_args = entry.get("InOperandList", {}).get("args", [])
    if isinstance(in_args, list):
        for arg in in_args:
            if not isinstance(arg, list) or len(arg) < 2:
                continue
            reg_info, op_name = arg[0], arg[1]
            if not isinstance(reg_info, dict):
                continue
            reg_class = reg_info.get("def", "")
            prefix = REG_CLASS_TO_PREFIX.get(reg_class)
            if prefix is None:
                continue
            pos_name = OPERAND_TO_POSITION.get(op_name)
            if pos_name is None:
                continue
            src.append(prefix + pos_name)

    return dst, src


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

# Instructions whose LLVM def names start with these prefixes are not real
# RISC-V instructions and should be excluded.
SKIP_PREFIXES = (
    "Pseudo",
    "SDT_",
    "Select_",
)


def _should_include(name: str) -> bool:
    """Return True if the instruction with the given LLVM def name should be
    included in the generated descriptor."""
    if name.startswith(SKIP_PREFIXES):
        return False
    # Skip vector instructions (they have their own extension handling)
    if name.startswith("V") and not name.startswith(("VFMV_F_S", "VFMV_S_F")):
        # Exclude all V-prefixed instructions except the scalar move helpers
        # that occasionally leak into the F predicate set.
        return False
    return True


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _generate_module(ext: str, entries: list[tuple[str, list[str], list[str]]]) -> str:
    """Generate the Python source for an ISA descriptor module."""
    ext_lower = ext.lower()
    var_name = f"ALL_RV64{ext}"

    lines: list[str] = []
    lines.append(f'"""RV64{ext} extension instruction register flow definitions."""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dfg.instruction import ISARegistry, RegisterFlow")
    lines.append("")
    lines.append("")

    # Emit the instruction list
    lines.append(f"# RV64{ext} instructions (auto-generated by gen_isadesc.py)")
    lines.append(f"{var_name}: list[tuple[str, RegisterFlow]] = [")

    for mnemonic, dst, src in entries:
        dst_repr = repr(dst)
        src_repr = repr(src)
        lines.append(f'    ("{mnemonic}", RegisterFlow({dst_repr}, {src_repr})),')

    lines.append("]")
    lines.append("")
    lines.append("")

    # Emit build_registry
    lines.append("")
    lines.append("def build_registry(registry: ISARegistry) -> None:")
    lines.append(
        f'    """Register all RV64{ext} instructions into the given ISA registry."""'
    )
    lines.append(f'    registry.load_extension("{ext}", {var_name})')
    lines.append("")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Python ISA descriptor from llvm-tblgen JSON.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the llvm-tblgen JSON file.",
    )
    parser.add_argument(
        "--ext",
        required=True,
        choices=list(EXTENSION_PREDICATES.keys()),
        help="ISA extension to generate (e.g. F, M).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output Python file path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load JSON
    with open(args.json_path, "r") as f:
        data = json.load(f)

    # Filter and extract instructions for the given extension
    entries: list[tuple[str, list[str], list[str]]] = []
    for name, entry in sorted(data.items()):
        if not isinstance(entry, dict):
            continue
        if not _should_include(name):
            continue
        if not _has_extension(entry, args.ext):
            continue

        mnemonic = llvm_name_to_mnemonic(name)
        dst, src = _extract_flow(entry)

        # Skip instructions with no register flow at all (pure system)
        if not dst and not src:
            continue

        entries.append((mnemonic, dst, src))

    if not entries:
        print(
            f"Warning: no instructions found for extension {args.ext}",
            file=sys.stderr,
        )

    # Generate and write the module
    source = _generate_module(args.ext, entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(source)

    print(f"Generated {args.output} with {len(entries)} instructions")


if __name__ == "__main__":
    main()
