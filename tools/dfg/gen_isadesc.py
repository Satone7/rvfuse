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

# Ensure dfg package imports work whether this file is invoked directly
# (e.g. ``python3 dfg/gen_isadesc.py`` from tools/) or via pytest.
if __name__ == "__main__" and __package__ is None:
    _script_dir = str(Path(__file__).resolve().parent.parent)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

from dfg.instruction import InstructionFormat  # noqa: E402


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
    # Vector register classes (V extension)
    "VR":       "v",
    "VRM1":     "v",
    "VRM2":     "v",
    "VRM4":     "v",
    "VRM8":     "v",
    "VRN":      "v",
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
    # Vector operand names (V extension)
    "vs3": "rs3",
    "vl": "rd",
    "vtype": "rd",
    "vstart": "rd",
    "vxrm": "rd",
    "vxsat": "rd",
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
    "V": {"HasStdExtV"},
    # Future extensions can be added here:
    # "D": {"HasStdExtD"},
    # "C": {"HasStdExtC"},
    # "A": {"HasStdExtA"},
}

# Instructions that also carry any of these predicates are excluded from the
# extension's descriptor, even if they carry the extension's own predicate.
# This prevents F from pulling in vector-FP instructions that are already
# covered by the V descriptor.
EXTENSION_EXCLUDED_PREDICATES: dict[str, set[str]] = {
    "F": {"HasStdExtV"},
}

# LLVM def names that override the exclusion -- these are included even if they
# carry a normally-excluded predicate.
EXTENSION_EXCLUSION_OVERRIDES: dict[str, set[str]] = {
    "F": {"VFMV_F_S", "VFMV_S_F"},
}

EXTENSION_REG_CLASS: dict[str, str] = {
    "F": "float",
    "M": "integer",
    "V": "vector",
}


def _has_extension(entry: dict, ext: str, name: str = "") -> bool:
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
    if not required.issubset(pred_names):
        return False

    # Exclude instructions that carry predicates from a different extension
    # (e.g. skip vector-FP instructions when generating F descriptor),
    # unless the instruction is in the override allowlist.
    excluded = EXTENSION_EXCLUDED_PREDICATES.get(ext)
    if excluded and excluded & pred_names:
        overrides = EXTENSION_EXCLUSION_OVERRIDES.get(ext, set())
        if name not in overrides:
            return False

    return True


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


def _extract_encoding(entry: dict, reg_class: str) -> InstructionFormat:
    """Extract encoding layout metadata from a tablegen instruction entry.

    Parses the Inst array (LSB-first) to determine which fields are present,
    the instruction format type, and fixed encoding bits (opcode, funct3,
    funct7).
    """
    inst = entry.get("Inst", [])
    if len(inst) < 32:
        return InstructionFormat(format_type="R", opcode=0, reg_class=reg_class)

    def bv(el):
        """Return int value if element is a fixed bit, else None for variable."""
        return el if isinstance(el, int) else None

    def extract_field(start: int, width: int) -> int | None:
        bits = [bv(inst[start + i]) for i in range(width)]
        if any(b is None for b in bits):
            return None
        return sum(b * (1 << i) for i, b in enumerate(bits))

    def has_var_field(start: int, width: int) -> bool:
        return any(bv(inst[start + i]) is None for i in range(width))

    # Opcode: the Opcode field is stored in reverse order
    opcode_bits = entry.get("Opcode", [0] * 7)
    opcode = int("".join(str(b) for b in reversed(opcode_bits)), 2)

    funct3 = extract_field(12, 3)
    funct7 = extract_field(25, 7)

    has_rd = has_var_field(7, 5)
    has_rs1 = has_var_field(15, 5)
    has_rs2 = has_var_field(20, 5)

    may_load = bool(entry.get("mayLoad", 0))
    may_store = bool(entry.get("mayStore", 0))

    # R4 format: only for 4-operand FP instructions (opcode 0x43 MADD,
    # 0x47 MSUB, 0x4b NMSUB, 0x4f NMADD).  Bits 27-31 are rs3 only in
    # this context; for I/S formats those same bit positions are part of
    # the immediate.
    _R4_OPCODES = {0x43, 0x47, 0x4b, 0x4f}
    has_rs3 = opcode in _R4_OPCODES and has_var_field(27, 5)

    has_imm = not has_rs2 and not has_rs3
    imm_bits = 12 if has_imm and not may_load and not may_store else 0
    if may_store:
        imm_bits = 12
        has_imm = True
    if may_load:
        imm_bits = 12
        has_imm = True

    if has_rs3:
        format_type = "R4"
    elif may_store:
        format_type = "S"
    elif may_load:
        format_type = "I"
    elif has_imm:
        format_type = "I"
    else:
        format_type = "R"

    return InstructionFormat(
        format_type=format_type, opcode=opcode, funct3=funct3, funct7=funct7,
        has_rd=has_rd, has_rs1=has_rs1, has_rs2=has_rs2, has_rs3=has_rs3,
        has_imm=has_imm, imm_bits=imm_bits, may_load=may_load, may_store=may_store,
        reg_class=reg_class,
    )


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
    return True


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _generate_module(
    ext: str,
    entries: list[tuple[str, list[str], list[str], InstructionFormat]],
) -> str:
    """Generate the Python source for an ISA descriptor module."""
    ext_lower = ext.lower()
    var_name = f"ALL_RV64{ext}"

    lines: list[str] = []
    lines.append(f'"""RV64{ext} extension instruction register flow definitions."""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow")
    lines.append("")
    lines.append("")

    # Emit the instruction list
    lines.append(f"# RV64{ext} instructions (auto-generated by gen_isadesc.py)")
    lines.append(f"{var_name}: list[tuple[str, RegisterFlow]] = [")

    for mnemonic, dst, src, enc in entries:
        dst_repr = repr(dst)
        src_repr = repr(src)
        enc_repr = _format_encoding(enc)
        if enc_repr:
            lines.append(f'    ("{mnemonic}", RegisterFlow({dst_repr}, {src_repr}, encoding={enc_repr})),')
        else:
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


def _format_encoding(enc: InstructionFormat) -> str:
    """Format an InstructionFormat as a Python source expression."""
    parts = [
        f'format_type={enc.format_type!r}',
        f'opcode=0x{enc.opcode:02x}',
    ]
    if enc.funct3 is not None:
        parts.append(f'funct3=0x{enc.funct3:02x}')
    else:
        parts.append('funct3=None')
    if enc.funct7 is not None:
        parts.append(f'funct7=0x{enc.funct7:02x}')
    else:
        parts.append('funct7=None')
    parts.extend([
        f'has_rd={enc.has_rd!r}',
        f'has_rs1={enc.has_rs1!r}',
        f'has_rs2={enc.has_rs2!r}',
        f'has_rs3={enc.has_rs3!r}',
        f'has_imm={enc.has_imm!r}',
        f'imm_bits={enc.imm_bits}',
        f'may_load={enc.may_load!r}',
        f'may_store={enc.may_store!r}',
        f'reg_class={enc.reg_class!r}',
    ])
    return f'InstructionFormat({", ".join(parts)})'


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

    reg_class = EXTENSION_REG_CLASS.get(args.ext, "integer")

    # Filter and extract instructions for the given extension
    entries: list[tuple[str, list[str], list[str], InstructionFormat]] = []
    for name, entry in sorted(data.items()):
        if not isinstance(entry, dict):
            continue
        if not _should_include(name):
            continue
        if not _has_extension(entry, args.ext, name=name):
            continue

        mnemonic = llvm_name_to_mnemonic(name)
        dst, src = _extract_flow(entry)

        # Skip instructions with no register flow at all (pure system)
        if not dst and not src:
            continue

        enc = _extract_encoding(entry, reg_class)
        entries.append((mnemonic, dst, src, enc))

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
