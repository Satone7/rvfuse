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
    "I": set(),  # Base ISA: no required predicates
    "F": {"HasStdExtF"},
    "M": {"HasStdExtM"},
    "M_ZMMUL": {"HasStdExtZmmul"},
    "V": {"HasVInstructions"},
}

# Instructions that also carry any of these predicates are excluded from the
# extension's descriptor, even if they carry the extension's own predicate.
# This prevents F from pulling in vector-FP instructions that are already
# covered by the V descriptor.
EXTENSION_EXCLUDED_PREDICATES: dict[str, set[str]] = {
    "F": {"HasVInstructions"},
}

# LLVM def names that override the exclusion -- these are included even if they
# carry a normally-excluded predicate.
EXTENSION_EXCLUSION_OVERRIDES: dict[str, set[str]] = {
    "F": {"VFMV_F_S", "VFMV_S_F"},
}

EXTENSION_REG_CLASS: dict[str, str] = {
    "I": "integer",
    "F": "float",
    "M": "integer",
    "M_ZMMUL": "integer",
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

    # Exclude pseudo instructions (some don't start with "Pseudo" prefix)
    if "Pseudo" in superclasses or "PseudoInstExpansion" in superclasses:
        return False

    # Exclude compressed (16-bit) instructions — their Inst array is < 32 bits
    # and doesn't follow standard RISC-V encoding layout.  Only filter when
    # the Inst field is actually present (some test fixtures omit it).
    inst = entry.get("Inst")
    if isinstance(inst, list) and len(inst) > 0 and len(inst) < 32:
        return False

    preds = entry.get("Predicates", [])
    if not isinstance(preds, list):
        return False

    pred_names = {p.get("def", "") for p in preds if isinstance(p, dict)}
    if not required.issubset(pred_names):
        return False

    # I extension: must NOT have any extension-specific predicates
    if ext == "I":
        _I_EXCLUDE_PREDS = {
            "HasStdExtF", "HasStdExtD", "HasStdExtM", "HasStdExtZmmul",
            "HasStdExtV", "HasStdExtC", "HasStdExtA", "HasStdExtB",
            "HasVInstructions", "HasStdExtZfinx", "HasStdExtZdinx",
            "HasStdExtZbkb", "HasStdExtZbs", "HasStdExtZba",
            "HasStdExtZbb", "HasStdExtZbc", "HasStdExtZknd",
            "HasStdExtZkne", "HasStdExtZknh", "HasStdExtZksed",
            "HasStdExtZksh", "HasStdExtZfh", "HasStdExtZfhmin",
            "HasTHeadBa", "HasTHeadBb", "HasTHeadBs", "HasTHeadCMO",
            "HasTHeadCondMov", "HasTHeadFmIdx", "HasTHeadMac",
            "HasTHeadMemIdx", "HasTHeadMemPair", "HasTHeadSync",
            "HasStdExtZcb",
        }
        if pred_names & _I_EXCLUDE_PREDS:
            return False
        # Also exclude IsRV64-only check: include it, but nothing else
        if pred_names - {"IsRV64"}:
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

    # Branch-format fixup: when there are no output operands but two GPR
    # inputs (rs1, rs2), the parser in _extract_registers will assign the
    # first register token to "rd" and the second to "rs1" positionally
    # (since the third token is an immediate and gets skipped).  Remap so
    # src positions match what the parser produces.
    if not dst and len(src) == 2 and src == ["rs1", "rs2"]:
        src = ["rd", "rs1"]

    return dst, src


def _extract_encoding(entry: dict, reg_class: str) -> InstructionFormat:
    """Extract encoding layout metadata from a tablegen instruction entry.

    Parses the Inst array (LSB-first) to determine which fields are present,
    the instruction format type, and fixed encoding bits (opcode, funct3,
    funct7).

    The Inst array contains either integers (fixed bits) or dicts with a 'var'
    key naming the variable field (e.g. 'rs2', 'imm12', 'shamt').  We use these
    var names to distinguish register fields from immediate fields.
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

    def field_var_name(start: int, width: int) -> str | None:
        """Return the var name for a contiguous field, or None if mixed/fixed."""
        names = set()
        for i in range(width):
            el = inst[start + i]
            if isinstance(el, int):
                return None  # fixed bit in this range
            name = el.get("var", "") if isinstance(el, dict) else ""
            names.add(name)
        if len(names) == 1:
            return names.pop()
        return None  # mixed variable names (e.g. rs2 + imm12 in S-type)

    # Opcode: bits [0:7] of the Inst array (LSB-first, already in binary)
    opcode = extract_field(0, 7)
    if opcode is None:
        opcode = 0

    funct3 = extract_field(12, 3)
    funct7 = extract_field(25, 7)

    # Determine has_rd / has_rs1 by checking var names, not just variability.
    # This correctly handles cases like auipc where bits 15-19 are imm20,
    # not rs1, and compressed instructions with mixed var names.
    _REG_VARS = {"rd", "rs1", "rs2", "vs1", "vs2", "vs3", "vd"}
    rd_var = field_var_name(7, 5)
    has_rd = rd_var in _REG_VARS
    rs1_var = field_var_name(15, 5)
    has_rs1 = rs1_var in _REG_VARS

    may_load = bool(entry.get("mayLoad", 0))
    may_store = bool(entry.get("mayStore", 0))

    # Determine if bits 20-24 are actually rs2 or an immediate.
    # The var name tells us: "rs2" or "vs2" → register, anything else → imm.
    rs2_var = field_var_name(20, 5)
    has_rs2 = rs2_var in ("rs2", "vs2")

    # R4 format: only for 4-operand FP instructions (opcode 0x43 MADD,
    # 0x47 MSUB, 0x4b NMSUB, 0x4f NMADD).  Bits 27-31 are rs3 only in
    # this context.
    _R4_OPCODES = {0x43, 0x47, 0x4b, 0x4f}
    has_rs3 = opcode in _R4_OPCODES and has_var_field(27, 5)

    # Determine has_imm and imm_bits.
    has_imm = False
    imm_bits = 0

    # Check for any variable bits in the immediate region (bits 20-31)
    # that are NOT rs2/rs3.  If rs2 occupies bits 20-24, check bits 25-31
    # for immediate content.
    if may_load or may_store:
        has_imm = True
        imm_bits = 12
    elif not has_rs2 and not has_rs3:
        # No rs2 at bits 20-24 → these bits plus 25-31 are all immediate
        # (I-type: 12-bit immediate at bits 20-31)
        has_imm = True
        imm_bits = 12
    elif has_rs2 and not has_rs3:
        # rs2 at 20-24, check if bits 25-31 are an immediate (not funct7)
        if funct7 is None:
            # Variable bits in 25-31 that aren't a single funct7 → immediate
            # (B-type, S-type upper imm, etc.)
            has_imm = True
            imm_bits = 7  # upper immediate bits 25-31

    # Determine format type from opcode + flags.
    # RISC-V opcode → format mapping:
    #   0x33, 0x3b           → R-type (ALU reg-reg)
    #   0x13, 0x1b           → I-type (ALU immediate)
    #   0x03, 0x07           → I-type (loads)
    #   0x67                 → I-type (jalr)
    #   0x73                 → I-type (system)
    #   0x23, 0x27           → S-type (stores)
    #   0x63                 → B-type (branches)
    #   0x6f                 → J-type (jal)
    #   0x37, 0x17           → U-type (lui/auipc)
    #   0x53                 → R-type or R4-type (FP)
    #   0x57                 → V-type (vector — treat as R)
    _I_OPCODES = {0x03, 0x07, 0x13, 0x1b, 0x67, 0x73}
    _S_OPCODES = {0x23, 0x27}
    _B_OPCODES = {0x63}
    _J_OPCODES = {0x6f}
    _U_OPCODES = {0x17, 0x37}
    _V_OPCODES = {0x57}

    if has_rs3:
        format_type = "R4"
    elif opcode in _V_OPCODES:
        format_type = "V"
    elif may_store or opcode in _S_OPCODES:
        format_type = "S"
    elif may_load or opcode in _I_OPCODES:
        format_type = "I"
    elif opcode in _B_OPCODES:
        format_type = "B"
    elif opcode in _J_OPCODES:
        format_type = "J"
    elif opcode in _U_OPCODES:
        format_type = "U"
    else:
        format_type = "R"

    # For I/J/U/B/S formats, set has_imm and imm_bits based on format
    if format_type in ("I", "J", "U", "B", "S"):
        has_imm = True
        imm_bits = {"I": 12, "S": 12, "B": 12, "J": 20, "U": 20}[format_type]

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
    "anonymous",
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
