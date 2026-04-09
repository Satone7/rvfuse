"""Encoding conflict validator for fusion scheme proposals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dfg.instruction import ISARegistry


@dataclass
class ValidationResult:
    """Result of validating a proposed fusion encoding.

    Attributes:
        passed: Whether the encoding passed all validation checks.
        conflicts: List of conflict descriptions if validation failed.
        warnings: List of non-fatal warnings about the encoding.
        suggested_alternatives: List of alternative encodings to consider.
    """
    passed: bool
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_alternatives: list[str] = field(default_factory=list)


def validate_encoding(
    opcode: int,
    funct3: int | None,
    funct7: int | None,
    reg_class: str,
    registry: ISARegistry,
) -> ValidationResult:
    """Validate a proposed fusion encoding against existing ISA instructions."""
    conflicts: list[str] = []
    warnings: list[str] = []
    suggested_alternatives: list[str] = []

    # Track which instructions use this opcode for suggestions
    opcode_users: list[tuple[str, int | None, int | None, str]] = []

    # Scan all registered instructions for conflicts
    for mnemonic, flow in registry._flows.items():
        encoding = flow.encoding
        if encoding is None:
            continue

        # Track opcode users
        if encoding.opcode == opcode:
            opcode_users.append((
                mnemonic,
                encoding.funct3,
                encoding.funct7,
                encoding.reg_class,
            ))

        # Check full encoding match (hard conflict)
        if encoding.opcode == opcode:
            match_funct3 = (funct3 is None or encoding.funct3 is None or
                           encoding.funct3 == funct3)
            match_funct7 = (funct7 is None or encoding.funct7 is None or
                           encoding.funct7 == funct7)

            if match_funct3 and match_funct7:
                conflict_desc = f"opcode 0x{opcode:02X}"
                if funct3 is not None:
                    conflict_desc += f" funct3 0x{funct3:X}"
                if funct7 is not None:
                    conflict_desc += f" funct7 0x{funct7:02X}"
                conflict_desc += f" conflicts with {mnemonic}"
                conflicts.append(conflict_desc)

    # Check register class mismatch against opcode region
    reg_class_conflict = _check_reg_class_mismatch(opcode, reg_class, opcode_users)
    if reg_class_conflict:
        conflicts.append(reg_class_conflict)

    # Check partial field usage (warnings)
    partial_warnings = _check_partial_usage(opcode, funct3, funct7, opcode_users)
    warnings.extend(partial_warnings)

    # Generate suggested alternatives if there are conflicts
    if conflicts:
        suggested = _suggest_alternatives(reg_class, registry, opcode_users)
        suggested_alternatives.extend(suggested)

    passed = len(conflicts) == 0
    return ValidationResult(
        passed=passed,
        conflicts=conflicts,
        warnings=warnings,
        suggested_alternatives=suggested_alternatives,
    )


def _check_reg_class_mismatch(
    opcode: int,
    reg_class: str,
    opcode_users: list[tuple[str, int | None, int | None, str]],
) -> str | None:
    """Check if the opcode region is reserved for a different register class."""
    opcode_reg_class: dict[int, str] = {
        0x33: "integer",   # OP
        0x3B: "integer",   # OP-32
        0x13: "integer",   # OP-IMM
        0x1B: "integer",   # OP-IMM-32
        0x53: "float",     # OP-FP
        0x07: "integer",   # LOAD-FP (uses FP registers but integer opcode space)
        0x27: "integer",   # STORE-FP
        0x43: "float",     # MADD
        0x47: "float",     # MSUB
        0x4B: "float",     # NMSUB
        0x4F: "float",     # NMADD
        0x57: "vector",    # OP-V
    }

    expected_class = opcode_reg_class.get(opcode)
    if expected_class is None:
        return None

    if expected_class != reg_class:
        return (f"register class mismatch: opcode 0x{opcode:02X} is "
                f"{expected_class} region, but pattern is {reg_class}")

    return None


def _check_partial_usage(
    opcode: int,
    funct3: int | None,
    funct7: int | None,
    opcode_users: list[tuple[str, int | None, int | None, str]],
) -> list[str]:
    """Check for partial field usage (soft warnings)."""
    warnings: list[str] = []

    if funct3 is not None:
        funct3_users = [m for m, f3, f7, rc in opcode_users if f3 == funct3]
        if funct3_users and funct7 is not None:
            warnings.append(
                f"funct3 0x{funct3:X} is used by {', '.join(funct3_users)} "
                f"-- ensure funct7 0x{funct7:02X} is unique"
            )

    if funct7 is not None:
        funct7_users = [m for m, f3, f7, rc in opcode_users if f7 == funct7]
        if funct7_users and funct3 is not None:
            warnings.append(
                f"funct7 0x{funct7:02X} is used by {', '.join(funct7_users)} "
                f"-- ensure funct3 0x{funct3:X} is unique"
            )

    return warnings


def _suggest_alternatives(
    reg_class: str,
    registry: ISARegistry,
    opcode_users: list[tuple[str, int | None, int | None, str]],
) -> list[str]:
    """Suggest alternative encoding slots for the given register class."""
    suggestions: list[str] = []

    candidate_opcodes: list[int] = []
    if reg_class == "integer":
        candidate_opcodes = [0x0B, 0x33, 0x3B]
    elif reg_class == "float":
        candidate_opcodes = [0x43, 0x47, 0x4B, 0x4F, 0x53]
    elif reg_class == "vector":
        candidate_opcodes = [0x57]

    for op in candidate_opcodes:
        used_funct3: set[int] = set()
        for mnemonic, flow in registry._flows.items():
            encoding = flow.encoding
            if encoding and encoding.opcode == op and encoding.funct3 is not None:
                used_funct3.add(encoding.funct3)

        for f3 in range(8):
            if f3 not in used_funct3:
                suggestions.append(f"opcode 0x{op:02X} funct3 0x{f3:X}")
                if len(suggestions) >= 3:
                    return suggestions

    return suggestions