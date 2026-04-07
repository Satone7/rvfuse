"""Data models for RISC-V instructions and ISA registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Instruction:
    """A single RISC-V instruction parsed from .disas output."""

    address: int
    mnemonic: str
    operands: str
    raw_line: str


@dataclass
class ResolvedFlow:
    """Register flow with actual register names resolved from operands."""

    dst_regs: list[str]
    src_regs: list[str]


@dataclass
class RegisterFlow:
    """Describes which register positions are dst/src for an instruction.

    Positions use RISC-V spec names (rd, rs1, rs2, rs3, etc.) which are
    resolved to actual register names from the operand string via resolve().
    When kinds is empty, all operands default to "integer".
    """

    dst_regs: list[str]
    src_regs: list[str]
    kinds: dict[str, str] = field(default_factory=dict)

    def resolve(self, operands: str) -> ResolvedFlow:
        """Map positional names to actual register names from the operand string."""
        regs = _extract_registers(operands, kinds=_BUILTIN_KINDS)
        dst = [regs[p][0] for p in self.dst_regs if p in regs]
        src = [regs[p][0] for p in self.src_regs if p in regs]
        return ResolvedFlow(dst_regs=dst, src_regs=src)


@dataclass
class BasicBlock:
    """A basic block with ID, start address, and instruction list."""

    bb_id: int
    vaddr: int
    instructions: list[Instruction] = field(default_factory=list)


@dataclass
class DFGNode:
    """A node in the DFG, wrapping an instruction with its position index."""

    instruction: Instruction
    index: int


@dataclass
class DFGEdge:
    """A data dependency edge (RAW: read after write) in the DFG."""

    src_index: int
    dst_index: int
    register: str


@dataclass
class DFG:
    """Complete DFG for a single basic block."""

    bb: BasicBlock
    nodes: list[DFGNode] = field(default_factory=list)
    edges: list[DFGEdge] = field(default_factory=list)
    source: str = "script"


@dataclass
class RegisterKind:
    """Describes a register name space (integer, float, vector, ...).

    Attributes:
        name: Kind identifier, e.g. "integer", "float", "vector".
        pattern: Compiled regex matching register names in this space.
        position_prefix: Prefix for position names extracted from operands.
            E.g. "f" means operands map to "frd", "frs1", "frs2", "frs3".
    """

    name: str
    pattern: re.Pattern
    position_prefix: str


def _extract_registers(operands: str, kinds: list[RegisterKind] | None = None) -> dict[str, tuple[str, str]]:
    """Extract register names from an operand string.

    Handles formats:
      - "rd,rs1,rs2" -> {"rd": ("a0", "integer"), "rs1": ("a1", "integer"), ...}
      - "rs2,offset(rs1)" -> {"rs2": ("a0", "integer"), "rs1": ("s0", "integer")}
      - "fmadd.s dyn,ft2,fa4,ft0,ft2" -> {"frd": ("ft2","float"), "frs1": ("fa4","float"), ...}
      - "flw fa4,0(a6)" -> {"frd": ("fa4","float"), "rs1": ("a6","integer")}
    Returns a mapping from position name to (register_name, kind_name).
    """
    if kinds is None:
        kinds = _BUILTIN_KINDS
    regs: dict[str, tuple[str, str]] = {}
    parts = [p.strip() for p in operands.split(",")]
    if not parts:
        return regs

    # Detect memory format: "rs2, offset(rs1)" or "rd, offset(rs1)"
    if len(parts) == 2 and "(" in parts[1]:
        first = parts[0].strip()
        match = re.match(r"(-?\d+)\((\w+)\)", parts[1].strip())
        if match:
            offset_reg_name = match.group(2)
            offset_reg_kind = _match_kind(offset_reg_name, kinds)
            if offset_reg_kind:
                regs[offset_reg_kind.position_prefix + "rs1"] = (
                    offset_reg_name,
                    offset_reg_kind.name,
                )
            # Both rd and rs2 use the first operand — RegisterFlow.resolve()
            # selects only the positions each instruction specifies.
            first_kind = _match_kind(first, kinds)
            if first_kind:
                regs[first_kind.position_prefix + "rd"] = (first, first_kind.name)
                regs[first_kind.position_prefix + "rs2"] = (first, first_kind.name)
        return regs

    # Standard format: "rd, rs1, rs2, rs3" or "rd, rs1, imm"
    position_names = ["rd", "rs1", "rs2", "rs3"]
    pos_idx = 0
    for token in parts:
        token = token.strip()
        # Skip immediates
        if re.match(r"^-?\d+$", token) or re.match(r"^0x[0-9a-fA-F]+$", token):
            continue
        # Skip rounding modes
        if token in _ROUNDING_MODES:
            continue
        # Try matching against registered kinds
        matched_kind = _match_kind(token, kinds)
        if matched_kind is None:
            continue
        if pos_idx >= len(position_names):
            break
        pos_name = matched_kind.position_prefix + position_names[pos_idx]
        regs[pos_name] = (token, matched_kind.name)
        pos_idx += 1
    return regs


def _match_kind(
    token: str,
    kinds: list[RegisterKind],
) -> RegisterKind | None:
    """Find the first RegisterKind whose pattern matches *token*."""
    for kind in kinds:
        if kind.pattern.match(token):
            return kind
    return None


# Builtin register kinds
INTEGER_KIND = RegisterKind(
    name="integer",
    pattern=re.compile(
        r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|[sf]\d+|jt?\d*)$"
    ),
    position_prefix="",
)

FLOAT_KIND = RegisterKind(
    name="float",
    pattern=re.compile(
        r"^(f\d+|ft\d+|fs\d+|fa\d+|fv\d+)$"
    ),
    position_prefix="f",
)

_BUILTIN_KINDS: list[RegisterKind] = [INTEGER_KIND, FLOAT_KIND]

_ROUNDING_MODES = frozenset({"dyn", "rne", "rtz", "rdn", "rup", "rmm"})


class ISARegistry:
    """Registry mapping instruction mnemonics to their register flow rules."""

    def __init__(self) -> None:
        self._flows: dict[str, RegisterFlow] = {}

    def register(self, mnemonic: str, flow: RegisterFlow) -> None:
        """Register a single instruction's register flow."""
        self._flows[mnemonic] = flow

    def load_extension(
        self, _name: str, instructions: list[tuple[str, RegisterFlow]]
    ) -> None:
        """Bulk-load instructions for an ISA extension."""
        for mnemonic, flow in instructions:
            self.register(mnemonic, flow)

    def get_flow(self, mnemonic: str) -> RegisterFlow | None:
        """Get register flow for a mnemonic, or None if unknown."""
        return self._flows.get(mnemonic)

    def is_known(self, mnemonic: str) -> bool:
        """Check if a mnemonic is registered."""
        return mnemonic in self._flows
