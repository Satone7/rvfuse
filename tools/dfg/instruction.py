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
    """

    dst_regs: list[str]
    src_regs: list[str]

    def resolve(self, operands: str) -> ResolvedFlow:
        """Map positional names to actual register names from the operand string."""
        regs = _extract_registers(operands)
        dst = [regs.get(p) for p in self.dst_regs if regs.get(p)]
        src = [regs.get(p) for p in self.src_regs if regs.get(p)]
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


def _extract_registers(operands: str) -> dict[str, str]:
    """Extract register names from an operand string.

    Handles formats:
      - "rd,rs1,rs2" -> {"rd": "a0", "rs1": "a1", "rs2": "a2"}
      - "rs2,offset(rs1)" -> {"rs2": "a0", "rs1": "s0"}
      - "rd,offset(rs1)" -> {"rd": "a0", "rs1": "s0"}
    Returns a mapping from position name to register name.
    """
    regs: dict[str, str] = {}
    parts = [p.strip() for p in operands.split(",")]
    if not parts:
        return regs

    # Detect memory format: "rs2, offset(rs1)" or "rd, offset(rs1)"
    # This has exactly 2 comma-separated parts, second contains "("
    if len(parts) == 2 and "(" in parts[1]:
        first = parts[0].strip()
        match = re.match(r"(-?\d+)\((\w+)\)", parts[1].strip())
        if match:
            offset_reg = match.group(2)
            regs["rs1"] = offset_reg
            # Both rd and rs2 are set to the first operand because loads use rd
            # and stores use rs2 for the first operand. RegisterFlow.resolve()
            # selects only the positions each instruction's flow definition
            # specifies, so the dual-assignment is safe.
            regs["rd"] = first
            regs["rs2"] = first
        return regs

    # Standard format: "rd, rs1, rs2" or "rd, rs1, imm"
    position_names = ["rd", "rs1", "rs2", "rs3"]
    for i, part in enumerate(parts):
        if i >= len(position_names):
            break
        token = part.strip()
        # Skip immediates (numbers, hex values)
        if re.match(r"^-?\d+$", token) or re.match(r"^0x[0-9a-fA-F]+$", token):
            continue
        # Only accept valid RISC-V register names (x0-x31 or ABI names)
        if not re.match(r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|[sf]\d+)$", token):
            continue
        regs[position_names[i]] = token
    return regs


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
