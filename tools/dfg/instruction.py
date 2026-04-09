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
    config_regs: list[str] = field(default_factory=list)


@dataclass
class InstructionFormat:
    """RISC-V instruction encoding layout metadata.

    Attributes:
        format_type: RISC-V instruction format ("R", "I", "S", "B", "U", "J", "R4", "V").
        opcode: 7-bit opcode value (e.g., 0x33 for OP, 0x53 for OP-FP, 0x57 for OP-V).
        funct3: 3-bit funct3 value, or None if variable.
        funct7: 7-bit funct7 value, or None if variable.
        has_rd: Whether the rd/destination field is present.
        has_rs1: Whether the rs1 field is present.
        has_rs2: Whether the rs2 field is present.
        has_rs3: Whether the rs3 field is present (R4-type only).
        has_imm: Whether an immediate field is present.
        imm_bits: Immediate field width in bits (0 if no imm).
        may_load: Whether the instruction accesses memory (load).
        may_store: Whether the instruction accesses memory (store).
        reg_class: Register class ("integer", "float", "vector").
    """

    format_type: str
    opcode: int
    funct3: int | None = None
    funct7: int | None = None
    has_rd: bool = True
    has_rs1: bool = True
    has_rs2: bool = True
    has_rs3: bool = False
    has_imm: bool = False
    imm_bits: int = 0
    may_load: bool = False
    may_store: bool = False
    reg_class: str = "integer"


@dataclass
class RegisterFlow:
    """Describes which register positions are dst/src for an instruction.

    Positions use RISC-V spec names (rd, rs1, rs2, rs3, etc.) which are
    resolved to actual register names from the operand string via resolve().
    """

    dst_regs: list[str]
    src_regs: list[str]
    config_regs: list[str] = field(default_factory=list)
    encoding: InstructionFormat | None = None

    def resolve(self, operands: str) -> ResolvedFlow:
        """Map positional names to actual register names from the operand string."""
        regs = _extract_registers(operands, kinds=_BUILTIN_KINDS)
        dst = [regs[p][0] for p in self.dst_regs if p in regs]
        src = [regs[p][0] for p in self.src_regs if p in regs]
        cfg = _resolve_config_regs(self.config_regs)
        return ResolvedFlow(dst_regs=dst, src_regs=src, config_regs=cfg)


@dataclass
class BasicBlock:
    """A basic block with ID, start address, and instruction list."""

    bb_id: int
    vaddr: int
    instructions: list[Instruction] = field(default_factory=list)
    vec_config: "VectorConfig | None" = None


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
class VectorConfig:
    """Vector configuration state at a point in a basic block."""

    vlen: int
    sew: int             # 8, 16, 32, or 64
    lmul: int            # 1, 2, 4, or 8
    vl: int | None
    tail_policy: str     # "undisturbed" or "agnostic"
    mask_policy: str     # "undisturbed" or "agnostic"
    change_points: list[tuple[int, VectorConfig]] = field(default_factory=list)


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
        match = re.match(r"(-?\d+)?\((\w+)\)", parts[1].strip())
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
            # Also map rs3 for store instructions (e.g. vse32.v, fmadd.s memory).
            first_kind = _match_kind(first, kinds)
            if first_kind:
                regs[first_kind.position_prefix + "rd"] = (first, first_kind.name)
                regs[first_kind.position_prefix + "rs2"] = (first, first_kind.name)
                if first_kind.name == "vector":
                    regs[first_kind.position_prefix + "rs3"] = (first, first_kind.name)
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


def _resolve_config_regs(config_positions: list[str]) -> list[str]:
    """Resolve implicit config register position names to actual register names.

    Config regs are implicit CSR writes (e.g. vsetvli writes vl, vtype)
    that are not present in the operand string. Position names follow the
    RegisterKind prefix convention (e.g. "cvl" -> prefix "c" -> name "vl").
    """
    result: list[str] = []
    for pos in config_positions:
        for kind in _BUILTIN_KINDS:
            pfx = kind.position_prefix
            if pfx and pos.startswith(pfx):
                name = pos[len(pfx):]
                if kind.pattern.match(name):
                    result.append(name)
                    break
    return result


def _expand_grouping(
    resolved: ResolvedFlow,
    config: VectorConfig | None,
) -> ResolvedFlow:
    """Expand vector register names to physical register groups based on LMUL.

    If config is None or LMUL is 1, returns the flow unchanged.
    Only registers matching v{1-31} pattern are expanded.
    """
    if config is None or config.lmul <= 1:
        return resolved

    _vec_re = re.compile(r"^v([1-9]|[1-2]\d|3[0-1])$")
    lmul = config.lmul

    def _expand(regs: list[str]) -> list[str]:
        result: list[str] = []
        for r in regs:
            m = _vec_re.match(r)
            if m:
                base = int(m.group(1))
                if base + lmul > 32:
                    continue  # group exceeds register file
                for i in range(lmul):
                    result.append(f"v{base + i}")
            else:
                result.append(r)
        return result

    return ResolvedFlow(
        dst_regs=_expand(resolved.dst_regs),
        src_regs=_expand(resolved.src_regs),
    )


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
        r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|s\d+|jt?\d*)$"
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

VECTOR_KIND = RegisterKind(
    name="vector",
    pattern=re.compile(
        r"^v([1-9]|[1-2]\d|3[0-1])$"
    ),
    position_prefix="v",
)

MASK_KIND = RegisterKind(
    name="mask",
    pattern=re.compile(
        r"^v0$"
    ),
    position_prefix="v",
)

CSR_VEC_KIND = RegisterKind(
    name="csr_vec",
    pattern=re.compile(
        r"^(vl|vtype|vstart|vxrm|vxsat)$"
    ),
    position_prefix="c",
)

_BUILTIN_KINDS: list[RegisterKind] = [INTEGER_KIND, FLOAT_KIND, VECTOR_KIND, MASK_KIND, CSR_VEC_KIND]

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

    def get_encoding(self, mnemonic: str) -> InstructionFormat | None:
        """Get encoding format for a mnemonic, or None if unknown or not set."""
        flow = self._flows.get(mnemonic)
        if flow is None:
            return None
        return flow.encoding
