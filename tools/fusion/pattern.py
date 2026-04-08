"""Fusible instruction pattern model and normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dfg.instruction import ISARegistry

# Register classification patterns (matching dfg/instruction.py)
_INTEGER_RE = re.compile(r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|s\d+|jt?\d*)$")
_FLOAT_RE = re.compile(r"^(f\d+|ft\d+|fs\d+|fa\d+|fv\d+)$")

# Control flow mnemonics
_CONTROL_FLOW_MNEMONICS = frozenset({
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
    "jal", "jalr", "j", "jr", "call", "ret",
})


def classify_register(reg_name: str) -> str | None:
    """Return 'integer', 'float', or None for a register name."""
    if _INTEGER_RE.match(reg_name):
        return "integer"
    if _FLOAT_RE.match(reg_name):
        return "float"
    return None


def is_control_flow(mnemonic: str) -> bool:
    """Return True if the mnemonic is a branch, jump, or call."""
    return mnemonic in _CONTROL_FLOW_MNEMONICS


def normalize_chain(
    chain: list[tuple[str, str]],
    edges_between: list[dict],
    registry: ISARegistry,
) -> Pattern:
    """Normalize a concrete instruction chain to a Pattern template.

    Args:
        chain: List of (mnemonic, operands) tuples.
        edges_between: RAW edges within the chain, each with 'src', 'dst', 'register'.
        registry: ISA registry for resolving operand role positions.

    Returns:
        A Pattern with normalized opcodes and chain_registers.

    Raises:
        ValueError: If any mnemonic is unknown in the registry.
    """
    opcodes = [mn for mn, _ in chain]
    first_flow = registry.get_flow(chain[0][0])
    if first_flow is None:
        raise ValueError(f"Unknown mnemonic: {chain[0][0]}")
    first_resolved = first_flow.resolve(chain[0][1])
    all_regs = first_resolved.dst_regs + first_resolved.src_regs
    reg_class = classify_register(all_regs[0]) if all_regs else None

    chain_regs: list[list[str]] = []
    for pair_idx in range(len(chain) - 1):
        pair_edges = [
            e for e in edges_between
            if e["src"] == pair_idx and e["dst"] == pair_idx + 1
        ]
        roles_for_pair: list[list[str]] = []
        for edge in pair_edges:
            reg_name = edge["register"]
            src_flow = registry.get_flow(chain[pair_idx][0])
            if src_flow is None:
                raise ValueError(f"Unknown mnemonic: {chain[pair_idx][0]}")
            src_resolved = src_flow.resolve(chain[pair_idx][1])
            dst_role = _find_role(reg_name, src_resolved.dst_regs, src_flow.dst_regs)

            dst_flow = registry.get_flow(chain[pair_idx + 1][0])
            if dst_flow is None:
                raise ValueError(f"Unknown mnemonic: {chain[pair_idx + 1][0]}")
            dst_resolved = dst_flow.resolve(chain[pair_idx + 1][1])
            src_role = _find_role(reg_name, dst_resolved.src_regs, dst_flow.src_regs)

            if dst_role and src_role:
                roles_for_pair.append([dst_role, src_role])
        chain_regs.extend(roles_for_pair)

    return Pattern(
        opcodes=opcodes,
        register_class=reg_class or "unknown",
        chain_registers=chain_regs,
    )


def _find_role(
    reg_name: str,
    resolved_regs: list[str],
    role_names: list[str],
) -> str | None:
    """Find the role name corresponding to a concrete register name."""
    for i, resolved in enumerate(resolved_regs):
        if resolved == reg_name and i < len(role_names):
            return role_names[i]
    return None


@dataclass
class Pattern:
    """A normalized fusible instruction template.

    Attributes:
        opcodes: Ordered list of instruction mnemonics in the chain.
        register_class: "integer" or "float".
        chain_registers: For each consecutive pair, a list of (dst_role, src_role)
            tuples describing which operand positions carry the RAW dependency.
            E.g. [["frd", "frs1"]] means the frd output of instruction i
            feeds into the frs1 input of instruction i+1.
    """

    opcodes: list[str]
    register_class: str
    chain_registers: list[list[str]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.opcodes)

    @property
    def template_key(self) -> tuple:
        """Hashable key for grouping identical patterns across BBs.

        Converts lists to tuples so the key is hashable (usable in dicts/sets).
        """
        chain_tuple = tuple(tuple(pair) for pair in self.chain_registers)
        return (tuple(self.opcodes), self.register_class, chain_tuple)
