"""Fusible instruction pattern model and normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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
