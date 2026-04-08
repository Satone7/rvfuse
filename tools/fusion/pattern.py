"""Fusible instruction pattern model and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field


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
