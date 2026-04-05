"""Parser for QEMU BBV .disas text files into BasicBlock lists."""

from __future__ import annotations

import re
from pathlib import Path

from dfg.instruction import BasicBlock, Instruction

_BB_HEADER_RE = re.compile(
    r"^BB\s+(\d+)\s+\(vaddr:\s+(0x[0-9a-fA-F]+),\s+(\d+)\s+insns\):"
)
_INSN_RE = re.compile(r"^(\s+)(0x[0-9a-fA-F]+):\s+(\S+)(.*)?$")


def parse_disas(source: str | Path) -> list[BasicBlock]:
    """Parse a .disas file into a list of BasicBlock objects.

    Args:
        source: Either a string with file contents, or a Path/string path
                to a .disas file on disk.

    Returns:
        List of BasicBlock objects in the order they appear in the file.
    """
    if isinstance(source, Path):
        text = source.read_text()
    elif _looks_like_path(source):
        text = Path(source).read_text()
    else:
        text = source

    blocks: list[BasicBlock] = []
    current_bb: BasicBlock | None = None

    for line in text.splitlines():
        header_match = _BB_HEADER_RE.match(line)
        if header_match:
            bb_id = int(header_match.group(1))
            vaddr = int(header_match.group(2), 16)
            current_bb = BasicBlock(bb_id=bb_id, vaddr=vaddr)
            blocks.append(current_bb)
            continue

        insn_match = _INSN_RE.match(line)
        if insn_match and current_bb is not None:
            address = int(insn_match.group(2), 16)
            mnemonic = insn_match.group(3)
            raw_operands = insn_match.group(4)

            # Strip leading/trailing whitespace and remove trailing comments
            operands = raw_operands.strip()
            comment_pos = operands.find("#")
            if comment_pos != -1:
                operands = operands[:comment_pos].strip()

            current_bb.instructions.append(
                Instruction(
                    address=address,
                    mnemonic=mnemonic,
                    operands=operands,
                    raw_line=line.rstrip("\n"),
                )
            )

    return blocks


def _looks_like_path(source: str) -> bool:
    """Heuristic: treat a plain string as a file path if it ends with .disas or .txt."""
    return source.strip().endswith((".disas", ".txt")) and "\n" not in source
