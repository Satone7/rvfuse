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


# Pattern to extract SEW and LMUL from VSETVLI/VSETVL operands.
# Matches patterns like "e32,m2" or "e32,m2,tu,mu" in operand string.
_VSEW_LMUL_RE = re.compile(r"e(\d+),m(f?\d+)")


def _parse_sew_lmul(operands: str) -> tuple[int, int, str, str] | None:
    """Extract (sew, lmul, tail_policy, mask_policy) from vsetvli operands.

    Returns None if operands don't contain a valid SEW/LMUL encoding.
    """
    match = _VSEW_LMUL_RE.search(operands)
    if not match:
        return None
    sew = int(match.group(1))
    lmul_str = match.group(2)
    if lmul_str.startswith("f"):
        # Fractional LMUL -- for DFG purposes, treat as LMUL=1
        lmul = 1
    else:
        lmul = int(lmul_str)
    # Detect tail/mask policies
    tail_policy = "agnostic" if "tu" in operands else "undisturbed"
    mask_policy = "agnostic" if "mu" in operands else "undisturbed"
    return (sew, lmul, tail_policy, mask_policy)


def _annotate_vector_config(blocks: list[BasicBlock]) -> None:
    """Post-process parsed basic blocks to extract vector configuration.

    Scans each BB for VSETVLI/VSETVL/VSETIVLI instructions and builds
    a VectorConfig with the parsed SEW/LMUL and change points.
    """
    from dfg.instruction import VectorConfig

    for bb in blocks:
        config: VectorConfig | None = None

        for idx, insn in enumerate(bb.instructions):
            if insn.mnemonic in ("vsetvli", "vsetvl", "vsetivli"):
                parsed = _parse_sew_lmul(insn.operands)
                if parsed is not None:
                    sew, lmul, tail_policy, mask_policy = parsed
                    new_config = VectorConfig(
                        vlen=0,  # Unknown until runtime; set to 0 as placeholder
                        sew=sew,
                        lmul=lmul,
                        vl=None,
                        tail_policy=tail_policy,
                        mask_policy=mask_policy,
                        change_points=[],
                    )
                    if config is None:
                        config = new_config
                        bb.vec_config = config
                    else:
                        config.change_points.append((idx, new_config))

        # Also set config if any vector instructions exist but no vsetvli
        # was found (use default LMUL=1, SEW=32)
        if config is None:
            for insn in bb.instructions:
                if insn.mnemonic.startswith("v"):
                    bb.vec_config = VectorConfig(
                        vlen=0, sew=32, lmul=1, vl=None,
                        tail_policy="undisturbed", mask_policy="undisturbed",
                        change_points=[],
                    )
                    break
