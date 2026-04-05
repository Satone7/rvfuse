"""Tests for the .disas parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfg.instruction import BasicBlock, Instruction
from dfg.parser import parse_disas

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DISAS = FIXTURES_DIR / "sample_disas.txt"


class TestParseSingleBB:
    """Test parsing a single basic block."""

    def test_returns_one_basic_block(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111f4, 2 insns):\n"
            "  0x111f4: addi                    sp,sp,-32\n"
            "  0x111f6: sd                      ra,24(sp)\n"
        )
        result = parse_disas(text)
        assert len(result) == 1

    def test_bb_id_and_vaddr(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111f4, 2 insns):\n"
            "  0x111f4: addi                    sp,sp,-32\n"
            "  0x111f6: sd                      ra,24(sp)\n"
        )
        bb = parse_disas(text)[0]
        assert bb.bb_id == 1
        assert bb.vaddr == 0x111F4

    def test_instruction_count(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111f4, 2 insns):\n"
            "  0x111f4: addi                    sp,sp,-32\n"
            "  0x111f6: sd                      ra,24(sp)\n"
        )
        bb = parse_disas(text)[0]
        assert len(bb.instructions) == 2


class TestParseMultipleBBs:
    """Test parsing multiple basic blocks separated by blank lines."""

    def test_returns_two_basic_blocks(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111f4, 2 insns):\n"
            "  0x111f4: addi                    sp,sp,-32\n"
            "  0x111f6: sd                      ra,24(sp)\n"
            "\n"
            "BB 2 (vaddr: 0x11158, 2 insns):\n"
            "  0x11158: addi                    sp,sp,-32\n"
            "  0x1115a: sd                      ra,24(sp)\n"
        )
        result = parse_disas(text)
        assert len(result) == 2

    def test_each_bb_has_correct_id(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111f4, 2 insns):\n"
            "  0x111f4: addi                    sp,sp,-32\n"
            "  0x111f6: sd                      ra,24(sp)\n"
            "\n"
            "BB 2 (vaddr: 0x11158, 2 insns):\n"
            "  0x11158: addi                    sp,sp,-32\n"
            "  0x1115a: sd                      ra,24(sp)\n"
        )
        result = parse_disas(text)
        assert result[0].bb_id == 1
        assert result[1].bb_id == 2


class TestTrailingComment:
    """Test that trailing comments are stripped from operands."""

    def test_comment_stripped_from_operands(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111fc, 1 insns):\n"
            "  0x111fc: auipc                   ra,0                    # 0x111fc\n"
        )
        bb = parse_disas(text)[0]
        insn = bb.instructions[0]
        assert insn.mnemonic == "auipc"
        assert "#" not in insn.operands
        assert insn.operands.strip() == "ra,0"

    def test_comment_stripped_j_instruction(self) -> None:
        text = (
            "BB 4 (vaddr: 0x1117a, 1 insns):\n"
            "  0x1117a: j                       2                       # 0x1117c\n"
        )
        bb = parse_disas(text)[0]
        insn = bb.instructions[0]
        assert insn.mnemonic == "j"
        assert "#" not in insn.operands
        assert insn.operands.strip() == "2"


class TestEmptyString:
    """Test that empty input returns empty list."""

    def test_empty_string(self) -> None:
        result = parse_disas("")
        assert result == []

    def test_whitespace_only(self) -> None:
        result = parse_disas("   \n\n  \n")
        assert result == []


class TestNoOperands:
    """Test instructions with no operands (ret, ecall)."""

    def test_ret_no_operands(self) -> None:
        text = (
            "BB 99 (vaddr: 0x11234, 1 insns):\n"
            "  0x11234: ret\n"
        )
        bb = parse_disas(text)[0]
        insn = bb.instructions[0]
        assert insn.mnemonic == "ret"
        assert insn.operands == ""

    def test_ecall_no_operands(self) -> None:
        text = (
            "BB 99 (vaddr: 0x11234, 1 insns):\n"
            "  0x11234: ecall\n"
        )
        bb = parse_disas(text)[0]
        insn = bb.instructions[0]
        assert insn.mnemonic == "ecall"
        assert insn.operands == ""


class TestMvPseudoInstruction:
    """Test that the mv pseudo-instruction is parsed correctly."""

    def test_mv_parsed(self) -> None:
        text = (
            "BB 2 (vaddr: 0x11160, 1 insns):\n"
            "  0x11160: mv                      a0,zero\n"
        )
        bb = parse_disas(text)[0]
        insn = bb.instructions[0]
        assert insn.mnemonic == "mv"
        assert insn.operands.strip() == "a0,zero"


class TestParseFromFilePath:
    """Test parsing from a file path (Path object)."""

    def test_parse_file_path(self) -> None:
        result = parse_disas(SAMPLE_DISAS)
        assert len(result) > 0
        # First BB in the fixture
        assert result[0].bb_id == 1
        assert result[0].vaddr == 0x111F4

    def test_parse_string_path(self) -> None:
        result = parse_disas(str(SAMPLE_DISAS))
        assert len(result) > 0


class TestAddressParsedAsInt:
    """Test that instruction addresses are parsed as integers."""

    def test_address_is_int(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111f4, 1 insns):\n"
            "  0x111f4: addi                    sp,sp,-32\n"
        )
        bb = parse_disas(text)[0]
        insn = bb.instructions[0]
        assert isinstance(insn.address, int)
        assert insn.address == 0x111F4


class TestRawLinePreserved:
    """Test that the raw line is preserved in Instruction."""

    def test_raw_line_matches_input(self) -> None:
        text = (
            "BB 1 (vaddr: 0x111f4, 1 insns):\n"
            "  0x111f4: addi                    sp,sp,-32\n"
        )
        bb = parse_disas(text)[0]
        insn = bb.instructions[0]
        assert insn.raw_line == "  0x111f4: addi                    sp,sp,-32"
