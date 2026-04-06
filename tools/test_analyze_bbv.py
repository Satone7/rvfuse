#!/usr/bin/env python3
"""Tests for analyze_bbv.py"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from analyze_bbv import generate_report, parse_bbv, resolve_addresses


class TestParseBbv(unittest.TestCase):
    def test_basic_parsing(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            f.write("0x10000 42\n")
            f.write("0x10050 15\n")
            f.write("0x10200 100\n")
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(len(blocks), 3)
            self.assertEqual(blocks[0], (0x10000, 42))
            self.assertEqual(blocks[1], (0x10050, 15))
            self.assertEqual(blocks[2], (0x10200, 100))
        finally:
            Path(path).unlink()

    def test_skips_comments_and_empty_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            f.write("# comment\n\n0x20000 10\n  \n0x20100 20\n")
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0], (0x20000, 10))
        finally:
            Path(path).unlink()

    def test_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(blocks, [])
        finally:
            Path(path).unlink()

    def test_decimal_addresses(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            f.write("65536 42\n")
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(blocks, [(65536, 42)])
        finally:
            Path(path).unlink()

    def test_native_qemu_bbv_format(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bb", delete=False
        ) as f:
            # Multiple intervals that reference the same BBs
            f.write("T:0:100 :1:50 :2:200\n")
            f.write("T:0:30 :2:70\n")
            bb_path = f.name

        disas_path = Path(bb_path).with_suffix(".disas")
        try:
            disas_path.write_text(
                "BB 0 (vaddr: 0x10000, 1 insns):\n"
                "  0x10000: addi t0, t0, 1\n\n"
                "BB 1 (vaddr: 0x10050, 1 insns):\n"
                "  0x10050: addi t1, t1, 2\n\n"
                "BB 2 (vaddr: 0x10200, 1 insns):\n"
                "  0x10200: addi t2, t2, 3\n\n"
            )
            blocks = parse_bbv(bb_path)
            # Counts are aggregated: BB0=100+30, BB1=50, BB2=200+70
            by_addr = dict(blocks)
            self.assertEqual(by_addr[0x10000], 130)
            self.assertEqual(by_addr[0x10050], 50)
            self.assertEqual(by_addr[0x10200], 270)
            self.assertEqual(len(blocks), 3)
        finally:
            Path(bb_path).unlink()
            disas_path.unlink(missing_ok=True)

    def test_native_qemu_bbv_missing_disas(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bb", delete=False
        ) as f:
            f.write("T:0:100\n")
            path = f.name

        try:
            with self.assertRaises(SystemExit):
                parse_bbv(path)
        finally:
            Path(path).unlink()


class TestResolveAddresses(unittest.TestCase):
    def test_empty_blocks(self):
        resolved = resolve_addresses([], "/fake/elf")
        self.assertEqual(resolved, [])

    @patch("analyze_bbv.subprocess.run")
    def test_calls_addr2line_and_parses_output(self, mock_run):
        # First call: readelf to find .text range
        # Second call: addr2line
        mock_run.side_effect = [
            MagicMock(
                stdout="  [13] .text     PROGBITS 0000000000001000 001000 000500",
                returncode=0,
            ),
            MagicMock(
                stdout="main_func\n/app/main.c:10\n"
                       "helper\n/app/util.c:25\n",
                returncode=0,
            ),
        ]
        blocks = [(0x1100, 42), (0x1200, 15)]
        resolved = resolve_addresses(blocks, "/fake/elf")

        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0], (0x1100, 42, "main_func (/app/main.c:10)"))
        self.assertEqual(resolved[1], (0x1200, 15, "helper (/app/util.c:25)"))

    @patch("analyze_bbv.subprocess.run")
    def test_fallback_on_addr2line_failure(self, mock_run):
        mock_run.side_effect = FileNotFoundError("not found")
        resolved = resolve_addresses([(0x1000, 42)], "/fake/elf")
        # When readelf fails, address can't be classified as main ELF,
        # so it falls to the unknown-so path
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0][0], 0x1000)
        self.assertEqual(resolved[0][1], 42)


class TestGenerateReport(unittest.TestCase):
    def test_sorted_by_count_descending(self):
        resolved = [
            (0x1000, 10, "func_a (a.c:1)"),
            (0x2000, 100, "func_b (b.c:2)"),
            (0x3000, 50, "func_c (c.c:3)"),
        ]
        report = generate_report(resolved, top_n=3)
        self.assertLess(report.index("func_b"), report.index("func_c"))
        self.assertLess(report.index("func_c"), report.index("func_a"))

    def test_respects_top_n(self):
        resolved = [(i, 100 - i, f"func_{i}") for i in range(10)]
        report = generate_report(resolved, top_n=3)
        self.assertIn("func_0", report)
        self.assertNotIn("func_3", report)

    def test_percentage_calculation(self):
        resolved = [
            (0x1000, 75, "func_a (a.c:1)"),
            (0x2000, 25, "func_b (b.c:2)"),
        ]
        report = generate_report(resolved, top_n=2)
        self.assertIn("75.00%", report)
        self.assertIn("25.00%", report)

    def test_empty_input(self):
        report = generate_report([])
        self.assertIn("Total basic blocks: 0", report)
        self.assertIn("Total executions:   0", report)


if __name__ == "__main__":
    unittest.main()
