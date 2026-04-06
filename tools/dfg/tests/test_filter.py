"""Tests for report-driven BB filtering."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from dfg.filter import load_report, select_addresses
from dfg.instruction import BasicBlock, Instruction
from dfg.parser import parse_disas


def _sample_disas() -> str:
    return textwrap.dedent("""\
        BB 1 (vaddr: 0x111f4, 2 insns):
          0x111f4: addi  sp,sp,-32
          0x111f6: sd    ra,24(sp)

        BB 2 (vaddr: 0x11200, 1 insns):
          0x11200: lw    a0,-28(s0)

        BB 3 (vaddr: 0x11210, 1 insns):
          0x11210: addi  a0,a0,1
    """)


def _sample_report_json() -> str:
    return json.dumps({
        "total_blocks": 3,
        "total_executions": 200,
        "blocks": [
            {"rank": 1, "address": "0x111f4", "count": 120,
             "pct": 60.0, "cumulative_pct": 60.0,
             "location": "func_a (a.c:1)"},
            {"rank": 2, "address": "0x11200", "count": 60,
             "pct": 30.0, "cumulative_pct": 90.0,
             "location": "func_b (b.c:2)"},
            {"rank": 3, "address": "0x11210", "count": 20,
             "pct": 10.0, "cumulative_pct": 100.0,
             "location": "func_c (c.c:3)"},
        ],
    })


class TestLoadReport:
    def test_loads_valid_json(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        data = load_report(p)
        assert data["total_blocks"] == 3
        assert len(data["blocks"]) == 3

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_report(tmp_path / "nonexistent.json")

    def test_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_report(p)


class TestSelectAddresses:
    def test_top_n_mode(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        addrs = select_addresses(p, top_n=2, coverage=None)
        assert addrs == {0x111F4, 0x11200}

    def test_coverage_mode_includes_exceeding_block(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        # 80% coverage: BB1 (60%) + BB2 (30% -> cumulative 90% >= 80%) — both included
        addrs = select_addresses(p, top_n=None, coverage=80)
        assert addrs == {0x111F4, 0x11200}

    def test_coverage_100_includes_all(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        addrs = select_addresses(p, top_n=None, coverage=100)
        assert addrs == {0x111F4, 0x11200, 0x11210}

    def test_coverage_exact_match(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        # 90% = exact cumulative of BB2 — BB1 and BB2 included
        addrs = select_addresses(p, top_n=None, coverage=90)
        assert addrs == {0x111F4, 0x11200}

    def test_top_n_defaults_to_20(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        addrs = select_addresses(p, top_n=20, coverage=None)
        # Only 3 blocks in report, so all are returned
        assert len(addrs) == 3

    def test_empty_report_returns_empty(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(json.dumps({"total_blocks": 0, "total_executions": 0, "blocks": []}))
        addrs = select_addresses(p, top_n=20, coverage=None)
        assert addrs == set()

    def test_neither_top_n_nor_coverage_raises(self, tmp_path):
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        with pytest.raises(ValueError, match="Either top_n or coverage"):
            select_addresses(p, top_n=None, coverage=None)


class TestFilterIntegration:
    def test_matched_blocks_produced(self, tmp_path):
        """BBs whose vaddr appears in the report address set are kept."""
        p = tmp_path / "report.json"
        p.write_text(_sample_report_json())
        addrs = select_addresses(p, top_n=2, coverage=None)
        blocks = parse_disas(_sample_disas())
        filtered = [bb for bb in blocks if bb.vaddr in addrs]
        assert len(filtered) == 2
        assert {bb.vaddr for bb in filtered} == {0x111F4, 0x11200}

    def test_unmatched_addresses_skipped(self, tmp_path):
        """Report addresses not in .disas produce no match (warning, no abort)."""
        report = json.dumps({
            "total_blocks": 1,
            "total_executions": 100,
            "blocks": [
                {"rank": 1, "address": "0xDEAD", "count": 100,
                 "pct": 100.0, "cumulative_pct": 100.0,
                 "location": "nowhere"},
            ],
        })
        p = tmp_path / "report.json"
        p.write_text(report)
        addrs = select_addresses(p, top_n=1, coverage=None)
        blocks = parse_disas(_sample_disas())
        filtered = [bb for bb in blocks if bb.vaddr in addrs]
        assert len(filtered) == 0


class TestMutualExclusion:
    """--report and --bb-filter are mutually exclusive."""

    def test_report_and_bb_filter_rejected(self):
        from dfg.__main__ import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--disas", "x.disas", "--report", "r.json", "--bb-filter", "1"])

    def test_report_accepted_alone(self):
        from dfg.__main__ import parse_args
        args = parse_args(["--disas", "x.disas", "--report", "r.json"])
        assert args.report == Path("r.json")

    def test_report_with_coverage(self):
        from dfg.__main__ import parse_args
        args = parse_args(["--disas", "x.disas", "--report", "r.json", "--coverage", "80"])
        assert args.coverage == 80

    def test_report_with_top(self):
        from dfg.__main__ import parse_args
        args = parse_args(["--disas", "x.disas", "--report", "r.json", "--top", "5"])
        assert args.top == 5

    def test_coverage_without_report_rejected(self):
        from dfg.__main__ import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--disas", "x.disas", "--coverage", "80"])
