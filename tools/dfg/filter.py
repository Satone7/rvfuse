"""Report-driven BB filtering for selective DFG generation.

Reads JSON hotspot reports from analyze_bbv.py and selects which
basic block addresses should be processed.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_report(report_path: Path) -> dict:
    """Load and validate a JSON hotspot report.

    Args:
        report_path: Path to JSON file produced by analyze_bbv --json-output.

    Returns:
        Parsed JSON as a dict with 'total_blocks', 'total_executions', 'blocks'.

    Raises:
        FileNotFoundError: if report_path does not exist.
        ValueError: if the file is not valid JSON.
    """
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")
    try:
        data = json.loads(report_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in report: {report_path}: {e}") from e
    if "blocks" not in data:
        raise ValueError(f"Report missing 'blocks' key: {report_path}")
    return data


def select_addresses(
    report_path: Path,
    top_n: int | None = 20,
    coverage: int | None = None,
) -> set[int]:
    """Select basic block addresses from a hotspot report.

    Args:
        report_path: Path to JSON report.
        top_n: If coverage is None, include this many top-ranked blocks (default 20).
        coverage: If set, include blocks up to and including the one whose
                  cumulative_pct first reaches or exceeds this value.

    Returns:
        Set of integer addresses for blocks that should be processed.
    """
    if top_n is None and coverage is None:
        raise ValueError("Either top_n or coverage must be specified")

    data = load_report(report_path)
    blocks = data["blocks"]
    if not blocks:
        return set()

    selected = blocks

    if coverage is not None:
        # Include blocks up to and including the one that first exceeds threshold
        cutoff = []
        for b in blocks:
            cutoff.append(b)
            if b["cumulative_pct"] >= coverage:
                break
        selected = cutoff
    elif top_n is not None:
        selected = blocks[:top_n]

    return {int(b["address"], 16) for b in selected}
