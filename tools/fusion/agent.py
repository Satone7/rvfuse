"""Agent layer for fusion pattern analysis.

Calls Claude CLI subprocess to analyze miner output and produce
recommendations on which patterns have the highest fusion value.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("fusion.agent")


def _build_analysis_prompt(patterns: list[dict]) -> str:
    """Build a prompt for Claude to analyze fusion patterns."""
    top = patterns[:20]
    pattern_text = json.dumps(top, indent=2)

    prompt = (
        "fusion-discover: Analyze the following RISC-V instruction fusion "
        "pattern candidates extracted from real workload DFGs. For each "
        "pattern, evaluate its fusion potential based on:\n"
        "1. Frequency (higher = more impact from fusion)\n"
        "2. Dependency tightness (RAW chain density, register reuse)\n"
        "3. Hardware feasibility (same execution unit, operand count)\n\n"
        "Return JSON with keys:\n"
        "- 'top_recommendations': list of objects with 'pattern_rank' (int), "
        "'recommendation' (str: 'Strong candidate'/'Moderate'/'Weak'), "
        "'rationale' (str), 'notes' (str, optional)\n"
        "- 'missed_patterns': list of pattern descriptions the miner might "
        "have missed (empty list if none)\n"
        "- 'summary': str, 2-3 sentence overall assessment\n\n"
        f"Patterns (total: {len(patterns)}):\n{pattern_text}"
    )
    return prompt


def _invoke_claude(prompt: str, model: str | None = None) -> str | None:
    """Call Claude CLI subprocess. Returns stdout or None on failure."""
    cmd: list[str] = ["claude"]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--print", prompt])

    logger.debug("Agent command: %s", " ".join(cmd[:-1]) + " <prompt>")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        logger.warning("Claude CLI not found — skipping agent analysis")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out — skipping agent analysis")
        return None

    if result.returncode != 0:
        logger.warning("Claude CLI failed (rc=%d) — skipping agent analysis", result.returncode)
        return None

    return result.stdout.strip() or None


def run_agent(
    patterns: list[dict],
    output_path: Path,
    model: str | None = None,
) -> dict | None:
    """Run agent analysis on miner output and append to JSON file."""
    prompt = _build_analysis_prompt(patterns)
    response = _invoke_claude(prompt, model=model)

    if response is None:
        logger.info("Agent analysis skipped (CLI unavailable or failed)")
        return None

    try:
        analysis = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Agent response was not valid JSON — storing as text summary")
        analysis = {
            "top_recommendations": [],
            "missed_patterns": [],
            "summary": response,
        }

    output_data = json.loads(output_path.read_text())
    output_data["analysis"] = analysis
    output_path.write_text(json.dumps(output_data, indent=2) + "\n")

    logger.info("Agent analysis appended to %s", output_path)

    if "summary" in analysis:
        print(f"\nAgent Analysis Summary:\n  {analysis['summary']}")

    return analysis
