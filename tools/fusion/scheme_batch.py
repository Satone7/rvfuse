"""Batch fusion scheme generation via parallel Claude CLI invocations.

Automates F3 (Fusion Scheme Specification): generates encoding schemes for
each candidate in parallel, validates them, and produces a summary report.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dfg.instruction import ISARegistry

logger = logging.getLogger("fusion.scheme_batch")

DEFAULT_MODEL = "opus"
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT = 300

# -- Prompt templates --------------------------------------------------------

_SCHEME_SYSTEM_PROMPT = """\
You are a RISC-V instruction fusion encoding architect. Given a ranked fusion \
candidate, generate a complete fusion encoding scheme in Markdown.

## Process (5 steps)

1. **Parse the candidate**: Extract instruction opcodes, register class, RAW \
dependency chain, score breakdown, and hardware feasibility status.

2. **Identify encoding space**: Find an unused opcode+funct3+funct7 slot \
compatible with the candidate's register class.

3. **Draft the encoding**: Assign opcode, funct3, funct7, rd, rs1, rs2 fields \
that preserve the original computation semantics.

4. **Generate the Markdown scheme**: Fill the template below.

5. **Self-check**: Verify no encoding conflicts with existing RISC-V instructions.

## Encoding Space Guidance

- Integer: opcode 0x0B (custom-0), 0x33 (OP), 0x3B (OP-32)
- Float: opcode 0x43 (MADD), 0x47 (MSUB), 0x4B (NMSUB), 0x4F (NMADD), 0x53 (OP-FP)
- Vector: opcode 0x57 (OP-V)

## Handling Infeasible Candidates

If the candidate hardware status is "infeasible":
- Acknowledge the infeasibility and its root cause
- Still propose an encoding scheme as if the constraint could be relaxed
- Include a "Decomposition" section suggesting how to break the pattern into \
multiple feasible fused instructions
- Note what hardware changes would make the pattern feasible

## Critical Output Requirement

You MUST include an HTML comment with the proposed encoding values in this \
exact format (for automated validation):

    <!-- encoding: opcode=0xNN funct3=0xN funct7=0xNN reg_class=CLASS -->

Place this comment immediately after the "## Encoding Layout" section heading.

## Markdown Template

# Fusion Scheme: <MNEMONIC>

## Candidate Summary
- Source pattern: <INSTRUCTION_SEQUENCE>
- Frequency: <COUNT>
- Score: <TOTAL> (freq=<FREQ>, tight=<TIGHT>, hw=<HW>)
- Register class: <CLASS>
- Hardware status: <FEASIBLE/CONSTRAINED/INFEASIBLE>

## Encoding Layout

<!-- encoding: opcode=0xNN funct3=0xN funct7=0xNN reg_class=CLASS -->

| Field   | Bits  | Value | Description         |
|---------|-------|-------|---------------------|
| opcode  | [6-7] | 0x??  | <OPCODE_REGION>     |
| funct3  | [3]   | 0x??  | <SUB_OP>            |
| funct7  | [7]   | 0x??  | <VARIANT>           |
| rd      | [5]   | dst   | Destination         |
| rs1     | [5]   | src1  | Source operand 1    |
| rs2     | [5]   | src2  | Source operand 2    |

### Encoding Justification
<WHY_THIS_ENCODING_SPACE_WAS_CHOSEN>

## Instruction Semantics

### Assembly Syntax
`<MNEMONIC> rd, rs1, rs2  # <DESCRIPTION>`

### Operation
rd = rs1 <OP> rs2  // <SEMANTIC_DESCRIPTION>

### Register Flow
- Reads: rs1, rs2
- Writes: rd
- Preserves: <ANY_IMPLICITLY_PRESERVED>

### Latency Estimate
- Original: <N> cycles (<BREAKDOWN>)
- Fused: <M> cycles (estimated)
- Saving: <N-M> cycles per occurrence

## Decomposition (for infeasible patterns)
<SUGGESTED_BREAKDOWN_INTO_MULTIPLE_FEASIBLE_FUSED_INSTRUCTIONS>

## Constraint Compliance

| Constraint           | Status     | Notes                  |
|---------------------|------------|------------------------|
| Encoding space       | PASS/FAIL  | <VALIDATOR_CHECK>      |
| Register class       | PASS       | <CLASS_MATCH>          |
| Operand count        | PASS/FAIL  | <WITHIN_2SRC+1DST>     |
| No config write      | PASS       | <FOR_V:_NO_VSETVLI>    |
| No load/store        | PASS       | Memory-free pattern    |

## Validation Log
- <SELF_CHECK_RESULTS>
"""

_SUMMARY_PROMPT_TEMPLATE = """\
You are reviewing a batch of RISC-V fusion encoding schemes generated for \
instruction fusion research. Analyze ALL {count} schemes below and produce \
a consolidated summary report in Markdown.

## Required Sections

1. **Overview Table**: One row per candidate with columns: \
Rank, Pattern, Register Class, Proposed Encoding, Feasibility, Key Notes

2. **Cross-Candidate Analysis**:
   - Encoding conflicts between proposals (same opcode+funct3+funct7)
   - Shared constraints across candidates
   - Pattern frequency distribution

3. **Recommendations**:
   - Which candidates to prioritize for Phase 3 simulation
   - Which need multi-instruction decomposition
   - Suggested next steps

4. **Overall Assessment**: 2-3 sentence summary of the candidate set quality

## Individual Schemes

{scheme_texts}
"""


def _build_scheme_prompt(candidate: dict, index: int) -> str:
    """Build a complete prompt for generating a fusion scheme for one candidate."""
    candidate_json = json.dumps(candidate, indent=2, ensure_ascii=False)
    return (
        f"{_SCHEME_SYSTEM_PROMPT}\n\n"
        f"---\n\n"
        f"## Candidate #{index:03d}\n\n"
        f"```json\n{candidate_json}\n```\n\n"
        f"Generate the fusion scheme now."
    )


def _build_summary_prompt(schemes: list[dict]) -> str:
    """Build a prompt synthesizing all individual schemes into a summary."""
    parts: list[str] = []
    for s in schemes:
        header = f"### Candidate #{s['index']:03d}"
        content = s.get("content", s.get("error", "No content"))
        validation = ""
        if s.get("validation"):
            v = s["validation"]
            status = "PASS" if v.passed else "FAIL"
            validation = f"\n**Validation: {status}**"
            if v.conflicts:
                validation += f" — {', '.join(v.conflicts)}"
        parts.append(f"{header}{validation}\n\n{content}")

    return _SUMMARY_PROMPT_TEMPLATE.format(
        count=len(schemes),
        scheme_texts="\n\n---\n\n".join(parts),
    )


def _invoke_claude(prompt: str, model: str, timeout: int = DEFAULT_TIMEOUT) -> str | None:
    """Call claude -p --model <model> with the given prompt."""
    cmd: list[str] = ["claude", "-p", "--model", model, "--dangerously-skip-permissions"]

    logger.debug("Invoking claude -p --model %s (prompt %d chars)", model, len(prompt))

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        logger.error("Claude CLI not found — skipping")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after %ds", timeout)
        return None

    if result.returncode != 0:
        logger.error("Claude CLI failed (rc=%d): %s", result.returncode, result.stderr[:500])
        return None

    return result.stdout.strip() or None


def _parse_encoding_from_scheme(markdown_text: str) -> dict | None:
    """Extract opcode/funct3/funct7/reg_class from the HTML comment block."""
    pattern = r"<!--\s*encoding:\s*opcode=(0x[0-9a-fA-F]+)\s+funct3=(0x[0-9a-fA-F]+)\s+funct7=(0x[0-9a-fA-F]+)\s+reg_class=(\w+)\s*-->"
    match = re.search(pattern, markdown_text)
    if not match:
        return None
    return {
        "opcode": int(match.group(1), 16),
        "funct3": int(match.group(2), 16),
        "funct7": int(match.group(3), 16),
        "reg_class": match.group(4),
    }


def _validate_and_annotate(
    scheme_text: str,
    registry: ISARegistry,
) -> tuple[object | None, str]:
    """Parse encoding from scheme, validate, annotate Validation Log section."""
    from fusion.scheme_validator import validate_encoding

    enc = _parse_encoding_from_scheme(scheme_text)
    if enc is None:
        return None, scheme_text

    result = validate_encoding(
        opcode=enc["opcode"],
        funct3=enc["funct3"],
        funct7=enc["funct7"],
        reg_class=enc["reg_class"],
        registry=registry,
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "PASS" if result.passed else "FAIL"
    log_entry = f"\n- {ts} {status} — opcode=0x{enc['opcode']:02X} funct3=0x{enc['funct3']:X} funct7=0x{enc['funct7']:02X} reg_class={enc['reg_class']}"
    if result.conflicts:
        log_entry += f"\n  Conflicts: {'; '.join(result.conflicts)}"
    if result.warnings:
        log_entry += f"\n  Warnings: {'; '.join(result.warnings)}"
    if result.suggested_alternatives:
        log_entry += f"\n  Alternatives: {'; '.join(result.suggested_alternatives)}"

    # Append to Validation Log section
    if "## Validation Log" in scheme_text:
        scheme_text = scheme_text.replace(
            "## Validation Log\n",
            f"## Validation Log\n{log_entry}\n",
            1,
        )
    else:
        scheme_text += f"\n## Validation Log\n{log_entry}\n"

    return result, scheme_text


def _process_one_candidate(
    candidate: dict,
    index: int,
    model: str,
    timeout: int,
    output_dir: Path,
    registry: ISARegistry | None,
) -> dict:
    """Process a single candidate: generate scheme, save, validate."""
    prompt = _build_scheme_prompt(candidate, index)
    content = _invoke_claude(prompt, model, timeout)

    if content is None:
        error_md = (
            f"# Fusion Scheme: Candidate #{index:03d}\n\n"
            f"## Error\nScheme generation failed (Claude CLI unavailable or timed out).\n"
        )
        out_path = output_dir / f"{index:03d}.md"
        out_path.write_text(error_md)
        return {"index": index, "path": out_path, "content": error_md,
                "validation": None, "error": "Claude CLI failed"}

    # Validate and annotate
    validation_result = None
    if registry is not None:
        validation_result, content = _validate_and_annotate(content, registry)

    out_path = output_dir / f"{index:03d}.md"
    out_path.write_text(content)

    return {
        "index": index,
        "path": out_path,
        "content": content,
        "validation": validation_result,
        "error": None,
    }


def run_scheme_batch(
    candidates_json: Path,
    output_dir: Path,
    top: int = 20,
    model: str = DEFAULT_MODEL,
    workers: int = DEFAULT_WORKERS,
    timeout: int = DEFAULT_TIMEOUT,
    registry: ISARegistry | None = None,
) -> Path:
    """Main entry point: load candidates, process in parallel, generate summary.

    Args:
        candidates_json: Path to fusion_candidates.json.
        output_dir: Directory for per-candidate .md files and summary.
        top: Number of top candidates to process.
        model: Claude model name.
        workers: Number of parallel Claude processes.
        timeout: Timeout per Claude invocation (seconds).
        registry: ISARegistry for validation (optional).

    Returns:
        Path to summary.md.
    """
    data = json.loads(candidates_json.read_text())
    candidates = data.get("candidates", data if isinstance(data, list) else [])
    candidates = candidates[:top]

    if not candidates:
        logger.warning("No candidates found in %s", candidates_json)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.md"
        summary_path.write_text("# Fusion Scheme Summary\n\nNo candidates to process.\n")
        return summary_path

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Processing %d candidates with %d workers (model=%s)",
        len(candidates), workers, model,
    )
    print(f"fusion.scheme: Generating schemes for {len(candidates)} candidates "
          f"(workers={workers}, model={model})...")

    # Parallel execution
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx, cand in enumerate(candidates):
            future = pool.submit(
                _process_one_candidate,
                candidate=cand,
                index=idx,
                model=model,
                timeout=timeout,
                output_dir=output_dir,
                registry=registry,
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["error"]:
                    logger.warning("Candidate %03d: %s", idx, result["error"])
                else:
                    logger.info("Candidate %03d: done (%s)", idx, result["path"].name)
            except Exception as exc:
                logger.error("Candidate %03d failed: %s", idx, exc)
                results.append({"index": idx, "error": str(exc),
                                "content": None, "validation": None})

    # Sort by index for deterministic output
    results.sort(key=lambda r: r["index"])

    # Count successes
    success_count = sum(1 for r in results if r["error"] is None)
    print(f"fusion.scheme: {success_count}/{len(results)} schemes generated successfully")

    # Generate summary
    logger.info("Generating summary report...")
    summary_prompt = _build_summary_prompt(results)
    summary_text = _invoke_claude(summary_prompt, model, timeout=timeout * 2)

    if summary_text is None:
        # Fallback: generate minimal summary from results
        lines = ["# Fusion Scheme Summary\n"]
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        lines.append(f"Candidates processed: {len(results)} "
                     f"(success: {success_count}, failed: {len(results) - success_count})\n")
        lines.append("\n## Per-Candidate Status\n")
        for r in results:
            status = "OK" if r["error"] is None else f"FAILED ({r['error']})"
            lines.append(f"- #{r['index']:03d}: {status}")
        summary_text = "\n".join(lines)

    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_text)
    logger.info("Summary written to %s", summary_path)

    print(f"\nFusion Scheme Generation Complete")
    print(f"  Schemes: {success_count}/{len(results)}")
    print(f"  Output:  {output_dir}/")
    print(f"  Summary: {summary_path}")

    return summary_path
