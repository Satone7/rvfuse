"""Agent dispatcher: sends BB/DFG data to Claude Code CLI for verification
or DFG generation when the script-based builder cannot handle a block.

The agent is advisory: check() defaults to "pass" on any failure, and
generate() returns None so the caller can fall back.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

from dfg.instruction import BasicBlock, DFG, DFGEdge, DFGNode, Instruction
from dfg.output import dfg_to_json


@dataclass
class CheckResult:
    """Result of an agent check on a DFG."""

    verdict: str
    issues: list[dict] = field(default_factory=list)

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"


def _format_bb_for_prompt(bb: BasicBlock) -> str:
    """Format a BasicBlock as human-readable text for a prompt."""
    lines = [f"BB {bb.bb_id} (vaddr: 0x{bb.vaddr:x}, {len(bb.instructions)} insns):"]
    for insn in bb.instructions:
        lines.append(f"  0x{insn.address:x}: {insn.mnemonic} {insn.operands}")
    return "\n".join(lines)


def _json_to_dfg(data: dict, bb: BasicBlock) -> DFG:
    """Convert agent JSON response dict into a DFG object.

    When a node's index is within range of bb.instructions, reuses the
    existing Instruction object.  Otherwise, constructs a new one from
    the JSON data.
    """
    nodes: list[DFGNode] = []
    for n in data.get("nodes", []):
        idx = n["index"]
        if 0 <= idx < len(bb.instructions):
            insn = bb.instructions[idx]
        else:
            insn = Instruction(
                address=int(n.get("address", "0x0"), 16),
                mnemonic=n.get("mnemonic", "unknown"),
                operands=n.get("operands", ""),
                raw_line="",
            )
        nodes.append(DFGNode(instruction=insn, index=idx))

    edges: list[DFGEdge] = []
    for e in data.get("edges", []):
        edges.append(DFGEdge(
            src_index=e["src"],
            dst_index=e["dst"],
            register=e["register"],
        ))

    return DFG(bb=bb, nodes=nodes, edges=edges, source="agent")


class AgentDispatcher:
    """Dispatches BB/DFG data to the Claude Code CLI for verification or
    generation.  All errors are handled gracefully -- the agent is advisory.
    """

    def __init__(self, enabled: bool = True, model: str | None = None) -> None:
        self.enabled = enabled
        self._model = model

    # -- public API ----------------------------------------------------------

    def check(self, dfg: DFG) -> CheckResult:
        """Send BB + DFG to agent for verification.

        Returns pass if agent is disabled or unavailable.
        """
        if not self.enabled:
            return CheckResult(verdict="pass")

        bb_text = _format_bb_for_prompt(dfg.bb)
        dfg_text = dfg_to_json(dfg)
        prompt = (
            "dfg-check: Review the following basic block and its data-flow "
            "graph for correctness.  Return JSON with keys 'verdict' "
            "('pass' or 'fail') and 'issues' (list of dicts with 'msg').\n\n"
            f"{bb_text}\n\nDFG JSON:\n{dfg_text}"
        )

        raw = self._invoke_claude(prompt)
        if raw is None:
            return CheckResult(verdict="pass")

        try:
            parsed = json.loads(raw)
            return CheckResult(
                verdict=parsed.get("verdict", "pass"),
                issues=parsed.get("issues", []),
            )
        except (json.JSONDecodeError, ValueError):
            return CheckResult(verdict="pass")

    def generate(self, bb: BasicBlock) -> DFG | None:
        """Send unsupported BB to agent for DFG generation.

        Returns None if agent is disabled, unavailable, or failed.
        """
        if not self.enabled:
            return None

        bb_text = _format_bb_for_prompt(bb)
        prompt = (
            "dfg-generate: Generate a data-flow graph for the following "
            "basic block.  Return JSON with keys 'nodes' (list of "
            "{'index', 'address', 'mnemonic', 'operands'}) and 'edges' "
            "(list of {'src', 'dst', 'register'}).  Use the same address "
            "values as the input.\n\n"
            f"{bb_text}"
        )

        raw = self._invoke_claude(prompt)
        if raw is None:
            return None

        try:
            parsed = json.loads(raw)
            return _json_to_dfg(parsed, bb)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    # -- internals -----------------------------------------------------------

    def _invoke_claude(self, prompt: str) -> str | None:
        """Subprocess call to Claude Code CLI.

        Returns stdout on success, None on any failure.
        """
        cmd: list[str] = ["claude"]
        if self._model:
            cmd.extend(["--model", self._model])
        cmd.extend(["--print", prompt])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            # CLI not installed
            return None
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            return None

        return result.stdout.strip() or None
