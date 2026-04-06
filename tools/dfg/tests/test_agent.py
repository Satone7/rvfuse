#!/usr/bin/env python3
"""Tests for agent dispatcher (agent.py).

All subprocess calls are mocked via unittest.mock.patch.
"""

import json
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from dfg.agent import AgentDispatcher, CheckResult, _format_bb_for_prompt, _json_to_dfg
from dfg.instruction import BasicBlock, DFG, DFGEdge, DFGNode, Instruction


def _sample_bb() -> BasicBlock:
    """Return a minimal basic block for testing."""
    return BasicBlock(
        bb_id=1,
        vaddr=0x1000,
        instructions=[
            Instruction(0x1000, "addi", "sp,sp,-32", "  0x1000: addi sp,sp,-32"),
            Instruction(0x1004, "sd", "ra,24(sp)", "  0x1004: sd ra,24(sp)"),
        ],
    )


def _sample_dfg(bb: BasicBlock) -> DFG:
    """Return a sample DFG for the sample BB."""
    return DFG(
        bb=bb,
        nodes=[
            DFGNode(instruction=bb.instructions[0], index=0),
            DFGNode(instruction=bb.instructions[1], index=1),
        ],
        edges=[
            DFGEdge(src_index=0, dst_index=1, register="sp"),
        ],
        source="script",
    )


# ---------------------------------------------------------------------------
# CheckResult unit tests
# ---------------------------------------------------------------------------


class TestCheckResult(unittest.TestCase):
    def test_pass_result(self):
        cr = CheckResult(verdict="pass")
        self.assertTrue(cr.is_pass)
        self.assertEqual(cr.issues, [])

    def test_fail_result(self):
        cr = CheckResult(verdict="fail", issues=[{"msg": "missing edge"}])
        self.assertFalse(cr.is_pass)
        self.assertEqual(len(cr.issues), 1)


# ---------------------------------------------------------------------------
# AgentDispatcher -- disabled mode
# ---------------------------------------------------------------------------


class TestDispatcherDisabled(unittest.TestCase):
    def test_disabled_dispatcher_skips_check(self):
        d = AgentDispatcher(enabled=False)
        result = d.check(_sample_dfg(_sample_bb()))
        self.assertTrue(result.is_pass)

    def test_disabled_dispatcher_returns_none_for_generate(self):
        d = AgentDispatcher(enabled=False)
        result = d.generate(_sample_bb())
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# AgentDispatcher.check() with mocked subprocess
# ---------------------------------------------------------------------------


class TestDispatcherCheck(unittest.TestCase):
    @patch("dfg.agent.subprocess.run")
    def test_check_passes_on_agent_approval(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"verdict": "pass", "issues": []}),
        )
        d = AgentDispatcher(enabled=True)
        result = d.check(_sample_dfg(_sample_bb()))
        self.assertTrue(result.is_pass)

    @patch("dfg.agent.subprocess.run")
    def test_check_fails_on_agent_rejection(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "verdict": "fail",
                "issues": [{"msg": "missing edge for sp"}],
            }),
        )
        d = AgentDispatcher(enabled=True)
        result = d.check(_sample_dfg(_sample_bb()))
        self.assertFalse(result.is_pass)
        self.assertEqual(len(result.issues), 1)

    @patch("dfg.agent.subprocess.run")
    def test_check_returns_pass_on_unparseable_response(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json at all")
        d = AgentDispatcher(enabled=True)
        result = d.check(_sample_dfg(_sample_bb()))
        self.assertTrue(result.is_pass)

    @patch("dfg.agent.subprocess.run")
    def test_check_returns_pass_on_cli_failure(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        d = AgentDispatcher(enabled=True)
        result = d.check(_sample_dfg(_sample_bb()))
        self.assertTrue(result.is_pass)

    @patch("dfg.agent.subprocess.run")
    def test_check_prompt_contains_bb_data(self, mock_run: MagicMock):
        """The prompt sent to the CLI must contain 'dfg-check' and an instruction mnemonic."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"verdict": "pass", "issues": []}),
        )
        d = AgentDispatcher(enabled=True)
        d.check(_sample_dfg(_sample_bb()))
        # Verify the prompt passed to subprocess.run
        call_args = mock_run.call_args
        prompt = call_args[0][0][-1]  # last element is always the prompt
        self.assertIn("dfg-check", prompt)
        self.assertIn("addi", prompt)

    @patch("dfg.agent.subprocess.run")
    def test_check_passes_model_to_cli(self, mock_run: MagicMock):
        """When model is set, --model flag is included in subprocess call."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"verdict": "pass", "issues": []}),
        )
        d = AgentDispatcher(enabled=True, model="claude-opus-4-6")
        d.check(_sample_dfg(_sample_bb()))
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        self.assertIn("--model", cmd)
        self.assertIn("claude-opus-4-6", cmd)

    @patch("dfg.agent.subprocess.run")
    def test_check_omits_model_when_unset(self, mock_run: MagicMock):
        """When model is None, --model flag is NOT in subprocess call."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"verdict": "pass", "issues": []}),
        )
        d = AgentDispatcher(enabled=True)
        d.check(_sample_dfg(_sample_bb()))
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--model", cmd)


# ---------------------------------------------------------------------------
# AgentDispatcher.generate() with mocked subprocess
# ---------------------------------------------------------------------------


class TestDispatcherGenerate(unittest.TestCase):
    @patch("dfg.agent.subprocess.run")
    def test_generate_returns_dfg_on_success(self, mock_run: MagicMock):
        bb = _sample_bb()
        dfg_json = json.dumps({
            "nodes": [
                {"index": 0, "address": "0x1000", "mnemonic": "addi", "operands": "sp,sp,-32"},
                {"index": 1, "address": "0x1004", "mnemonic": "sd", "operands": "ra,24(sp)"},
            ],
            "edges": [
                {"src": 0, "dst": 1, "register": "sp"},
            ],
        })
        mock_run.return_value = MagicMock(returncode=0, stdout=dfg_json)
        d = AgentDispatcher(enabled=True)
        result = d.generate(bb)
        self.assertIsNotNone(result)
        assert result is not None  # for type narrowing
        self.assertEqual(result.source, "agent")
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0].register, "sp")

    @patch("dfg.agent.subprocess.run")
    def test_generate_returns_none_on_failure(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        d = AgentDispatcher(enabled=True)
        result = d.generate(_sample_bb())
        self.assertIsNone(result)

    @patch("dfg.agent.subprocess.run")
    def test_generate_returns_none_on_unparseable(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")
        d = AgentDispatcher(enabled=True)
        result = d.generate(_sample_bb())
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestFormatBbForPrompt(unittest.TestCase):
    def test_format_contains_instructions(self):
        bb = _sample_bb()
        text = _format_bb_for_prompt(bb)
        self.assertIn("addi", text)
        self.assertIn("sd", text)
        self.assertIn("0x1000", text)


class TestJsonToDfg(unittest.TestCase):
    def test_json_to_dfg_with_valid_index(self):
        """When node index is valid, reuse bb.instructions."""
        bb = _sample_bb()
        data = {
            "nodes": [
                {"index": 0, "address": "0x1000", "mnemonic": "addi", "operands": "sp,sp,-32"},
                {"index": 1, "address": "0x1004", "mnemonic": "sd", "operands": "ra,24(sp)"},
            ],
            "edges": [
                {"src": 0, "dst": 1, "register": "sp"},
            ],
        }
        dfg = _json_to_dfg(data, bb)
        self.assertEqual(len(dfg.nodes), 2)
        self.assertEqual(len(dfg.edges), 1)
        # Should reuse actual instruction objects
        self.assertIs(dfg.nodes[0].instruction, bb.instructions[0])
        self.assertIs(dfg.nodes[1].instruction, bb.instructions[1])

    def test_json_to_dfg_with_extra_node(self):
        """When node index is out of range, construct Instruction from JSON."""
        bb = _sample_bb()
        data = {
            "nodes": [
                {"index": 0, "address": "0x1000", "mnemonic": "addi", "operands": "sp,sp,-32"},
                {"index": 99, "address": "0x2000", "mnemonic": "ret", "operands": ""},
            ],
            "edges": [],
        }
        dfg = _json_to_dfg(data, bb)
        self.assertEqual(len(dfg.nodes), 2)
        # Index 99 is out of range, so a new Instruction is constructed
        self.assertEqual(dfg.nodes[1].instruction.mnemonic, "ret")
        self.assertEqual(dfg.nodes[1].instruction.address, 0x2000)


if __name__ == "__main__":
    unittest.main()
