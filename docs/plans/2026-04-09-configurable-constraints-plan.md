# Configurable Constraint System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every hardware constraint in `tools/fusion` independently enable/disable via CLI flags and a JSON config file, and add 3 new hardware-team constraints default-enabled.

**Architecture:** Add a `ConstraintConfig` dataclass holding per-constraint enabled flags. Inject it into `ConstraintChecker` so `check()` only runs enabled constraints. New hardware-team constraints (`encoding_32bit`, `operand_format`, `datatype_encoding_space`) are added as private methods. CLI gets `--constraints-config`, `--enable-constraint`, `--disable-constraint`, `--list-constraints` flags. A default `constraints.json` ships with the repo.

**Tech Stack:** Python 3 (stdlib `json` for config — no new dependencies), existing `argparse` CLI, existing `unittest` test framework.

**Design doc:** `docs/plans/2026-04-09-configurable-constraints-design.md`

---

### Task 1: ConstraintConfig dataclass (data structure)

**Files:**
- Modify: `tools/fusion/constraints.py:1-195` (add new class at top, before Verdict)
- Test: `tools/fusion/tests/test_constraints.py` (add new test class)

**Step 1: Write the failing test**

Add to `tools/fusion/tests/test_constraints.py` after imports:

```python
# After the imports block (line ~11)
from fusion.constraints import ConstraintConfig


class TestConstraintConfigDefaults(unittest.TestCase):
    def test_defaults_has_all_constraints(self):
        config = ConstraintConfig.defaults()
        # All 12 constraints must be present
        self.assertEqual(len(config.enabled), 12)

    def test_defaults_new_constraints_enabled(self):
        config = ConstraintConfig.defaults()
        for name in ["encoding_32bit", "operand_format", "datatype_encoding_space"]:
            self.assertTrue(config.enabled[name], f"{name} should default enabled")

    def test_defaults_old_constraints_disabled(self):
        config = ConstraintConfig.defaults()
        for name in ["no_load_store", "register_class_mismatch", "no_config_write",
                     "unknown_instruction", "too_many_destinations", "too_many_sources",
                     "has_immediate", "missing_encoding"]:
            self.assertFalse(config.enabled[name], f"{name} should default disabled")

    def test_all_constraints_metadata_complete(self):
        for name, (category, default, desc) in ConstraintConfig.ALL_CONSTRAINTS.items():
            self.assertIn(category, ("hard", "soft"))
            self.assertIsInstance(default, bool)
            self.assertIsInstance(desc, str)
            self.assertTrue(len(desc) > 0)


class TestConstraintConfigFromFile(unittest.TestCase):
    def test_from_file_missing_returns_defaults(self):
        # Non-existent file returns defaults
        config = ConstraintConfig.from_file(Path("/nonexistent/path.json"))
        self.assertEqual(config.enabled, ConstraintConfig.defaults().enabled)

    def test_from_file_partial_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"constraints": {"no_load_store": True}}, f)
            f.flush()
            config = ConstraintConfig.from_file(Path(f.name))
        self.assertTrue(config.enabled["no_load_store"])
        # Others use defaults
        self.assertFalse(config.enabled["register_class_mismatch"])
        self.assertTrue(config.enabled["encoding_32bit"])  # new constraint still enabled

    def test_from_file_invalid_json_returns_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            f.flush()
            config = ConstraintConfig.from_file(Path(f.name))
        self.assertEqual(config.enabled, ConstraintConfig.defaults().enabled)
```

Also add imports at top of test file:
```python
import tempfile
import json
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintConfigDefaults -v
```

Expected: FAIL with `ImportError: cannot import name 'ConstraintConfig' from 'fusion.constraints'`

**Step 3: Write minimal implementation**

Add to `tools/fusion/constraints.py` after imports (around line 15):

```python
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Constraint configuration
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """Per-constraint enable/disable configuration.

    Attributes:
        enabled: Dict mapping constraint name to bool (True=enabled).
    """

    enabled: dict[str, bool] = field(default_factory=dict)

    # Metadata for every constraint: name -> (category, default_enabled, description)
    ALL_CONSTRAINTS: ClassVar[dict[str, tuple[str, bool, str]]] = {
        # Hardware-team constraints (default: enabled)
        "encoding_32bit":          ("hard", True,  "指令编码限定为32位（压缩指令除外）"),
        "operand_format":          ("hard", True,  "操作数格式: 3源+1目的 或 2源+5位imm+1目的"),
        "datatype_encoding_space": ("hard", True,  "区分数据类型时需预留编码空间"),

        # Existing constraints (default: disabled)
        "no_load_store":           ("hard", False, "链中包含 load/store 指令"),
        "register_class_mismatch": ("hard", False, "指令寄存器类与模式不匹配"),
        "no_config_write":         ("hard", False, "链中包含配置寄存器写入指令"),
        "unknown_instruction":     ("hard", False, "opcode 在 ISA 注册表中不存在"),
        "too_many_destinations":   ("hard", False, "唯一目标字段数 > 1"),
        "too_many_sources":        ("hard", False, "唯一源字段数 > 3"),
        "has_immediate":           ("soft", False, "链中包含立即数操作数"),
        "missing_encoding":        ("soft", False, "指令缺少 InstructionFormat 元数据"),
    }

    @classmethod
    def defaults(cls) -> "ConstraintConfig":
        """Create config with all constraints at their default enabled state."""
        return cls(enabled={name: meta[1] for name, meta in cls.ALL_CONSTRAINTS.items()})

    @classmethod
    def from_file(cls, path: Path) -> "ConstraintConfig":
        """Load config from JSON file. Missing file or invalid JSON returns defaults."""
        path = Path(path)
        if not path.exists():
            return cls.defaults()

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls.defaults()

        # Start from defaults, override with file values
        config = cls.defaults()
        constraints = data.get("constraints", {})
        for name, value in constraints.items():
            if name in cls.ALL_CONSTRAINTS and isinstance(value, bool):
                config.enabled[name] = value
        return config

    def to_dict(self) -> dict:
        """Serialize for --list-constraints output."""
        return {
            "constraints": {
                name: {
                    "category": meta[0],
                    "default": meta[1],
                    "enabled": self.enabled.get(name, meta[1]),
                    "description": meta[2],
                }
                for name, meta in self.ALL_CONSTRAINTS.items()
            }
        }
```

Also add the `ClassVar` import to the dataclass imports at line ~12:
```python
from dataclasses import dataclass, field
from typing import ClassVar, Literal
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintConfigDefaults -v
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintConfigFromFile -v
```

Expected: PASS for both test classes.

**Step 5: Commit**

```bash
git add tools/fusion/constraints.py tools/fusion/tests/test_constraints.py
git commit -m "feat(fusion): add ConstraintConfig for per-constraint enable/disable"
```

---

### Task 2: New hardware-team constraint methods

**Files:**
- Modify: `tools/fusion/constraints.py` (add 3 new private methods to `ConstraintChecker`)
- Modify: `tools/fusion/tests/test_constraints.py` (add tests for new constraints)

**Step 1: Write the failing tests**

Add to `tools/fusion/tests/test_constraints.py`:

```python
class TestNewHardwareConstraints(unittest.TestCase):
    def setUp(self):
        self.registry = _make_registry()
        # Enable new constraints explicitly
        config = ConstraintConfig.defaults()
        config.enabled["encoding_32bit"] = True
        config.enabled["operand_format"] = True
        config.enabled["datatype_encoding_space"] = True
        self.checker = ConstraintChecker(self.registry, config=config)

    def test_encoding_32bit_passes_for_standard_instructions(self):
        # fadd.s has opcode 0x53 (low bits = 0x03, not compressed)
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        # Should NOT have encoding_32bit violation
        self.assertNotIn("encoding_32bit", verdict.violations)

    def test_encoding_32bit_detects_compressed(self):
        # Simulate a compressed instruction (opcode low bits = 0x00)
        registry = ISARegistry()
        registry.register("c.add", RegisterFlow(["rd"], ["rs1", "rs2"],
            encoding=InstructionFormat("R", 0x00, 0x0, 0x00, reg_class="integer")))
        config = ConstraintConfig.defaults()
        config.enabled["encoding_32bit"] = True
        checker = ConstraintChecker(registry, config=config)
        pattern = {"opcodes": ["c.add", "add"], "register_class": "integer",
                   "chain_registers": [["rd", "rs1"]]}
        verdict = checker.check(pattern)
        self.assertIn("encoding_32bit", verdict.violations)

    def test_operand_format_passes_3src_1dst_no_imm(self):
        # fmadd.s has 3 sources (frs1, frs2, frs3) + 1 destination (frd)
        pattern = {"opcodes": ["fmadd.s"], "register_class": "float",
                   "chain_registers": []}
        verdict = self.checker.check(pattern)
        # Single instruction chain, should pass operand_format
        self.assertNotIn("operand_format", verdict.violations)

    def test_operand_format_passes_2src_imm_1dst(self):
        # addi has 1 src + 1 imm + 1 dst -> but fused chain needs 2 src
        # Use a pattern where external src count matches format
        pattern = {"opcodes": ["addi", "add"], "register_class": "integer",
                   "chain_registers": [["rd", "rs1"]]}
        verdict = self.checker.check(pattern)
        # Should satisfy Mode B: 2 src + imm + 1 dst
        self.assertNotIn("operand_format", verdict.violations)

    def test_operand_format_detects_mismatch(self):
        # fadd.s + fmul.s: each has 2 src + 1 dst, chain passes rd->frs1
        # External: 2 src (frs2 from both) + 1 dst, no imm
        # Mode A requires 3 src, Mode B requires imm -> neither satisfied
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertIn("operand_format", verdict.violations)

    def test_datatype_encoding_space_single_type_ok(self):
        # All float instructions, single datatype
        pattern = {"opcodes": ["fadd.s", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = self.checker.check(pattern)
        self.assertNotIn("datatype_encoding_space", verdict.violations)
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestNewHardwareConstraints -v
```

Expected: FAIL with `AttributeError: 'ConstraintChecker' object has no attribute '_check_encoding_32bit'`

**Step 3: Write minimal implementation**

Add helper methods and 3 new constraint methods to `ConstraintChecker` class in `constraints.py`:

```python
# Inside ConstraintChecker class, add helper methods first:

    def _get_flows(self, opcodes: list[str]) -> list["RegisterFlow | None"]:
        """Get RegisterFlow for all opcodes."""
        return [self._registry.get_flow(op) for op in opcodes]

    def _get_encodings(self, opcodes: list[str]) -> list["InstructionFormat | None"]:
        """Get InstructionFormat for all opcodes."""
        flows = self._get_flows(opcodes)
        return [flow.encoding if flow else None for flow in flows]

    def _check_encoding_32bit(self, pattern: dict) -> tuple[str, str] | None:
        """Check that all instructions use 32-bit encoding (not compressed).

        Compressed instructions have opcode low 2 bits in {0x00, 0x01, 0x02}.
        Returns (violation_name, reason) tuple if violated, else None.
        """
        opcodes = pattern["opcodes"]
        encodings = self._get_encodings(opcodes)
        for i, enc in enumerate(encodings):
            if enc is None:
                continue
            # Compressed: opcode & 0x03 in {0, 1, 2}
            if (enc.opcode & 0x03) in (0x00, 0x01, 0x02):
                return ("encoding_32bit", f"{opcodes[i]} 是16位压缩指令，不符合32位编码要求")
        return None

    def _check_operand_format(self, pattern: dict) -> tuple[str, str] | None:
        """Check fused encoding operand format.

        Valid formats:
          - Mode A: 3 external sources + 1 destination (no immediate)
          - Mode B: 2 external sources + 5-bit immediate + 1 destination
        """
        opcodes = pattern["opcodes"]
        chain_regs = pattern.get("chain_registers", [])

        # Collect all dst/src from flows
        flows = self._get_flows(opcodes)
        all_dst: set[str] = set()
        all_src: set[str] = set()
        for flow in flows:
            if flow is None:
                continue
            all_dst.update(flow.dst_regs)
            all_src.update(flow.src_regs)

        # Chain-internal registers are passed between instructions, not external
        chain_internal_dst: set[str] = set()
        for pair in chain_regs:
            if pair:
                chain_internal_dst.add(pair[0][0])

        external_dst = all_dst - chain_internal_dst
        external_src = all_src - chain_internal_dst

        num_dst = len(external_dst)
        num_src = len(external_src)

        # Check for immediate
        encodings = self._get_encodings(opcodes)
        has_imm = any(enc.has_imm for enc in encodings if enc is not None)

        # Validate format
        valid_a = (num_src == 3 and num_dst == 1 and not has_imm)
        valid_b = (num_src == 2 and num_dst == 1 and has_imm)

        if not (valid_a or valid_b):
            return ("operand_format",
                f"操作数格式不符合: 外部源={num_src}, 外部目的={num_dst}, "
                f"有立即数={has_imm} (要求: 3源+1目的无imm 或 2源+1目的+5位imm)")
        return None

    def _check_datatype_encoding_space(self, pattern: dict) -> tuple[str, str] | None:
        """Check if chain involves multiple data types requiring encoding space."""
        opcodes = pattern["opcodes"]
        encodings = self._get_encodings(opcodes)

        datatype_indicators: list[str] = []
        for enc in encodings:
            if enc is None:
                continue
            if enc.opcode == 0x53:
                datatype_indicators.append(f"float_funct3={enc.funct3}")
            if enc.opcode == 0x57:
                datatype_indicators.append("vector_dtype")

        if len(set(datatype_indicators)) > 1:
            return ("datatype_encoding_space",
                f"链涉及多种数据类型标识 {datatype_indicators}，需预留编码空间字段")
        return None
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestNewHardwareConstraints -v
```

Expected: PASS (some tests may still fail if operand_format logic needs refinement — that's expected, we'll fix in Task 3)

**Step 5: Commit**

```bash
git add tools/fusion/constraints.py tools/fusion/tests/test_constraints.py
git commit -m "feat(fusion): add hardware-team constraint check methods"
```

---

### Task 3: Refactor ConstraintChecker.check() to use config

**Files:**
- Modify: `tools/fusion/constraints.py` (refactor `check()` method and `__init__`)
- Modify: `tools/fusion/tests/test_constraints.py` (update existing tests to use config)

**Step 1: Write the failing test**

Add test class verifying config-driven enable/disable:

```python
class TestConstraintCheckerConfigDriven(unittest.TestCase):
    """Verify that ConstraintChecker respects ConstraintConfig enabled flags."""

    def setUp(self):
        self.registry = _make_registry()

    def test_no_constraints_enabled_all_feasible(self):
        """With all constraints disabled, everything is feasible."""
        config = ConstraintConfig(enabled={name: False for name in ConstraintConfig.ALL_CONSTRAINTS})
        checker = ConstraintChecker(self.registry, config=config)
        # Even a load-store chain should be feasible when constraint is disabled
        pattern = {"opcodes": ["flw", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = checker.check(pattern)
        self.assertEqual(verdict.status, "feasible")

    def test_selectively_enable_no_load_store(self):
        """Only no_load_store enabled -> load chain is infeasible."""
        config = ConstraintConfig(enabled={name: False for name in ConstraintConfig.ALL_CONSTRAINTS})
        config.enabled["no_load_store"] = True
        checker = ConstraintChecker(self.registry, config=config)
        pattern = {"opcodes": ["flw", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]]}
        verdict = checker.check(pattern)
        self.assertEqual(verdict.status, "infeasible")
        self.assertIn("no_load_store", verdict.violations)

    def test_selectively_enable_has_immediate(self):
        """Only has_immediate enabled -> addi chain is constrained."""
        config = ConstraintConfig(enabled={name: False for name in ConstraintConfig.ALL_CONSTRAINTS})
        config.enabled["has_immediate"] = True
        checker = ConstraintChecker(self.registry, config=config)
        pattern = {"opcodes": ["addi", "add"], "register_class": "integer",
                   "chain_registers": [["rd", "rs1"]]}
        verdict = checker.check(pattern)
        self.assertEqual(verdict.status, "constrained")
        self.assertIn("has_immediate", verdict.violations)

    def test_is_enabled_helper(self):
        config = ConstraintConfig(enabled={"no_load_store": True, "has_immediate": False})
        checker = ConstraintChecker(self.registry, config=config)
        self.assertTrue(checker.is_enabled("no_load_store"))
        self.assertFalse(checker.is_enabled("has_immediate"))
        self.assertFalse(checker.is_enabled("nonexistent"))  # unknown defaults False
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerConfigDriven -v
```

Expected: FAIL — current `check()` runs all constraints regardless of config.

**Step 3: Write minimal implementation**

Replace `ConstraintChecker.__init__` and `check()` in `constraints.py`:

```python
class ConstraintChecker:
    """Check whether a fusion pattern satisfies hardware-level constraints.

    The checker evaluates a pattern dict against the constraints enabled in
    the provided :class:`ConstraintConfig`. Default config enables only the
    hardware-team constraints.
    """

    def __init__(
        self,
        registry: ISARegistry,
        config: ConstraintConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or ConstraintConfig.defaults()

    def is_enabled(self, name: str) -> bool:
        """Check whether a constraint is currently enabled."""
        return self._config.enabled.get(name, False)

    def check(self, pattern: dict) -> Verdict:
        """Evaluate *pattern* against enabled constraints.

        Args:
            pattern: Dict with keys ``opcodes`` (list[str]),
                ``register_class`` (str), and ``chain_registers``
                (list[list[str]]).

        Returns:
            A :class:`Verdict` with status and violation details.
        """
        opcodes: list[str] = pattern["opcodes"]
        reg_class: str = pattern["register_class"]

        reasons: list[str] = []
        hard_violations: list[str] = []
        soft_violations: list[str] = []

        # --- 1. Hardware-team constraints (new) -------------------------------

        if self.is_enabled("encoding_32bit"):
            result = self._check_encoding_32bit(pattern)
            if result:
                hard_violations.append(result[0])
                reasons.append(result[1])

        if self.is_enabled("operand_format"):
            result = self._check_operand_format(pattern)
            if result:
                hard_violations.append(result[0])
                reasons.append(result[1])

        if self.is_enabled("datatype_encoding_space"):
            result = self._check_datatype_encoding_space(pattern)
            if result:
                hard_violations.append(result[0])
                reasons.append(result[1])

        # --- 2. Existing constraints ------------------------------------------

        # Look up encodings (shared by several checks)
        encodings: list = []
        flows: list = []
        for opcode in opcodes:
            flow = self._registry.get_flow(opcode)
            if flow is None:
                encodings.append(None)
                flows.append(None)
                continue
            flows.append(flow)
            encodings.append(flow.encoding)

        # unknown_instruction
        if self.is_enabled("unknown_instruction"):
            for i, opcode in enumerate(opcodes):
                if flows[i] is None:
                    hard_violations.append("unknown_instruction")
                    reasons.append(f"unknown instruction: {opcode}")
                    break

        # If any instruction is unknown, return infeasible immediately.
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # missing_encoding (soft)
        if self.is_enabled("missing_encoding"):
            for i, enc in enumerate(encodings):
                if enc is None:
                    soft_violations.append("missing_encoding")
                    reasons.append(f"missing encoding metadata for: {opcodes[i]}")

        # register_class_mismatch
        if self.is_enabled("register_class_mismatch"):
            for i, enc in enumerate(encodings):
                if enc is not None and enc.reg_class != reg_class:
                    hard_violations.append("register_class_mismatch")
                    reasons.append(
                        f"{opcodes[i]} reg_class '{enc.reg_class}' != pattern '{reg_class}'"
                    )
                    break

        # no_load_store
        if self.is_enabled("no_load_store"):
            for i, enc in enumerate(encodings):
                if enc is not None and (enc.may_load or enc.may_store):
                    kind = "load" if enc.may_load else "store"
                    hard_violations.append("no_load_store")
                    reasons.append(f"{opcodes[i]} is a {kind} instruction")
                    break

        # no_config_write
        if self.is_enabled("no_config_write"):
            for i, flow in enumerate(flows):
                if flow is None:
                    continue
                if flow.config_regs:
                    hard_violations.append("no_config_write")
                    reasons.append(f"{opcodes[i]} writes config registers: {flow.config_regs}")
                    break
                enc = encodings[i]
                if enc is not None and enc.opcode == 0x57 and enc.funct3 in _CONFIG_FUNCT3:
                    hard_violations.append("no_config_write")
                    reasons.append(f"{opcodes[i]} is a vector config instruction")
                    break

        # Early exit on hard violations
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # too_many_destinations
        if self.is_enabled("too_many_destinations"):
            all_dst_fields: set[str] = set()
            for flow in flows:
                if flow is None:
                    continue
                all_dst_fields.update(flow.dst_regs)
            num_dst = len(all_dst_fields)
            if num_dst > 1:
                hard_violations.append("too_many_destinations")
                reasons.append(f"chain has {num_dst} unique destination fields (max 1)")

        # too_many_sources
        if self.is_enabled("too_many_sources"):
            all_src_fields: set[str] = set()
            for flow in flows:
                if flow is None:
                    continue
                all_src_fields.update(flow.src_regs)
            num_src = len(all_src_fields)
            if num_src > 3:
                hard_violations.append("too_many_sources")
                reasons.append(f"chain has {num_src} unique source fields (max 3)")

        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # has_immediate (soft)
        if self.is_enabled("has_immediate"):
            for i, enc in enumerate(encodings):
                if enc is not None and enc.has_imm:
                    soft_violations.append("has_immediate")
                    reasons.append(f"{opcodes[i]} carries an immediate operand")
                    break

        # --- 3. Determine final verdict --------------------------------------
        if soft_violations:
            return Verdict(status="constrained", reasons=reasons, violations=soft_violations)

        return Verdict(status="feasible", reasons=[], violations=[])
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerConfigDriven -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/fusion/constraints.py tools/fusion/tests/test_constraints.py
git commit -m "refactor(fusion): ConstraintChecker.check() uses ConstraintConfig"
```

---

### Task 4: Update existing tests to explicitly enable tested constraints

**Files:**
- Modify: `tools/fusion/tests/test_constraints.py:66-149` (TestConstraintCheckerFeasible, TestConstraintCheckerInfeasible, TestConstraintCheckerConstrained)

**Step 1: Identify failing tests**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerFeasible -v
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerInfeasible -v
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerConstrained -v
```

Expected: FAIL — old constraints now default disabled, tests expect them enabled.

**Step 2: Fix tests by explicitly enabling constraints**

Replace `setUp` methods in each test class:

```python
class TestConstraintCheckerFeasible(unittest.TestCase):
    def setUp(self):
        registry = _make_registry()
        # Enable old constraints for backward-compatible tests
        config = ConstraintConfig.defaults()
        config.enabled["no_load_store"] = True
        config.enabled["register_class_mismatch"] = True
        config.enabled["no_config_write"] = True
        config.enabled["unknown_instruction"] = True
        self.checker = ConstraintChecker(registry, config=config)

class TestConstraintCheckerInfeasible(unittest.TestCase):
    def setUp(self):
        registry = _make_registry()
        config = ConstraintConfig.defaults()
        config.enabled["no_load_store"] = True
        config.enabled["register_class_mismatch"] = True
        config.enabled["no_config_write"] = True
        config.enabled["unknown_instruction"] = True
        self.checker = ConstraintChecker(registry, config=config)

class TestConstraintCheckerConstrained(unittest.TestCase):
    def setUp(self):
        registry = _make_registry()
        config = ConstraintConfig.defaults()
        config.enabled["has_immediate"] = True
        config.enabled["missing_encoding"] = True
        self.checker = ConstraintChecker(registry, config=config)
```

**Step 3: Run tests to verify they pass**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerFeasible -v
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerInfeasible -v
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_constraints.py::TestConstraintCheckerConstrained -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add tools/fusion/tests/test_constraints.py
git commit -m "test(fusion): update existing tests to explicitly enable constraints"
```

---

### Task 5: Update Scorer to accept ConstraintConfig

**Files:**
- Modify: `tools/fusion/scorer.py:47-60` (Scorer.__init__)
- Modify: `tools/fusion/tests/test_scorer.py` (update Scorer instantiations)

**Step 1: Write failing test**

Add to `tools/fusion/tests/test_scorer.py`:

```python
class TestScorerWithConfig(unittest.TestCase):
    def test_scorer_accepts_config(self):
        registry = _make_registry()
        config = ConstraintConfig.defaults()
        scorer = Scorer(registry, max_frequency=100000, config=config)
        self.assertIsNotNone(scorer._checker)
        self.assertEqual(scorer._checker._config, config)

    def test_scorer_default_config(self):
        registry = _make_registry()
        scorer = Scorer(registry, max_frequency=100000)
        # Should use ConstraintConfig.defaults()
        self.assertIsNotNone(scorer._checker._config)
        self.assertTrue(scorer._checker.is_enabled("encoding_32bit"))

    def test_scorer_custom_config_affects_check(self):
        registry = _make_registry()
        config = ConstraintConfig(enabled={name: False for name in ConstraintConfig.ALL_CONSTRAINTS})
        scorer = Scorer(registry, max_frequency=100000, config=config)
        # Pattern that would be infeasible with default config
        pattern = {"opcodes": ["flw", "fmul.s"], "register_class": "float",
                   "chain_registers": [["frd", "frs1"]], "total_frequency": 50000,
                   "occurrence_count": 10}
        result = scorer.score_pattern(pattern)
        # With all constraints disabled, should have hw_score > 0
        self.assertGreater(result["score_breakdown"]["hw_score"], 0.0)
```

Also add import:
```python
from fusion.constraints import ConstraintConfig
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_scorer.py::TestScorerWithConfig -v
```

Expected: FAIL — Scorer.__init__ doesn't accept config parameter.

**Step 3: Write minimal implementation**

Modify `tools/fusion/scorer.py`:

```python
# Update imports
from fusion.constraints import ConstraintChecker, ConstraintConfig, Verdict

# Update Scorer.__init__ (lines 47-60)
class Scorer:
    def __init__(
        self,
        registry: ISARegistry,
        max_frequency: int = 1,
        weights: dict[str, float] | None = None,
        config: ConstraintConfig | None = None,  # NEW
    ) -> None:
        self._registry = registry
        self._max_frequency = max(1, max_frequency)
        self._weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self._checker = ConstraintChecker(registry, config=config)  # NEW: pass config
```

Also update the `score()` function (line ~210) to accept config:

```python
def score(
    catalog_path: str | Path,
    registry: ISARegistry,
    output_path: str | Path | None = None,
    top: int | None = None,
    min_score: float = 0.0,
    weights: dict[str, float] | None = None,
    feasibility_only: bool = False,
    config: ConstraintConfig | None = None,  # NEW
) -> list[dict[str, Any]]:
    ...
    # In feasibility_only branch (line ~246):
    if feasibility_only:
        checker = ConstraintChecker(registry, config=config)  # pass config
        ...
    # In normal scoring branch (line ~272):
    scorer = Scorer(registry, max_frequency=max_freq, weights=weights, config=config)  # pass config
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_scorer.py::TestScorerWithConfig -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/fusion/scorer.py tools/fusion/tests/test_scorer.py
git commit -m "feat(fusion): Scorer accepts ConstraintConfig parameter"
```

---

### Task 6: Update integration tests

**Files:**
- Modify: `tools/fusion/tests/test_integration.py`

**Step 1: Run integration tests to see failures**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_integration.py -v
```

Expected: Some tests may fail because new constraints are enabled by default, causing different verdict outcomes.

**Step 2: Fix integration tests**

The key issue: `test_feasibility_only_mode` expects `flw + fmul.s` to be infeasible. With new default config:
- `encoding_32bit` enabled -> passes (standard 32-bit)
- `operand_format` enabled -> check this
- `datatype_encoding_space` enabled -> passes (both floats)

If operand_format makes it infeasible, that's correct. Otherwise, the test needs adjustment.

Update `test_feasibility_only_mode`:

```python
def test_feasibility_only_mode(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog_path = Path(tmpdir) / "patterns.json"
        output_path = Path(tmpdir) / "candidates.json"
        catalog_path.write_text(json.dumps(self.catalog))

        results = run_score(catalog_path=catalog_path, registry=self.registry,
                            output_path=output_path, feasibility_only=True)
        statuses = [r["hardware"]["status"] for r in results]
        # With new constraints enabled, check actual results
        # flw+fmul.s may be feasible or constrained depending on operand_format
        self.assertIn("infeasible", statuses)  # adjust if needed based on operand_format logic
```

If test fails, adjust assertion to match new behavior or explicitly disable new constraints:

```python
def test_feasibility_only_mode_old_constraints(self):
    """Test with only old constraints enabled (backward compatibility)."""
    config = ConstraintConfig.defaults()
    config.enabled["encoding_32bit"] = False
    config.enabled["operand_format"] = False
    config.enabled["datatype_encoding_space"] = False
    config.enabled["no_load_store"] = True  # enable old constraint

    with tempfile.TemporaryDirectory() as tmpdir:
        ...
        results = run_score(..., config=config)  # pass config
        statuses = [r["hardware"]["status"] for r in results]
        self.assertIn("infeasible", statuses)
```

**Step 3: Run tests to verify**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/test_integration.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add tools/fusion/tests/test_integration.py
git commit -m "test(fusion): update integration tests for configurable constraints"
```

---

### Task 7: Add CLI parameters for constraint configuration

**Files:**
- Modify: `tools/fusion/__main__.py:48-144` (parse_args and main)

**Step 1: Write failing test (manual CLI test)**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1 && python -m tools.fusion --list-constraints
```

Expected: FAIL with `error: unrecognized arguments: --list-constraints`

**Step 2: Add CLI arguments**

Modify `tools/fusion/__main__.py` parse_args function:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.fusion",
        description="Discover fusible instruction patterns from DFG output.",
    )
    parser.add_argument("command", choices=["discover", "score", "validate", "list-constraints"])
    # ... existing args ...

    # NEW: Constraint configuration args
    parser.add_argument(
        "--constraints-config",
        type=Path,
        default=None,
        help="Path to JSON file with constraint enable/disable config",
    )
    parser.add_argument(
        "--enable-constraint",
        action="append",
        default=None,
        help="Enable a specific constraint (repeatable)",
    )
    parser.add_argument(
        "--disable-constraint",
        action="append",
        default=None,
        help="Disable a specific constraint (repeatable)",
    )
    parser.add_argument(
        "--list-constraints",
        action="store_true",
        default=False,
        help="List all constraints with their default status and descriptions",
    )
    return parser.parse_args(argv)
```

Also add a helper function to build config from args:

```python
def _build_constraint_config(args: argparse.Namespace) -> ConstraintConfig:
    """Build ConstraintConfig from CLI args.

    Priority: --enable/disable flags > --constraints-config file > defaults
    """
    config = ConstraintConfig.defaults()

    # 1. Load from config file if specified
    if args.constraints_config:
        config = ConstraintConfig.from_file(args.constraints_config)

    # 2. Apply CLI enable/disable overrides
    for name in (args.enable_constraint or []):
        if name in ConstraintConfig.ALL_CONSTRAINTS:
            config.enabled[name] = True
        else:
            logging.warning("Unknown constraint '%s' in --enable-constraint", name)

    for name in (args.disable_constraint or []):
        if name in ConstraintConfig.ALL_CONSTRAINTS:
            config.enabled[name] = False
        else:
            logging.warning("Unknown constraint '%s' in --disable-constraint", name)

    return config
```

Add import for ConstraintConfig:
```python
from fusion.constraints import ConstraintConfig
```

**Step 3: Handle list-constraints command**

Add to main():
```python
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Handle --list-constraints
    if args.list_constraints or args.command == "list-constraints":
        config = _build_constraint_config(args)
        print("\nConstraint Configuration:")
        print("-" * 60)
        for name, meta in ConstraintConfig.ALL_CONSTRAINTS.items():
            category, default, desc = meta
            status = "ON" if config.enabled[name] else "OFF"
            print(f"{name:30} {category:5} {status:3}  {desc}")
        print("-" * 60)
        return

    # ... rest of main ...
```

**Step 4: Pass config to score/discover commands**

Update score command in main():
```python
if args.command == "score":
    config = _build_constraint_config(args)
    ...
    candidates = run_score(
        catalog_path=args.catalog, registry=registry, output_path=args.output,
        top=args.top, min_score=args.min_score, weights=weights,
        feasibility_only=args.feasibility_only,
        config=config,  # NEW
    )
```

**Step 5: Test CLI**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1 && python -m tools.fusion --list-constraints
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1 && python -m tools.fusion --list-constraints --enable-constraint no_load_store
```

Expected: Output listing all constraints with their status.

**Step 6: Commit**

```bash
git add tools/fusion/__main__.py
git commit -m "feat(fusion): add CLI args for constraint configuration"
```

---

### Task 8: Create default constraints.json config file

**Files:**
- Create: `tools/fusion/constraints.json`

**Step 1: Create config file**

```json
{
  "constraints": {
    "encoding_32bit": true,
    "operand_format": true,
    "datatype_encoding_space": true,
    "no_load_store": false,
    "register_class_mismatch": false,
    "no_config_write": false,
    "unknown_instruction": false,
    "too_many_destinations": false,
    "too_many_sources": false,
    "has_immediate": false,
    "missing_encoding": false
  }
}
```

**Step 2: Test loading the file**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -c "
from fusion.constraints import ConstraintConfig
from pathlib import Path
config = ConstraintConfig.from_file(Path('fusion/constraints.json'))
print('encoding_32bit:', config.enabled['encoding_32bit'])
print('no_load_store:', config.enabled['no_load_store'])
assert config.enabled['encoding_32bit'] == True
assert config.enabled['no_load_store'] == False
print('OK: default config loaded correctly')
"
```

Expected: `OK: default config loaded correctly`

**Step 3: Commit**

```bash
git add tools/fusion/constraints.json
git commit -m "feat(fusion): add default constraints.json config file"
```

---

### Task 9: Run full test suite and fix any regressions

**Files:**
- Any files that need fixes discovered during full test run

**Step 1: Run full test suite**

Run:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/fusion-v1.1/tools && python -m pytest fusion/tests/ -v
```

Expected: All tests PASS. If any fail, investigate and fix.

**Step 2: Fix any regressions**

Common issues to watch for:
- Tests that create `Scorer()` without config may get different behavior
- Tests that create `ConstraintChecker(registry)` now get defaults config (new constraints enabled)
- Integration tests that rely on specific verdict outcomes

**Step 3: Commit**

```bash
git add -A
git commit -m "fix(fusion): fix test regressions from configurable constraints"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-04-09-configurable-constraints-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?

