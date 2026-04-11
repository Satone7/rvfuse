"""Hardware constraint checking for fusion candidate patterns.

Implements a three-tier verdict model:
  - feasible: no hard or soft violations
  - constrained: soft violations only (may be fusible with extra encoding space)
  - infeasible: at least one hard violation (cannot be fused)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from dfg.instruction import ISARegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-register-writing detection (vsetvli, vsetivli, vsetvl).
# Detected by opcode 0x57 (OP-V) + funct3 0x7.
_CONFIG_FUNCT3 = frozenset({0x07})


@dataclass
class ConstraintConfig:
    """Per-constraint enable/disable configuration."""

    enabled: dict[str, bool] = field(default_factory=dict)

    ALL_CONSTRAINTS: ClassVar[dict[str, tuple[str, bool, str]]] = {
        "encoding_32bit":          ("hard", True,  "指令编码限定为32位（压缩指令除外）"),
        "operand_format":          ("hard", True,  "操作数格式: 3源+1目的 或 2源+5位imm+1目的"),
        "datatype_encoding_space": ("hard", True,  "区分数据类型时需预留编码空间"),
        "no_load_store":           ("hard", False, "链中包含 load/store 指令"),
        "register_class_mismatch": ("hard", False, "指令寄存器类与模式不匹配"),
        "no_config_write":         ("hard", False, "链中包含配置寄存器写入指令"),
        "unknown_instruction":     ("hard", False, "opcode 在 ISA 注册表中不存在"),
        "too_many_destinations":   ("hard", False, "唯一目标字段数 > 1"),
        "too_many_sources":        ("hard", False, "唯一源字段数 > 3"),
        "has_immediate":           ("soft", False, "链中包含立即数操作数"),
        "missing_encoding":        ("soft", False, "指令缺少 InstructionFormat 元数据"),
        "max_nodes":             ("hard", False, "子图节点数超过硬件可实现的融合上限"),
        "no_control_flow":       ("hard", False, "子图包含控制流指令（分支/跳转）"),
    }

    @classmethod
    def defaults(cls) -> "ConstraintConfig":
        return cls(enabled={name: meta[1] for name, meta in cls.ALL_CONSTRAINTS.items()})

    @classmethod
    def from_file(cls, path: Path) -> "ConstraintConfig":
        path = Path(path)
        if not path.exists():
            return cls.defaults()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls.defaults()
        config = cls.defaults()
        constraints = data.get("constraints", {})
        for name, value in constraints.items():
            if name not in cls.ALL_CONSTRAINTS:
                logger.warning("unknown constraint name in config file: %s", name)
            elif isinstance(value, bool):
                config.enabled[name] = value
        return config

    def to_dict(self) -> dict:
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


@dataclass
class Verdict:
    """Result of checking a fusion candidate against hardware constraints.

    Attributes:
        status: One of "feasible", "constrained", or "infeasible".
        reasons: Human-readable descriptions of all violations found.
        violations: Violation identifiers from _HARD_VIOLATIONS / _SOFT_VIOLATIONS.
    """

    status: Literal["feasible", "constrained", "infeasible"]
    reasons: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


class ConstraintChecker:
    """Check whether a fusion pattern satisfies hardware-level constraints.

    The checker evaluates a pattern dict (with keys ``opcodes``,
    ``register_class``, and ``chain_registers``) against the encoding
    metadata in an :class:`ISARegistry`.

    Verdict priority:
      1. Any hard violation -> ``infeasible``
      2. Any soft violation (but no hard) -> ``constrained``
      3. No violations -> ``feasible``
    """

    def __init__(self, registry: ISARegistry, config: ConstraintConfig | None = None) -> None:
        self._registry = registry
        self._config = config if config is not None else ConstraintConfig.defaults()

    def is_enabled(self, name: str) -> bool:
        """Check whether a constraint is currently enabled."""
        return self._config.enabled.get(name, False)

    def _check_encoding_32bit(
        self, opcodes: list[str], encodings: list,
    ) -> tuple[str, str] | None:
        """Check that all instructions use 32-bit encoding (not compressed).

        Compressed instructions have opcode low 2 bits in {0x00, 0x01, 0x02}.
        Returns (violation_name, reason) tuple if violated, else None.
        """
        for opcode, enc in zip(opcodes, encodings):
            if enc is None:
                continue
            # Compressed instructions have opcode bits [1:0] in {0, 1, 2}
            # Standard 32-bit instructions have opcode bits [1:0] = 0b11 (3)
            if enc.opcode & 0x03 in (0x00, 0x01, 0x02):
                return ("encoding_32bit", f"{opcode} is a compressed instruction (opcode 0x{enc.opcode:02x})")
        return None

    def _check_operand_format(
        self,
        opcodes: list[str],
        edges: list[dict],
        flows: list,
        encodings: list,
    ) -> tuple[str, str] | None:
        """Check fused encoding operand format.

        Mode A: 3 external sources + 1 destination (no immediate)
        Mode B: 2 external sources + 5-bit immediate + 1 destination

        External operands are those not passed through the subgraph.
        Uses edges to identify internal register roles — a role is
        internal if it appears as both a dst_role and a src_role
        across the subgraph edges.
        Returns (violation_name, reason) tuple if violated, else None.
        """
        # Collect internal register roles from subgraph edges.
        # A role carried between two instructions is both a dst_role
        # (output of producer) and a src_role (input of consumer).
        dst_roles: set[str] = set()
        src_roles: set[str] = set()
        for edge in edges:
            dst_roles.add(edge.get("dst_role", ""))
            src_roles.add(edge.get("src_role", ""))
        internal_regs: set[str] = dst_roles & src_roles

        # Collect all external sources and destinations
        external_srcs: set[str] = set()
        external_dsts: set[str] = set()
        has_immediate = False

        for i, flow in enumerate(flows):
            if flow is None:
                continue
            # External sources: src_regs not in internal_regs
            for src in flow.src_regs:
                if src not in internal_regs:
                    external_srcs.add(src)
            # External destinations: dst_regs not in internal_regs
            for dst in flow.dst_regs:
                if dst not in internal_regs:
                    external_dsts.add(dst)
            # Check for immediate that fits in 5 bits (Mode B requirement)
            enc = encodings[i]
            if enc is not None and enc.has_imm and enc.imm_bits and enc.imm_bits <= 5:
                has_immediate = True

        num_external_srcs = len(external_srcs)
        num_external_dsts = len(external_dsts)

        # Mode A: exactly 3 external sources + 1 destination (no immediate)
        # Mode B: exactly 2 external sources + 5-bit immediate + 1 destination
        valid_a = (num_external_srcs == 3 and num_external_dsts == 1 and not has_immediate)
        valid_b = (num_external_srcs == 2 and num_external_dsts == 1 and has_immediate)

        if not (valid_a or valid_b):
            return ("operand_format",
                f"操作数格式不符合: 外部源={num_external_srcs}, 外部目的={num_external_dsts}, "
                f"有5位立即数={has_immediate} (要求: 3源+1目的无imm 或 2源+1目的+≤5位imm)")

        return None

    def _check_datatype_encoding_space(
        self, opcodes: list[str], encodings: list,
    ) -> tuple[str, str] | None:
        """Check if chain involves multiple data types requiring encoding space.

        Float instructions: opcode 0x53
        Vector instructions: opcode 0x57

        Returns (violation_name, reason) tuple if violated, else None.
        """
        has_float = False
        has_vector = False

        for opcode, enc in zip(opcodes, encodings):
            if enc is None:
                continue
            if enc.opcode == 0x53:
                has_float = True
            if enc.opcode == 0x57:
                has_vector = True

        if has_float and has_vector:
            return ("datatype_encoding_space", "chain mixes float (0x53) and vector (0x57) instructions, requiring extra encoding space")

        return None

    def check(self, pattern: dict) -> Verdict:
        """Evaluate *pattern* against hardware constraints.

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

        # --- Look up encodings, check for unknown instructions ----------
        encodings: list = []
        flows: list = []
        for opcode in opcodes:
            flow = self._registry.get_flow(opcode)
            if flow is None:
                if self.is_enabled("unknown_instruction"):
                    hard_violations.append("unknown_instruction")
                    reasons.append(f"unknown instruction: {opcode}")
                encodings.append(None)
                flows.append(None)
                continue
            flows.append(flow)
            enc = flow.encoding
            encodings.append(enc)
            if enc is None and self.is_enabled("missing_encoding"):
                soft_violations.append("missing_encoding")
                reasons.append(f"missing encoding metadata for: {opcode}")

        # If any instruction is unknown, return infeasible immediately.
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Check register class compatibility --------------------------------
        if self.is_enabled("register_class_mismatch"):
            for i, enc in enumerate(encodings):
                if enc is not None and enc.reg_class != reg_class:
                    hard_violations.append("register_class_mismatch")
                    reasons.append(
                        f"{opcodes[i]} reg_class '{enc.reg_class}' != pattern '{reg_class}'"
                    )
                    break  # one is enough

        # --- Check for loads/stores -------------------------------------------
        if self.is_enabled("no_load_store"):
            for i, enc in enumerate(encodings):
                if enc is not None and (enc.may_load or enc.may_store):
                    kind = "load" if enc.may_load else "store"
                    hard_violations.append("no_load_store")
                    reasons.append(f"{opcodes[i]} is a {kind} instruction")
                    break

        # --- Check for config-register writes (e.g. vsetvli) ------------------
        if self.is_enabled("no_config_write"):
            for i, flow in enumerate(flows):
                if flow is None:
                    continue
                # Check explicit config_regs field
                if flow.config_regs:
                    hard_violations.append("no_config_write")
                    reasons.append(f"{opcodes[i]} writes config registers: {flow.config_regs}")
                    break
                # Check encoding-based detection: OP-V with funct3 0x7
                enc = encodings[i]
                if enc is not None and enc.opcode == 0x57 and enc.funct3 in _CONFIG_FUNCT3:
                    hard_violations.append("no_config_write")
                    reasons.append(f"{opcodes[i]} is a vector config instruction")
                    break

        # --- New hardware-team constraint checks -------------------------------
        if self.is_enabled("encoding_32bit"):
            result = self._check_encoding_32bit(opcodes, encodings)
            if result is not None:
                hard_violations.append(result[0])
                reasons.append(result[1])

        if self.is_enabled("operand_format"):
            edges: list[dict] = pattern.get("edges", [])
            result = self._check_operand_format(opcodes, edges, flows, encodings)
            if result is not None:
                hard_violations.append(result[0])
                reasons.append(result[1])

        if self.is_enabled("datatype_encoding_space"):
            result = self._check_datatype_encoding_space(opcodes, encodings)
            if result is not None:
                hard_violations.append(result[0])
                reasons.append(result[1])

        # If hard violations so far, return early.
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Count total unique dst and src operand fields across chain --------
        if self.is_enabled("too_many_destinations") or self.is_enabled("too_many_sources"):
            all_dst_fields: set[str] = set()
            all_src_fields: set[str] = set()
            for flow in flows:
                if flow is None:
                    continue
                all_dst_fields.update(flow.dst_regs)
                all_src_fields.update(flow.src_regs)

            num_dst = len(all_dst_fields)
            num_src = len(all_src_fields)

            if num_dst > 1 and self.is_enabled("too_many_destinations"):
                hard_violations.append("too_many_destinations")
                reasons.append(f"chain has {num_dst} unique destination fields (max 1)")
            if num_src > 3 and self.is_enabled("too_many_sources"):
                hard_violations.append("too_many_sources")
                reasons.append(f"chain has {num_src} unique source fields (max 3)")

        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Check subgraph size limit ---
        if self.is_enabled("max_nodes"):
            num_nodes = len(opcodes)
            max_allowed = 4  # Conservative default for ISA extension design
            if num_nodes > max_allowed:
                hard_violations.append("max_nodes")
                reasons.append(f"subgraph has {num_nodes} nodes (max {max_allowed})")

        # --- Check for control flow instructions ---
        if self.is_enabled("no_control_flow"):
            _CONTROL_FLOW = frozenset({
                "beq", "bne", "blt", "bge", "bltu", "bgeu",
                "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
                "jal", "jalr", "j", "jr", "call", "ret",
                "bgtu",
            })
            for i, opcode in enumerate(opcodes):
                if opcode in _CONTROL_FLOW:
                    hard_violations.append("no_control_flow")
                    reasons.append(f"{opcode} is a control flow instruction")
                    break

        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Check for immediates (soft) --------------------------------------
        if self.is_enabled("has_immediate"):
            for i, enc in enumerate(encodings):
                if enc is not None and enc.has_imm:
                    soft_violations.append("has_immediate")
                    reasons.append(f"{opcodes[i]} carries an immediate operand")
                    break

        # --- Determine final verdict ------------------------------------------
        if soft_violations:
            return Verdict(status="constrained", reasons=reasons, violations=soft_violations)

        return Verdict(status="feasible", reasons=[], violations=[])
