"""Hardware constraint checking for fusion candidate patterns.

Implements a three-tier verdict model:
  - feasible: no hard or soft violations
  - constrained: soft violations only (may be fusible with extra encoding space)
  - infeasible: at least one hard violation (cannot be fused)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from dfg.instruction import ISARegistry

# ---------------------------------------------------------------------------
# Violation categories
# ---------------------------------------------------------------------------

_HARD_VIOLATIONS: frozenset[str] = frozenset({
    "no_load_store",           # chain contains a load or store instruction
    "register_class_mismatch", # instruction encoding reg_class != pattern reg_class
    "no_config_write",         # chain contains a config-register-writing instruction (e.g. vsetvli)
    "unknown_instruction",     # opcode not found in the ISA registry
    "too_many_destinations",   # total unique dst fields across chain > 1
    "too_many_sources",        # total unique src fields across chain > 3
})

_SOFT_VIOLATIONS: frozenset[str] = frozenset({
    "has_immediate",           # at least one instruction in the chain carries an immediate
    "missing_encoding",        # at least one instruction has no InstructionFormat metadata
})

# Known config-register-writing instructions (vsetvli, vsetivli, vsetvl).
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
            if name in cls.ALL_CONSTRAINTS and isinstance(value, bool):
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
        opcodes: list[str] = pattern["opcodes"]
        encodings = self._get_encodings(opcodes)
        for opcode, enc in zip(opcodes, encodings):
            if enc is None:
                continue
            # Compressed instructions have opcode bits [1:0] in {0, 1, 2}
            # Standard 32-bit instructions have opcode bits [1:0] = 0b11 (3)
            if enc.opcode & 0x03 in (0x00, 0x01, 0x02):
                return ("encoding_32bit", f"{opcode} is a compressed instruction (opcode 0x{enc.opcode:02x})")
        return None

    def _check_operand_format(self, pattern: dict) -> tuple[str, str] | None:
        """Check fused encoding operand format.

        Mode A: 3 external sources + 1 destination (no immediate)
        Mode B: 2 external sources + 5-bit immediate + 1 destination

        External operands are those not passed through the chain.
        Uses chain_registers to identify internal registers.
        Returns (violation_name, reason) tuple if violated, else None.
        """
        opcodes: list[str] = pattern["opcodes"]
        chain_registers: list[list[str]] = pattern.get("chain_registers", [])

        flows = self._get_flows(opcodes)
        encodings = self._get_encodings(opcodes)

        # Collect all internal (chain-through) registers
        internal_regs: set[str] = set()
        for chain in chain_registers:
            if len(chain) >= 2:
                # First is the destination from previous, second is source to next
                internal_regs.add(chain[0])
                internal_regs.add(chain[1])

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
            # Check for immediate
            enc = encodings[i]
            if enc is not None and enc.has_imm:
                has_immediate = True

        num_external_srcs = len(external_srcs)
        num_external_dsts = len(external_dsts)

        # Mode A: 3 external sources + 1 destination (no immediate)
        # Mode B: 2 external sources + 5-bit immediate + 1 destination
        if not has_immediate:
            # Mode A check
            if num_external_srcs > 3:
                return ("operand_format", f"Mode A requires <=3 external sources, found {num_external_srcs}")
            if num_external_dsts > 1:
                return ("operand_format", f"Mode A requires <=1 external destination, found {num_external_dsts}")
        else:
            # Mode B check
            if num_external_srcs > 2:
                return ("operand_format", f"Mode B requires <=2 external sources, found {num_external_srcs}")
            if num_external_dsts > 1:
                return ("operand_format", f"Mode B requires <=1 external destination, found {num_external_dsts}")

        return None

    def _check_datatype_encoding_space(self, pattern: dict) -> tuple[str, str] | None:
        """Check if chain involves multiple data types requiring encoding space.

        Float instructions: opcode 0x53
        Vector instructions: opcode 0x57

        Returns (violation_name, reason) tuple if violated, else None.
        """
        opcodes: list[str] = pattern["opcodes"]
        encodings = self._get_encodings(opcodes)

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
        enabled = self._config.enabled

        # --- Look up encodings, check for unknown instructions ----------
        encodings: list = []
        flows: list = []
        for opcode in opcodes:
            flow = self._registry.get_flow(opcode)
            if flow is None:
                if enabled.get("unknown_instruction", False):
                    hard_violations.append("unknown_instruction")
                    reasons.append(f"unknown instruction: {opcode}")
                encodings.append(None)
                flows.append(None)
                continue
            flows.append(flow)
            enc = flow.encoding
            encodings.append(enc)
            if enc is None and enabled.get("missing_encoding", False):
                soft_violations.append("missing_encoding")
                reasons.append(f"missing encoding metadata for: {opcode}")

        # If any instruction is unknown, return infeasible immediately.
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Check register class compatibility --------------------------------
        if enabled.get("register_class_mismatch", False):
            for i, enc in enumerate(encodings):
                if enc is not None and enc.reg_class != reg_class:
                    hard_violations.append("register_class_mismatch")
                    reasons.append(
                        f"{opcodes[i]} reg_class '{enc.reg_class}' != pattern '{reg_class}'"
                    )
                    break  # one is enough

        # --- Check for loads/stores -------------------------------------------
        if enabled.get("no_load_store", False):
            for i, enc in enumerate(encodings):
                if enc is not None and (enc.may_load or enc.may_store):
                    kind = "load" if enc.may_load else "store"
                    hard_violations.append("no_load_store")
                    reasons.append(f"{opcodes[i]} is a {kind} instruction")
                    break

        # --- Check for config-register writes (e.g. vsetvli) ------------------
        if enabled.get("no_config_write", False):
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
        if enabled.get("encoding_32bit", False):
            result = self._check_encoding_32bit(pattern)
            if result is not None:
                hard_violations.append(result[0])
                reasons.append(result[1])

        if enabled.get("operand_format", False):
            result = self._check_operand_format(pattern)
            if result is not None:
                hard_violations.append(result[0])
                reasons.append(result[1])

        if enabled.get("datatype_encoding_space", False):
            result = self._check_datatype_encoding_space(pattern)
            if result is not None:
                hard_violations.append(result[0])
                reasons.append(result[1])

        # If hard violations so far, return early.
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Count total unique dst and src operand fields across chain --------
        if enabled.get("too_many_destinations", False) or enabled.get("too_many_sources", False):
            all_dst_fields: set[str] = set()
            all_src_fields: set[str] = set()
            for flow in flows:
                if flow is None:
                    continue
                all_dst_fields.update(flow.dst_regs)
                all_src_fields.update(flow.src_regs)

            num_dst = len(all_dst_fields)
            num_src = len(all_src_fields)

            if num_dst > 1 and enabled.get("too_many_destinations", False):
                hard_violations.append("too_many_destinations")
                reasons.append(f"chain has {num_dst} unique destination fields (max 1)")
            if num_src > 3 and enabled.get("too_many_sources", False):
                hard_violations.append("too_many_sources")
                reasons.append(f"chain has {num_src} unique source fields (max 3)")

        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Check for immediates (soft) --------------------------------------
        if enabled.get("has_immediate", False):
            for i, enc in enumerate(encodings):
                if enc is not None and enc.has_imm:
                    soft_violations.append("has_immediate")
                    reasons.append(f"{opcodes[i]} carries an immediate operand")
                    break

        # --- Determine final verdict ------------------------------------------
        if soft_violations:
            return Verdict(status="constrained", reasons=reasons, violations=soft_violations)

        return Verdict(status="feasible", reasons=[], violations=[])
