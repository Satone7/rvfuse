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

    def __init__(self, registry: ISARegistry) -> None:
        self._registry = registry

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
                hard_violations.append("unknown_instruction")
                reasons.append(f"unknown instruction: {opcode}")
                encodings.append(None)
                flows.append(None)
                continue
            flows.append(flow)
            enc = flow.encoding
            encodings.append(enc)
            if enc is None:
                soft_violations.append("missing_encoding")
                reasons.append(f"missing encoding metadata for: {opcode}")

        # If any instruction is unknown, return infeasible immediately.
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Check register class compatibility --------------------------------
        for i, enc in enumerate(encodings):
            if enc is not None and enc.reg_class != reg_class:
                hard_violations.append("register_class_mismatch")
                reasons.append(
                    f"{opcodes[i]} reg_class '{enc.reg_class}' != pattern '{reg_class}'"
                )
                break  # one is enough

        # --- Check for loads/stores -------------------------------------------
        for i, enc in enumerate(encodings):
            if enc is not None and (enc.may_load or enc.may_store):
                kind = "load" if enc.may_load else "store"
                hard_violations.append("no_load_store")
                reasons.append(f"{opcodes[i]} is a {kind} instruction")
                break

        # --- Check for config-register writes (e.g. vsetvli) ------------------
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

        # If hard violations so far, return early.
        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Count total unique dst and src operand fields across chain --------
        all_dst_fields: set[str] = set()
        all_src_fields: set[str] = set()
        for flow in flows:
            if flow is None:
                continue
            all_dst_fields.update(flow.dst_regs)
            all_src_fields.update(flow.src_regs)

        # The chain register dependencies count as "shared" -- the output of
        # one instruction feeds the input of the next, so those don't add
        # extra destination/source requirements for the fused encoding.
        # However, any *additional* dst or src fields beyond the chain do.

        # Count how many dst positions are actual destinations (not chain-through).
        # All dst_regs from the chain are part of the fused instruction's
        # output set.  Similarly, all src_regs that are NOT fed by another
        # instruction in the chain are external inputs.
        #
        # Simplified model: count the unique dst fields across all instructions
        # and unique src fields.  >1 dst or >3 src is infeasible.
        num_dst = len(all_dst_fields)
        num_src = len(all_src_fields)

        if num_dst > 1:
            hard_violations.append("too_many_destinations")
            reasons.append(f"chain has {num_dst} unique destination fields (max 1)")
        if num_src > 3:
            hard_violations.append("too_many_sources")
            reasons.append(f"chain has {num_src} unique source fields (max 3)")

        if hard_violations:
            return Verdict(status="infeasible", reasons=reasons, violations=hard_violations)

        # --- Check for immediates (soft) --------------------------------------
        has_immediate = False
        for i, enc in enumerate(encodings):
            if enc is not None and enc.has_imm:
                has_immediate = True
                soft_violations.append("has_immediate")
                reasons.append(f"{opcodes[i]} carries an immediate operand")
                break

        # --- Determine final verdict ------------------------------------------
        if soft_violations:
            return Verdict(status="constrained", reasons=reasons, violations=soft_violations)

        return Verdict(status="feasible", reasons=[], violations=[])
