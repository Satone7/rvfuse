# Configurable Constraint System Design

**Date**: 2026-04-09
**Status**: Approved
**Author**: Claude (brainstorming skill)

---

## Overview

Modify `tools/fusion/constraints.py` to support per-constraint enable/disable configuration. Hardware team constraints default enabled, existing constraints default disabled.

---

## Requirements

1. Each constraint independently configurable (enable/disable)
2. Hardware team constraints (from spec) default enabled:
   - `encoding_32bit`: 32-bit instruction encoding (except compressed)
   - `operand_format`: 3-src+1-dst OR 2-src+5bit-imm+1-dst
   - `datatype_encoding_space`: Reserve encoding space for data types
3. Existing constraints default disabled (smooth transition)
4. CLI + config file mechanism (CLI overrides config)
5. Similar old/new constraints can be reused/adjusted

---

## Design

### 1. ConstraintConfig Data Structure

```python
@dataclass
class ConstraintConfig:
    enabled: dict[str, bool]

    ALL_CONSTRAINTS: ClassVar[dict[str, tuple[str, bool, str]]] = {
        # Hardware team (default: enabled)
        "encoding_32bit":          ("hard", True,  "指令编码限定为32位（压缩指令除外）"),
        "operand_format":          ("hard", True,  "操作数格式: 3源+1目的 或 2源+5位imm+1目的"),
        "datatype_encoding_space": ("hard", True,  "区分数据类型时需预留编码空间"),

        # Existing (default: disabled)
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
        return cls(enabled={k: v[1] for k, v in cls.ALL_CONSTRAINTS.items()})

    @classmethod
    def from_file(cls, path: Path) -> "ConstraintConfig":
        # Load from JSON, missing keys use defaults

    def to_dict(self) -> dict:
        # Serialize for --list-constraints output
```

### 2. New Constraint Check Logic

#### encoding_32bit

Check all instructions use 32-bit encoding (exclude compressed instructions).

```python
def _check_encoding_32bit(self, pattern) -> tuple[str, str] | None:
    for opcode in pattern["opcodes"]:
        enc = self._registry.get_flow(opcode)?.encoding
        if enc and (enc.opcode & 0x03) in (0x00, 0x01, 0x02):
            return ("encoding_32bit", f"{opcode} is 16-bit compressed")
    return None
```

#### operand_format

Fused encoding must satisfy one of:
- Mode A: 3 source + 1 destination (no immediate)
- Mode B: 2 source + 5-bit immediate + 1 destination

```python
def _check_operand_format(self, pattern) -> tuple[str, str] | None:
    # Count external operands (exclude chain-internal registers)
    external_dst, external_src, has_imm = self._analyze_external_operands(pattern)

    valid_a = (external_src == 3 and external_dst == 1 and not has_imm)
    valid_b = (external_src == 2 and external_dst == 1 and has_imm)

    if not (valid_a or valid_b):
        return ("operand_format", f"format mismatch: src={external_src}, dst={external_dst}, imm={has_imm}")
    return None
```

#### datatype_encoding_space

Check if chain involves multiple data types, flag encoding space requirement.

```python
def _check_datatype_encoding_space(self, pattern) -> tuple[str, str] | None:
    # Detect data type indicators from encoding (funct3 for floats, vector dtype)
    datatypes = self._collect_datatype_indicators(pattern)
    if len(set(datatypes)) > 1:
        return ("datatype_encoding_space", f"multiple datatypes {datatypes}, need encoding space")
    return None
```

### 3. CLI + Config File

#### Config file (JSON)

Default path: `tools/fusion/constraints.json`

```json
{
  "constraints": {
    "encoding_32bit": true,
    "operand_format": true,
    "no_load_store": false
  }
}
```

#### CLI parameters

```
--constraints-config <path>   Specify config file
--enable-constraint <name>    Enable constraint (repeatable)
--disable-constraint <name>   Disable constraint (repeatable)
--list-constraints            List all constraints with defaults
```

Priority: CLI flags > config file > code defaults

### 4. ConstraintChecker Modification

```python
class ConstraintChecker:
    def __init__(self, registry, config=None):
        self._registry = registry
        self._config = config or ConstraintConfig.defaults()

    def check(self, pattern) -> Verdict:
        # Execute only enabled constraints
        if self.is_enabled("encoding_32bit"):
            result = self._check_encoding_32bit(pattern)
            # ...

        if self.is_enabled("no_load_store"):
            # existing logic
```

### 5. Backward Compatibility

| Change | Impact |
|--------|--------|
| Default behavior | Old tests pass if explicitly enable tested constraint |
| `Scorer` | New `config` parameter, optional |
| `__main__.py` | Build config from args, pass to Scorer |
| Tests | Explicitly enable constraints per test case |

---

## Files Modified

| File | Change |
|------|--------|
| `constraints.py` | Add `ConstraintConfig`, split `check()` into methods |
| `scorer.py` | Accept `config` parameter |
| `__main__.py` | Add CLI args, build config |
| `test_constraints.py` | Explicitly enable tested constraints |

---

## Next Step

Invoke `writing-plans` skill to create implementation plan.