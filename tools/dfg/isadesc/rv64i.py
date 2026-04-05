"""RV64I base instruction set register flow definitions."""

from __future__ import annotations

from dfg.instruction import ISARegistry, RegisterFlow

# R-type: rd = rs1 op rs2
R_TYPE: list[tuple[str, RegisterFlow]] = [
    ("add",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("sub",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("and",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("or",   RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("xor",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("sll",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("srl",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("sra",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("slt",  RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("sltu", RegisterFlow(["rd"], ["rs1", "rs2"])),
]

# I-type immediate: rd = rs1 op imm
I_TYPE_IMM: list[tuple[str, RegisterFlow]] = [
    ("addi",  RegisterFlow(["rd"], ["rs1"])),
    ("andi",  RegisterFlow(["rd"], ["rs1"])),
    ("ori",   RegisterFlow(["rd"], ["rs1"])),
    ("xori",  RegisterFlow(["rd"], ["rs1"])),
    ("slti",  RegisterFlow(["rd"], ["rs1"])),
    ("sltiu", RegisterFlow(["rd"], ["rs1"])),
]

# I-type shift: rd = rs1 op shamt
I_TYPE_SHIFT: list[tuple[str, RegisterFlow]] = [
    ("slli", RegisterFlow(["rd"], ["rs1"])),
    ("srli", RegisterFlow(["rd"], ["rs1"])),
    ("srai", RegisterFlow(["rd"], ["rs1"])),
]

# W-suffix R-type: rd = rs1 op rs2 (32-bit)
W_R_TYPE: list[tuple[str, RegisterFlow]] = [
    ("addw", RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("subw", RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("sllw", RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("srlw", RegisterFlow(["rd"], ["rs1", "rs2"])),
    ("sraw", RegisterFlow(["rd"], ["rs1", "rs2"])),
]

# W-suffix I-type: rd = rs1 op imm (32-bit)
W_I_TYPE: list[tuple[str, RegisterFlow]] = [
    ("addiw", RegisterFlow(["rd"], ["rs1"])),
    ("slliw", RegisterFlow(["rd"], ["rs1"])),
    ("srliw", RegisterFlow(["rd"], ["rs1"])),
    ("sraiw", RegisterFlow(["rd"], ["rs1"])),
]

# Loads: rd = mem[rs1 + offset]
LOADS: list[tuple[str, RegisterFlow]] = [
    ("lb",  RegisterFlow(["rd"], ["rs1"])),
    ("lbu", RegisterFlow(["rd"], ["rs1"])),
    ("lh",  RegisterFlow(["rd"], ["rs1"])),
    ("lhu", RegisterFlow(["rd"], ["rs1"])),
    ("lw",  RegisterFlow(["rd"], ["rs1"])),
    ("lwu", RegisterFlow(["rd"], ["rs1"])),
    ("ld",  RegisterFlow(["rd"], ["rs1"])),
]

# Stores: mem[rs1 + offset] = rs2
STORES: list[tuple[str, RegisterFlow]] = [
    ("sb", RegisterFlow([], ["rs2", "rs1"])),
    ("sh", RegisterFlow([], ["rs2", "rs1"])),
    ("sw", RegisterFlow([], ["rs2", "rs1"])),
    ("sd", RegisterFlow([], ["rs2", "rs1"])),
]

# Branches (2 register): operands are rs1, rs2, offset
# _extract_registers maps positionally: pos0 -> "rd", pos1 -> "rs1"
# so we reference those position names as sources.
BRANCHES_2REG: list[tuple[str, RegisterFlow]] = [
    ("beq",  RegisterFlow([], ["rd", "rs1"])),
    ("bne",  RegisterFlow([], ["rd", "rs1"])),
    ("blt",  RegisterFlow([], ["rd", "rs1"])),
    ("bge",  RegisterFlow([], ["rd", "rs1"])),
    ("bltu", RegisterFlow([], ["rd", "rs1"])),
    ("bgeu", RegisterFlow([], ["rd", "rs1"])),
]

# Branches (1 register or pseudo): single reg at pos0 -> "rd"
# Two-reg pseudo-branches (bgt, ble, etc.) use pos0+pos1 -> "rd","rs1"
BRANCHES_1REG: list[tuple[str, RegisterFlow]] = [
    ("beqz", RegisterFlow([], ["rd"])),
    ("bnez", RegisterFlow([], ["rd"])),
    ("bgtz", RegisterFlow([], ["rd"])),
    ("blez", RegisterFlow([], ["rd"])),
    ("bgt",  RegisterFlow([], ["rd", "rs1"])),
    ("ble",  RegisterFlow([], ["rd", "rs1"])),
    ("bgtu", RegisterFlow([], ["rd", "rs1"])),
    ("bleu", RegisterFlow([], ["rd", "rs1"])),
]

# Jumps
JUMPS: list[tuple[str, RegisterFlow]] = [
    ("jal",  RegisterFlow(["rd"], [])),
    ("jalr", RegisterFlow(["rd"], ["rs1"])),
]

# Pseudo-instructions
PSEUDO: list[tuple[str, RegisterFlow]] = [
    ("j",    RegisterFlow([], [])),
    ("ret",  RegisterFlow([], [])),
    ("call", RegisterFlow(["ra"], [])),
    ("tail", RegisterFlow([], [])),
    ("nop",  RegisterFlow([], [])),
    ("mv",   RegisterFlow(["rd"], ["rs1"])),
    ("li",   RegisterFlow(["rd"], [])),
    ("la",   RegisterFlow(["rd"], [])),
    ("not",  RegisterFlow(["rd"], ["rs1"])),
    ("neg",  RegisterFlow(["rd"], ["rs1"])),
    ("negw", RegisterFlow(["rd"], ["rs1"])),
    ("seqz", RegisterFlow(["rd"], ["rs1"])),
    ("snez", RegisterFlow(["rd"], ["rs1"])),
    ("sltz", RegisterFlow(["rd"], ["rs1"])),
    ("sgtz", RegisterFlow(["rd"], ["rs1"])),
]

# Upper immediate
UPPER_IMM: list[tuple[str, RegisterFlow]] = [
    ("auipc", RegisterFlow(["rd"], [])),
    ("lui",   RegisterFlow(["rd"], [])),
]

# System
SYSTEM: list[tuple[str, RegisterFlow]] = [
    ("ecall",   RegisterFlow([], [])),
    ("ebreak",  RegisterFlow([], [])),
    ("fence",   RegisterFlow([], [])),
    ("fence.i", RegisterFlow([], [])),
]

# Combined list of all RV64I instructions
ALL_RV64I: list[tuple[str, RegisterFlow]] = (
    R_TYPE
    + I_TYPE_IMM
    + I_TYPE_SHIFT
    + W_R_TYPE
    + W_I_TYPE
    + LOADS
    + STORES
    + BRANCHES_2REG
    + BRANCHES_1REG
    + JUMPS
    + PSEUDO
    + UPPER_IMM
    + SYSTEM
)


def build_registry(registry: ISARegistry) -> None:
    """Register all RV64I instructions into the given ISA registry."""
    registry.load_extension("I", ALL_RV64I)
