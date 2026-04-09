"""RV64I base instruction set register flow definitions."""

from __future__ import annotations

from dfg.instruction import ISARegistry, InstructionFormat, RegisterFlow


def _ef(
    format_type: str,
    opcode: int,
    funct3: int | None = None,
    funct7: int | None = None,
    *,
    has_rd: bool = True,
    has_rs1: bool = True,
    has_rs2: bool = True,
    has_imm: bool = False,
    imm_bits: int = 0,
    may_load: bool = False,
    may_store: bool = False,
) -> InstructionFormat:
    """Shorthand for creating an integer-class InstructionFormat."""
    return InstructionFormat(
        format_type=format_type, opcode=opcode, funct3=funct3, funct7=funct7,
        has_rd=has_rd, has_rs1=has_rs1, has_rs2=has_rs2,
        has_imm=has_imm, imm_bits=imm_bits,
        may_load=may_load, may_store=may_store,
        reg_class="integer",
    )


# R-type: rd = rs1 op rs2  (opcode=0x33)
R_TYPE: list[tuple[str, RegisterFlow]] = [
    ("add",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x0, 0x00))),
    ("sub",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x0, 0x20))),
    ("and",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x7, 0x00))),
    ("or",   RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x6, 0x00))),
    ("xor",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x4, 0x00))),
    ("sll",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x1, 0x00))),
    ("srl",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x5, 0x00))),
    ("sra",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x5, 0x20))),
    ("slt",  RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x2, 0x00))),
    ("sltu", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x33, 0x3, 0x00))),
]

# I-type immediate: rd = rs1 op imm  (opcode=0x13)
I_TYPE_IMM: list[tuple[str, RegisterFlow]] = [
    ("addi",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
    ("andi",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x7, has_rs2=False, has_imm=True, imm_bits=12))),
    ("ori",   RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x6, has_rs2=False, has_imm=True, imm_bits=12))),
    ("xori",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x4, has_rs2=False, has_imm=True, imm_bits=12))),
    ("slti",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_rs2=False, has_imm=True, imm_bits=12))),
    ("sltiu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x3, has_rs2=False, has_imm=True, imm_bits=12))),
]

# I-type shift: rd = rs1 op shamt  (opcode=0x13, 5-bit imm)
I_TYPE_SHIFT: list[tuple[str, RegisterFlow]] = [
    ("slli", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x1, 0x00, has_rs2=False, has_imm=True, imm_bits=5))),
    ("srli", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x5, 0x00, has_rs2=False, has_imm=True, imm_bits=5))),
    ("srai", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x5, 0x20, has_rs2=False, has_imm=True, imm_bits=5))),
]

# W-suffix R-type: rd = rs1 op rs2 (32-bit)  (opcode=0x3b)
W_R_TYPE: list[tuple[str, RegisterFlow]] = [
    ("addw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x0, 0x00))),
    ("subw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x0, 0x20))),
    ("sllw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x1, 0x00))),
    ("srlw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x5, 0x00))),
    ("sraw", RegisterFlow(["rd"], ["rs1", "rs2"], encoding=_ef("R", 0x3b, 0x5, 0x20))),
]

# W-suffix I-type: rd = rs1 op imm (32-bit)  (opcode=0x1b)
W_I_TYPE: list[tuple[str, RegisterFlow]] = [
    ("addiw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
    ("slliw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x1, 0x00, has_rs2=False, has_imm=True, imm_bits=5))),
    ("srliw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x5, 0x00, has_rs2=False, has_imm=True, imm_bits=5))),
    ("sraiw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x1b, 0x5, 0x20, has_rs2=False, has_imm=True, imm_bits=5))),
]

# Loads: rd = mem[rs1 + offset]  (opcode=0x03)
LOADS: list[tuple[str, RegisterFlow]] = [
    ("lb",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x0, has_rs2=False, has_imm=True, imm_bits=12, may_load=True))),
    ("lbu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x4, has_rs2=False, has_imm=True, imm_bits=12, may_load=True))),
    ("lh",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x1, has_rs2=False, has_imm=True, imm_bits=12, may_load=True))),
    ("lhu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x5, has_rs2=False, has_imm=True, imm_bits=12, may_load=True))),
    ("lw",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x2, has_rs2=False, has_imm=True, imm_bits=12, may_load=True))),
    ("lwu", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x6, has_rs2=False, has_imm=True, imm_bits=12, may_load=True))),
    ("ld",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x03, 0x3, has_rs2=False, has_imm=True, imm_bits=12, may_load=True))),
]

# Stores: mem[rs1 + offset] = rs2  (opcode=0x23)
STORES: list[tuple[str, RegisterFlow]] = [
    ("sb", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x0, has_rd=False, has_imm=True, imm_bits=12, may_store=True))),
    ("sh", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x1, has_rd=False, has_imm=True, imm_bits=12, may_store=True))),
    ("sw", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x2, has_rd=False, has_imm=True, imm_bits=12, may_store=True))),
    ("sd", RegisterFlow([], ["rs2", "rs1"], encoding=_ef("S", 0x23, 0x3, has_rd=False, has_imm=True, imm_bits=12, may_store=True))),
]

# Branches (2 register): operands are rs1, rs2, offset
# _extract_registers maps positionally: pos0 -> "rd", pos1 -> "rs1"
# so we reference those position names as sources.  (opcode=0x63)
BRANCHES_2REG: list[tuple[str, RegisterFlow]] = [
    ("beq",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x0, has_rd=False, has_imm=True, imm_bits=12))),
    ("bne",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x1, has_rd=False, has_imm=True, imm_bits=12))),
    ("blt",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x4, has_rd=False, has_imm=True, imm_bits=12))),
    ("bge",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x5, has_rd=False, has_imm=True, imm_bits=12))),
    ("bltu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x6, has_rd=False, has_imm=True, imm_bits=12))),
    ("bgeu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x7, has_rd=False, has_imm=True, imm_bits=12))),
]

# Branches (1 register or pseudo): single reg at pos0 -> "rd"
# Two-reg pseudo-branches (bgt, ble, etc.) use pos0+pos1 -> "rd","rs1"
BRANCHES_1REG: list[tuple[str, RegisterFlow]] = [
    ("beqz", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x0, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("bnez", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x1, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("bgtz", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x4, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("blez", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x5, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("bgt",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x6, has_rd=False, has_imm=True, imm_bits=12))),
    ("ble",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x7, has_rd=False, has_imm=True, imm_bits=12))),
    ("bgtu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x7, has_rd=False, has_imm=True, imm_bits=12))),
    ("bleu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x6, has_rd=False, has_imm=True, imm_bits=12))),
]

# Jumps
JUMPS: list[tuple[str, RegisterFlow]] = [
    ("jal",  RegisterFlow(["rd"], [], encoding=_ef("J", 0x6f, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("jalr", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x67, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
]

# Pseudo-instructions
PSEUDO: list[tuple[str, RegisterFlow]] = [
    ("j",    RegisterFlow([], [], encoding=_ef("J", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("ret",  RegisterFlow([], [], encoding=_ef("I", 0x67, 0x0, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("call", RegisterFlow(["ra"], [], encoding=_ef("U", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("tail", RegisterFlow([], [], encoding=_ef("U", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("nop",  RegisterFlow([], [], encoding=_ef("I", 0x13, 0x0, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("mv",   RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
    ("li",   RegisterFlow(["rd"], [], encoding=_ef("I", 0x13, 0x0, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    ("la",   RegisterFlow(["rd"], [], encoding=_ef("U", 0x17, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("not",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x4, has_rs2=False, has_imm=True, imm_bits=12))),
    ("neg",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("R", 0x33, 0x0, 0x20))),
    ("negw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("R", 0x3b, 0x0, 0x20))),
    ("seqz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x3, has_rs2=False, has_imm=True, imm_bits=12))),
    ("snez", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x7, has_rs2=False, has_imm=True, imm_bits=12))),
    ("sltz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_rs2=False, has_imm=True, imm_bits=12))),
    ("sgtz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_rs2=False, has_imm=True, imm_bits=12))),
]

# Upper immediate
UPPER_IMM: list[tuple[str, RegisterFlow]] = [
    ("auipc", RegisterFlow(["rd"], [], encoding=_ef("U", 0x17, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    ("lui",   RegisterFlow(["rd"], [], encoding=_ef("U", 0x37, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
]

# System
SYSTEM: list[tuple[str, RegisterFlow]] = [
    ("ecall",   RegisterFlow([], [], encoding=_ef("I", 0x73, 0x0, has_rd=False, has_rs1=False, has_rs2=False))),
    ("ebreak",  RegisterFlow([], [], encoding=_ef("I", 0x73, 0x0, has_rd=False, has_rs1=False, has_rs2=False))),
    ("fence",   RegisterFlow([], [], encoding=_ef("I", 0x0f, 0x0, has_rd=False, has_rs2=False, has_imm=True, imm_bits=4))),
    ("fence.i", RegisterFlow([], [], encoding=_ef("I", 0x0f, 0x1, has_rd=False, has_rs1=False, has_rs2=False))),
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
