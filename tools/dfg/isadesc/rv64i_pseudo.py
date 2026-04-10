"""RV64I pseudo-instructions (hand-written).

These are assembler-level aliases that LLVM tablegen does not emit
as standalone instructions. They are needed because QEMU disassembly
output may emit these pseudo-mnemonics.

Each pseudo-instruction's RegisterFlow mirrors its underlying real
instruction's register usage.
"""

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
    """Shorthand for integer-class InstructionFormat."""
    return InstructionFormat(
        format_type=format_type, opcode=opcode, funct3=funct3, funct7=funct7,
        has_rd=has_rd, has_rs1=has_rs1, has_rs2=has_rs2,
        has_imm=has_imm, imm_bits=imm_bits,
        may_load=may_load, may_store=may_store,
        reg_class="integer",
    )


# Pseudo-instructions — assembler aliases not present in LLVM tablegen
ALL_RV64I_PSEUDO: list[tuple[str, RegisterFlow]] = [
    # mv rd, rs1  ->  addi rd, rs1, 0
    ("mv",   RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x0, has_rs2=False, has_imm=True, imm_bits=12))),
    # nop  ->  addi x0, x0, 0
    ("nop",  RegisterFlow([], [], encoding=_ef("I", 0x13, 0x0, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # ret  ->  jalr x0, ra, 0
    ("ret",  RegisterFlow([], [], encoding=_ef("I", 0x67, 0x0, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # j offset  ->  jal x0, offset
    ("j",    RegisterFlow([], [], encoding=_ef("J", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    # la rd, sym  ->  auipc rd, ...; addi rd, rd, ...
    ("la",   RegisterFlow(["rd"], [], encoding=_ef("U", 0x17, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    # li rd, imm  ->  lui/addi sequence
    ("li",   RegisterFlow(["rd"], [], encoding=_ef("I", 0x13, 0x0, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # not rd, rs  ->  xori rs, -1
    ("not",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x4, has_rs2=False, has_imm=True, imm_bits=12))),
    # neg rd, rs  ->  sub rd, x0, rs
    ("neg",  RegisterFlow(["rd"], ["rs1"], encoding=_ef("R", 0x33, 0x0, 0x20))),
    # negw rd, rs  ->  subw rd, x0, rs
    ("negw", RegisterFlow(["rd"], ["rs1"], encoding=_ef("R", 0x3b, 0x0, 0x20))),
    # seqz rd, rs  ->  sltiu rd, rs, 1
    ("seqz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x3, has_rs2=False, has_imm=True, imm_bits=12))),
    # snez rd, rs  ->  sltu rd, x0, rs  (R-type, opcode 0x33, funct3 0x3)
    ("snez", RegisterFlow(["rd"], ["rs1"], encoding=_ef("R", 0x33, 0x3, 0x00))),
    # sltz rd, rs  ->  slt rd, rs, x0
    ("sltz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_rs2=False, has_imm=True, imm_bits=12))),
    # sgtz rd, rs  ->  slt rd, x0, rs
    ("sgtz", RegisterFlow(["rd"], ["rs1"], encoding=_ef("I", 0x13, 0x2, has_rs2=False, has_imm=True, imm_bits=12))),
    # call -> auipc ra + jalr (multi-instruction)
    ("call", RegisterFlow(["ra"], [], encoding=_ef("U", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    # tail -> auipc ra + jr (multi-instruction)
    ("tail", RegisterFlow([], [], encoding=_ef("U", 0x6f, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=20))),
    # ecall -> environment call (system instruction, no registers)
    ("ecall", RegisterFlow([], [], encoding=_ef("I", 0x73, 0x0, has_rd=False, has_rs1=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # bgt rs1, rs2, offset  ->  blt rs2, rs1, offset
    # Parser assigns first two register tokens to rd/rs1 positionally
    ("bgt",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x4, has_rd=False, has_imm=True, imm_bits=12))),
    # bnez rs1, offset  ->  bne rs1, x0, offset
    # Parser assigns first register token to rd positionally
    ("bnez", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x1, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # beqz rs1, offset  ->  beq rs1, x0, 0
    ("beqz", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x0, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # bgtz rs, offset  ->  blt x0, rs, offset
    ("bgtz", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x4, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # blez rs, offset  ->  bge x0, rs, offset
    ("blez", RegisterFlow([], ["rd"], encoding=_ef("B", 0x63, 0x5, has_rd=False, has_rs2=False, has_imm=True, imm_bits=12))),
    # ble rs1, rs2, offset  ->  bge rs2, rs1, offset
    ("ble",  RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x5, has_rd=False, has_imm=True, imm_bits=12))),
    # bgtu rs1, rs2, offset  ->  bltu rs2, rs1, offset
    ("bgtu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x6, has_rd=False, has_imm=True, imm_bits=12))),
    # bleu rs1, rs2, offset  ->  bgeu rs2, rs1, offset
    ("bleu", RegisterFlow([], ["rd", "rs1"], encoding=_ef("B", 0x63, 0x7, has_rd=False, has_imm=True, imm_bits=12))),
    # ebreak -> environment breakpoint (system instruction, no registers)
    ("ebreak", RegisterFlow([], [], encoding=_ef("I", 0x73, 0x0, has_rd=False, has_rs1=False, has_rs2=False))),
    # fence pred, succ -> ordering barrier (has fencearg operands, no GPR)
    ("fence", RegisterFlow([], [], encoding=_ef("I", 0x0f, 0x0, has_rd=False, has_rs2=False, has_imm=True, imm_bits=4))),
    # fence.i -> instruction fence (no registers)
    ("fence.i", RegisterFlow([], [], encoding=_ef("I", 0x0f, 0x1, has_rd=False, has_rs1=False, has_rs2=False))),
]


def build_registry(registry: ISARegistry) -> None:
    """Register all RV64I pseudo-instructions."""
    for mnemonic, flow in ALL_RV64I_PSEUDO:
        registry.register(mnemonic, flow)
