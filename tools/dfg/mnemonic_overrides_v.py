# RVV extension mnemonic overrides: LLVM tablegen def name -> QEMU disassembly mnemonic.
#
# Generated from riscv_instrs.json (HasStdExtV, non-Pseudo, Instruction superclass).
# All 375 named V-extension instructions follow the default conversion rule
# (lowercase + underscore -> dot), so no overrides are needed.
#
# The 21 InstAlias entries (vl1r.v, vneg.v, vnot.v, etc.) are excluded because
# they have InstAlias (not Instruction) superclass and are filtered out by
# gen_isadesc.py's _should_include() function.
RVV_MNEMONIC_OVERRIDES: dict[str, str] = {}
