# RVV 差距分析：vec_dot_q5_0_q8_0

## 对比：x86 AVX2 vs ARM NEON vs RISC-V RVV

### 算法概述

Q5_0 点积需要四个步骤：
1. **半字节解包（Nibble unpack）**：从 16 个打包字节中提取 32 个 4-bit 值
2. **符号扩展（Sign extension）**：根据 qh 位掩码应用第 5 位（bit=0 处减去 0x10）
3. **乘累加（Multiply-accumulate）**：计算 int8 × int8 点积
4. **缩放（Scale）**：乘以组合 delta 因子

### 指令级对比

| 步骤 | x86 AVX2 | ARM NEON | RISC-V RVV |
|------|----------|----------|------------|
| **半字节解包** | `PSHUFB` + `VPSRLDQ` + `VPAND` | `VAND` + `VSHR` | `VAND.VI` + `VSRL.VI` |
| **qh 符号扩展** | `VPSHUFB` + `VPCMPEQB` + `VPANDNOT` + `VPOR` | 查表（`table_b2b_1`）+ `VSUB` | `VLM` + `VMNAND` + `VSUB.MU` |
| **乘法** | `VPMADDUBSW` + `VPMADDWD` | `VDOT.S32`（dotprod 扩展） | `VWMUL.VV` |
| **归约** | 水平求和（`VPHADDD`） | `VADDV` | `VWREDSUM.VS` |
| **融合乘加** | `VFMADD`（融合） | `VMLA`（标量） | `FMADD`（标量） |

### 寄存器压力与吞吐量

| 指标 | x86 AVX2 | ARM NEON | RVV（VLEN=512） | RVV（VLEN=128） |
|------|----------|----------|-----------------|-----------------|
| 寄存器宽度 | 256-bit YMM | 128-bit Q | 512-bit V | 128-bit V |
| 每次迭代元素数 | 32 | 64（2 个块） | 32（1 个块） | 32（1 个块） |
| 使用的 LMUL | 不适用 | 不适用 | m1（mf2 加载） | m2 |
| 掩码寄存器 | 不适用（隐式） | 不适用（隐式） | v0（b8） | v0（b4） |
| 累加器 | 1（标量） | 2（并行向量） | 1（标量） | 1（标量） |

### 关键差距：qh 符号扩展

qh 符号扩展步骤揭示了最显著的架构差异：

**x86 AVX2**（4 条指令）：
```asm
VPSHUFB  xmm_qh, shuf_mask     ; 通过 shuffle 将位广播到字节
VPCMPEQB bytes, all_ones         ; 比较以获取 0x00/0xFF 掩码
VPANDNOT qh_mask, 0xF0           ; 取反并掩码
VPOR     result, nibbles         ; 与半字节值进行或运算
```
- `VPSHUFB` 可使用 256 位 shuffle 掩码任意排列字节——极其灵活
- qh→字节扩展是通过预计算掩码实现的单指令操作

**ARM NEON**（查表 + 减法）：
```c
// 查表：每个 qh 字节索引 → 8 字节预计算模式
tmp[i] = table_b2b_1[(qh >> (i*8)) & 0xFF];  // 0 → 0x00, 1 → 0x10
qhl = vld1q_s8((const int8_t *)(tmp + 0));
result = vsubq_s8(nibbles, qhl);               // 单次减法
```
- 使用 **256 字节查找表**（2^8 个条目）——需要内存但速度极快
- 基于减法的方法很优雅：bit=1 时执行 `value - 0x10`，bit=0 时执行 `value - 0x00`
- 然而，查表加载增加了内存压力

**RISC-V RVV**（掩码减法——最接近 ARM 的方法）：
```c
vbool8_t qh = __riscv_vlm_v_b8(x->qh, vl);     // 将 32 位加载为掩码
qh = __riscv_vmnand_mm_b8(qh, qh, vl);          // 取反所有位
vint8m1_t v0f = __riscv_vsub_vx_i8m1_mu(qh, v0c, v0c, 0x10, vl);  // 掩码减法
```
- 使用 **向量按掩码加载**（`VLM`）——将按字节加载的 qh 直接转换为掩码寄存器
- **掩码减法**（`VSUB.MU`）——仅在掩码激活（取反后）处执行减法
- 无需查找表——完全基于寄存器
- 这实际上比 x86 和 ARM 的方法都**更简洁**

### 差距评估

#### RVV 优势（对比 x86 AVX2 / ARM NEON）

1. **可配置 VLEN**：单一代码路径即可处理 128/256/512 位——x86 需要独立的 SSE/AVX2 路径；ARM 需要特性检测
2. **基于掩码的符号扩展**：`VLM` + `VMNAND` + `VSUB.MU` 共 3 条指令，优于 AVX2 的 4 条。对比 ARM 没有查找表的内存开销
3. **可扩展的元素数量**：VLEN=512 使用 m2 可处理 64 个元素——吞吐量可能达到 AVX2（32 个元素）的 2 倍
4. **自然的半字节解包**：`VAND.VI` + `VSRL.VI` 与 ARM NEON 方法完全相同，效率相当

#### RVV 劣势 / 差距

1. **缺少融合乘减与符号提取**：x86 `VPMADDUBSW` 将符号提取 + 无符号×有符号乘法 + 累加合并为一条指令。RVV 需要独立的符号扩展 + `VWMUL.VV` + `VWREDSUM.VS`（3 条指令 vs 1 条）。

2. **缺少整数点积指令**：ARM `VDOT.S32` 在一条指令中计算 `sum(dp[i]*dq[i])`（4×int8→int32）。RVV `VWMUL.VV` 产生 int16 中间结果，需要独立的归约步骤。这需要 2 条指令 vs 1 条。

3. **归约瓶颈**：`VWREDSUM.VS` 要求标量累加器，无法并行累加。ARM NEON 使用 2 个向量累加器（sumv0、sumv1）以获得更好的流水线利用率。RVV 可以通过 `vslideup` + `vadd` 维护跨迭代的向量累加器来改善。

4. **缺少字节重排（PSHUFB 等价物）**：RVV 缺少通用的字节级重排指令。这限制了 x86 上可能实现的一些优化。最接近的是 `vrgather` 指令，但需要索引计算。

5. **跨 VLEN 的掩码寄存器类型不匹配**：VLEN=128 需要 `b4` 掩码类型；VLEN>=256 需要 `b8` 掩码类型。这要求运行时 VLEN 检测和双代码路径——是一种本可避免的复杂度。

### 面向 Q5_0 点积的 RVV 扩展建议

| 差距 | 建议扩展 | 收益 |
|------|---------|------|
| 无 int8 点积 | `vdot.s8`（4×i8→i32） | 用单条指令替代 VWMUL+VWREDSUM；与 ARM VDOT 对齐 |
| 掩码乘减 | `vmsub.vx`（掩码标量减法） | 将 VMNAND+VSUB.MU 合并为单条符号扩展指令 |
| 可配置掩码类型 | 根据目标 LMUL 自动提升 b4↔b8 | 消除掩码操作的运行时 VLEN 分支 |
| 跨迭代向量累加 | `vredsum.vs` 支持向量（非标量）源 | 支持类似 ARM NEON 双累加器的多迭代向量累加 |

### 指令数对比（每个 Q5_0 块）

| 步骤 | x86 AVX2 | ARM NEON | RVV VLEN=512 | RVV VLEN=128 |
|------|----------|----------|-------------|-------------|
| 半字节解包 | 3 | 2 | 4 | 3 |
| qh 符号扩展 | 4 | 2（查表） | 3 | 3 |
| 乘法 | 1（PMADDUBSW） | 1（VDOT） | 1（VWMUL） | 1（VWMUL） |
| 归约 | 1（PMADDWD） | 1（VADDV） | 2（VWREDSUM+MV） | 2（VWREDSUM+MV） |
| 缩放 | 2（FP16+FMUL） | 2（FP16+VMLA） | 2（FP16+FMUL） | 2（FP16+FMUL） |
| **合计** | **11** | **8** | **12** | **11** |

### 吞吐量对比（每条指令处理的元素数）

| 平台 | 每次迭代元素数 | 每次迭代指令数 | 吞吐量（元素/指令） |
|------|--------------|----------------|-------------------|
| x86 AVX2 | 32 | 11 | 2.91 |
| ARM NEON | 64 | 8 | 8.00 |
| RVV VLEN=512 | 32 | 12 | 2.67 |
| RVV VLEN=128 | 32 | 11 | 2.91 |

ARM NEON 由于 `VDOT.S32`（乘法+归约仅需 1 条指令）和双块处理（每次迭代 64 个元素）而领先。RVV 通过增加 `vdot.s8` 扩展和多块迭代可以达到甚至超越此水平。

### 总结

RVV 的 `vec_dot_q5_0_q8_0` 实现在功能上正确，指令数与 x86 AVX2 相当。基于掩码的符号扩展方法（`VLM`+`VMNAND`+`VSUB.MU`）简洁优雅，避免了 ARM NEON 查找表的内存开销。主要差距在于缺少整数点积指令（`vdot.s8`），该指令可用单条操作替代 2 条指令的 `VWMUL`+`VWREDSUM` 序列——这是对 Q5_0 点积性能影响最大的单一扩展。
