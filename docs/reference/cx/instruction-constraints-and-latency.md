# 指令约束及执行延迟

## 1. 指令约束

1. 除压缩指令外，其他指令编码限定为32位
2. 指令中的操作数：最多可以是3个源操作数（rs1、rs2、rs3）+1个目的操作数（rd），或者2个源操作数（rs1、rs2）+1个不超过5位的立即数（imm）+1个目的操作数（rd）
3. 指令如果区分不同数据类型，例如int8、fp16等等，需要考虑保留相应的编码空间

## 2. 指令执行延迟

> **符号说明：**
> - `x - x` 表示不定周期，是一个范围
> - `x + x` 表示拆分为微码后对应的周期
> - `(x、x)` 表示不同的 SEW

---

### 2.1 RVG 指令执行延迟

| 指令名称 | 执行延迟 |
|----------|----------|
| add | 1 |
| addi | 1 |
| addiw | 1 |
| addw | 1 |
| amoadd.d | 5 |
| amoadd.w | 5 |
| amoand.d | 5 |
| amoand.w | 5 |
| amomax.d | 5 |
| amomax.w | 5 |
| amomaxu.d | 5 |
| amomaxu.w | 5 |
| amomin.d | 5 |
| amomin.w | 5 |
| amominu.d | 5 |
| amominu.w | 5 |
| amoor.d | 5 |
| amoor.w | 5 |
| amoswap.d | 5 |
| amoswap.w | 5 |
| amoxor.d | 5 |
| amoxor.w | 5 |
| and | 1 |
| andi | 1 |
| auipc | 2 |
| beq | 2 |
| bge | 2 |
| bgeu | 2 |
| blt | 2 |
| bltu | 2 |
| bne | 2 |
| csrrc | 阻塞执行不定周期 |
| csrrci | 阻塞执行不定周期 |
| csrrs | 阻塞执行不定周期 |
| csrrsi | 阻塞执行不定周期 |
| csrrw | 阻塞执行不定周期 |
| csrrwi | 阻塞执行不定周期 |
| div | 15 |
| divu | 15 |
| divuw | 13 |
| divw | 13 |
| ebreak | 1 |
| ecall | 1 |
| fadd.d | 3 |
| fadd.s | 3 |
| fclass.d | 1+1 |
| fclass.s | 1+1 |
| fcvt.d.l | 3 |
| fcvt.d.lu | 3 |
| fcvt.d.s | 3 |
| fcvt.d.w | 3 |
| fcvt.d.wu | 3 |
| fcvt.l.d | 3+1 |
| fcvt.l.s | 3+1 |
| fcvt.lu.d | 3+1 |
| fcvt.lu.s | 3+1 |
| fcvt.s.d | 3 |
| fcvt.s.l | 3+1 |
| fcvt.s.lu | 3+1 |
| fcvt.s.w | 3+1 |
| fcvt.s.wu | 3+1 |
| fcvt.w.d | 3+1 |
| fcvt.w.s | 3+1 |
| fcvt.wu.d | 3+1 |
| fcvt.wu.s | 3+1 |
| fdiv.d | 4-17 |
| fdiv.s | 4-10 |
| fence | 阻塞执行不定周期 |
| fence.i | 阻塞执行不定周期 |
| feq.d | 1+1 |
| feq.s | 1+1 |
| fld | 4 |
| fle.d | 3+1 |
| fle.s | 3+1 |
| flt.d | 3+1 |
| flt.s | 3 |
| flw | 4 |
| fmadd.d | 5 |
| fmadd.s | 5 |
| fmax.d | 3 |
| fmax.s | 3 |
| fmin.d | 3 |
| fmin.s | 3 |
| fmsub.d | 5 |
| fmsub.s | 5 |
| fmul.d | 4 |
| fmul.s | 4 |
| fmv.d.x | 3 |
| fmv.w.x | 3 |
| fmv.x.d | 1+1 |
| fmv.x.w | 1+1 |
| fnmadd.d | 5 |
| fnmadd.s | 5 |
| fnmsub.d | 5 |
| fnmsub.s | 5 |
| fsd | 4 |
| fsgnj.d | 3 |
| fsgnj.s | 3 |
| fsgnjn.d | 3 |
| fsgnjn.s | 3 |
| fsgnjx.d | 3 |
| fsgnjx.s | 3 |
| fsqrt.d | 4-17 |
| fsqrt.s | 4-10 |
| fsub.d | 3 |
| fsub.s | 3 |
| fsw | 4 |
| jal | 4 |
| jalr | 4 |
| lb | 3 |
| lbu | 3 |
| ld | 3 |
| lh | 3 |
| lhu | 3 |
| lr.d | 3 |
| lr.w | 3 |
| lui | 3 |
| lw | 3 |
| lwu | 3 |
| mul | 5 |
| mulh | 5 |
| mulhsu | 5 |
| mulhu | 5 |
| mulw | 5 |
| or | 1 |
| ori | 1 |
| rem | 15 |
| remu | 15 |
| remuw | 13 |
| remw | 13 |
| sb | 4 |
| sc.d | 4 |
| sc.w | 4 |
| sd | 4 |
| sh | 4 |
| sll | 1 |
| slli | 1 |
| slliw | 1 |
| sllw | 1 |
| slt | 1 |
| slti | 1 |
| sltiu | 1 |
| sltu | 1 |
| sra | 1 |
| srai | 1 |
| sraiw | 1 |
| sraw | 1 |
| srl | 1 |
| srli | 1 |
| srliw | 1 |
| srlw | 1 |
| sub | 1 |
| subw | 1 |
| sw | 4 |
| xor | 1 |
| xori | 1 |

---

### 2.2 RVV 指令执行延迟

#### 2.2.1 配置指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vsetvli.v | 2 |
| vsetivli.v | 2 |
| vsetvl.v | 2 |

#### 2.2.2 向量加载/存储

| 指令名称 | 执行延迟 |
|----------|----------|
| vle8.v | 3 |
| vle16.v | 3 |
| vle32.v | 3 |
| vle64.v | 3 |
| vse8.v | 4 |
| vse16.v | 4 |
| vse32.v | 4 |
| vse64.v | 4 |
| vlm.v | 3 |
| vsm.v | 3 |

#### 2.2.3 向量跨步加载/存储

| 指令名称 | 执行延迟 |
|----------|----------|
| vlse8.v | 3 |
| vlse16.v | 3 |
| vlse32.v | 3 |
| vlse64.v | 3 |
| vsse8.v | 4 |
| vsse16.v | 4 |
| vsse32.v | 4 |
| vsse64.v | 4 |

#### 2.2.4 向量索引加载/存储（有序/无序）

| 指令名称 | 执行延迟 |
|----------|----------|
| vloxei8.v | 拆分微码，与向量长度、编组相关 |
| vloxei16.v | 拆分微码，与向量长度、编组相关 |
| vloxei32.v | 拆分微码，与向量长度、编组相关 |
| vloxei64.v | 拆分微码，与向量长度、编组相关 |
| vsoxei8.v | 拆分微码，与向量长度、编组相关 |
| vsoxei16.v | 拆分微码，与向量长度、编组相关 |
| vsoxei32.v | 拆分微码，与向量长度、编组相关 |
| vsoxei64.v | 拆分微码，与向量长度、编组相关 |
| vluxei8.v | 拆分微码，与向量长度、编组相关 |
| vluxei16.v | 拆分微码，与向量长度、编组相关 |
| vluxei32.v | 拆分微码，与向量长度、编组相关 |
| vluxei64.v | 拆分微码，与向量长度、编组相关 |
| vsuxei8.v | 拆分微码，与向量长度、编组相关 |
| vsuxei16.v | 拆分微码，与向量长度、编组相关 |
| vsuxei32.v | 拆分微码，与向量长度、编组相关 |
| vsuxei64.v | 拆分微码，与向量长度、编组相关 |

#### 2.2.5 向量容错加载

| 指令名称 | 执行延迟 |
|----------|----------|
| vle8ff.v | 拆分微码，与向量长度、编组相关 |
| vle16ff.v | 拆分微码，与向量长度、编组相关 |
| vle32ff.v | 拆分微码，与向量长度、编组相关 |
| vle64ff.v | 拆分微码，与向量长度、编组相关 |

#### 2.2.6 向量分段加载/存储（vlseg/vsseg）

| 指令名称 | 执行延迟 |
|----------|----------|
| vlseg[2-8]e[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |
| vsseg[2-8]e[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |
| vlseg[2-8]e[8/16/32/64]ff.v | 拆分微码，与向量长度、编组相关 |

#### 2.2.7 向量分段跨步加载/存储（vlsseg/vssseg）

| 指令名称 | 执行延迟 |
|----------|----------|
| vlsseg[2-8]e[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |
| vssseg[2-8]e[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |

#### 2.2.8 向量分段索引加载/存储（vluxseg/vsuxseg/vloxseg/vsoxseg）

| 指令名称 | 执行延迟 |
|----------|----------|
| vluxseg[2-8]ei[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |
| vsuxseg[2-8]ei[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |
| vloxseg[2-8]ei[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |
| vsoxseg[2-8]ei[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |

#### 2.2.9 向量单位跨步加载/存储

| 指令名称 | 执行延迟 |
|----------|----------|
| vl[1/2/4/8]re[8/16/32/64].v | 拆分微码，与向量长度、编组相关 |
| vs[1/2/4/8]r.v | 拆分微码，与向量长度、编组相关 |

#### 2.2.10 向量算术指令 — 整数加减法

| 指令名称 | 执行延迟 |
|----------|----------|
| vadd.vv | 3 |
| vadd.vx | 4+3 |
| vadd.vi | 3 |
| vsub.vv | 3 |
| vsub.vx | 4+3 |
| vrsub.vx | 4+3 |
| vrsub.vi | 3 |
| vwaddu.vv | 4 |
| vwaddu.vx | 4+4 |
| vwadd.vv | 4 |
| vwadd.vx | 4+4 |
| vwsubu.vv | 4 |
| vwsubu.vx | 4+4 |
| vwsub.vv | 4 |
| vwsub.vx | 4+4 |
| vwaddu.wv | 4 |
| vwaddu.wx | 4+4 |
| vwadd.wv | 4 |
| vwadd.wx | 4+4 |
| vwsubu.wv | 4 |
| vwsubu.wx | 4+4 |
| vwsub.wv | 4 |
| vwsub.wx | 4+4 |
| vzext.vf2 | 3 |
| vsext.vf2 | 3 |
| vzext.vf4 | 3 |
| vsext.vf4 | 3 |
| vzext.vf8 | 3 |
| vsext.vf8 | 3 |

#### 2.2.11 向量进位指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vadc.vvm | 3 |
| vadc.vxm | 3 |
| vadc.vim | 3 |
| vmadc.vvm | 3 |
| vmadc.vxm | 4+3 |
| vmadc.vim | 3 |
| vmadc.vv | 3 |
| vmadc.vx | 4+3 |
| vmadc.vi | 3 |
| vsbc.vvm | 3 |
| vsbc.vxm | 4+3 |
| vmsbc.vvm | 3 |
| vmsbc.vxm | 4+3 |
| vmsbc.vv | 3 |
| vmsbc.vx | 4+3 |

#### 2.2.12 向量位运算

| 指令名称 | 执行延迟 |
|----------|----------|
| vand.vv | 3 |
| vand.vx | 4+3 |
| vand.vi | 3 |
| vor.vv | 3 |
| vor.vx | 4+3 |
| vor.vi | 3 |
| vxor.vv | 3 |
| vxor.vx | 4+3 |
| vxor.vi | 3 |
| vsll.vv | 3 |
| vsll.vx | 4+3 |
| vsll.vi | 3 |
| vsrl.vv | 3 |
| vsrl.vx | 4+3 |
| vsrl.vi | 3 |
| vsra.vv | 3 |
| vsra.vx | 4+3 |
| vsra.vi | 3 |
| vnsrl.wv | 4 |
| vnsrl.wx | 4+3 |
| vnsrl.wi | 4 |
| vnsra.wv | 4 |
| vnsra.wx | 4+3 |
| vnsra.wi | 4 |

#### 2.2.13 向量比较指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vmseq.vv | 3 |
| vmseq.vx | 4+3 |
| vmseq.vi | 3 |
| vmsne.vv | 3 |
| vmsne.vx | 4+3 |
| vmsne.vi | 3 |
| vmsltu.vv | 3 |
| vmsltu.vx | 4+3 |
| vmslt.vv | 3 |
| vmslt.vx | 4+3 |
| vmsleu.vv | 3 |
| vmsleu.vx | 4+3 |
| vmsleu.vi | 3 |
| vmsle.vv | 3 |
| vmsle.vx | 4+3 |
| vmsle.vi | 3 |
| vmsgtu.vx | 4+3 |
| vmsgtu.vi | 3 |
| vmsgt.vx | 4+3 |
| vmsgt.vi | 3 |
| vminu.vv | 3 |
| vminu.vx | 4+3 |
| vmin.vv | 3 |
| vmin.vx | 4+3 |
| vmaxu.vv | 3 |
| vmaxu.vx | 4+3 |
| vmax.vv | 3 |
| vmax.vx | 4+3 |

#### 2.2.14 向量整数乘除法

| 指令名称 | 执行延迟 |
|----------|----------|
| vmul.vv | 4(64、32)/3(16、8) |
| vmul.vx | 4+4(64、32)/3(16、8) |
| vmulh.vv | 4(64、32)/3(16、8) |
| vmulh.vx | 4+4(64、32)/3(16、8) |
| vmulhu.vv | 4(64、32)/3(16、8) |
| vmulhu.vx | 4+4(64、32)/3(16、8) |
| vmulhsu.vv | 4(64、32)/3(16、8) |
| vmulhsu.vx | 4+4(64、32)/3(16、8) |
| vdivu.vv | 8 |
| vdivu.vx | 4+8 |
| vdiv.vv | 8 |
| vdiv.vx | 4+8 |
| vremu.vv | 8 |
| vremu.vx | 4+8 |
| vrem.vv | 8 |
| vrem.vx | 4+8 |
| vwmulu.vv | 5(64、32)/4(16、8) |
| vwmulu.vx | 4+5(64、32)/4(16、8) |
| vwmulsu.vv | 5(64、32)/4(16、8) |
| vwmulsu.vx | 4+5(64、32)/4(16、8) |
| vwmul.vv | 5(64、32)/4(16、8) |
| vwmul.vx | 4+5(64、32)/4(16、8) |
| vmacc.vv | 4 |
| vmacc.vx | 4+4 |
| vnmsac.vv | 4 |
| vnmsac.vx | 4+4 |
| vmadd.vv | 4 |
| vmadd.vx | 4+4 |
| vnmsub.vv | 4 |
| vnmsub.vx | 4+4 |
| vwmaccu.vv | 5 |
| vwmaccu.vx | 4+5 |
| vwmacc.vv | 5 |
| vwmacc.vx | 4+5 |
| vwmaccus.vx | 4+5 |
| vwmaccsu.vv | 5 |
| vwmaccsu.vx | 4+5 |

#### 2.2.15 向量合并与移动

| 指令名称 | 执行延迟 |
|----------|----------|
| vmerge.vvm | 3 |
| vmerge.vxm | 4+3 |
| vmerge.vim | 3 |
| vmv.v.v | 3 |
| vmv.v.x | 3 |
| vmv.v.i | 3 |

#### 2.2.16 向量饱和算术指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vsaddu.vv | 3 |
| vsaddu.vx | 7 |
| vsaddu.vi | 3 |
| vsadd.vv | 3 |
| vsadd.vx | 7 |
| vsadd.vi | 3 |
| vssubu.vv | 3 |
| vssubu.vx | 7 |
| vssub.vv | 3 |
| vssub.vx | 7 |
| vaaddu.vv | 3 |
| vaaddu.vx | 7 |
| vaadd.vv | 3 |
| vaadd.vx | 7 |
| vasubu.vv | 3 |
| vasubu.vx | 7 |
| vasub.vv | 3 |
| vasub.vx | 7 |
| vsmul.vv | 4 |
| vsmul.vx | 8 |
| vssrl.vv | 3 |
| vssrl.vx | 7 |
| vssrl.vi | 3 |
| vssra.vv | 3 |
| vssra.vx | 7 |
| vssra.vi | 3 |
| vnclipu.wv | 4 |
| vnclipu.wx | 8 |
| vnclipu.wi | 4 |
| vnclip.wv | 4 |
| vnclip.wx | 8 |
| vnclip.wi | 4 |

#### 2.2.17 向量浮点算术指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vfadd.vv | 3 |
| vfadd.vf | 3 |
| vfsub.vv | 3 |
| vfsub.vf | 3 |
| vfrsub.vf | 3 |
| vfwadd.vv | 3 |
| vfwadd.vf | 3 |
| vfwsub.vv | 3 |
| vfwsub.vf | 3 |
| vfwadd.wv | 3 |
| vfwadd.wf | 3 |
| vfwsub.wv | 3 |
| vfwsub.wf | 3 |
| vfmul.vv | 5 |
| vfmul.vf | 5 |
| vfdiv.vv | 4-17 |
| vfdiv.vf | 4-17 |
| vfrdiv.vf | 4-17 |
| vfwmul.vv | 4 |
| vfwmul.vf | 4 |
| vfmacc.vv | 5 |
| vfmacc.vf | 5 |
| vfnmacc.vv | 5 |
| vfnmacc.vf | 5 |
| vfmsac.vv | 5 |
| vfmsac.vf | 5 |
| vfnmsac.vv | 5 |
| vfnmsac.vf | 5 |
| vfmadd.vv | 5 |
| vfmadd.vf | 5 |
| vfnmadd.vv | 5 |
| vfnmadd.vf | 5 |
| vfmsub.vv | 5 |
| vfmsub.vf | 5 |
| vfnmsub.vv | 5 |
| vfnmsub.vf | 5 |
| vfwmacc.vv | 5 |
| vfwmacc.vf | 5 |
| vfwnmacc.vv | 5 |
| vfwnmacc.vf | 5 |
| vfwmsac.vv | 5 |
| vfwmsac.vf | 5 |
| vfwnmsac.vv | 5 |
| vfwnmsac.vf | 5 |
| vfsqrt.v | 4-17 |
| vfrsqrt7.v | 4-17 |
| vfrec7.v | 4-17 |
| vfmin.vv | 3 |
| vfmin.vf | 3 |
| vfmax.vv | 3 |
| vfmax.vf | 3 |
| vfsgnj.vv | 3 |
| vfsgnj.vf | 3 |
| vfsgnjn.vv | 3 |
| vfsgnjn.vf | 3 |
| vfsgnjx.vv | 3 |
| vfsgnjx.vf | 3 |

#### 2.2.18 向量浮点比较与分类

| 指令名称 | 执行延迟 |
|----------|----------|
| vmfeq.vv | 2 |
| vmfeq.vf | 2 |
| vmfne.vv | 2 |
| vmfne.vf | 2 |
| vmflt.vv | 2 |
| vmflt.vf | 2 |
| vmfle.vv | 2 |
| vmfle.vf | 2 |
| vmfgt.vf | 2 |
| vmfge.vf | 2 |
| vfclass.v | 2 |
| vfmerge.vfm | 2 |
| vfmv.v.f | 3 |

#### 2.2.19 向量浮点转换指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vfcvt.xu.f.v | 4 |
| vfcvt.x.f.v | 4 |
| vfcvt.rtz.xu.f.v | 4 |
| vfcvt.rtz.x.f.v | 4 |
| vfcvt.f.xu.v | 4 |
| vfcvt.f.x.v | 4 |
| vfwcvt.xu.f.v | 4 |
| vfwcvt.x.f.v | 4 |
| vfwcvt.rtz.xu.f.v | 4 |
| vfwcvt.rtz.x.f.v | 4 |
| vfwcvt.f.xu.v | 4 |
| vfwcvt.f.x.v | 4 |
| vfwcvt.f.f.v | 4 |
| vfncvt.xu.f.w | 4 |
| vfncvt.x.f.w | 4 |
| vfncvt.rtz.xu.f.w | 4 |
| vfncvt.rtz.x.f.w | 4 |
| vfncvt.f.xu.w | 4 |
| vfncvt.f.x.w | 4 |
| vfncvt.f.f.w | 4 |
| vfncvt.rod.f.f.w | 4 |

#### 2.2.20 向量归约指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vredsum.vs | 4 |
| vredand.vs | 3 |
| vredor.vs | 3 |
| vredxor.vs | 3 |
| vredminu.vs | 4 |
| vredmin.vs | 4 |
| vredmaxu.vs | 4 |
| vredmax.vs | 4 |
| vwredsumu.vs | 4 |
| vwredsum.vs | 4 |
| vfredosum.vs | 5-129 |
| vfredusum.vs | 5-10 |
| vfredmax.vs | 4 |
| vfredmin.vs | 4 |
| vfwredosum.vs | 5-129 |
| vfwredusum.vs | 5-10 |

#### 2.2.21 向量掩码指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vmand.mm | 3 |
| vmnand.mm | 3 |
| vmandn.mm | 3 |
| vmxor.mm | 3 |
| vmor.mm | 3 |
| vmnor.mm | 3 |
| vmorn.mm | 3 |
| vmxnor.mm | 3 |
| vcpop.m | 3 |
| vfirst.m | 3 |
| vmsbf.m | 3 |
| vmsof.m | 3 |
| vmsif.m | 3 |
| viota.m | 3 |
| vid.v | 3 |

#### 2.2.22 向量标量移动与滑动指令

| 指令名称 | 执行延迟 |
|----------|----------|
| vmv.x.s | 3 |
| vmv.s.x | 4 |
| vfmv.f.s | 3 |
| vfmv.s.f | 3 |
| vslide1up.vx | 4+3 |
| vfslide1up.vf | 3 |
| vslide1down.vx | 4+3 |
| vfslide1down.vf | 3 |
| vslideup.vx | 4+3 |
| vslideup.vi | 3 |
| vslidedown.vx | 4+3 |
| vslidedown.vi | 3 |
| vrgather.vv | 3 |
| vrgather.vx | 4+3 |
| vrgather.vi | 3 |
| vrgatherei16.vv | 4 |
| vcompress.vm | 3 |
| vmv1r.v | 3 |
| vmv2r.v | 3 |
| vmv4r.v | 3 |
| vmv8r.v | 3 |
