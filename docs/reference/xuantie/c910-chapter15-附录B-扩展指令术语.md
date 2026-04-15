# 玄铁 C910 平头哥扩展指令参考

> 来源：玄铁 C910 用户手册 第十五章 附录 B（PDF pages 241–337）

## 查阅说明

本文档为 C910 自定义扩展指令的完整语义参考，按功能子集分为 6 组，共 136 条指令。

### 前置条件

- Cache / 同步 / 算术 / 位操作 / 存储子集：需 `mxstatus.theadisaee==1`，否则 `IllegalInstr`
- 浮点半精度子集：需 `mstatus.fs != 2'b00`，否则 `IllegalInstr`

### 指令速查索引

| 子集 | 助记符 | 数量 |
|------|--------|------|
| 附录 B-1 Cache 指令术语 | DCACHE.CALL, DCACHE.CIALL, DCACHE.CIPA, DCACHE.CISW, DCACHE.CIVA, DCACHE.CPA, DCACHE.CPAL1, DCACHE.CVA, DCACHE.CVAL1, DCACHE.IPA, DCACHE.ISW, DCACHE.IVA, DCACHE.IALL, ICACHE.IALL, ICACHE.IALLS, ICACHE.IPA, ICACHE.IVA, L2CACHE.CALL, L2CACHE.CIALL, L2CACHE.IALL, DCACHE.CSW | 21 |
| 附录 B-2 多核同步指令术语 | SFENCE.VMAS, SYNC, SYNC.I, SYNC.IS, SYNC.S | 5 |
| 附录 B-3 算术运算指令术语 | ADDSL, MULA, MULAH, MULAW, MULS, MULSH, MULSW, MVEQZ, MVNEZ, SRRI, SRRIW | 11 |
| 附录 B-4 位操作指令术语 | EXT, EXTU, FF0, FF1, REV, REVW, TST, TSTNBZ | 8 |
| 附录 B-5 存储指令术语 | FLRD, FLRW, FLURD, FLURW, FSRD, FSRW, FSURD, FSURW, LBIA, LBIB, LBUIA, LBUIB, LDD, LDIA, LDIB, LHIA, LHIB, LHUIA, LHUIB, LRB, LRBU, LRD, LRH, LRHU, LRW, LRWU, LURB, LURBU, LURD, LURH, LURHU, LURW, LURWU, LWD, LWIA, LWIB, LWUD, LWUIA, LWUIB, SBIA, SBIB, SDD, SDIA, SDIB, SHIA, SHIB, SRB, SRD, SRH, SRW, SURB, SURD, SURH, SURW, SWIA, SWIB, SWD | 57 |
| 附录 B-6 浮点半精度指令术语 | FADD.H, FCLASS.H, FCVT.D.H, FCVT.H.D, FCVT.H.L, FCVT.H.LU, FCVT.H.S, FCVT.H.W, FCVT.H.WU, FCVT.L.H, FCVT.LU.H, FCVT.S.H, FCVT.W.H, FCVT.WU.H, FDIV.H, FEQ.H, FLE.H, FLH, FLT.H, FMADD.H, FMAX.H, FMIN.H, FMSUB.H, FMUL.H, FMV.H.X, FMV.X.H, FNMADD.H, FNMSUB.H, FSGNJ.H, FSGNJN.H, FSGNJX.H, FSH, FSQRT.H, FSUB.H | 34 |

### 条目格式约定

每条指令以如下格式记录：

- **MNEMONIC** — 中文简述
  - **语法**: 汇编助记符 + 操作数
  - **语义**: RTL 行为（`←` 赋值, `>>>` 循环右移, `&` 按位与, `|` 按位或）
  - **权限**: M/S/U mode 可用性
  - **异常**: `IllegalInstr` / `LoadAlign` / `StorePage` 等，`Flags:` 标注受影响的状态位
  - **说明**: 仅在有特殊约束时出现（省略 = 仅标准的 theadisaee 前置条件）

---

## 15.1 B-1 Cache 指令术语

### DCACHE.CALL — DCACHE 清全部脏表项

- **语法**: `dcache.call`
- **语义**: `clear 所有 L1 dcache 表项，将所有 dirty 表项写回到下一级存储，仅操作当前核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### DCACHE.CIALL — DCACHE 清全部脏表项后无效

- **语法**: `dcache.ciall`
- **语义**: `将所有 L1 dcache dirty 表项写回到下一级存储后，无效所有表项。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### DCACHE.CIPA — DCACHE 按物理地址清脏表项并无效

- **语法**: `dcache.cipa rs1`
- **语义**:
  - `将 rs1 中物理地址所属的 dcache/L2cache 表项写回下级存储并无效该表项, 操作所有核和`
  - `L2CACHE。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### DCACHE.CISW — DCACHE 按 way/set 清脏表项并无效

- **语法**: `dcache.cisw rs1`
- **语义**:
  - `按照 rs1 中指定的 way/set 将 L1 dache dirty 表项写回到下一级存储并无效该表项，仅操作当前`
  - `核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr
- **说明**: C910 dcache 为两路组相联，rs1[31] 为 way 编码，rs1[w:6] 为 set 编码。当 dcache 为 32K 时,w 为 13，dcache 为 64K 时，w 为 14。

### DCACHE.CIVA — DCACHE 按虚拟地址清脏表项并无效

- **语法**: `dcache.civa rs1`
- **语义**:
  - `将 rs1 指定虚拟地址所属的 dcache/L2 cache 表项写回到下级存储，并无效该表项，操作当前核`
  - `和 L2CACHE，并根据虚拟地址共享属性决定是否广播到其他核。`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr/LoadPage
- **说明**: mxstatus.theadisaee=1，mxstatus.ucme =1，U mode 下可以执行该指令。 mxstatus.theadisaee=1，mxstatus.ucme =0，U mode 下执行该指令产生非法指令异常。

### DCACHE.CPA — DCACHE 按物理地址清脏表项

- **语法**: `dcache.cpa rs1`
- **语义**: `将 rs1 中物理地址所对应的 dcache/l2cache 表项写回到下一级存储，操作所有核和 L2CACHE。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### DCACHE.CPAL1 — L1DCACHE 按物理地址清脏表项

- **语法**: `dcache.cpal1 rs1
操作： 将 rs1 中物理地址所对应的 dcache 表项写回到下一级存储，操作所有核 L1CACHE。`
- **语义**: `(见说明)`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### DCACHE.CVA — DCACHE 按虚拟地址清脏表项

- **语法**: `dcache.cva rs1`
- **语义**:
  - `将 rs1 中虚拟地址所对应的 dcache/l2cache 表项写回到下一级存储，操作当前核和 L2CACHE，`
  - `并根据虚拟地址共享属性决定是否广播到其他核`
- **权限**: M mode/S mode
- **异常**: IllegalInstr/LoadPage

### DCACHE.CVAL1 — L1DCACHE 按虚拟地址清脏表项

- **语法**: `dcache.cval1 rs1`
- **语义**: `将 rs1 中虚拟地址所对应的 dcache 表项写回到下一级存储，操作所有核 L1CACHE`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr/LoadPage
- **说明**: mxstatus.theadisaee=1，mxstatus.ucme=0，Umode下执行该指令产生非法指令异常。

### DCACHE.IPA — DCACHE 按物理地址无效

- **语法**: `dcache.ipa rs1`
- **语义**: `将 rs1 中物理地址所对应的 dcache/l2 cache 表项无效，操作所有核和 L2CACHE。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### DCACHE.ISW — DCACHE 按 set/way 无效

- **语法**: `dcache.isw rs1`
- **语义**: `无效指定 SET 和 WAY 的 dcache 表项，仅操作当前核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr
- **说明**: C910 dcache 为两路组相联，rs1[31] 为 way 编码，rs1[w:6] 为 set 编码。当 dcache 为 32K 时,w 为 13，dcache 为 64K 时，w 为 14。

### DCACHE.IVA — DCACHE 按虚拟地址无效

- **语法**: `dcache.iva rs1`
- **语义**:
  - `将 rs1 中虚拟地址所对应的 dcache/l2 cache 表项无效，操作当前核和 L2CACHE，并根据虚拟`
  - `地址共享属性决定是否广播到其他核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr/LoadPage

### DCACHE.IALL — DCACHE 无效所有表项

- **语法**: `dcache.iall`
- **语义**: `无效所有 L1 dcache 表项，仅操作当前核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### ICACHE.IALL — ICACHE 无效所有表项

- **语法**: `icache.iall`
- **语义**: `无效所有 icache 表项，仅操作当前核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### ICACHE.IALLS — ICACHE 广播无效所有表项

- **语法**: `icache.ialls`
- **语义**: `无效所有 icache 表项，并广播其他核去无效各自所有 icache 表项，操作所有核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### ICACHE.IPA — ICACHE 按物理地址无效表项

- **语法**: `icache.ipa rs1`
- **语义**: `将 rs1 中物理地址所对应的 icache 表项无效，操作所有核。`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### ICACHE.IVA — ICACHE 按虚拟地址无效表项

- **语法**: `icache.iva rs1`
- **语义**:
  - `将 rs1 中虚拟地址所对应的 icache 表项无效，操作当前核，并根据虚拟地址共享属性决定是否广`
  - `播到其他核。`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr/LoadPage
- **说明**: mxstatus.theadisaee=1，mxstatus.ucme=1，U mode 下可以执行该指令。 mxstatus.theadisaee=1，mxstatus.ucme=0，U mode 下执行该指令产生非法指令异常。

### L2CACHE.CALL — L2CACHE 清所有脏表项

- **语法**: `l2cache.call`
- **语义**: `将 l2cache 中所有 dirty 表项写回到下一级存储`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### L2CACHE.CIALL — L2CACHE 清所有脏表项并无效

- **语法**: `l2cache.ciall`
- **语义**: `将 l2cache 中所有 dirty 表项写回到下一级存储后无效所有 l2 表项`
- **权限**: M mode/S mode
- **异常**: IllegalInstr

### L2CACHE.IALL — L2CACHE 无效

- **语法**: `l2cache.iall`
- **语义**: `将 l2cache 中所有表项无效`
- **权限**: M mode/S mode
- **异常**: IllegalInstr
- **说明**: mxstatus.cskisayee=1，U mode 下执行该指令产生非法指令异常。

### DCACHE.CSW — DCACHE 按 set/way 清脏表项

- **语法**: `dcache.csw rs1`
- **语义**: `按 SET 和 WAY 将 dcache 中的脏表项回写到下一级存储器`
- **权限**: M mode/S mode
- **异常**: IllegalInstr
- **说明**: C910 dcache 为两路组相联，rs1[31] 为 way 编码，rs1[w:6] 为 set 编码。当 dcache 为 32K 时,w 为 13， dcache 为 64K 时，w 为 14。

## 15.2 B-2 多核同步指令术语

### SFENCE.VMAS — 虚拟内存同步广播

- **语法**: `sfence.vmas rs1,rs2`
- **语义**: `虚拟内存的无效和同步操作，需要广播到 cluster 里的其他核`
- **权限**: M mode/S mode
- **异常**: IllegalInstr
- **说明**: rs1：虚拟地址，rs2：asid rs1=x0，rs2=x0 时，无效 TLB 中所有表项，并广播到 cluster 中的其他核 rs1!=x0，rs2=x0 时，无效 TLB 中所有命中 rs1 虚拟地址的表项，并广播到 cluster 中的其 他核。 rs1=x0，rs2!=x0 时，无效 TLB 中所有命中 rs2 进程号的表项，并广播到 cluster 中的其他 核。 rs1!=x0，rs2!=x0 时，无效 TLB 中所有命中 rs1 虚拟地址和 rs2 进程号的表项，并广播到 cluster 中的其他核。

### SYNC — 同步

- **语法**: `sync`
- **语义**: `该指令保证前序所有指令比该指令早退休，后续所有指令比该指令晚退休`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### SYNC.I — 同步清空

- **语法**: `sync.i`
- **语义**:
  - `该指令保证前序所有指令比该指令早退休，后续所有指令比该指令晚退休, 该指令退休时清空流`
  - `水线`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### SYNC.IS — 同步清空广播

- **语法**: `sync.is`
- **语义**:
  - `该指令保证前序所有指令比该指令早退休，后续所有指令比该指令晚退休, 该指令退休时清空流`
  - `水线，并将该请求广播给其他核`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### SYNC.S — 同步广播

- **语法**: `sync.s`
- **语义**:
  - `该指令保证前序所有指令比该指令早退休，后续所有指令比该指令晚退休, 并将该请求广播给其`
  - `他核`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

## 15.3 B-3 算术运算指令术语

### ADDSL — 寄存器移位相加

- **语法**: `addsl rd rs1, rs2, imm2`
- **语义**: `rd ← rs1+ rs2<<imm2`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MULA — 乘累加

- **语法**: `mula rd, rs1, rs2`
- **语义**: `rd ← rd+ (rs1 * rs2)[63:0]`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MULAH — 低 16 位乘累加

- **语法**: `mulah rd, rs1, rs2`
- **语义**:
  - `tmp[31:0] ← rd[31:0]+ (rs1[15:0] * rs[15:0])`
  - `rd ←sign_extend(tmp[31:0])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MULAW — 低 32 位乘累加

- **语法**: `mulaw rd, rs1, rs2`
- **语义**:
  - `tmp[31:0] ← rd[31:0]+ (rs1[31:0] * rs[31:0])[31:0]`
  - `rd ←sign_extend(tmp[31:0])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MULS — 乘累减

- **语法**: `muls rd, rs1, rs2`
- **语义**: `rd ← rd- (rs1 * rs2)[63:0]`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MULSH — 低 16 位乘累减

- **语法**: `mulsh rd, rs1, rs2`
- **语义**:
  - `tmp[31:0] ← rd[31:0]- (rs1[15:0] * rs[15:0])`
  - `rd ←sign_extend(tmp[31:0])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MULSW — 低 32 位乘累减

- **语法**: `mulaw rd, rs1, rs2`
- **语义**:
  - `tmp[31:0] ← rd[31:0]- (rs1[31:0] * rs[31:0])`
  - `rd ←sign_extend(tmp[31:0])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MVEQZ — 寄存器为 0 传送

- **语法**: `mveqz rd, rs1, rs2
操作： if (rs2 == 0)
rd ← rs1
else
rd ← rd`
- **语义**: `(见说明)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### MVNEZ — 寄存器非 0 传送

- **语法**: `mvnez rd, rs1, rs2`
- **语义**:
  - `if (rs2 != 0)`
  - `rd ← rs1`
  - `else`
  - `rd ← rd`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### SRRI — 循环右移

- **语法**: `srri rd, rs1, imm6`
- **语义**:
  - `rd ← rs1 >>>> imm6`
  - `rs1 原值右移，左侧移入右侧移出位`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### SRRIW — 低 32 位循环右移

- **语法**: `srriw rd, rs1, imm5`
- **语义**:
  - `rd ← sign_extend(rs1[31:0] >>>> imm5)`
  - `rs1[31:0] 原值右移，左侧移入右侧移出位`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

## 15.4 B-4 位操作指令术语

### EXT — 寄存器连续位提取符号位扩展

- **语法**: `ext rd, rs1, imm1,imm2`
- **语义**: `rd←sign_extend(rs1[imm1:imm2])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr
- **说明**: 若 imm1<imm2，该指令行为不可预测

### EXTU — 寄存器连续位提取零扩展

- **语法**: `extu rd, rs1, imm1,imm2`
- **语义**: `rd←zero_extend(rs1[imm1:imm2])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr
- **说明**: 若 imm1<imm2，该指令行为不可预测

### FF0 — 快速找 0 

- **语法**: `ff0 rd, rs1`
- **语义**:
  - `从 rs1 最高位开始查找第一个为 0 的位，结果写回 rd。如果 rs1 的最高位为 0，则结果为 0，如`
  - `果 rs1 中没有 0，结果为 64`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### FF1 — 快速找 1 

- **语法**: `ff1 rd, rs1`
- **语义**:
  - `从 rs1 最高位开始查找第一个为 1 的位，将该位的索引写回 rd。如果 rs1 的最高位为 1，则结果`
  - `为 0，如果 rs1 中没有 1，结果为 64`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### REV — 字节倒序

- **语法**: `rev rd, rs1`
- **语义**:
  - `rd[63:56] ←rs1[7:0]`
  - `rd[55:48] ←rs1[15:8]`
  - `rd[47:40] ←rs1[23:16]`
  - `rd[39:32] ←rs1[31:24]`
  - `rd[31:24] ←rs1[39:32]`
  - `rd[23:16] ←rs1[47:40]`
  - `rd[15:8] ←rs1[55:48]`
  - `rd[7:0] ←rs1[63:56]`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### REVW — 低 32 位字节倒序

- **语法**: `revw rd, rs1`
- **语义**:
  - `tmp[31:24] ←rs1[7:0]`
  - `tmp [23:16] ←rs1[15:8]`
  - `tmp [15:8] ←rs1[23:16]`
  - `tmp [7:0] ←rs1[31:24]`
  - `rd ←sign_extend(tmp[31:0])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### TST — 比特为 0 测试

- **语法**: `tst rd, rs1, imm6`
- **语义**:
  - `if(rs1[imm6] == 1)`
  - `rd←1`
  - `else`
  - `rd←0`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

### TSTNBZ — 字节为 0 测试

- **语法**: `tstnbz rd, rs1`
- **语义**:
  - `rd[63:56] ← (rs1[63:56] == 0) ? 8’hff : 8’h0`
  - `rd[55:48] ← (rs1[55:48] == 0) ? 8’hff : 8’h0`
  - `rd[47:40] ← (rs1[47:40] == 0) ? 8’hff : 8’h0`
  - `rd[39:32] ← (rs1[39:32] == 0) ? 8’hff : 8’h0`
  - `rd[31:24] ← (rs1[31:24] == 0) ? 8’hff : 8’h0`
  - `rd[23:16] ← (rs1[23:16] == 0) ? 8’hff : 8’h0`
  - `rd[15:8] ← (rs1[15:8] == 0) ? 8’hff : 8’h0`
  - `rd[7:0] ← (rs1[7:0] == 0) ? 8’hff : 8’h0`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr

## 15.5 B-5 存储指令术语

### FLRD — 浮点寄存器移位双字加载

- **语法**: `flrd rd, rs1, rs2, imm2`
- **语义**: `rd ←mem[(rs1+rs2<<imm2)+7: (rs1+rs2<<imm2)]`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/ IllegalInstr
- **说明**: mxstatus.theadisaee=1’b0 或 mstatus.fs =2’b00 时，该指令产生非法指令异常

### FLRW — 浮点寄存器移位字加载

- **语法**: `flrw rd, rs1, rs2, imm2`
- **语义**: `rd ←one_extend(mem[(rs1+rs2<<imm2)+3: (rs1+rs2<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage，IllegalInstr
- **说明**: mxstatus.theadisaee=1’b0 或 mstatus.fs =2’b00 时，该指令产生非法指令异常

### FLURD — 浮点寄存器低 32 位移位双字加载

- **语法**: `flurd rd, rs1, rs2, imm2`
- **语义**: `rd ←mem[(rs1+rs2[31:0]<<imm2)+7: (rs1+rs2[31:0]<<imm2)]`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算 mxstatus.theadisaee=1’b0 或 mstatus.fs = 2’b00 时，该指令产生非法指令异常

### FLURW — 浮点寄存器低 32 位移位字加载

- **语法**: `flurw rd, rs1, rs2, imm2`
- **语义**: `rd ←one_extend(mem[(rs1+rs2[31:0]<<imm2)+3: (rs1+rs2[31:0]<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算 mxstatus.theadisaee=1’b0 或 mstatus.fs = 2’b00 时，该指令产生非法指令异常

### FSRD — 浮点寄存器移位双字存储

- **语法**: `fsrd rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2<<imm2)+7: (rs1+rs2<<imm2)] ←rd[63:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: mxstatus.theadisaee=1’b0 或 mstatus.fs =2’b00 时，该指令产生非法指令异常

### FSRW — 浮点寄存器移位字存储

- **语法**: `fsrw rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2<<imm2)+3: (rs1+rs2<<imm2)] ←rd[31:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: mxstatus.theadisaee=1’b0 或 mstatus.fs =2’b00 时，该指令产生非法指令异常

### FSURD — 浮点寄存器低 32 位移位双字存储

- **语法**: `fsurd rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2[31:0]<<imm2)+7: (rs1+rs2[31:0]<<imm2)] ←rd[63:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算 mxstatus.theadisaee=1’b0 或 mstatus.fs = 2’b00 时，该指令产生非法指令异常

### FSURW — 浮点寄存器低 32 位移位字存储

- **语法**: `fsurw rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2[31:0]<<imm2)+3: (rs1+rs2[31:0]<<imm2)] ←rd[31:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算 mxstatus.theadisaee=1’b0 或 mstatus.fs = 2’b00 时，该指令产生非法指令异常

### LBIA — 符号位扩展字节加载基地址自增

- **语法**: `lbia rd, (rs1), imm5,imm2`
- **语义**:
  - `rd ←sign_extend(mem[rs1])`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/非法指令 异常。
- **说明**: rd 和 rs1 不可相等

### LBIB — 基地址自增符号位扩展字节加载

- **语法**: `lbib rd, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `rd ←sign_extend(mem[rs1])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LBUIA — 零扩展字节加载基地址自增

- **语法**: `lbuia rd, (rs1), imm5,imm2`
- **语义**:
  - `rd ←zero_extend(mem[rs1])`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LBUIB — 基地址自增零扩展字节加载

- **语法**: `lbuib rd, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `rd ←zero_extend(mem[rs1])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LDD — 双寄存器加载

- **语法**: `ldd rd1,rd2, (rs1),imm2`
- **语义**:
  - `address←rs1 + zero_extend(imm2<<4)`
  - `rd1←mem[address+7:address]`
  - `rd2←mem[address+15:address+8]`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rd1,rd2 ,rs1 互相不可相等

### LDIA — 符号位扩展双字加载基地址自增

- **语法**: `ldia rd, (rs1), imm5,imm2`
- **语义**:
  - `rd ←sign_extend(mem[rs1+7:rs1])`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LDIB — 基地址自增符号位扩展双字加载

- **语法**: `ldib rd, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `rd ←sign_extend(mem[rs1+7:rs1])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LHIA — 符号位扩展半字加载基地址自增

- **语法**: `lhia rd, (rs1), imm5,imm2`
- **语义**:
  - `rd ←sign_extend(mem[rs1+1:rs1])`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LHIB — 基地址自增符号位扩展半字加载

- **语法**: `lhib rd, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `rd ←sign_extend(mem[rs1+1:rs1])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LHUIA — 零扩展半字加载基地址自增

- **语法**: `lhuia rd, (rs1), imm5,imm2`
- **语义**:
  - `rd ←zero_extend(mem[rs1+1:rs1])`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LHUIB — 基地址自增零扩展半字加载

- **语法**: `lhuib rd, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `rd ←zero_extend(mem[rs1+1:rs1])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LRB — 寄存器移位符号位扩展字节加载

- **语法**: `lrb rd, rs1, rs2, imm2`
- **语义**: `rd ←sign_extend(mem[(rs1+rs2<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LRBU — 寄存器移位零扩展扩展字节加载

- **语法**: `lrbu rd, rs1, rs2, imm2`
- **语义**: `rd ←zero_extend(mem[(rs1+rs2<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LRD — 寄存器移位双字加载

- **语法**: `lrd rd, rs1, rs2, imm2`
- **语义**: `rd ←mem[(rs1+rs2<<imm2)+7: (rs1+rs2<<imm2)]`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LRH — 寄存器移位符号位扩展半字加载

- **语法**: `lrh rd, rs1, rs2, imm2`
- **语义**: `rd ←sign_extend(mem[(rs1+rs2<<imm2)+1: (rs1+rs2<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LRHU — 寄存器移位零扩展扩展半字加载

- **语法**: `lrhu rd, rs1, rs2, imm2`
- **语义**: `rd ←zero_extend(mem[(rs1+rs2<<imm2)+1: (rs1+rs2<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LRW — 寄存器移位符号位扩展字加载

- **语法**: `lrw rd, rs1, rs2, imm2`
- **语义**: `rd ←sign_extend(mem[(rs1+rs2<<imm2)+3: (rs1+rs2<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LRWU — 寄存器移位零扩展扩展字加载

- **语法**: `lrwu rd, rs1, rs2, imm2`
- **语义**: `rd ←zero_extend(mem[(rs1+rs2<<imm2)+3: (rs1+rs2<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LURB — 寄存器低 32 位移位符号位扩展字节加载

- **语法**: `lurb rd, rs1, rs2, imm2`
- **语义**: `rd ←sign_extend(mem[ (rs1+rs2[31:0]<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算

### LURBU — 寄存器低 32 位移位零扩展字节加载

- **语法**: `lurbu rd, rs1, rs2, imm2`
- **语义**: `rd ←zero_extend(mem[(rs1+rs2[31:0]<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算

### LURD — 寄存器低 32 位移位双字加载

- **语法**: `lurd rd, rs1, rs2, imm2`
- **语义**: `rd ←mem[(rs1+rs2[31:0]<<imm2)+7: (rs1+rs2[31:0]<<imm2)]`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算

### LURH — 寄存器低 32 位移位符号位扩展半字加载

- **语法**: `lurh rd, rs1, rs2, imm2`
- **语义**:
  - `rd ←sign_extend(mem[(rs1+rs2[31:0]<<imm2)+1:`
  - `(rs1+rs2[31:0]<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算

### LURHU — 寄存器低 32 位移位零扩展半字加载

- **语法**: `lurhu rd, rs1, rs2, imm2`
- **语义**:
  - `rd ←zero_extend(mem[(rs1+rs2[31:0]<<imm2)+1:`
  - `(rs1+rs2[31:0]<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算

### LURW — 寄存器低 32 位移位符号位扩展字加载

- **语法**: `lurw rd, rs1, rs2, imm2`
- **语义**:
  - `rd ←sign_extend(mem[(rs1+rs2[31:0]<<imm2)+3:`
  - `(rs1+rs2[31:0]<<imm2)])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数，高位 [63:32] 补 0 进行地址运算

### LURWU — 寄存器低 32 位移位零扩展字加载

- **语法**: `lwd rd1, rd2, (rs1),imm2`
- **语义**:
  - `address←rs1+zero_extend(imm2<<3)`
  - `rd1 ←sign_extend(mem[address+3: address])`
  - `rd2 ←sign_extend(mem[address+7: address+4])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rd1,rd2 ,rs1 互相不可相等

### LWD — 符号位扩展双寄存器字加载

- **语法**: `lwd rd, imm7(rs1)`
- **语义**:
  - `address←rs1+sign_extend(imm7)`
  - `rd ←sign_extend(mem[address+31: address])`
  - `rd+1 ←sign_extend(mem[address+63: address+32])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr

### LWIA — 符号位扩展字加载基地址自增

- **语法**: `lwia rd, (rs1), imm5,imm2`
- **语义**:
  - `rd ←sign_extend(mem[rs1+3:rs1])`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LWIB — 基地址自增符号位扩展字加载

- **语法**: `lwib rd, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `rd ←sign_extend(mem[rs1+3:rs1])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LWUD — 零扩展双寄存器字加载

- **语法**: `lwud rd1,rd2, (rs1),imm2`
- **语义**:
  - `address←rs1+zero_extend(imm2<<3)`
  - `rd1 ←zero_extend(mem[address+3: address])`
  - `rd2 ←zero_extend(mem[address+7: address+4])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr
- **说明**: rd1,rd2 ,rs1 互相不可相等

### LWUIA — 零扩展字加载基地址自增

- **语法**: `lwuia rd, (rs1), imm5,imm2`
- **语义**:
  - `rd ←zero_extend(mem[rs1+3:rs1])`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### LWUIB — 基地址自增零扩展字加载

- **语法**: `lwuib rd, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `rd ←zero_extend(mem[rs1+3:rs1])`
- **权限**: M mode/S mode/U mode
- **异常**: LoadAlign/LoadAccess/LoadPage/IllegalInstr。
- **说明**: rd 和 rs1 不可相等

### SBIA — 字节存储基地址自增

- **语法**: `sbia rs2, (rs1), imm5,imm2`
- **语义**:
  - `mem[rs1]←rs2[7:0]`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SBIB — 基地址自增字节存储

- **语法**: `sbib rs2, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `mem[rs1] ←rs2[7:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SDD — 双寄存器存储

- **语法**: `sdd rd1,rd2, (rs1),imm2`
- **语义**:
  - `address←rs1 + zero_extend(imm2<<4)`
  - `mem[address+7:address] ←rd1`
  - `mem[address+15:address+8]←rd2`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SDIA — 双字存储基地址自增

- **语法**: `sdia rs2, (rs1), imm5,imm2`
- **语义**:
  - `mem[rs1+7:rs1]←rs2[63:0]`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SDIB — 基地址自增双字存储

- **语法**: `sdib rs2, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `mem[rs1+7:rs1] ←rs2[63:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SHIA — 半字存储基地址自增

- **语法**: `shia rs2, (rs1), imm5,imm2`
- **语义**:
  - `mem[rs1+1:rs1]←rs2[15:0]`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SHIB — 基地址自增半字存储

- **语法**: `shib rs2, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `mem[rs1+1:rs1] ←rs2[15:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SRB — 寄存器移位字节存储

- **语法**: `srb rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2<<imm2)] ←rd[7:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SRD — 寄存器移位双字存储

- **语法**: `srd rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2<<imm2)+7: (rs1+rs2<<imm2)] ←rd[63:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SRH — 寄存器移位半字存储

- **语法**: `srh rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2<<imm2)+1: (rs1+rs2<<imm2)] ←rd[15:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SRW — 寄存器移位字存储

- **语法**: `srw rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2<<imm2)+3: (rs1+rs2<<imm2)] ←rd[31:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SURB — 寄存器低 32 位移位字节存储

- **语法**: `surb rd, rs1, rs2, imm2`
- **语义**: `mem[ (rs1+rs2[31:0]<<imm2)] ←rd[7:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数, 高位 [63:32] 补 0 进行地址运算

### SURD — 寄存器低 32 位移位双字存储

- **语法**: `surd rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2[31:0]<<imm2)+7: (rs1+rs2[31:0]<<imm2)] ←rd[63:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数, 高位 [63:32] 补 0 进行地址运算

### SURH — 寄存器低 32 位移位半字存储

- **语法**: `surh rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2[31:0]<<imm2)+1: (rs1+rs2[31:0]<<imm2)] ←rd[15:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数, 高位 [63:32] 补 0 进行地址运算

### SURW — 寄存器低 32 位移位字存储

- **语法**: `surw rd, rs1, rs2, imm2`
- **语义**: `mem[(rs1+rs2[31:0]<<imm2)+3: (rs1+rs2[31:0]<<imm2)] ←rd[31:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr
- **说明**: rs2[31:0] 是无符号数, 高位 [63:32] 补 0 进行地址运算

### SWIA — 字存储基地址自增

- **语法**: `swia rs2, (rs1), imm5,imm2`
- **语义**:
  - `mem[rs1+3:rs1]←rs2[31:0]`
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SWIB — 基地址自增字存储

- **语法**: `swib rs2, (rs1), imm5,imm2`
- **语义**:
  - `rs1←rs1 + sign_extend(imm5 << imm2)`
  - `mem[rs1+3:rs1] ←rs2[31:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### SWD — 双寄存器低 32 位存储

- **语法**: `swd rd1,rd2,(rs1),imm2`
- **语义**:
  - `address←rs1+ zero_extend(imm2<<3)`
  - `mem[address+3:address] ←rd1[31:0]`
  - `mem[address+7:address+4]←rd2[31:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

## 15.6 B-6 浮点半精度指令术语

### FADD.H — 半精度浮点加法

- **语法**: `fadd.h fd, fs1, fs2, rm`
- **语义**: `fd ← fs1 + fs2`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fadd.h fd, fs1,fs2,rne。 3’b001: 向零舍入，对应的汇编指令 fadd.h fd, fs1,fs2,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fadd.h fd, fs1,fs2,rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fadd.h fd, fs1,fs2,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fadd.h fd, fs1,fs2,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fadd.h fd, fs1,fs2。

### FCLASS.H — 半精度浮点分类

- **语法**: `fclass.h rd, fs1`
- **语义**:
  - `if ( fs1 = -inf)`
  - `rd ← 64’h1`
  - `if ( fs1 = -norm)`
  - `rd ← 64’h2`
  - `if ( fs1 = -subnorm)`
  - `rd ← 64’h4`
  - `if ( fs1 = -zero)`
  - `rd ← 64’h8`
  - `if ( fs1 = +zero)`
  - `rd ← 64’h10`
  - `if ( fs1 = +subnorm)`
  - `rd ← 64’h20`
  - `if ( fs1 = +norm)`
  - `rd ← 64’h40`
  - `if ( fs1 = +inf)`
  - `rd ← 64’h80`
  - `if ( fs1 = sNaN)`
  - `rd ← 64’h100`
  - `if ( fs1 = qNaN)`
  - `rd ← 64’h200`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FCVT.D.H — 半精度浮点转换成双精度浮点

- **语法**: `fcvt.d.h fd, fs1`
- **语义**: `fd ← half_convert_to_double(fs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FCVT.H.D — 双精度浮点转换成半精度浮点

- **语法**: `fcvt.h.d fd, fs1, rm`
- **语义**: `fd ← double_convert_to_half(fs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/UF/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.h.d fd,fs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.h.d fd,fs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.h.d fd,fs1,rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.h.d fd,fs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.h.d fd,fs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.h.d fd, fs1。

### FCVT.H.L — 有符号长整型转换成半精度浮点数

- **语法**: `fcvt.h.l fd, rs1, rm`
- **语义**: `fd ← signed_long_convert_to_half(rs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NX/OF
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.h.l fd,rs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.h.l fd,rs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.h.l fd,rs1,fdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.h.l fd,rs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.h.l fd,rs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.h.l fd, rs1。

### FCVT.H.LU — 无符号长整型转换成半精度浮点数

- **语法**: `fcvt.h.lu fd, rs1, rm`
- **语义**: `fd ← unsigned_long_convert_to_half_fp(rs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NX/OF
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.h.lu fd,rs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.h.lu fd, rs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.h.lu fd, rs1,fdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.h.lu fd, rs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.h.lu fd, rs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.h.lu fd, rs1。

### FCVT.H.S — 单精度浮点转换成半精度浮点

- **语法**: `fcvt.h.s fd, fs1, rm`
- **语义**: `fd ← single_convert_to_half(fs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/UF/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.h.s fd,fs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.h.s fd,fs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.h.s fd,fs1,fdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.h.s fd,fs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.h.s fd,fs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.h.s fd, fs1。

### FCVT.H.W — 有符号整型转换成半精度浮点数

- **语法**: `fcvt.h.w fd, rs1, rm`
- **语义**: `fd ← signed_int_convert_to_half(rs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NX/OF
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.h.w fd,rs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.h.w fd,rs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.h.w fd,rs1,fdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.h.w fd,rs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.h.w fd,rs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.h.w fd, rs1。

### FCVT.H.WU — 无符号整型转换成半精度浮点数

- **语法**: `fcvt.h.wu fd, rs1, rm`
- **语义**: `fd ← unsigned_int_convert_to_half_fp(rs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NX/OF
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.h.wu fd,rs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.h.wu fd,rs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.h.wu fd,rs1,fdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.h.wu fd,rs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.h.wu fd,rs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.h.wu fd, rs1。

### FCVT.L.H — 半精度浮点转换成有符号长整型

- **语法**: `fcvt.l.h rd, fs1, rm`
- **语义**: `rd ← half_convert_to_signed_long(fs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.l.h rd,fs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.l.h rd,fs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.l.h rd,fs1,rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.l.h rd,fs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.l.h rd,fs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.l.h rd, fs1。

### FCVT.LU.H — 半精度浮点转换成无符号长整型

- **语法**: `fcvt.lu.h rd, fs1, rm`
- **语义**: `rd ← half_convert_to_unsigned_long(fs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.lu.h rd,fs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.lu.h rd,fs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.lu.h rd,fs1,rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.lu.h rd,fs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.lu.h rd,fs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.lu.h rd, fs1。

### FCVT.S.H — 半精度浮点转换成单精度浮点

- **语法**: `fcvt.s.h fd, fs1`
- **语义**: `fd ← half_convert_to_single(fs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FCVT.W.H — 半精度浮点转换成有符号整型

- **语法**: `fcvt.w.h rd, fs1, rm`
- **语义**:
  - `tmp ← half_convert_to_signed_int(fs1)`
  - `rd←sign_extend(tmp)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.w.h rd,fs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.w.h rd,fs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.w.h rd,fs1,rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.w.h rd,fs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.w.h rd,fs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.w.h rd, fs1。

### FCVT.WU.H — 半精度浮点转换成无符号整型

- **语法**: `fcvt.wu.h rd, fs1, rm`
- **语义**:
  - `tmp ← half_convert_to_unsigned_int(fs1)`
  - `rd←sign_extend(tmp)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fcvt.wu.h rd,fs1,rne。 3’b001: 向零舍入，对应的汇编指令 fcvt.wu.h rd,fs1,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fcvt.wu.h rd,fs1,rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fcvt.wu.h rd,fs1,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fcvt.wu.h rd,fs1,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fcvt.wu.h rd, fs1。

### FDIV.H — 半精度浮点除法

- **语法**: `fdiv.h fd, fs1, fs2, rm`
- **语义**: `fd ← fs1 / fs2`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/DZ/OF/UF/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fdiv.h fs1,fs2,rne。 3’b001: 向零舍入，对应的汇编指令 fdiv.h fd fs1,fs2,rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fdiv.h fd, fs1,fs2,rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fdiv.h fd, fs1,fs2,rup。 3’b100: 就近向大值舍入，对应的汇编指令 fdiv.h fd, fs1,fs2,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fdiv.h fd, fs1,fs2。

### FEQ.H — 半精度浮点比较相等

- **语法**: `feq.h rd, fs1, fs2`
- **语义**:
  - `if(fs1 == fs2)`
  - `rd ← 1`
  - `else`
  - `rd ← 0`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV

### FLE.H — 半精度浮点比较小于等于

- **语法**: `fle.h rd, fs1, fs2`
- **语义**:
  - `if(fs1 <= fs2)`
  - `rd ← 1`
  - `else`
  - `rd ← 0`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV

### FLH — 半精度浮点加载

- **语法**: `flh fd, imm12(rs1)`
- **语义**:
  - `address←rs1+sign_extend(imm12)`
  - `fd[15:0] ← mem[(address+1):address]`
  - `fd[63:16] ← 48’hffffffffffff`
- **权限**: M mode/S mode/U mode
- **异常**: Flags: none

### FLT.H — 半精度浮点比较小于

- **语法**: `flt.h rd, fs1, fs2`
- **语义**:
  - `if(fs1 < fs2)`
  - `rd ← 1`
  - `else`
  - `rd ← 0`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV

### FMADD.H — 半精度浮点乘累加

- **语法**: `fmadd.h fd, fs1, fs2, fs3, rm`
- **语义**: `fd ← fs1*fs2 + fs3`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/UF/IX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fmadd.h fd,fs1, fs2, fs3, rne。 3’b001: 向零舍入，对应的汇编指令 fmadd.h fd,fs1, fs2, fs3, rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fmadd.h fd,fs1, fs2, fs3, rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fmadd.h fd,fs1, fs2, fs3, rup。 3’b100: 就近向大值舍入，对应的汇编指令 fmadd.h fd,fs1, fs2, fs3, rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fmadd.h fd,fs1, fs2, fs3。

### FMAX.H — 半精度浮点取最大值

- **语法**: `fmax.h fd, fs1, fs2`
- **语义**:
  - `if(fs1 >= fs2)`
  - `fd ← fs1`
  - `else`
  - `fd ← fs2`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV

### FMIN.H — 半精度浮点取最小值

- **语法**: `fmin.h fd, fs1, fs2`
- **语义**:
  - `if(fs1 >= fs2)`
  - `fd ← fs2`
  - `else`
  - `fd ← fs1`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV

### FMSUB.H — 半精度浮点乘累减

- **语法**: `fmsub.h fd, fs1, fs2, fs3, rm`
- **语义**: `fd ← fs1*fs2 - fs3`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/UF/IX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fmsub.h fd,fs1, fs2, fs3, rne。 3’b001: 向零舍入，对应的汇编指令 fmsub.h fd,fs1, fs2, fs3, rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fmsub.h fd,fs1, fs2, fs3, rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fmsub.h fd,fs1, fs2, fs3, rup。 3’b100: 就近向大值舍入，对应的汇编指令 fmsub.h fd, fs1, fs2, fs3,rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fmsub.h fd,fs1, fs2, fs3。

### FMUL.H — 半精度浮点乘法

- **语法**: `fmul.h fd, fs1, fs2, rm`
- **语义**: `fd ← fs1 * fs2`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/UF/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fmul.h fd, fs1, fs2, rne。 3’b001: 向零舍入，对应的汇编指令 fmul.h fd, fs1, fs2, rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fmul.h fd, fs1, fs2, rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fmul.h fd, fs1, fs2, rup。 3’b100: 就近向大值舍入，对应的汇编指令 fmul.h fd, fs1,fs2 , rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fmul.h fs1,fs2。

### FMV.H.X — 半精度浮点写传送

- **语法**: `fmv.h.x fd, rs1`
- **语义**:
  - `fd[15:0] ← rs1[15:0]`
  - `fd[63:16] ← 48’hffffffffffff`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FMV.X.H — 半精度浮点寄存器读传送

- **语法**: `fmv.x.h rd, fs1`
- **语义**:
  - `tmp[15:0] ← fs1[15:0]`
  - `rd ← sign_extend(tmp[15:0])`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FNMADD.H — 半精度浮点乘累加取负

- **语法**: `fnmadd.h fd, fs1, fs2, fs3, rm`
- **语义**: `fd ←-( fs1*fs2 + fs3)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/UF/IX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fnmadd.h fd,fs1, fs2, fs3, rne。 3’b001: 向零舍入，对应的汇编指令 fnmadd.h fd,fs1, fs2, fs3, rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fnmadd.h fd,fs1, fs2, fs3, rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fnmadd.h fd,fs1, fs2, fs3, rup。 3’b100: 就近向大值舍入，对应的汇编指令 fnmadd.h fd,fs1, fs2, fs3, rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fnmadd.h fd,fs1, fs2, fs3。

### FNMSUB.H — 半精度浮点乘累减取负

- **语法**: `fnmsub.h fd, fs1, fs2, fs3, rm`
- **语义**: `fd ← -(fs1*fs2 - fs3)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/UF/IX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fnmsub.h fd,fs1, fs2, fs3, rne。 3’b001: 向零舍入，对应的汇编指令 fnmsub.h fd,fs1, fs2, fs3, rtz。 3’b010: 向负无穷舍入，对应的汇编指令 fnmsub.h fd,fs1, fs2, fs3, rdn。 3’b011: 向正无穷舍入，对应的汇编指令 fnmsub.h fd,fs1, fs2, fs3, rup。 3’b100: 就近向大值舍入，对应的汇编指令 fnmsub.h fd,fs1, fs2, fs3, rmm。 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fnmsub.h fd,fs1, fs2, fs3。

### FSGNJ.H — 半精度浮点符号注入

- **语法**: `fsgnj.h fd, fs1, fs2`
- **语义**:
  - `fd[14:0] ← fs1[14:0]`
  - `fd[15] ← fs2[15]`
  - `fd[63:16] ← 48’hffffffffffff`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FSGNJN.H — 半精度浮点符号取反注入

- **语法**: `fsgnjn.h fd, fs1, fs2`
- **语义**:
  - `fd[14:0] ← fs1[14:0]`
  - `fd[15] ← ! fs2[15]`
  - `fd[63:16] ← 48’hffffffffffff`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FSGNJX.H — 半精度浮点符号异或注入

- **语法**: `fsgnjx.h fd, fs1, fs2`
- **语义**:
  - `fd[14:0] ← fs1[14:0]`
  - `fd[15] ← fs1[15] ^ fs2[15]`
  - `fd[63:16] ← 48’hffffffffffff`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: none

### FSH — 半精度浮点存储

- **语法**: `fsh fs2, imm12(fs1)`
- **语义**:
  - `address←fs1+sign_extend(imm12)`
  - `mem[(address+1):address] ← fs2[15:0]`
- **权限**: M mode/S mode/U mode
- **异常**: StoreAlign/StoreAccess/StorePage/IllegalInstr

### FSQRT.H — 半精度浮点开方

- **语法**: `fsqrt.h fd, fs1, rm`
- **语义**: `fd ← sqrt(fs1)`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fsqrt.h fd, fs1,rne 3’b001: 向零舍入，对应的汇编指令 fsqrt.h fd, fs1,rtz 3’b010: 向负无穷舍入，对应的汇编指令 fsqrt.h fd, fs1,rdn 3’b011: 向正无穷舍入，对应的汇编指令 fsqrt.h fd, fs1,rup 3’b100: 就近向大值舍入，对应的汇编指令 fsqrt.h fd, fs1,rmm 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fsqrt.h fd, fs1。

### FSUB.H — 半精度浮点减法

- **语法**: `fsub.h fd, fs1, fs2, rm`
- **语义**: `fd ← fs1 - fs2`
- **权限**: M mode/S mode/U mode
- **异常**: IllegalInstr Flags: NV/OF/NX
- **说明**: rm 决定舍入模式: 3’b000: 就近向偶数舍入，对应的汇编指令 fsub.h fd, fs1,fs2,rne 3’b001: 向零舍入，对应的汇编指令 fsub.h fd, fs1,fs2,rtz 3’b010: 向负无穷舍入，对应的汇编指令 fsub.h fd, fs1,fs2,rdn 3’b011: 向正无穷舍入，对应的汇编指令 fsub.h fd, fs1,fs2,rup 3’b100: 就近向大值舍入，对应的汇编指令 fsub.h fd, fs1,fs2,rmm 3’b101: 暂未使用，不会出现该编码。 3’b110: 暂未使用，不会出现该编码。 3’b111: 动态舍入，根据浮点控制寄存器 fcsr 中的 rm 位来决定舍入模式，对应的汇编指 令 fsub.h fd, fs1,fs2。
