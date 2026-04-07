# YOLOv11n 推理标量指令融合分析报告

> 基于 ONNX Runtime 标量构建 + YOLOv11n 推理的 BBV 热点分析
> 仅考虑标量 ISA 扩展与微架构融合，不涉及 RISC-V V 扩展向量化

<style>
table { font-size: 9pt; table-layout: fixed; width: 100%; word-wrap: break-word; overflow-wrap: break-word; }
th, td { padding: 0.3em 0.5em; vertical-align: top; }
table.compat { font-size: 8pt; }
table.compat th, table.compat td { text-align: center; padding: 0.2em 0.3em; }
</style>

---

## 1. 分析背景与方法

### 1.1 实验环境

| 项目 | 值 |
|------|-----|
| 推理框架 | ONNX Runtime（标量编译） |
| 模型 | YOLOv11n |
| 目标架构 | RISC-V 64（RV64GCV，本研究仅使用 F/D 扩展） |
| 仿真环境 | Xuantie QEMU（BBV 插件采样） |
| 分析工具 | analyze_bbv.py → DFG 生成 → Agent融合分析 |
| 数据生成 | setup.sh (Steps 0-6) |

### 1.2 BBV 热点分布

Top 4 基本块覆盖 **82.42%** 的所有基本块执行次数：

| Rank | BB ID | 指令数 | BBV 占比 | 累计占比 | 功能描述 |
|------|-------|--------|---------|---------|---------|
| 1 | 31047 | 33 | 36.20% | 36.20% | 卷积内循环（完整） |
| 2 | 31294 | 23 | 30.75% | 66.95% | 卷积内循环前半（加载+第1轮FMA+部分第2轮FMA） |
| 3 | 31293 | 9 | 12.27% | 79.21% | 卷积内循环后半（第2轮FMA+指针递增+分支） |
| 4 | 31180 | 63 | 3.21% | 82.42% | 归一化/激活后处理 |

> BB 31294 与 BB 31293 构成**分裂循环体**：31294 负责加载+第一轮FMA+部分第二轮FMA，31293 完成剩余FMA+指针递增+条件分支。两者应作为一个逻辑循环体分析。

### 1.3 成本模型

假设 **每条标量指令 cost = 1**（忽略流水线延迟、缓存缺失等微架构因素）。

加权指令开销计算：
```
Weighted_Cost(BB) = BBV占比 × 指令数
```

原始热点区域总加权成本：

| BB | BBV占比 | 指令数 | 加权成本 |
|----|---------|--------|---------|
| 31047 | 36.20% | 33 | 1194.6 |
| 31294 | 30.75% | 23 | 707.3 |
| 31293 | 12.27% | 9 | 110.4 |
| 31180 | 3.21% | 63 | 202.2 |
| **合计** | **82.42%** | | **2214.5** |

---

## 2. 识别的融合模式

### 2.1 模式总览

| # | 模式名称 | 类型 | 出现的 BB | 频率 |
|---|---------|------|----------|------|
| A | 同基址连续加载 (Load Pair) | ISA 扩展 | 31047, 31294 | 2/4 |
| B | 乘数复用 (Dual-FMA) | ISA 扩展 | 31047, 31294, 31293, 31180 | 4/4 |
| C | 计数器递减+条件分支 (C&B) | 微架构/ISA | 31047, 31294+31293, 31180 | 3/4 |
| D | 指针递增 (addi 多路) | ISA 扩展 | 31047, 31293 | 2/4 |
| E | 加载-使用 (Load+FMA) | 微架构 | 31047, 31294 | 2/4 |
| F | 带自动递增的加载 (Post-inc Load) | ISA 扩展 | 31047, 31294 | 2/4 |
| G | 多指针递增+分支 (loop.end) | ISA 扩展 | 31293 | 1/4 |
| H | 连续栈存储 (Store Pair) | ISA 扩展 | 31180 | 1/4 |
| I | 除法-加法链 (Div-Add) | 微架构 | 31180 | 1/4 |
| J | 深度 FMA 链融合 (FMA Chain) | ISA 扩展 | 31180 | 1/4 |

### 2.2 各模式详细描述

#### 模式 A：同基址连续加载 (Load Pair / flw2)

将两条使用同一基址寄存器、偏移量连续的 `flw` 融合为一条 `flw2`。

```
# 融合前                          # 融合后
flw  frd1, offset(rs1)           flw2  frd1, frd2, offset(rs1)
flw  frd2, offset+4(rs1)
```

出现位置：
- **BB 31047**: 6 对可融合（基址 a6×2, a7×1, a5×3）→ 节省 6 条
- **BB 31294**: 6 对可融合（与 BB 31047 完全相同的加载模式）→ 节省 6 条

#### 模式 B：乘数复用 (Dual-FMA / fmadd2.s)

将两条共享同一乘数的 `fmadd.s` 融合为一条 `fmadd2.s`。这是**出现频率最高**的模式，覆盖全部 4 个 BB。

```
# 融合前                                      # 融合后
fmadd.s frd1, frs_shared, frs2a, frs3a       fmadd2.s frd1, frd2, frs_shared, frs2a, frs3a, frs2b, frs3b
fmadd.s frd2, frs_shared, frs2b, frs3b
```

出现位置：

| BB | 乘数 | 复用次数 | 可融合对数 | 节省指令 |
|----|------|---------|-----------|---------|
| 31047 | fa4×4, fa5×4, ft1×4, fa3×4 | 各4次 | 8对 | -8 |
| 31294 | fa4×4, fa5×4, ft1×2 | 4+4+2 | 5对 | -5（注1） |
| 31293 | fa3×3, ft3×2 | 3+2 | 2对 | -2（注2） |
| 31180 | fa2×9, fa3×9, fa4×9, fa5×9 | 各约4-5对/轮 × 5轮 | 21对 | -21 |

> 注1：BB 31294 第二轮 FMA 截断，ft1 仅复用 2 次。阶段 2 为 4 对，阶段 5 为 1 对。
> 注2：BB 31293 的 fa3 复用 3 次，可升级为 Triple-FMA（fmadd3.s）节省 3 条。

#### 模式 C：计数器递减+条件分支 (Compare-and-Branch)

将 `addi rs1, rs1, -imm` 与紧接的 `bgtu`/`bleu` 合并。

```
# 融合前                          # 融合后
addi  t1, t1, -2                 bgtui.dec  t1, t3, offset, 2
bgtu  t1, t3, offset
```

出现位置：

| BB | 指令 | 备注 |
|----|------|------|
| 31047 | insn 20 (addi -2) + insn 32 (bgtu) | 同一 BB 内 |
| 31294+31293 | 31294: insn 20 (addi -2) → 31293: insn 8 (bgtu) | **跨 BB** |
| 31180 | insn 45 (addi -4) + insn 62 (bleu) | 同一 BB 内，但中间隔 16 条指令 |

#### 模式 D：指针递增

卷积循环底部的多条 `addi` 指针递增。

```
addi  a6, a6, 8       ← 权重指针
addi  a5, a5, 32      ← 激活值指针
addi  a7, a7, 8       ← 权重指针
```

出现位置：BB 31047 (insn 29-31)、BB 31293 (insn 5-7)

#### 模式 E：加载-使用 (Load+FMA Macro-fusion)

微架构层面将 `flw` 与紧接的 `fmadd.s` 融合为单微操作，数据通过旁路传递。

出现位置：
- **BB 31047**: 30+ 对 load-use 关系
- **BB 31294**: 20 对 load-use 关系

此模式**不减少指令数**，但消除寄存器文件写-读往返延迟（约 1-2 周期/对）。

#### 模式 F：带自动递增的加载 (Post-increment Load)

将加载与基址指针递增合并。与模式 A (Load Pair) **互斥**（二选一）。

```
# 融合前                          # 融合后
flw   frd, offset(rs1)           flw.post  frd, offset(rs1), stride
...
addi  rs1, rs1, stride
```

出现位置：BB 31047（-3 条）、BB 31294（-3 条，指针递增在 31293 中需跨 BB 协调）

#### 模式 G：多指针递增+分支 (loop.end)

将循环尾部的所有整数操作合并为一条 `loop.end` 指令。

```
# 融合前（BB 31293）              # 融合后
addi  a6, a6, 8                  loop.end  a6, a5, a7, t1, t4, offset, 8, 32, 8
addi  a5, a5, 32                 # 语义: a6+=8, a5+=32, a7+=8; if t1>t4 goto offset
addi  a7, a7, 8
bgtu  t1, t4, offset
```

仅出现在 BB 31293（-4 条，含 bgtu）。与 Compare-and-Branch（31294 中的 addi -2）正交可叠加。

#### 模式 H：连续栈存储 (Store Pair / fsw2)

与 Load Pair 对称，将连续的 `fsw` 融合。

```
# 融合前                          # 融合后
fsw  frs1, 0(sp)                 fsw2  frs1, frs2, 0(sp)
fsw  frs2, 4(sp)
```

仅出现在 BB 31180（2 对 → -2 条）。

#### 模式 I：除法-加法链 (Div-Add Macro-fusion)

将 `fdiv.s` 与紧接的 `fadd.s` 合并为单微操作。

```
# 融合前                          # 融合后
fdiv.s  frd1, frs1, frs2         [fdiv+fadd] frd3, frs1, frs2, frs3
fadd.s  frd3, frd1, frs3
```

仅出现在 BB 31180（4 组 → 不减少指令数，消除 4 次寄存器文件往返延迟）。

#### 模式 J：深度 FMA 链融合 (FMA Chain Fusion)

将同一累加器链上多轮连续的 FMA 融合为一条链式指令。仅出现在 BB 31180。

```
# 融合前：5 轮依赖链                # 融合后
fmadd.s ft4, fa2, fs3, fs2        fmadd_chain.s ft0, ft0, fa2, {fs3,fs2,fs1,fs0,ft11}
fmadd.s ft4, ft4, fa2, fs1
fmadd.s ft4, ft4, fa2, fs0
fmadd.s ft4, fa2, ft4, ft11
fmul.s  ft0, ft0, ft4
```

BB 31180 有 8 条累加器链（4 组 × 2 子链），每条 4-5 轮 → 总计节省 ~32 条。与 Dual-FMA **互斥**（二选一）。

---

## 3. 各 BB 融合方案

### 3.1 BB 31047 — 卷积内循环（完整）

| 方案 | 融合后指令数 | 减少 | 难度 |
|------|------------|------|------|
| 原始 | 33 | — | — |
| + Load Pair | 27 | -6 | 低 |
| + Dual-FMA | 19 | -8 | 中 |
| + Compare-and-Branch | 18 | -1 | 低 |
| **最优组合** | **18** | **-15 (-45.5%)** | |
| 替代: Post-increment Load 替换 Load Pair | 21 | -12 (-36.4%) | |
| 叠加 Load+FMA Macro-fusion | 18 | 0 (降延迟) | 中 |

### 3.2 BB 31294 + BB 31293 — 卷积内循环（分裂）

合并分析（循环体共 32 条指令）：

| 方案 | 31294 指令数 | 31293 指令数 | 合计 | 减少 | 难度 |
|------|-----------|-----------|------|------|------|
| 原始 | 23 | 9 | 32 | — | — |
| + Load Pair (31294) | 17 | 9 | 26 | -6 | 低 |
| + Dual-FMA (31294+31293) | 11 | 7 | 18 | -8 | 中 |
| + Compare-and-Branch (跨BB) | 10 | 7 | 17 | -1 | 低 |
| + loop.end (31293) | 10 | 4 | 14 | -3 | 中 |
| **最优组合** | **10** | **4** | **14** | **-18 (-56.3%)** | |
| 替代: Triple-FMA 替换 Dual-FMA (31293) | 10 | 4 | 14 | 同上 | 中-高 |

### 3.3 BB 31180 — 归一化/激活后处理

| 方案 | 指令数 | 减少 | 难度 |
|------|--------|------|------|
| 原始 | 63 | — | — |
| + Dual-FMA | 42 | -21 | 中 |
| + Compare-and-Branch | 41 | -1 | 低 |
| + Store Pair | 39 | -2 | 低 |
| + Div-Add Macro-fusion | 39 | 0 (降延迟) | 中 |
| **最优组合 (Dual-FMA)** | **35** | **-28 (-44.4%)** | |
| 替代: FMA Chain Fusion 替换 Dual-FMA | 24 | -39 (-61.9%) | 高 |

---

## 4. 融合方案兼容性矩阵

<table class="compat">
<tr>
  <th></th><th>LP</th><th>PIL</th><th>DF</th><th>C&amp;B</th><th>addi3</th><th>loop</th><th>SP</th><th>FC</th>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">Load Pair (LP)</td>
  <td>—</td><td style="color:#b33;">互斥</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">Post-inc Load (PIL)</td>
  <td style="color:#b33;">互斥</td><td>—</td><td>✓</td><td>✓</td><td style="color:#b33;">互斥</td><td style="color:#b33;">互斥</td><td>✓</td><td>✓</td>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">Dual-FMA (DF)</td>
  <td>✓</td><td>✓</td><td>—</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td style="color:#b33;">互斥</td>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">C&amp;B</td>
  <td>✓</td><td>✓</td><td>✓</td><td>—</td><td>✓</td><td style="color:#666;">被包含</td><td>✓</td><td>✓</td>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">addi3</td>
  <td>✓</td><td style="color:#b33;">互斥</td><td>✓</td><td>✓</td><td>—</td><td style="color:#666;">被包含</td><td>✓</td><td>✓</td>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">loop.end</td>
  <td>✓</td><td style="color:#b33;">互斥</td><td>✓</td><td style="color:#666;">包含</td><td style="color:#666;">包含</td><td>—</td><td>✓</td><td>✓</td>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">Store Pair (SP)</td>
  <td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>—</td><td>✓</td>
</tr>
<tr>
  <td style="text-align:left;font-weight:bold;">FMA Chain (FC)</td>
  <td>✓</td><td>✓</td><td style="color:#b33;">互斥</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>—</td>
</tr>
</table>

<p style="font-size: 8pt; color: #6B5344; text-indent: 0;">
缩写：LP=Load Pair, PIL=Post-inc Load, DF=Dual-FMA, SP=Store Pair, FC=FMA Chain。<br/>
<span style="color:#b33;">红色</span>=互斥，<span style="color:#666;">灰色</span>=包含/被包含，✓=兼容。
</p>

---

## 5. 分阶段实施路径

### 阶段 1：微架构融合（无需修改 ISA）

| 方案 | 覆盖 BB | 效果 |
|------|---------|------|
| Compare-and-Branch | 31047, 31294+31293, 31180 | 每循环减少 1 条 addi |
| Load+FMA Macro-fusion | 31047, 31294 | 消除 ~50 对 load-use 延迟（每对 ~1-2 周期） |
| Div-Add Macro-fusion | 31180 | 消除 4 对 div-use 延迟 |

此阶段**零指令数减少**，但减少流水线气泡，预计降低关键路径 5-15%。

### 阶段 2：低难度 ISA 扩展

| 方案 | 新增指令 | 覆盖 BB | 每循环节省 |
|------|---------|---------|-----------|
| Load Pair (flw2) | 1 | 31047, 31294 | -6 × 2 = -12 |
| Compare-and-Branch (bgtui.dec) | 1 | 31047, 31294, 31180 | -1 × 3 = -3 |
| Store Pair (fsw2) | 1 | 31180 | -2 |

共需 **3 条新指令**，覆盖 4 个 BB。

### 阶段 3：中难度 ISA 扩展

| 方案 | 新增指令 | 覆盖 BB | 每循环节省 |
|------|---------|---------|-----------|
| Dual-FMA (fmadd2.s) | 1 | 31047, 31294, 31293, 31180 | -8 -5 -2 -21 = -36 |
| loop.end | 1 | 31293 | -3 |

共需 **2 条新指令**（累计 5 条）。

### 阶段 4：高难度 ISA 扩展（可选）

| 方案 | 新增指令 | 覆盖 BB | 每循环节省 |
|------|---------|---------|-----------|
| FMA Chain Fusion | 1 | 31180 | -32（替代 Dual-FMA 的 -21） |
| Post-increment Load | 1 | 31047, 31294 | 替代 Load Pair（净收益 ±0） |
| Triple-FMA (fmadd3.s) | 1 | 31293 | -3（替代 Dual-FMA 的 -2） |

---

## 6. 整体收益评估

### 6.1 各阶段加权成本节省

基于 cost=1 模型，Weighted_Cost = BBV占比 × 指令数：

#### 原始成本

| BB | BBV占比 | 原始指令 | 加权成本 |
|----|---------|---------|---------|
| 31047 | 36.20% | 33 | 1194.6 |
| 31294 | 30.75% | 23 | 707.3 |
| 31293 | 12.27% | 9 | 110.4 |
| 31180 | 3.21% | 63 | 202.2 |
| **合计** | **82.42%** | | **2214.5** |

#### 阶段 1：微架构融合（仅降低延迟，不减少指令数）

| BB | 融合后指令 | 加权成本 | 节省 |
|----|-----------|---------|------|
| 31047 | 33 | 1194.6 | 0 |
| 31294 | 23 | 707.3 | 0 |
| 31293 | 9 | 110.4 | 0 |
| 31180 | 63 | 202.2 | 0 |
| **合计** | | **2214.5** | **0 (0.0%)** |

> 注：阶段 1 不改变指令数，但减少 ~50+ 次 load-use 寄存器往返（每次 ~1-2 周期），
> 在周期级模型中预计可减少 5-15% 执行时间。

#### 阶段 2：+ 低难度 ISA 扩展

| BB | 融合后指令 | 加权成本 | 节省 |
|----|-----------|---------|------|
| 31047 | 33 - 6(Load Pair) - 1(C&B) = **26** | 941.2 | 253.4 |
| 31294 | 23 - 6(Load Pair) - 1(C&B) = **16** | 492.0 | 215.3 |
| 31293 | 9（C&B 在 31294 侧已计算） = **9** | 110.4 | 0 |
| 31180 | 63 - 2(Store Pair) - 1(C&B) = **60** | 192.6 | 9.6 |
| **合计** | | **1736.2** | **478.3 (21.6%)** |

#### 阶段 3：+ 中难度 ISA 扩展（Dual-FMA + loop.end）

| BB | 融合后指令 | 加权成本 | 节省 |
|----|-----------|---------|------|
| 31047 | 26 - 8(Dual-FMA) = **18** | 651.6 | 543.0 |
| 31294 | 16 - 5(Dual-FMA) = **11** | 338.3 | 369.0 |
| 31293 | 9 - 2(Dual-FMA) - 3(loop.end) = **4** | 49.1 | 61.3 |
| 31180 | 60 - 21(Dual-FMA) = **39** | 125.2 | 77.4 |
| **合计** | | **1164.2** | **1050.3 (47.4%)** |

#### 阶段 4：+ 高难度扩展（FMA Chain 替代 BB 31180 的 Dual-FMA）

| BB | 融合后指令 | 加权成本 | 节省 |
|----|-----------|---------|------|
| 31047 | **18** | 651.6 | 543.0 |
| 31294 | **11** | 338.3 | 369.0 |
| 31293 | **4** | 49.1 | 61.3 |
| 31180 | 60 - 32(FMA Chain) = **28** | 89.9 | 112.3 |
| **合计** | | **1128.9** | **1085.6 (49.0%)** |

> 阶段 4 vs 阶段 3：BB 31180 额外节省 39 - 28 = 11 条指令，但仅占 3.21% BBV，
> 加权收益仅 +35.3 (1.6%)。投入产出比较低。

### 6.2 收益汇总

```
                        加权成本     节省        节省比例
原始：                  2214.5       —           —
+ 阶段1 (微架构)：       2214.5       0           0.0%（降延迟）
+ 阶段2 (低难度ISA)：    1736.2      478.3       21.6%
+ 阶段3 (中难度ISA)：    1164.2     1050.3       47.4%
+ 阶段4 (高难度ISA)：    1128.9     1085.6       49.0%
```

### 6.3 对总执行时间的影响

上述节省仅覆盖 Top 4 BB（82.42% 的基本块执行次数）。设冷路径（剩余 17.58% 的 BB）平均每块 ~10 条指令，则冷路径加权成本约为 175.8。

```
Total 原始 ≈ 2214.5 + 175.8 = 2390.3
Total 阶段3 ≈ 1164.2 + 175.8 = 1340.0
总节省 = (2390.3 - 1340.0) / 2390.3 ≈ 43.9%
```

**阶段 3 的中难度 ISA 扩展方案（Load Pair + Compare-and-Branch + Dual-FMA + loop.end + Store Pair）可将 YOLOv11n 推理总指令数减少约 44%。**

### 6.4 各方案的独立贡献排名

| 方案 | 独立加权节省 | 占总节省比 | 覆盖 BB 数 |
|------|------------|-----------|-----------|
| Dual-FMA (fmadd2.s) | 579.0 | 55.1% | 4 |
| Load Pair (flw2) | 468.7 | 44.6% | 2 |
| loop.end | 36.8 | 3.5% | 1 |
| Compare-and-Branch | 25.1 | 2.4% | 3 |
| Store Pair (fsw2) | 6.4 | 0.6% | 1 |
| Post-increment Load | ~0 | 0% | 2 |
| FMA Chain Fusion | 35.3 | 3.4% | 1 |

> Dual-FMA 和 Load Pair 贡献了绝大多数收益（99.7%），是优化的核心目标。

---

## 7. 推荐实施优先级

### 第一优先级：Dual-FMA (fmadd2.s)

**理由**：
- 覆盖全部 4 个热点 BB，加权节省最大（579.0，占 55.1%）
- 融合规则统一（共享乘数 + 两个独立目标和操作数）
- 硬件实现：增加一个 FMA 端口或复制乘数读取即可支持
- 指令编码：需要 7 个寄存器操作数（2 dst + 5 src），可使用 R4-type 或自定义 48-bit 编码

### 第二优先级：Load Pair (flw2) / Store Pair (fsw2)

**理由**：
- 加权节省 468.7（占 44.6%），仅次于 Dual-FMA
- 实现简单：两个寄存器文件读端口 + 连续地址计算
- Load Pair 和 Store Pair 共享数据通路设计
- 指令编码仅需 3 个操作数，兼容标准 R-type 编码

### 第三优先级：Compare-and-Branch (bgtui.dec / bleui.dec)

**理由**：
- 涵盖 3/4 个 BB（31047, 31294+31293, 31180），模式统一
- x86 (CMP+JCC) 和 ARM (CBZ/CBNZ) 的成熟做法
- 可先以微架构融合方式实现，验证效果后再决定是否升级为 ISA 扩展
- 编码简单：标准 R-type + 小立即数

### 第四优先级：loop.end

**理由**：
- 仅覆盖 BB 31293，但将该块从 9 条压缩到 4 条（-44%）
- 与 Compare-and-Branch + Dual-FMA 正交，可叠加
- 编码挑战较大（9+ 个操作数），需自定义编码
- 收益有限（加权 36.8），建议在核心方案验证后再实施

### 不推荐：Post-increment Load

**理由**：
- 与 Load Pair 互斥，且节省量相同（-3 条/循环）
- 跨 BB 协调复杂（31294 加载 + 31293 递增）
- 需要 rs1 同时作为源和隐式目标，增加寄存器写入冲突风险
- 编码需要额外立即数字段

### 不推荐：FMA Chain Fusion（相对 Dual-FMA）

**理由**：
- 加权收益仅 +35.3（1.6%），投入产出比低
- 编码空间压力大（需编码 5+ 个源操作数）
- 仅适用于 BB 31180 的深度链模式，通用性差
- Dual-FMA 已覆盖 BB 31180 的主要 FMA（-21 条）

---

## 8. ISA 扩展建议

### 8.1 推荐新增指令

| 指令 | 编码 | 语义 | 优先级 |
|------|------|------|--------|
| `fmadd2.s` | 自定义 (7 reg ops) | frd1,frd2 = frs1×frs2a+frs3a, frs1×frs2b+frs3b | P0 |
| `flw2` | R4-type 或扩展 | frd1,frd2 = mem[rs1+offset], mem[rs1+offset+4] | P0 |
| `fsw2` | R4-type 或扩展 | mem[rs1+offset], mem[rs1+offset+4] = frs1,frs2 | P0 |
| `bgtui.dec` / `bleui.dec` | 自定义扩展 | rs1-=imm; if rs1>rs2 goto offset | P1 |
| `loop.end` | 48-bit 或查表编码 | rs1+=s1, rs2+=s2, rs3+=s3; if cond goto off | P2 |

### 8.2 RegisterFlow 定义

```python
FUSION_OPS = [
    ("fmadd2.s", RegisterFlow(
        dst_regs=['frd1', 'frd2'],
        src_regs=['frs1', 'frs2a', 'frs3a', 'frs2b', 'frs3b']
    )),
    ("flw2", RegisterFlow(
        dst_regs=['frd1', 'frd2'],
        src_regs=['rs1']     # rs1 = base address
    )),
    ("fsw2", RegisterFlow(
        dst_regs=[],          # 写内存
        src_regs=['frs1', 'frs2', 'rs1']
    )),
    ("bgtui.dec", RegisterFlow(
        dst_regs=['rs1', 'pc'],   # rs1 递减, pc 条件更新
        src_regs=['rs1', 'rs2']
    )),
    ("loop.end", RegisterFlow(
        dst_regs=['rs1', 'rs2', 'rs3', 'pc'],
        src_regs=['rs1', 'rs2', 'rs3', 't_cond', 't_limit']
    )),
]
```

---

## 9. 结论

针对 ONNX Runtime 标量版 + YOLOv11n 推理的热点分析，Top 4 基本块（BB 31047/31294/31293/31180）覆盖 82.42% 的执行次数，存在大量可融合的指令模式。

**核心发现**：

1. **Dual-FMA (fmadd2.s) 是收益最大的单一方案**，贡献总节省的 55.1%。YOLO 推理的卷积和归一化内核中广泛存在乘数复用模式。

2. **Load Pair (flw2) 是第二大贡献者**，占 44.6%。与 Dual-FMA 正交，两者合计覆盖 99.7% 的指令数节省。

3. **仅实施阶段 3（Load Pair + Dual-FMA + Compare-and-Branch + loop.end + Store Pair）**，即可将热点区域的加权指令成本降低 **47.4%**，总执行指令数降低约 **44%**。

4. 所需新增指令仅 **3-5 条**，其中前 3 条（fmadd2.s、flw2、fsw2）即可覆盖绝大部分收益。

5. 高难度方案（FMA Chain Fusion、Post-increment Load）的边际收益有限（<2%），不建议优先实施。
