# 拟扩展向量指令编码方式分析

## 背景

对于拟扩展的向量类型指令（如 `vdot.s8`、`vmulsub.vx`），存在两种编码实现方式：

1. **Fixed-width Encoding**：元素位宽直接编码进指令，类似 x86/ARM NEON
2. **vset-configured Encoding**：参照 RVV 方式，元素位宽由 `vsetvli` 配置

本文从软件角度分析两种方式的利弊。

---

## 方案对比

### 方案一：Fixed-width Encoding

元素位宽直接编码进指令 opcode，类似 x86 `VPMADDUBSW`（固定 u8×i8→i16）或 ARM `SDOT`（固定 4×i8→i32）。

```asm
# 假设的编码方式
vdot.s8   v0, v1, v2    # 固定：int8 点积，4 elements → int32
vdot.s16  v0, v1, v2    # 固定：int16 点积，需要单独指令
vmulsub.b v0, v1, x0    # 固定：byte 宽度的掩码乘减
vmulsub.h v0, v1, x0    # 固定：halfword，需要单独指令
```

特点：
- 每种位宽占用独立的 opcode 空间
- 指令行为自解释，不依赖外部配置状态
- 扩展新位宽需新增指令编码

### 方案二：vset-configured Encoding

与现有 RVV 指令一致，SEW/LMUL/VL 由 `vsetvli` 决定。

```asm
# SEW=8 时：int8 点积
vsetvli t0, a0, e8, m1
vdot.vv   v0, v1, v2    # 行为由 vset 配置决定

# SEW=16 时：int16 点积（如果语义允许）
vsetvli t0, a0, e16, m1
vdot.vv   v0, v1, v2    # 同一指令，不同行为
```

特点：
- 单一 opcode，位宽由运行时配置决定
- 与现有 RVV 编程模型一致
- 编译器需追踪/插入 vset 状态

---

## 利弊分析

### 1. 编码空间

| 指标 | Fixed-width | vset-configured |
|------|-------------|-----------------|
| 指令编码开销 | 每种位宽占用独立 opcode 空间 | 单一 opcode，位宽由 vset 决定 |
| 扩展 N 种位宽 | 需 N 个 opcode | 仅需 1 个 opcode |
| 适合场景 | 语义与位宽强绑定 | 通用算术操作 |

**编码空间并非决定性因素**——RVV 保留了大量编码空间用于未来扩展。32 位指令编码有足够的 `funct3`/`funct6`/`funct7` 位来区分变���。

### 2. 编译器支持

这是**软件最关心的因素**。

#### Fixed-width 的编译器影响

```c
// 需要为每种位宽提供独立 intrinsic
int32_t result = __riscv_vdot_s8_i32(v0, v1, v2);   // int8 版
int32_t result = __riscv_vdot_s16_i64(v0, v1, v2);  // int16 版（如果存在）
```

- 每种变体需要一个独立的 intrinsic 函数名
- 编译器不需要追踪/插入 `vsetvli`——**简单**
- 但增加了 API 数量

#### vset-configured 的编译器影响

```c
// 单一 intrinsic，编译器自动处理 vset
vint32m1_t result = __riscv_vdot_vv_i32m1(v0, v1, v2, vl);
// 编译器负责确保 SEW=8 已设置
```

- LLVM/GCC 已有成熟的 vset 插入和省略逻辑（`RVVRegTile`、`RVVPseudos`）
- 遵循现有 RVV 编程模型——**编译器改动最小**
- 如果上下文中 SEW 不正确，编译器需额外插入 `vsetvli`（2 周期开销）

**结论**：vset-configured 编译器改动更小，复用现有基础设施。

### 3. 运行时性能

以 `vec_dot_q5_0_q8_0` 的 VLEN=128 路径为例，当前代码中已有 vset 调用：

```c
// 第一段：处理低半字 (SEW=8, LMUL=1, VL=16)
size_t vl = qk / 2;  // 16
vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);  // 隐式 vset e8,m1

// 第二段：处理全部 32 元素 (SEW=8, LMUL=2, VL=32)
vl = qk;  // 32
vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);  // 隐式 vset e8,m2
vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);  // SEW=8 已设置
```

**关键观察**：在 llama.cpp 的量化 kernel 中，SEW 在整个循环中始终为 8。`vsetvli` 被调用的原因是改变 LMUL（m1→m2→m4），而非改变 SEW。

因此对于 `vdot.s8`：

| 指标 | Fixed-width | vset-configured |
|------|-------------|-----------------|
| 额外 vsetvli 开销 | 0 周期（不需要） | 0 周期（SEW=8 已在上下文中正确） |
| 循环内摊销 | 无需摊销 | 已被其他操作摊销 |

**结论**：在此场景下，两种方式性能无差异。

### 4. 语义一致性

这是最关键的区分点——需要按指令类型分别讨论。

#### 对于点积指令 `vdot`：语义与位宽强绑定

ARM `SDOT` 和 x86 `VPMADDUBSW` 都是固定宽度的，原因在于：
- `4×i8→i32` 的 4:1 缩减比是 int8 特有的
- `2×i16→i32` 需要完全不同的硬件实现
- 将两者编码为同一指令在硬件层面没有复用价值

```
vdot (i8):  4个 int8 乘积累加 → 1个 int32   (4:1 reduction)
vdot (i16): 2个 int16 乘积累加 → 1个 int32   (2:1 reduction)  ← 完全不同的数据通路
```

**结论**：点积指令更适合 fixed-width 编码。

#### 对于通用操作 `vmulsub`：语义与位宽无关

掩码乘减（masked subtract）的本质是"对选中元素减去标量值"，这在 byte/halfword/word 上语义完全一致：

```c
// byte 版本
vint8m1_t r = vsub(v, 0x10);   // 对每个 i8 元素减去 16

// halfword 版本
vint16m1_t r = vsub(v, 0x10);  // 对每个 i16 元素减去 16
```

这是纯粹的算术操作，与现有的 `vsub.vx`、`vadd.vx` 性质相同。

**结论**：通用操作更适合 vset-configured 编码。

### 5. 可编程性与灵活性

| 指标 | Fixed-width | vset-configured |
|------|-------------|-----------------|
| 新位宽支持 | 需新增指令和编码 | 仅需编译器支持新 SEW 值 |
| 跨位宽代码 | 需 `#if` 选择不同 intrinsic | 单一代码路径，运行时决定 |
| 调试可读性 | 指令名直接标明位宽 | 需结合 vset 上下文理解 |

### 6. 调试与可观测性

在 perf/trace 场景下：

```asm
# Fixed-width — 自解释
vdot.s8   v0, v1, v2

# vset-configured — 需追踪上下文
vsetvli t0, zero, e8, m1
...
vdot.vv   v0, v1, v2      # 此时 SEW=8，但 50 条指令前可能被改为 e16
```

**Fixed-width 在 trace 分析中优势明显**——这正是 QEMU BBV profiling 和 hotspot 分析时会遇到的场景。

---

## 推荐：按指令类型分情况选择

| 指令类型 | 推荐方式 | 理由 |
|---------|---------|------|
| **`vdot`**（点积） | **Fixed-width** | 语义与位宽强绑定（4:1 vs 2:1 缩减比不同）；ARM/x86 均采用此方式；trace 可读性好 |
| **`vmulsub`**（掩码乘减） | **vset-configured** | 纯算术操作，语义与位宽无关；复用现有 vset 基础设施；编译器改动最小 |
| **通用算术扩展** | **vset-configured** | 与现有 RVV 一致性；避免编码空间膨胀 |
| **特殊语义操作** | **Fixed-width** | 当操作的语义含义（如缩减比、饱和行为）因位宽而异时 |

---

## 核心原则

> **如果操作对不同位宽有不同的数学语义，用 fixed-width；如果只是相同的操作作用于不同宽度的数据，用 vset-configured。**

判断标准：

1. **缩减比是否因位宽而变？**
   - 点积：i8 是 4:1，i16 是 2:1 → **不同** → Fixed-width
   - 加减乘：不涉及缩减 → **相同** → vset-configured

2. **硬件实现是否完全不同？**
   - 点积：4-element 累加器 vs 2-element 累加器 → **不同** → Fixed-width
   - 掩码减法：同一减法器单元，仅宽度不同 → **相同** → vset-configured

3. **trace 分析中是否需要自解释？**
   - 高频热点操作 → Fixed-width（便于 perf/BBV 分析）
   - 低频辅助操作 → vset-configured（编码空间更紧凑）

---

## 附录：llama.cpp vec_dot 系列指令编码建议

基于上述分析，针对 llama.cpp 性能热点分析中识别的四个量化 kernel：

| Kernel | 拟扩展指令 | 推荐编码方式 |
|--------|-----------|-------------|
| Q5_0×Q8_0 | `vdot.s8` | Fixed-width |
| Q6_K×Q8_K | `vdot.s8`（复用） | Fixed-width |
| Q4_K×Q8_K | `vdot.s8`（复用） | Fixed-width |
| Q8_0×Q8_0 | `vdot.s8`（复用） | Fixed-width |

四个 kernel 均涉及 int8×int8 点积，可共享同一 `vdot.s8` 指令（fixed-width 编码）。

---

*文档生成时间: 2026-04-17*