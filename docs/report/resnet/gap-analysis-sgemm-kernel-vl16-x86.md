# SGEMM Kernel 多平台向量实现对比与RVV扩展指令建议 (x86平台分析)

## 概述

**分析目标**: SGEMM (单精度通用矩阵乘法) 内核 K 循环热点路径
**基准实现**: RVV VL=16 (VLEN=512bit, SEW=32bit, LMUL=1)
**分析平台**: x86 AVX (256-bit), x86 FMA3 (256-bit), x86 AVX-512F (512-bit)
**BBV数据**: 未提供，收益为理论估算

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vfmacc.vf_mem | AVX-512F | BB内减少25% (K循环, 12行模式) | 中 | 需单独 flw 加载 A 元素 |
| P1 | 已有 vse32.v + mask | AVX-512F | 部分列存储从标量提取改为向量掩码存储，BB内减少约80% (部分列BB) | 低 | RVV已有掩码存储能力，但当前实现未使用 |

**收益计算方式**（理论估算，无BBV profiling数据）：
- BB内收益仅反映单个基本块范围内的指令减少比例
- 建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算整体收益
- 所有计算已归一化到 RVV VLEN=512bit, SEW=32bit
- AVX/AVX2 (256-bit) 归一化因子: 2x（2条AVX指令 = 1条RVV指令的元素吞吐量）
- AVX-512F (512-bit) 归一化因子: 1x（等宽，直接对比）

---

## 基准RVV实现分析

### 循环结构

RVV sgemm-kernel-vl16 的核心循环结构：

```
N-loop (列, 每16列一块):
  清零累加器
  K-loop (展开2, 每步处理2个K):
    Row 0:
      flw  fa0, 0(a0)          # 加载 A[m,k]
      vle32.v v0, (b_ptr)      # 加载 B[k, 0:15]
      flw  fa1, 0(a0+4)        # 加载 A[m,k+1]
      vfmacc.vf v2, fa0, v0    # acc += A[m,k] * B[k,:]
      vle32.v v1, (b_ptr+64)   # 加载 B[k+1, 0:15]
      vfmacc.vf v2, fa1, v1    # acc += A[m,k+1] * B[k+1,:]
    Row 1:
      flw  fa2, 0(a1)          # 加载 A[m+1,k]
      flw  fa3, 0(a1+4)        # 加载 A[m+1,k+1]
      vfmacc.vf v3, fa2, v0    # acc += A[m+1,k] * B[k,:]  (重用 B 加载)
      vfmacc.vf v3, fa3, v1    # acc += A[m+1,k+1] * B[k+1,:] (重用 B 加载)
    a0 += 8; a1 += 8; b_ptr += 128
  Alpha 乘法: vfmul.vf (每行1条)
  输出: vse32.v (全块) / 标量提取 (部分块)
```

### 每 K 对 (2个K步) 指令计数 — 2行 x 16列

| 指令 | 数量 | 说明 |
|------|------|------|
| flw (A元素) | 4 | 每行2个K步各1个 |
| vle32.v (B向量) | 4 | 2个K步各1个 (16 floats each) |
| vfmacc.vf (FMA) | 4 | 每行2个K步各1个 |
| 指针更新 | ~3 | add a0, add a1, add b_ptr |
| **合计** | **~15** | **2行 x 16列 x 2K步 = 64 FLOPs** |

> 注: Row 1 复用 Row 0 的 B 向量加载，因此 B 加载只需要 4 条而非 8 条。

### 每 K 步归一化到 12行 x 16列 的指令计数

由于 RVV 每次调用只处理 2 行，处理 12 行需要 6 次独立调用：

| 项目 | 值 |
|------|-----|
| 每调用指令数 (2行, 1 K步) | ~7 (2 flw + 1 vle32.v + 2 vfmacc.vf + 2 指针更新) |
| 12行需要调用次数 | 6 |
| RVV 总指令数 (12行, 1 K步, 16列) | 6 x 7 = **42** |
| 对应 FLOPs | 12 x 16 x 1 = **192** |

---

## 各平台对比分析

### 1. x86 AVX (256-bit, 无FMA)

**核心特点**：
- 256-bit YMM 寄存器，每寄存器 8 x float32
- 无 FMA 指令，使用分离的 `vmulps` + `vaddps` 实现乘加
- 最多处理 2 行，每行 2 个 YMM = 16 列
- K 循环展开 4 (主循环每次处理 4 个 K 步)

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vbroadcastss ymm, [mem]` | 从内存加载标量并广播到 YMM 所有 lane | RVV 无等价指令，需 `flw` + 手动使用标量寄存器 |
| `vmulps` + `vaddps` | 分离的乘法和加法 | RVV 有 `vfmacc.vf` 融合乘加，此方向无差距 |
| `vmaskmovps` | 掩码存储 (部分列) | RVV 有 `vse32.v` + mask，但当前实现未使用 |

**ComputeBlockAvxBy16 宏分析 (2行, 16列, 1 K步)**：

```asm
# RowCount=2, 2个YMM = 16列
vmovaps     ymm0, [edx+VectorOffset]       # 加载 B 列 0-7
vmovaps     ymm1, [edx+VectorOffset+32]    # 加载 B 列 8-15
vbroadcastss ymm3, [ecx+BroadcastOffset]   # 加载并广播 A[row0,k]
vmulps      ymm2, ymm3, ymm0               # A * B_lo
vaddps      ymm4, ymm2, ymm4               # 累加 C[row0]_lo
vmulps      ymm2, ymm3, ymm1               # A * B_hi
vaddps      ymm5, ymm2, ymm5               # 累加 C[row0]_hi
vbroadcastss ymm3, [ecx+ebx+BroadcastOffset] # 加载并广播 A[row1,k]
vmulps      ymm2, ymm3, ymm0               # A * B_lo (重用 B)
vaddps      ymm6, ymm2, ymm6               # 累加 C[row1]_lo
vmulps      ymm2, ymm3, ymm1               # A * B_hi (重用 B)
vaddps      ymm7, ymm2, ymm7               # 累加 C[row1]_hi
```

**指令计数 (2行 x 16列 x 1 K步, 原始)**：

| 指令 | 数量 |
|------|------|
| vmovaps (B加载) | 2 |
| vbroadcastss (A广播) | 2 |
| vmulps (乘法) | 4 |
| vaddps (加法) | 4 |
| **合计** | **12** |

**归一化到 RVV VLEN=512**：归一化因子 = 2x (AVX 256-bit vs RVV 512-bit)

| 项目 | 值 |
|------|-----|
| 归一化后 AVX 指令数 | 12 x 2 = **24** (等效 512-bit) |
| RVV 基准指令数 (2行, 1 K步) | **7** |
| RVV 优势 | (24 - 7) / 24 = **减少71%** |

> **分析**: AVX 无 FMA 是最大劣势。分离的 `vmulps` + `vaddps` 使指令数翻倍。RVV 的 `vfmacc.vf` 在此维度上已超越 AVX1。AVX 的 `vbroadcastss` 虽然是内存广播指令，但与 FMA 缺陷相比影响较小。此平台不构成 RVV 扩展的紧迫需求。

---

### 2. x86 FMA3 (256-bit, 有FMA)

**核心特点**：
- 256-bit YMM 寄存器，每寄存器 8 x float32
- 使用 `vfmadd231ps` (FMA3) 融合乘加
- 最多处理 6 行，每行 2 个 YMM = 16 列
- K 循环展开 4 (主循环每次处理 4 个 K 步)
- `ComputeBlockFma3By2`: 处理 2 个 YMM (32 列，用于 N>=16 的列块)

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vbroadcastss ymm, [mem]` | 从内存加载标量并广播到 YMM 所有 lane | RVV 无内存广播指令，需 `flw` 后使用标量寄存器 |
| `vfmadd231ps` | 融合乘加: acc += src1 * src2 | RVV 有等价 `vfmacc.vf` |
| `vmaskmovps` | 掩码加载/存储 (部分列) | RVV 有 `vse32.v` + mask |

**ComputeBlockFma3By2 宏分析 (N行, 32列, 1 K步)**：

```asm
# N=6 示例
vmovaps     ymm0, [rsi+VectorOffset]       # 加载 B 块 lo (8 floats)
vmovaps     ymm1, [rsi+VectorOffset+32]    # 加载 B 块 hi (8 floats)
# 每行:
vbroadcastss ymm3, [rdi+row_offset]        # 加载并广播 A[row,k]
vfmadd231ps ymm_acc_lo, ymm3, ymm0         # C[row]_lo += A * B_lo
vfmadd231ps ymm_acc_hi, ymm3, ymm1         # C[row]_hi += A * B_hi
```

**指令计数 (6行 x 16列 x 1 K步)**：

注: FMA3 By2 模式处理 32 列 (2 个 YMM)，我们取 16 列对应量。

对于 6行 x 16列 (1 个 YMM = 8 floats, 需要 2 个 YMM):

每 K 步 (ComputeBlockFma3By2 被调用一次，处理 2 个 YMM = 32 列):
- B 加载: 2 (vmovapf ymm0, ymm1)
- 每行: 1 `vbroadcastss` + 2 `vfmadd231ps` = 3 条
- 6行: 2 + 6 x 3 = **20** 条指令 → 6行 x 32列 x 1K = 192 FLOPs

归一化到 16 列 (等效工作量的一半): 10 条指令 → 6行 x 16列 x 1K = 96 FLOPs

| 项目 | 值 |
|------|-----|
| 归一化因子 (256-bit → 512-bit) | 2x |
| 归一化后 FMA3 指令数 (6行, 16列, 1 K步) | 10 x 2 = **20** |
| RVV 等效指令数 (6行, 16列, 1 K步) | 3 x 6/2 = **9** (3次2行调用, 每次~7条, 但B加载有复用) |

> 注: FMA3 的 `vbroadcastss [mem]` 对比 RVV 的 `flw + vfmacc.vf`: FMA3 每行需要 1 条广播指令，RVV 每行需要 1 条 `flw`。两者等价 — `vbroadcastss [mem]` 和 `flw` 都是 1 条标量加载指令，区别仅在于广播发生在寄存器级还是作为 FMA 的一部分。FMA3 的广播结果保存到临时 YMM 寄存器 (ymm3)，然后传给 `vfmadd231ps`，占用 1 个向量寄存器。

**与 RVV 的关键差异**: `vbroadcastss` 直接从内存广播到向量寄存器。RVV 的 `vfmacc.vf` 使用标量浮点寄存器 (f 寄存器) 作为操作数，广播是隐式的（硬件将标量广播到所有 lane）。因此 `vbroadcastss [mem]` + `vfmadd231ps` 与 `flw` + `vfmacc.vf` 在指令数上完全等价（都是 2 条指令）。此平台不产生新的扩展需求。

---

### 3. x86 AVX-512F (512-bit, 有FMA + 广播FMA)

**核心特点**：
- 512-bit ZMM 寄存器，每寄存器 16 x float32 (与 RVV VLEN=512 等宽)
- 使用 `vfmadd231pf` (FMA) 和 `vfmadd231pf_bcst` (广播 FMA，`{1to16}`)
- 最多处理 12 行 (使用 zmm4-zmm27 共 24 个 ZMM 累加器)
- K 循环展开 4 (主循环每次处理 4 个 K 步)
- 支持 opmask (k1-k7) 掩码存储
- `vfmadd231pf_bcst`: **单指令完成标量内存加载 + 广播 + FMA**

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vfmadd231pf_bcst zmm, zmm, [mem]{1to16}` | 从内存加载1个float32，广播到16 lane，执行 FMA: acc += scalar * vector | **RVV 无等价指令。** RVV 的 `vfmacc.vf` 需要标量寄存器操作数，必须先 `flw` 到 f 寄存器 |
| `vmovupf [mem]{k1}, zmm` | 掩码向量存储 (仅存储 mask=1 的 lane) | RVV 有 `vse32.v` + mask，但当前 sgemm 实现未使用，回退到标量提取 |
| `vfmadd213pf zmm{k1}, zmm, [mem]` | 掩码 FMA (用于 C += alpha * acc 的归约) | RVV 有 `vfmacc.vv` + mask，可直接使用 |

#### 3.1 ComputeBlockAvx512FBy1 — 单 ZMM 模式 (16列)

这是最关键的对比点。`ComputeBlockAvx512FBy1` 展示了 AVX-512 的广播-FMA 能力。

```asm
# 12行 x 16列 x 1 K步
vmovapf     zmm0, [rsi+VectorOffset]        # 加载 B (16 floats)
vfmadd231pf_bcst zmm5, zmm0, [rdi+0]        # row0: load A[m,k], broadcast, FMA
vfmadd231pf_bcst zmm7, zmm0, [rdi+lda]      # row1: load A[m+1,k], broadcast, FMA
vfmadd231pf_bcst zmm9, zmm0, [rdi+lda*2]    # row2: load A[m+2,k], broadcast, FMA
vfmadd231pf_bcst zmm11, zmm0, [rbx+0]       # row3: ...
vfmadd231pf_bcst zmm13, zmm0, [rbx+lda]
vfmadd231pf_bcst zmm15, zmm0, [rbx+lda*2]
vfmadd231pf_bcst zmm17, zmm0, [r13+0]       # row6 (9行偏移)
vfmadd231pf_bcst zmm19, zmm0, [r13+lda]
vfmadd231pf_bcst zmm21, zmm0, [r13+lda*2]
vfmadd231pf_bcst zmm23, zmm0, [r14+0]       # row9 (12行偏移)
vfmadd231pf_bcst zmm25, zmm0, [r14+lda]
vfmadd231pf_bcst zmm27, zmm0, [r14+lda*2]
```

**指令计数 (12行 x 16列 x 1 K步, AVX-512 By1)**：

| 指令 | 数量 |
|------|------|
| vmovapf (B 加载) | 1 |
| vfmadd231pf_bcst (广播FMA) | 12 (每行1条) |
| **合计** | **13** |
| FLOPs | 12 x 16 = 192 |

**RVV 基准等效 (12行 x 16列 x 1 K步)**：

由于 RVV 只处理 2 行，需要 6 次调用:

| 指令 | 每调用 (2行) | 6次调用合计 |
|------|-------------|------------|
| flw (A 元素) | 2 | 12 |
| vle32.v (B 加载) | 1 | 6 |
| vfmacc.vf (FMA) | 2 | 12 |
| 指针更新 | ~2 | ~12 |
| **合计** | **~7** | **~42** |
| FLOPs | 32 | 192 |

**核心差异分析**：

| 对比项 | AVX-512 By1 | RVV 基准 | 差异 |
|--------|------------|----------|------|
| 总指令数 (12行, 16列, 1K) | 13 | 42 | RVV 多 223% |
| A 元素加载 | 0 (内嵌于 vfmadd231pf_bcst) | 12 (独立 flw) | RVV 多 12 条 |
| B 向量加载 | 1 (所有行共享) | 6 (每次调用各1条) | RVV 多 5 条 |
| FMA 指令 | 12 (每行1条) | 12 (每行1条) | **相同** |
| 指针更新 | 0 (地址编码在指令中) | ~12 | RVV 多 12 条 |

**分析**: AVX-512 在此场景有三个优势:
1. **`vfmadd231pf_bcst` 消除了独立的标量加载**: 每行节省 1 条 `flw`
2. **B 向量在所有行之间共享**: 只需 1 次 `vmovapf`，而 RVV 每次 2 行调用需要 1 次 `vle32.v`
3. **12 行同时处理**: 大幅减少循环开销

其中第 2 和第 3 点是软件架构差异（多行处理策略），不直接对应指令集扩展。第 1 点是纯 ISA 差距。

#### 3.2 ComputeBlockAvx512FBy2 — 双 ZMM 模式 (32列)

```asm
# 12行 x 32列 x 1 K步
vmovapf     zmm0, [rsi+0]                   # 加载 B 列 0-15
vmovapf     zmm1, [rsi+64]                  # 加载 B 列 16-31
# 每行:
vbroadcastsf zmm3, [rdi+row_offset]          # 加载并广播 A[row,k]
vfmadd231pf zmm_acc_even, zmm3, zmm0         # C[row]_lo += A * B_lo
vfmadd231pf zmm_acc_odd, zmm3, zmm1         # C[row]_hi += A * B_hi
```

**指令计数 (12行 x 32列 x 1 K步, AVX-512 By2)**：

| 指令 | 数量 |
|------|------|
| vmovapf (B 加载) | 2 |
| vbroadcastsf (A 广播) | 12 (每行1条) |
| vfmadd231pf (FMA) | 24 (每行2条) |
| **合计** | **38** |
| FLOPs | 12 x 32 = 384 |

**RVV 等效 (12行 x 16列 x 2K步, 归一化到相同 FLOPs)**:

| 指令 | 6次调用合计 |
|------|------------|
| flw (A 元素) | 24 (6调用 x 2行 x 2K步) |
| vle32.v (B 加载) | 12 (6调用 x 2K步) |
| vfmacc.vf (FMA) | 24 (6调用 x 2行 x 2K步) |
| 指针更新 | ~24 |
| **合计** | **~84** |

> 注: 此处 By2 模式下 AVX-512 使用的 `vbroadcastsf` + `vfmadd231pf` 是分离的两条指令（与 By1 模式的 `vfmadd231pf_bcst` 不同），因为 B 向量已在寄存器中，广播值需要被复用于 2 次 FMA。

#### 3.3 K 循环展开效率

AVX-512 K 循环展开 4 (ComputeBlockLoop 每次迭代处理 4 个 K 步):

**12行 x 16列, 4个K步, AVX-512 By1 模式**:

| 指令 | 每K步 | 4K步合计 |
|------|-------|---------|
| vmovapf (B 加载) | 1 | 4 |
| vfmadd231pf_bcst | 12 | 48 |
| 循环控制 (sub, jae) | ~0.5 | ~2 |
| **合计** | **~13** | **~54** |
| FLOPs | 192 | 768 |

**12行 x 16列, 4个K步, AVX-512 By2 模式 (32列宽度)**:

每次迭代: 4 次 By2 调用 + 2 次 B 指针更新 + 1 次 A 指针更新 + 循环控制

| 指令 | 每次 By2 调用 | 4次合计 |
|------|-------------|--------|
| vmovapf (B 加载) | 2 | 8 |
| vbroadcastsf (A 广播) | 12 | 48 |
| vfmadd231pf (FMA) | 24 | 96 |
| B 指针更新 | 0.5 | 2 |
| A 指针更新 | 0.25 | 1 |
| 循环控制 | 0.5 | 2 |
| **合计** | **~38.75** | **~157** |
| FLOPs | 384 | 1536 |

#### 3.4 掩码存储 (部分列处理)

AVX-512 处理 N < 16 的尾部列:

```asm
# 计算掩码
mov   ebp, 1
shl   ebp, cl              # cl = 剩余列数
dec   ebp
kmovw k1, ebp              # 设置 opmask

# 掩码 FMA (C += alpha * acc)
vfmadd213pf zmm5{k1}, zmm31, [rdx]
vfmadd213pf zmm7{k1}, zmm31, [rdx+rax]
...

# 掩码存储
vmovupf [rdx]{k1}, zmm5
vmovupf [rdx+rax]{k1}, zmm7
...
```

每行仅需 1 条掩码存储指令处理任意剩余列数。

**RVV 现状**: `vse32.v` 支持掩码存储 (`vse32.v v0, (addr), v0.t`)，但当前实现使用标量提取逐个写回。

---

## RVV扩展指令建议详细说明

### [P0] vfmacc.vf_mem — 标量内存广播FMA

**指令定义**：
```
vfmacc.vf_mem vd, rs1, (rs2)    # vd[i] += MEM[rs2] * vs1[i], i = 0..VL-1
                                 # 等价于: flw ft0, 0(rs2); vfmacc.vf vd, ft0, vs1
```

- 格式: 自定义 R 型扩展，使用 2 个源寄存器 + 1 个向量寄存器
- 语义: 从内存地址 rs2 加载 1 个 SEW 宽度的标量，广播到所有活跃 lane，与向量寄存器 vs1 执行乘加运算，结果写入 vd
- 约束: rs2 寻址方式为基址寄存器 + 可选偏移 (与 `flw` 一致)，vd != vs1 (避免 WAW 冲突)
- 对比 x86: 等价于 AVX-512 的 `vfmadd231pf_bcst zmm, zmm, [mem]{1to16}`

**应用场景**：
- SGEMM K 循环中 A 元素的加载与 FMA 运算合并
- 任何需要标量 x 向量 FMA 的场景，且标量来自内存

**性能对比**：

以 12行 x 16列 x 1 K步为例:

| 实现 | 指令序列 | 指令数 |
|------|---------|--------|
| RVV 当前 (vfmacc.vf) | `flw fa0,[A]; vfmacc.vf vd,fa0,vs1` | 2 per row |
| RVV 扩展 (vfmacc.vf_mem) | `vfmacc.vf_mem vd,vs1,[A]` | 1 per row |
| AVX-512 | `vfmadd231pf_bcst zmm,zmm,[A]{1to16}` | 1 per row |

**单次 K 步 12行模式指令对比**:

| 指令类别 | RVV 当前 | RVV + vfmacc.vf_mem | 减少 |
|----------|---------|---------------------|------|
| A 加载 | 12 flw | 0 | -12 |
| B 加载 | 6 vle32.v | 6 vle32.v | 0 |
| FMA | 12 vfmacc.vf | 12 vfmacc.vf_mem | 0 |
| 指针更新 | ~12 | ~12 | 0 |
| **合计** | **~42** | **~30** | **-29%** |

> 注: 如果 RVV 实现扩展为 12 行共享 B 向量加载 (消除 5 次冗余 B 加载)，则:
> - 当前: ~42 → 扩展后 (仅 vfmacc.vf_mem): ~30 → 再加上 B 共享: ~25
> - 与 AVX-512 的 13 条相比仍有差距，主要来自 RVV 的多调用架构开销

**BB 内收益估算 (K 循环 BB)**:

假设 K 循环 BB 在 2 行模式下每 K 对约 15 条指令:
- `flw` 占 4 条 (A 加载)
- 引入 `vfmacc.vf_mem` 后: 消除 4 条 `flw`，FMA 指令数不变
- 新 BB 指令数: 15 - 4 = 11
- **BB 内减少: (15 - 11) / 15 = 27%**

假设 K 循环 BB 在 12 行模式下约 42 条指令 (1 K 步):
- `flw` 占 12 条
- 引入后: 42 - 12 = 30 条
- **BB 内减少: (42 - 30) / 42 = 29%**

保守取值: **BB 内减少 25%**

---

### [P1] 利用已有 RVV 掩码存储能力优化部分列处理

**说明**：这不是一个新的扩展指令建议，而是指出 RVV 已有的掩码存储能力未被当前实现利用。

**RVV 已有能力**:
```asm
# 设置活跃 lane 数
vsetvli zero, a0, e32, m1    # VL = 剩余列数
# 掩码存储
vse32.v v0, (addr), v0.t      # 仅存储 VL 个元素
```

**当前实现的问题**: sgemm-kernel-vl16 对 N < 16 的尾部列使用标量提取逐个写回:
```c
// 当前 C 代码模式 (简化)
for (int j = 0; j < remaining; j++) {
    C[row * ldc + col_offset + j] = accumulator[j];  // 逐个标量提取+存储
}
```

**改进方案**: 使用 `vse32.v` + 尾部 VL 进行掩码存储:
```asm
vsetvli t0, a0, e32, m1      # t0 = min(remaining, VLMAX)
vse32.v v_acc, (c_ptr), v0.t  # 一次存储 remaining 个 float32
```

**性能对比 (假设 N=10 剩余列, 2行)**:

| 实现 | 指令数 |
|------|--------|
| 当前 (标量提取) | ~2 (vsetvli) + ~20 (10次 flw + 10次 sw) = ~22 |
| 改进 (掩码存储) | ~2 (vsetvli) + 2 (每行 1 次 vse32.v) = ~4 |
| **BB 内减少** | **(22 - 4) / 22 = 82%** |

> 注: 实际收益取决于部分列出现的频率。在全 16 列对齐的情况下为 0。仅在 N 不是 16 的倍数时才有收益。

---

## 收益估算总结

### 各扩展指令收益链

| 扩展指令 | 目标场景 | 原指令数 | 新指令数 | BB内减少 | 影响范围 |
|----------|---------|----------|----------|----------|---------|
| vfmacc.vf_mem | K循环 (2行模式) | 15 | 11 | -27% | 所有K循环迭代 |
| vfmacc.vf_mem | K循环 (12行模式, 理论) | 42 | 30 | -29% | 若RVV实现扩展到12行 |
| 掩码存储优化 | 部分列输出 | ~22 | ~4 | -82% | 仅N非16倍数时 |

### 累计收益估算

- 无BBV profiling数据，无法计算整体收益
- K循环是GEMM的计算主体，通常占总执行时间的 70-90%（基于典型的GEMM profiling数据）
- 在 K 循环内部，`flw` (A 元素加载) 约占 K 循环指令的 27%
- `vfmacc.vf_mem` 的保守整体收益估算: 27% x 80% (K循环占比) = **~22%** (理论上限，未考虑其他瓶颈)
- 但实际受限于:
  - 内存延迟: `vfmacc.vf_mem` 虽然节省了指令槽，但标量加载的延迟仍需等待
  - 指令解码带宽: 减少 1 条指令释放的解码带宽可能被其他指令消费
  - **保守调整后**: 整体收益上限约 **10-15%** (按收益计算文档的保守原则)
- 掩码存储优化: 仅在 N 非 16 倍数时生效，影响有限，估计整体收益 < **2%**

---

## 附录

### A. FMA指令对比表

| 特性 | x86 AVX | x86 FMA3 | x86 AVX-512F | RVV |
|------|---------|----------|--------------|-----|
| FMA 指令 | 无 (分离 vmul+vadd) | `vfmadd231ps` | `vfmadd231pf` | `vfmacc.vf` |
| 标量 x 向量 FMA | `vbroadcastss` + `vmulps` + `vaddps` (3条) | `vbroadcastss` + `vfmadd231ps` (2条) | `vfmadd231pf_bcst` **(1条)** | `flw` + `vfmacc.vf` (2条) |
| 标量来源 | 内存或寄存器 | 内存或寄存器 | 内存 (广播修饰符) | 寄存器 (f 寄存器) |
| 掩码 FMA | 无 | 无 | `{k1}` opmask | `v0.t` mask |
| 广播到 lane 数 | 8 (YMM) | 8 (YMM) | 16 (ZMM `{1to16}`) | VL (可配置) |

### B. 数据重排指令对比表

| 特性 | x86 AVX | x86 FMA3 | x86 AVX-512F | RVV |
|------|---------|----------|--------------|-----|
| 向量加载 | `vmovaps`/`vmovups` (256-bit) | `vmovaps`/`vmovups` (256-bit) | `vmovapf`/`vmovupf` (512-bit) | `vle32.v` (VLEN-bit) |
| 标量广播 | `vbroadcastss [mem]` | `vbroadcastss [mem]` | `vbroadcastsf [mem]` | 无直接等价 (需 flw) |
| 部分存储 | `vmaskmovps` (掩码) | `vmaskmovps` (掩码) | `vmovupf {k1}` (opmask) | `vse32.v` + mask (已有，未使用) |

### C. 寄存器使用对比

| 特性 | x86 FMA3 (256-bit) | x86 AVX-512F (512-bit) | RVV VLEN=512 |
|------|--------------------|-----------------------|---------------|
| 最大行数 | 6 (ymm4-ymm15, 12个YMM) | 12 (zmm4-zmm27, 24个ZMM) | 2 (当前实现) |
| 累加器寄存器数 | 12 | 24 | 2 (可扩展，受限于 LMUL) |
| 临时寄存器 | ymm0-ymm3 | zmm0-zmm3 | 编译器分配 |
| B 向量复用 | By2: vmovapf ymm0,ymm1 被6行共享 | By2: vmovapf zmm0,zmm1 被12行共享 | By2: vle32.v v0,v1 仅被2行共享 |

---

## 结论

1. **最高优先级 (P0): `vfmacc.vf_mem` 标量内存广播FMA** — AVX-512 的 `vfmadd231pf_bcst` 将标量加载、广播和 FMA 合并为单条指令。RVV 需要 2 条指令 (`flw` + `vfmacc.vf`)。在 K 循环中，A 元素加载约占指令总数的 27-29%。保守估计整体收益 10-15%。

2. **中等优先级 (P1): 掩码存储优化** — RVV 已有 `vse32.v` + mask 能力，但当前 sgemm 实现未使用。修复此问题即可在部分列场景获得约 82% 的 BB 内指令减少，但整体影响有限 (N 非 16 倍数时才触发)。

3. **AVX (无FMA) 和 FMA3 (256-bit) 不构成紧迫的 ISA 扩展需求** — AVX 无 FMA，RVV 已有 `vfmacc.vf` 优势。FMA3 的 `vbroadcastss` + `vfmadd231ps` 在指令数上与 RVV 的 `flw` + `vfmacc.vf` 等价（均为 2 条），差距仅在广播目标寄存器类型（向量 vs 标量）。

4. **12 行处理能力是软件架构优势，非 ISA 差距** — AVX-512 能同时处理 12 行，是因为有 32 个 ZMM 寄存器。RVV 的向量寄存器数量 (32 个) 理论上也够用，但需要调整 LMUL 和软件架构来支持。这属于编译器和软件优化范畴，不需要 ISA 扩展。

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1   |           |        |      |

最终审查结论：待审查
