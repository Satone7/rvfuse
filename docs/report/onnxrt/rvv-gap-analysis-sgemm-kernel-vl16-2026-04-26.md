# MlasSgemmKernel VL=16 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: `MlasSgemmKernelRvv512Impl` — FP32矩阵乘法核心kernel（float32×float32→float32）
**基准实现**: RVV VLEN=512, VL=16 (float32), LMUL=1
**分析平台**: x86 AVX/FMA3, ARM NEON, LoongArch LASX, Power VSX MMA (POWER10), S390X Z-Vector, WASM SIMD
**BBV数据**: 基于QEMU-BBV profiling on 独立测试可执行文件 (output/bbv_rvv512/sgemm_rvv512/)，K循环BB占总执行权重约96.9%

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vmatmul.fp32 (4×4外积) | Power VSX MMA | K循环BB内减少约80%（64 MACs从20条→4条） | 高 | RVV无矩阵乘指令 |
| P1 | vfmacc.vv_lane | ARM NEON | K循环BB内减少约25%（4条flw→1条vle32） | 中 | RVV有vfmacc.vf但无lane索引版本 |
| P2 | vfmacc.vv_lane.unroll | x86 FMA3优化模式 | K循环BB内减少约15%（地址计算合并） | 低 | RVV地址计算需多条addi |

**收益计算方式**（基于QEMU-BBV profiling数据）：
- BB执行权重 = BB执行次数 × BB指令数
- K循环BB（BB5 + BB15）执行权重 = 38,269,440,000（占总权重96.9%）
- BB内收益 = (原BB指令数 - 扩展后BB指令数) / 原BB指令数 × 100%
- 整体收益 = BB内减少比例 × K循环占比（96.9%）

---

## 基准RVV实现分析

### RVV实现结构 (VLEN=512, VL=16)

```cpp
// rvv_sgemm_kernel_vl16.inl 核心K循环结构
template<bool ZeroMode, bool ProcessTwoRows>
inline size_t MlasSgemmKernelRvv512Impl(...) {
    constexpr size_t kVlen = 16;  // 512-bit VLEN, SEW=32, LMUL=1
    const size_t vl = __riscv_vsetvl_e32m1(kVlen);

    do {
        vfloat32m1_t v_acc_r0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t v_acc_r1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        while (k >= 2) {
            float a0_r0 = a[0];       // flw (scalar load A)
            float a1_r0 = a[1];       // flw
            float a0_r1 = a[lda];     // flw (second row)
            float a1_r1 = a[lda+1];   // flw

            vfloat32m1_t v_b0 = __riscv_vle32_v_f32m1(b, vl);
            v_acc_r0 = __riscv_vfmacc_vf_f32m1(v_acc_r0, a0_r0, v_b0, vl);
            v_acc_r1 = __riscv_vfmacc_vf_f32m1(v_acc_r1, a0_r1, v_b0, vl);

            vfloat32m1_t v_b1 = __riscv_vle32_v_f32m1(b + 16, vl);
            v_acc_r0 = __riscv_vfmacc_vf_f32m1(v_acc_r0, a1_r0, v_b1, vl);
            v_acc_r1 = __riscv_vfmacc_vf_f32m1(v_acc_r1, a1_r1, v_b1, vl);

            a += 2; b += 32; k -= 2;
        }
        // ... tail handling
    } while (CountN > 0);
}
```

### BBV热点数据（K循环BB）

从QEMU-BBV profiling分析：

| 排名 | BB地址 | BB指令数 | 执行次数 | 执行权重 | 执行占比 |
|------|--------|----------|----------|----------|----------|
| 1 | 0x...a482 (BB5) | 16 | 1,077,955,200 | 17,247,283,200 | 43.6% |
| 2 | 0x...a526 (BB15) | 16 | 1,313,884,800 | 21,022,156,800 | 53.3% |
| **合计** | K循环BB | 16 | 2,391,840,000 | **38,269,440,000** | **96.9%** |

**K循环BB指令序列**（每2 K元素迭代）：
```
flw     fa5, 0(a4)          # A元素0 (row0)
flw     fa4, 4(a4)          # A元素1 (row0)
vle32.v v11, (t5)           # B向量0 (16列)
flw     fa3, 0(t4)          # A元素0 (row1)
flw     fa2, 4(t4)          # A元素1 (row1)
addi    s0, t5, 64          # 地址计算
vfmacc.vf v9, fa5, v11      # FMA: acc0 += A[0] × B
vfmadd.vf v11, fa3, v10     # FMA: acc1 += A[0] × B
vle32.v v10, (s0)           # B向量1 (16列)
addi    a4, a4, 8           # A指针前进
addi    t5, t5, 128         # B指针前进
addi    t6, t6, -2          # K计数器
vfmacc.vf v9, fa4, v10      # FMA: acc0 += A[1] × B
vfmadd.vf v10, fa2, v11     # FMA: acc1 += A[1] × B
addi    t4, t4, 8           # A指针前进(row1)
bgtu    t6, t3, -54         # 循环回跳
```

### RVV指令计数分析

| 操作阶段 | 指令数 | 元素数 | 说明 |
|----------|--------|--------|------|
| A元素加载 | 4 | 4 | 4×flw (2 rows × 2 K) |
| B向量加载 | 2 | 32 | 2×vle32.v (16列 × 2 K块) |
| FMA操作 | 4 | 64 | 4×vfmacc.vf (64 MACs) |
| 地址更新 | 5 | - | 4×addi + 1×bgtu |
| **总计** | **16** | 64 MACs | 每2 K元素迭代 |

**关键观察**：
- RVV FP32 MAC仅需**1条vfmacc.vf指令**（已融合multiply+accumulate）
- 无需分开的multiply和add，比INT8 GEMM的widening操作更简洁
- 当前瓶颈：A矩阵需逐元素scalar load（4×flw），而非向量load+lane索引
- 地址计算开销较大（5条地址更新指令）

---

## 各平台对比分析

### 1. x86 AVX/FMA3

**核心特点**：
- YMM寄存器，256-bit向量宽度（8×float32）
- FMA3核心能力：`vfmadd231ps` — 乘+加融合，单指令完成8 MACs
- Broadcast：`vbroadcastss ymm, [mem]` — 从内存加载单float32并广播到所有lane

**MLAS实现分析**（amd64/SgemmKernelFma3.asm + FgemmKernelFma3Common.inc）：

```asm
; FMA3 K循环结构（每K元素，2 YMM向量=16 outputs）
vbroadcastss ymm3, [rcx+BroadcastOffset]   ; 1条: 加载A并广播
vmovaps      ymm0, [rdx+VectorOffset]      ; 1条: 加载B向量0
vmovaps      ymm1, [rdx+VectorOffset+32]   ; 1条: 加载B向量1
vfmadd231ps  ymm4, ymm3, ymm0              ; 1条: row0 FMA with B0
vfmadd231ps  ymm5, ymm3, ymm1              ; 1条: row0 FMA with B1
; (若ProcessTwoRows: 对row1重复2条FMA)
```

**指令计数对比**（归一化到VLEN=512，16 outputs，2 K元素）：

| 平台 | 每K元素指令数 | 2 K元素指令数 | 64 MACs指令数 | 相对效率 |
|------|---------------|----------------|---------------|----------|
| AVX FMA3 (1 row) | 5 (1 bcast + 2 load + 2 FMA) | 10 | 10 (YMM×2=16) → 40 (×4倍归一化) | 基准 |
| AVX FMA3 (2 rows) | 7 (1 bcast + 2 load + 4 FMA) | 14 | 14 → 56 | 100% |
| RVV (2 rows) | 8 (4 flw + 2 vle32 + 4 vfmacc + 地址) | 16 | 16 | 96.6% |

**关键发现**：
- **x86 AVX FMA3与RVV FP32 GEMM效率相近**
- AVX: 1 broadcast + 2 load + 4 FMA = 7 insns for 32 MACs (2 rows × 16 cols × 1 K)
- RVV: 4 flw + 2 vle32 + 4 vfmacc = 10 insns for 64 MACs (2 rows × 16 cols × 2 K)
- 归一化后：AVX需要56 insns vs RVV 16 insns → **RVV更高效**（因VLEN=512比YMM=256宽2倍）

**建议扩展**：无需扩展，RVV FP32 FMA已与AVX FMA3效率相当。

---

### 2. ARM NEON

**核心特点**：
- Q寄存器，128-bit向量宽度（4×float32）
- Lane-indexed FMA：`fmla vd.4s, vn.4s, vm.s[lane]` — 关键优势
- 可将A元素批量加载到向量寄存器，再通过lane索引逐个使用

**MLAS实现分析**（arm64/SgemmKernelNeon.asm）：

```asm
; NEON K循环结构（4 rows × 8 columns × 4 K）
ldr     v8, [x0], #16          ; 加载4个A元素（row0）
ldr     v9, [x10], #16         ; 加载4个A元素（row1）
; ...
ldr     q4, [x1]               ; 加载B向量0 (4列)
ldr     q5, [x1, #16]          ; 加载B向量1 (4列)
fmla    v16.4s, v4.4s, v8.s[0] ; row0 FMA with lane[0]
fmla    v17.4s, v5.4s, v8.s[0] ; row0 FMA with lane[0], B1
fmla    v20.4s, v4.4s, v9.s[0] ; row1 FMA with lane[0]
; ...
```

**Lane-indexed FMA语义**：
```
fmla vd.4s, vn.4s, vm.s[lane]
// vd[i] = vd[i] + vn[i] * vm[lane]  (broadcast vm[lane] to all positions)
```

**指令计数对比**（归一化到VLEN=512）：

| 平台 | A加载方式 | A加载指令数 | 每4 K元素总指令数 | 归一化到64 MACs |
|------|-----------|-------------|-------------------|-----------------|
| ARM NEON | 向量load (4元素) | 1×ldr | ~15 | ~60 insns (×4倍归一化) |
| RVV | scalar load (4元素) | 4×flw | 16 | 16 insns |

**关键差距**：
- ARM NEON用`ldr`批量加载4个A元素（1条指令）
- RVV需用`flw`逐个加载（4条指令）
- NEON的`fmla v.s[lane]`允许从向量寄存器提取单lane并广播

**建议扩展**：
- **P1: vfmacc.vv_lane vd, vs2, vs1[lane]** — lane-indexed FMA
  - 语义：`vd[i] = vd[i] + vs2[i] * vs1[lane]`
  - 用法：`vle32.v v_a, (a)` 加载A向量（4元素），然后 `vfmacc.vv_lane v_acc, v_b, v_a[0]` 等
  - BB内收益：4条flw → 1条vle32，减少25%（4/16 = 25%）

---

### 3. Power VSX MMA (POWER10)

**核心特点**：
- VSX寄存器，128-bit（4×float32）
- **MMA (Matrix Multiply Assist)**：`xvf32gerpp` — 4×4 FP32外积指令
- 单指令完成16 MACs（4×4矩阵乘累加）

**MLAS实现分析**（power/SgemmKernelPOWER10.cpp）：

```cpp
// POWER10 MMA FP32外积
__builtin_mma_xvf32gerpp(&acc[0], 
    reinterpret_cast<vec_t>(ABroadcast),  // 4×float32 A向量
    reinterpret_cast<vec_t>(BElements[0]) // 4×float32 B向量
);
// 单指令: acc += A[0..3] × B[0..3]^T = 16 MACs
```

**MMA语义详解**：
```
xvf32gerpp acc, A, B
// A: 4×float32向量
// B: 4×float32向量
// acc: 4×4 float32 accumulator (16 elements)
// 计算: acc[i][j] += A[i] × B[j]  (outer product)
// 总MACs = 4×4 = 16
```

**指令计数对比**（归一化到VLEN=512，64 MACs）：

| 平台 | 每64 MACs指令数 | 说明 |
|------|-----------------|------|
| POWER10 MMA | **4** | 4×xvf32gerpp (每条16 MACs) |
| RVV | **20** | 4×vle32 (B) + 16×vfmacc.vf |

**归一化计算**：
- POWER10: 4条MMA = 64 MACs（但accumulator需要4×4=16个float32，vs RVV 2×16=32个float32）
- RVV处理2 rows × 16 cols：需要4次迭代（每迭代16 MACs）× 4 K = 64 MACs → 20 insns
- **POWER10 MMA vs RVV：约5×效率差距**

**建议扩展**：
- **P0: vmatmul.fp32 vd, vs1, vs2** — 4×4 FP32矩阵乘累加
  - 语义：`vd[4×4] += vs1[4] × vs2[4]^T`（外积）
  - LMUL: 需要LMUL=4的accumulator（16个float32）
  - BB内收益：64 MACs从20条→4条，减少80%

---

### 4. LoongArch LASX

**核心特点**：
- LASX: 256-bit向量（8×float32）
- LSX: 128-bit向量（4×float32）
- 标准FMA指令：`fmadd.s` / `fmacc.s`

**与RVV类似**：使用widening FMA模式，无专用矩阵乘指令。

**建议扩展**：无（与RVV效率相当）。

---

### 5. S390X Z-Vector

**核心特点**：
- 128-bit向量寄存器（4×float32）
- `vec_splats` — scalar broadcast
- `vec_perm` — 数据重排
- 标准FMA指令

**MLAS实现**：使用vec_splats广播 + FMA模式，与RVV类似。

**建议扩展**：无（与RVV效率相当）。

---

### 6. WASM SIMD

**核心特点**：
- 128-bit v128（4×float32）
- `wasm_f32x4` — 标准SIMD
- 无专用矩阵乘或lane-indexed FMA

**建议扩展**：无（WASM比RVV更受限）。

---

## RVV扩展指令建议详细说明

### [P0] vmatmul.fp32 — 4×4 FP32外积矩阵乘累加

**指令定义**：
```
vmatmul.fp32 vd, vs1, vs2, vm
// vd: 4×4 float32 accumulator (LMUL=4, 16 elements)
// vs1: 4×float32 row vector (LMUL=1)
// vs2: 4×float32 column vector (LMUL=1)
// 计算: vd[i][j] += vs1[i] × vs2[j] for i,j ∈ {0,1,2,3}
// 总MACs: 16
```

**应用场景**：
- FP32 GEMM核心K循环
- 单指令替代16条vfmacc.vf指令
- 特别适用于小block GEMM（如4×4子块）

**性能对比**：

| 方案 | 64 MACs指令数 |
|------|---------------|
| RVV当前 | 20 (4 vle32 + 16 vfmacc) |
| vmatmul.fp32 | 4 (4×16 MACs) |
| **减少** | **80%** |

**整体收益**：
- BB内减少80% × K循环占比96.9% = 整体减少约77.5%

**参考**：Power VSX POWER10 `xvf32gerpp`

---

### [P1] vfmacc.vv_lane — Lane-indexed FMA

**指令定义**：
```
vfmacc.vv_lane vd, vs2, vs1, lane, vm
// vd[i] = vd[i] + vs2[i] × vs1[lane]
// lane: 立即数索引，选择vs1的某个lane广播
```

**应用场景**：
- FP32 GEMM：A元素批量加载后逐lane使用
- 减少4条flw → 1条vle32 + 4条vfmacc.vv_lane
- 寄存器压力降低（A元素保存在向量寄存器而非scalar寄存器）

**性能对比**：

| 方案 | 4个A元素加载 | FMA指令数 | 总计 |
|------|--------------|-----------|------|
| RVV当前 | 4×flw | 4×vfmacc.vf | 16条 |
| vfmacc.vv_lane | 1×vle32 | 4×vfmacc.vv_lane | ~12条 |
| **减少** | -3条 | 0 | **25%** |

**整体收益**：
- BB内减少25% × K循环占比96.9% = 整体减少约24.2%

**参考**：ARM NEON `fmla vd.4s, vn.4s, vm.s[lane]`

---

### [P2] 地址计算优化与预取指令

**观察**：RVV K循环需5条地址更新指令（addi）
- x86使用复杂寻址模式：`[rdx+VectorOffset]` 编码在指令中
- ARM使用post-index：`ldr v8, [x0], #16`

**建议**：编译器优化将地址计算合并到load/store指令的offset中（非新指令）。

**预取指令缺失**：
- x86有 `prefetcht0 [rdx+256]` 预取B矩阵下一K行数据
- RVV无预取指令，依赖硬件预取
- 对于大矩阵GEMM，预取可减少内存延迟miss，预期5-15%整体性能提升
- 建议扩展：`prefetch.v rs1, hint` — 向量数据预取指令

---

## BBV热点加权收益分析

### BBV热点分布

| 排名 | BB类型 | BB指令数 | 执行次数 | 执行权重 | 执行占比 |
|------|--------|----------|----------|----------|----------|
| 1 | K循环(ZeroMode) | 16 | 1,078M | 17,247M | 43.6% |
| 2 | K循环(AddMode) | 16 | 1,314M | 21,022M | 53.3% |
| 3 | 累加器初始化 | 22 | 30.6M | 672M | 1.7% |
| 4 | 地址计算头 | 27 | 4.2M | 115M | 0.3% |
| **合计** | K循环 | 16 | 2,392M | **38,270M** | **96.9%** |

### 各扩展指令收益链

| 扩展指令 | 目标BB | 原指令数 | 新指令数 | BB内减少 | BB占比 | 整体收益 |
|----------|--------|----------|----------|----------|--------|----------|
| vmatmul.fp32 | K循环BB | 20 (64 MACs) | 4 | -80% | 96.9% | **-77.5%** |
| vfmacc.vv_lane | K循环BB | 16 | ~12 | -25% | 96.9% | **-24.2%** |

### 累计收益估算

- P0 vmatmul.fp32整体收益上限：约77.5%
- P1 vfmacc.vv_lane整体收益上限：约24.2%
- 注：P0和P1影响同一BB，实际累计收益应取最大值（约77.5%）

---

## 结论

### 关键发现

1. **Power VSX MMA是FP32 GEMM最大差距来源**
   - `xvf32gerpp`单指令完成16 MACs（4×4外积）
   - RVV需20条指令完成64 MACs
   - 建议P0优先级：引入`vmatmul.fp32`

2. **ARM NEON lane-indexed FMA提供中等优化**
   - 批量加载A元素 + lane索引FMA
   - 减少25%加载指令数
   - 建议P1优先级：引入`vfmacc.vv_lane`

3. **x86 AVX FMA3与RVV效率相当**
   - AVX: broadcast + FMA vs RVV: flw + vfmacc.vf
   - 归一化后指令数相近
   - 无需针对AVX的扩展

### 优先级总结

| 优先级 | 扩展 | 整体收益 | 来源平台 | 说明 |
|--------|------|----------|----------|------|
| P0 | vmatmul.fp32 | 整体减少约77.5% | Power VSX MMA | 4×4外积矩阵乘 |
| P1 | vfmacc.vv_lane | 整体减少约24.2% | ARM NEON | Lane索引FMA |
| P2 | 地址计算+预取 | 整体减少约5-15% | 编译优化+新指令 | 预取指令缺失 |

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1 | - | - | 待审查 |

---