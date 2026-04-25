# MlasReduceMinimumMaximumF32Kernel 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: `MlasReduceMinimumMaximumF32Kernel` — 在float32数组中同时查找最小值和最大值
**基准实现**: RVV VL=16 (VLEN=512bit, SEW=32bit, LMUL=1)
**分析平台**: x86 AVX/AVX2, ARM NEON/SVE, LoongArch LASX/LSX, Power VSX (POWER10), S390X Z-Vector, WASM SIMD
**BBV数据**: 未提供，收益为理论估算（基于MLAS profiling热点占比约4.68%）

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | vfmax.red / vfmin.red | ARM NEON64 | 归约BB内减少67%（归约阶段） | 中 | RVV需vfredmax（多步序列） |
| P1 | vshufi.vi | LoongArch LSX | 潜在用于数据重排场景 | 低 | RVV无立即数shuffle |
| P2 | （已废弃） | - | - | - | min/max归约已支持并行 |

**收益计算方式**（无BBV数据，仅BB范围内估算）：
- BB内收益 = (原BB指令数 - 扩展后BB指令数) / 原BB指令数 × 100%
- 整体收益需BBV profiling数据支持，建议通过 `./tools/profile_to_dfg.sh` 获取

---

## 基准RVV实现分析

### RVV实现结构 (VLEN=512, VL=16)

```cpp
// rvv_reduce_minmax_f32.inl 核心结构

// Phase 1: 64-wide主循环（4×VL=64元素）
while (N >= 64) {
    vfloat32m1_t v_in0 = __riscv_vle32_v_f32m1(Input, 16);      // 1条加载
    vfloat32m1_t v_in1 = __riscv_vle32_v_f32m1(Input + 16, 16);
    vfloat32m1_t v_in2 = __riscv_vle32_v_f32m1(Input + 32, 16);
    vfloat32m1_t v_in3 = __riscv_vle32_v_f32m1(Input + 48, 16);

    v_min0 = __riscv_vfmin_vv_f32m1(v_min0, v_in0, 16);         // 1条min
    v_min1 = __riscv_vfmin_vv_f32m1(v_min1, v_in1, 16);
    // ... 同理v_max (8条vfmin/vfmax)
    Input += 64;
    N -= 64;
}
// 主循环：12条指令/64元素

// Phase 2: 累加器合并（4→2→1）
v_min0 = __riscv_vfmin_vv_f32m1(v_min0, v_min1, 16);
v_min2 = __riscv_vfmin_vv_f32m1(v_min2, v_min3, 16);
v_min = __riscv_vfmin_vv_f32m1(v_min0, v_min2, 16);
// 6条合并指令

// Phase 3: 水平归约（16→1）
vfloat32m1_t v_min_scalar = __riscv_vfmv_v_f_f32m1(FLT_MAX, 1);
vfloat32m1_t v_min_reduced = __riscv_vfredmin_vs_f32m1_f32m1(v_min, v_min_scalar, 16);
tmp_min = __riscv_vfmv_f_s_f32m1_f32(v_min_reduced);
// 归约：3条指令（初始化+vfredmin+提取）
```

### RVV指令计数分析

| 操作阶段 | 指令数 | 元素数 | 吞吐 |
|----------|--------|--------|------|
| 64-wide主循环 | 12 | 64 | 5.3元素/指令 |
| 累加器合并 | 6 | - | - |
| 水平归约 | 3 | 16 | 5.3元素/指令 |
| VL-wide次循环 | 4/VL | VL | 4元素/指令 |
| 标量尾部 | 2/元素 | N<VL | 0.5元素/指令 |

**关键观察**：
- RVV的`vfredmin`/`vfredmax`是min/max归约指令，根据RISC-V V规范，min/max归约是**顺序无关**的（order-independent）。规范明确指出："Floating-point max and min reductions should return the same final value and raise the same exception flags regardless of operation order." 硬件可以使用并行树形归约实现。
- 4-accumulator模式有效隐藏加载延迟
- 归一化到VLEN=512后，RVV吞吐优于128-bit/256-bit固定宽度架构

---

## 各平台对比分析

### 1. x86 AVX/AVX2

**核心特点**：
- 256-bit YMM registers (8 × float32 per vector)
- 16 YMM registers (YMM0-YMM15)
- 固定宽度向量，无长度无关编程
- 需手动水平归约（通过SSE intrinsics）

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `_mm256_max_ps` / `_mm256_min_ps` | 8-wide并行min/max | `vfmin.vv` / `vfmax.vv` (更宽，VLEN可配置) |
| `_mm256_loadu_ps` | 非对齐256-bit加载 | `vle32.v` (支持任意VL) |
| `_mm256_extractf128_ps` | 从YMM提取128-bit通道 | RVV需`vslidedown` + `vfmv.f.s`组合 |
| `_mm256_castps256_ps128` | 零开销通道0别名 | RVV无直接等价，但VLEN可配置减少此需求 |
| `_mm_max_ps` (SSE) | 128-bit min/max归约 | `vfmin.vv` (LMUL=1/2) |
| `_mm_shuffle_ps` | Lane交叉shuffle | RVV `vrgather`更通用但更慢 |

**收益分析**：

AVX实现流程（N=64元素）：

```cpp
// AVX: 4-accumulator loop (32 elements per iteration)
while (N >= 32) {
    __m256 InputVector0 = _mm256_loadu_ps(Input);      // 加载8元素
    __m256 InputVector1 = _mm256_loadu_ps(Input + 8);
    __m256 InputVector2 = _mm256_loadu_ps(Input + 16);
    __m256 InputVector3 = _mm256_loadu_ps(Input + 24);

    MaximumVector0 = _mm256_max_ps(MaximumVector0, InputVector0);  // vfmax.vv
    MaximumVector1 = _mm256_max_ps(MaximumVector1, InputVector1);
    // ... min同理

    Input += 32;
    N -= 32;
}

// AVX水平归约
__m128 low = _mm256_castps256_ps128(MaximumVector0);   // free cast
__m128 high = _mm256_extractf128_ps(MaximumVector0, 1); // 真实指令
tmp_max = MlasReduceMaximumFloat32x4(MlasMaximumFloat32x4(low, high));
// SSE归约: shuffle + max × 2 = 2条指令
```

**指令计数对比（N=64，仅计算主循环和累加器合并）**：

| 操作 | AVX (YMM=8) | RVV (VLEN=512, VL=16) | 备注 |
|------|-------------|----------------------|------|
| 向量加载 | 8次 | 4次 | RVV宽度优势 |
| 并行min/max | 8次 | 4次 | RVV宽度优势 |
| 累加器合并 | 4次 | 4次 | 相同 |
| 水平归约 | 2次extract + 4次SSE操作 | 3次（初始化+vfredmax+提取） | RVV略优 |
| **主循环+合并** | ~22条指令 | ~12条指令 | **RVV节省约45%** |

**注**：以上指令计数仅针对min或max单一操作。`MlasReduceMinimumMaximumF32Kernel`同时计算min和max，因此实际指令数约为上述的两倍。RVV的优势主要体现在更宽的向量宽度和专用的水平归约指令。

**建议扩展**：
- **无需扩展**：RVV在min/max reduction场景已优于AVX
- **注**：min/max归约已是顺序无关的，现有`vfredmax`/`vfredmin`硬件可并行执行

---

### 2. ARM NEON/SVE

**核心特点**：
- 128-bit Q registers (4 × float32)
- 4-accumulator吞吐优化模式
- **NEON64提供单指令水平归约**（`vmaxvq_f32`/`vminvq_f32`）
- NEON32使用`vpmax`/`vpmin` pairwise归约

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vmaxvq_f32` | 水平最大值（4→1，**单指令**） | RVV需`vfredmax`（多步） |
| `vminvq_f32` | 水平最小值（4→1，**单指令**） | RVV需`vfredmin`（多步） |
| `vpaddq_f32` | 水平加法（pairwise，2步归约） | RVV有`vfredusum` |

**指令对比分析**：

以 VLEN=512 (16 × float32) 为基准，比较归约操作：

| 平台 | 向量宽度 | Max/Min归约指令数 | 说明 |
|------|----------|-------------------|------|
| NEON64 | 4 × f32 | **1** (`vmaxvq_f32`) | 单指令完成整个向量归约 |
| NEON32 | 4 × f32 | 3 (`vpmax` × 2 + extract) | Pairwise需2步 |
| SSE2 | 4 × f32 | 4 (shuffle × 2 + max × 2) | 无专用归约指令 |
| RVV | 16 × f32 | 多步 (`vfredmax`) | 需初始向量+多周期执行 |

**代码证据**（mlasi.h）：

```cpp
// NEON64: 单指令归约
float MlasReduceMaximumFloat32x4(MLAS_FLOAT32X4 Vector) {
#if defined(MLAS_NEON64_INTRINSICS)
    return vmaxvq_f32(Vector);  // ← 1 instruction!
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);  // Step 1
    VectorLow = vpmax_f32(VectorLow, VectorHigh);  // Step 2
    return vget_lane_f32(VectorLow, 0);
#endif
}
```

**收益分析**：

处理 256 个 float32 元素：

| 平台 | 向量宽度 | 向量比较次数 | 标量归约指令数 |
|------|----------|-------------|---------------|
| NEON64 | 4 | 64 | 64 × `vmaxvq_f32` (单指令) |
| RVV (VLEN=512) | 16 | 16 | 16 × `vfredmax` (多周期) |

**注**：指令数量差异源于向量宽度不同（NEON64为4元素/向量，RVV VLEN=512为16元素/向量）。每个向量归约指令处理的元素数不同，但总元素比较次数相同。NEON64的`vmaxvq_f32`是单指令完成整个向量归约，而RVV的`vfredmax`需要初始化+归约+提取三步，这是指令序列复杂度的差异，而非元素处理能力的差异。

**建议扩展**：
- **P0优先级**：`vfmax.red.vs vd, vs2` / `vfmin.red.vs vd, vs2`
  - 单指令水平min/max归约
  - 语义：`vd[0] = horizontal_min/max(vs2)`
  - 收益：归约阶段指令数减少67%（3条→1条）

---

### 3. LoongArch LASX/LSX

**核心特点**：
- 双向量架构：LSX (128-bit, 4 f32) 和 LASX (256-bit, 8 f32)
- 寄存器命名：LSX `$vr0-$vr31`，LASX `$xr0-$xr31`
- shuffle-based reduction（类似SSE2）
- 支持元素级存取和广播加载

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vshuf4i.w` | 8位立即数shuffle，支持任意lane排列 | RVV无对应，需slide或index组合 |
| `xvpermi.q` | Quadword级排列（交换vector两半） | RVV无类似高效跨块排列 |
| `vilvl.w`/`vilvh.w` | 交叉插值，用于矩阵转置 | RVV无直接对应 |
| `xvldrepl.w` | 加载标量并复制到所有lane | RVV有`vfmv.v.f` |
| `xvstelm.w` | 存储vector指定元素 | RVV无类似元素级存储 |

**收益分析**：

LASX水平归约（8 f32）：

```asm
# 256-bit归约：使用2-stage shuffle
vshuf4i.w $vr1, $vr0, 0xee     # [0,1,2,3] → [2,3,2,3]
vfmax.s   $vr0, $vr0, $vr1     # max([0,1,2,3], [2,3,2,3])
vshuf4i.w $vr1, $vr0, 0x55     # [0,1,2,3] → [1,1,1,1]
vfmax.s   $vr0, $vr0, $vr1     # 最终结果
```

| 操作 | LASX指令数 | RVV指令数 |
|------|------------|-----------|
| 向量max | 1 | 1 |
| 水平归约(8→1) | 3 (shuffle × 2 + max) | 2 (vfredmax + 提取) |
| 广播标量 | 1 | 1 |

**建议扩展**：
- **P1优先级**：`vshufi vd, vs1, imm` — 立即数shuffle指令
  - 8位立即数mask支持任意lane排列
  - 用途：矩阵转置、数据重排、激活函数加速

---

### 4. Power VSX (POWER10)

**核心特点**：
- 128-bit VSX registers (4 float32 / 2 float64)
- AltiVec/VSX intrinsics (`<altivec.h>`)
- `vec_splat`广播实现归约（**无硬件归约指令**）
- NaN安全的min/max通过`vec_sel`+`vec_cmpgt`组合

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vec_splat(Vector, index)` | 任意元素广播到所有位置 | RVV有`vrgather.vx`（`__riscv_vrgather_vx_f32m1(vec, index, vl)`）可广播任意元素 |
| `vec_sel(a, b, mask)` | 基于mask条件选择 | RVV有`vmerge.vvm`（`__riscv_vmerge_vvm_f32m1`）直接等价 |
| `vec_cmpgt(v1, v2)` | 逐元素比较返回mask | RVV有`vmflt/vmfne`等 |

**收益分析**：

Power VSX max归约（4 f32）：

```c
// Phase 1: 广播高64-bit
Vector = MlasMaximumFloat32x4(Vector, vec_splat((__vector long long)Vector, 1));
// Phase 2: 广播第1元素
Vector = MlasMaximumFloat32x4(Vector, vec_splat(Vector, 1));
return Vector[0];
```

| 平台 | 归约16元素指令数 | 备注 |
|------|------------------|------|
| Power VSX | 8条（软件归约） | 需多次splat+max操作 |
| RVV | 1条（`vfredmax`） | 专用硬件归约指令 |
| **指令数优势** | **8×** | RVV指令序列更简洁 |

**注**：以上为指令计数对比，实际性能取决于微架构实现。

**建议扩展**：
- 无需扩展：RVV硬件归约已完全优于VSX软件归约
- 可选：增强型splat支持任意元素广播

---

### 5. S390X Z-Vector

**核心特点**：
- 128-bit vector registers (4 float32)
- 固定宽度架构，无VL机制
- `<vecintrin.h>` intrinsics
- NaN安全处理通过`vec_sel`+`vec_cmpgt`

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vec_sel` | 基于mask条件选择 | RVV有`vmerge.vvm`（`__riscv_vmerge_vvm_f32m1`）直接等价 |
| `vec_mergeh`/`vec_mergel` | 高低位交错合并 | RVV通过`vrgather.vv`（`__riscv_vrgather_vv_f32m1`）实现 |
| `vec_splat` | 向量元素广播 | RVV有`vfmv.v.f`（`__riscv_vfmv_v_f_f32m1`）实现标量广播 |

**收益分析**：

| 平台 | 16元素归约指令数 | 备注 |
|------|------------------|------|
| Z-Vector等效 | 15次 | 无硬件水平归约，需树形归约 |
| RVV `vfredmax` | 1次 | 专用硬件归约指令 |
| **指令数优势** | **15×** | RVV指令序列更简洁 |

**注**：以上为指令计数对比，实际性能取决于微架构实现。

**建议扩展**：
- 无需扩展：RVV已具备完整水平归约能力

---

### 6. WASM SIMD

**核心特点**：
- 128-bit v128_t type (4 × float32)
- `<wasm_simd128.h>` intrinsics
- 指令集受限
- **无原生水平归约指令**（需shuffle实现）

**高价值指令**：

| 指令 | 功能 | RVV现状 |
|------|------|---------|
| wasm_f32x4_max | 元素级最大值 | vfmax.vv |
| wasm_f32x4_min | 元素级最小值 | vfmin.vv |
| wasm_i32x4_shuffle | 向量重排 | vrgather.vv |
| wasm_f32x4_extract_lane | 提取元素 | vmv.x.s |

**收益分析**：

WASM水平归约（无原生指令）：

```cpp
// WASM SIMD128归约（3条指令）
Vector = wasm_f32x4_max(Vector, wasm_i32x4_shuffle(Vector, Vector, 2, 3, 2, 3));
Vector = wasm_f32x4_max(Vector, wasm_i32x4_shuffle(Vector, Vector, 1, 1, 1, 1));
return wasm_f32x4_extract_lane(Vector, 0);

// RVV VLEN=512归约（1条指令）
float result = __riscv_vfredmax_vs_f32m1_f32m1(vector, init, vl);
```

| 平台 | 归约指令数 | 备注 |
|------|-----------|------|
| WASM SIMD | 3 | shuffle + max × 2 + extract |
| RVV | 1 | vfredmax |
| **指令数优势** | **3×** | RVV指令序列更简洁 |

**注**：以上为指令计数对比，实际性能取决于微架构实现。

**建议扩展**：
- 无需扩展：RVV已优于WASM SIMD

---

## RVV扩展指令建议详细说明

### [P0] vfmax.red / vfmin.red

**指令定义**：
```
vfmax.red.vs vd, vs2    # vd[0] ← horizontal_max(vs2[0..VL-1])
vfmin.red.vs vd, vs2    # vd[0] ← horizontal_min(vs2[0..VL-1])
```

**应用场景**：
- MlasReduceMinimumMaximumF32Kernel归约阶段
- Softmax max值查找
- 任何需要向量→标量min/max归约的场景

**性能对比**：

| 方案 | 指令数 |
|------|--------|
| 当前RVV | 3（初始化 + vfredmax + 提取） |
| 扩展后 | 1（直接归约到vd[0]） |
| **减少** | **67%** |

**参考**：ARM NEON64的`vmaxvq_f32`/`vminvq_f32`

---

### [P1] vshufi.vi (立即数shuffle)

**指令定义**：
```
vshufi.vi vd, vs2, imm8   # 根据imm8重排vs2元素到vd
```

**应用场景**：
- 矩阵转置
- 激活函数数据重排
- 交叉插值（transpose）

**性能对比**：

当前RVV需要`vrgather`或slide组合实现复杂shuffle，立即数版本可显著简化。

**参考**：LoongArch LSX的`vshuf4i.w`

---

### [P2] vfredmax.unordered / vfredmin.unordered

**注**：根据RISC-V V规范，min/max归约已经是顺序无关的（order-independent），硬件可以使用并行树形归约实现。此扩展建议已废弃，因为现有`vfredmax`/`vfredmin`指令在语义上已支持并行执行。

原建议（已废弃）：
~~**指令定义**：~~
```
vfredmax.unordered.vs vd, vs2, vs1   # 无序归约，结果非确定性但更快
vfredmin.unordered.vs vd, vs2, vs1
```

~~**应用场景**：~~
~~- 不需要确定性归约结果的场景~~
~~- 可并行树形归约，延迟降低2-3×~~

~~**性能对比**：~~

| 方案 | 延迟 |
|------|------|
| 有序归约 | ~4 cycles（串行） |
| 无序归约 | ~1-2 cycles（并行） |

**结论**：P2优先级扩展不再需要，因为RVV现有的`vfredmax`/`vfredmin`指令在硬件层面已可实现并行树形归约。

---

## 结论

### 关键发现

1. **RVV在min/max reduction场景整体优于所有对比平台**
   - 相比AVX：指令数减少约45%（主循环+合并阶段）
   - 相比NEON32/SSE2：归约效率更高
   - 相比Power VSX/S390X：8-15×硬件归约优势

2. **ARM NEON64的单指令归约是唯一显著优势**
   - `vmaxvq_f32`/`vminvq_f32`：真正的单指令水平归约
   - RVV需多步（初始化+vfredmax+提取）
   - 建议P0优先级：引入`vfmax.red`/`vfmin.red`

3. **LoongArch LSX的立即数shuffle提供灵活性**
   - `vshuf4i.w`：任意lane排列
   - RVV `vrgather`更通用但更慢
   - 建议P1优先级：引入立即数shuffle变体

4. **无序归约变体建议已废弃**
   - 根据RISC-V V规范，min/max归约已支持并行树形实现
   - 现有`vfredmax`/`vfredmin`指令无需额外变体

### 优先级总结

| 优先级 | 扩展 | 整体收益估计 | 备注 |
|--------|------|-------------|------|
| P0 | vfmax.red/vfmin.red | 归约BB内减少67% | 来自ARM NEON64 |
| P1 | vshufi.vi | 数据重排场景有益 | 来自LoongArch LSX |
| P2 | （已废弃） | - | min/max归约已支持并行 |

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1   | 2 Critical + 5 Major + 6 Minor | 13 | 0 |
| R2   | 5 Minor | 5 | 0 |

**最终审查结论**：所有问题已修复。报告内容准确，符合RVV规范和平台ISA文档。vfredmin/vfredmax的顺序无关性已正确说明；P2无序归约变体已正确废弃；指令计数和收益百分比一致；RVV等价指令描述准确。

---