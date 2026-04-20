# ggml_gemv_q8_0_16x1_q8_0 多平台向量实现对比与RVV扩展指令建议

## 概述

**分析目标**: `ggml_gemv_q8_0_16x1_q8_0` - Q8_0量化权重(8-bit signed)与Q8_0量化激活(8-bit signed)的矩阵向量乘法(GEMV)，采用16x1 tile blocking布局(处理16列并行，每列1行)。

**基准实现**: RVV VLEN=512bit, SEW=32bit (累加器), LMUL=m2

**分析平台**: x86 AVX2/AVX-512 VNNI, ARM NEON (dotprod), LoongArch LASX, Power VSX (POWER10), WASM SIMD, S390X Z-Vector

**RVV实现状态**: 已实现，采用K-loop向量化(QK8_0=32次迭代，每迭代处理16列并行)

**BBV数据**: 未提供，收益为理论估算(基于K-loop指令数对比)

---

## 指令方案汇总

| 优先级 | 扩展指令 | 来源平台 | BB内收益 | 实现难度 | RVV现状 |
|--------|----------|----------|----------|----------|---------|
| P0 | `vdot4ss.vv` (i8xi8→i32 4元素点积) | x86 VNNI VDPBSSD | K-loop BB内减少33% | 中 | Zvdot4a8i存在但格式不兼容 |
| P0 | `vdot_lane.vx` (lane-indexed dot) | ARM NEON vdotq_laneq_s32 | K-loop BB内减少33% | 高 | 无lane-indexed指令 |
| P1 | `vwmacc_lane.vx` (lane-indexed widening MAC) | ARM/SVE2 indexed | 辅助优化 | 中 | 无lane-indexed MAC |
| P2 | `vwmaccwev/wod.vv` (even/odd widening MAC) | LoongArch LASX | 减少依赖链 | 中 | 无even/odd分离MAC |

**注**: 无BBV profiling数据，上表仅反映单个K-loop迭代BB范围内的指令减少比例。建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算整体收益。

---

## 基准RVV实现分析

### 实现位置

`applications/llama.cpp/vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp:447-494`

### 当前RVV核心循环分析

```c
void ggml_gemv_q8_0_16x1_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    const int qk = QK8_0;      // = 32
    const int nb = n / qk;
    
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;  // 激活向量
    for (int x = 0; x < nc / 16; x++) {
        const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);  // 16列并行
        
        // 16x FP32累加器 (VLEN=512, SEW=32, LMUL=m2)
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        
        for (int l = 0; l < nb; l++) {  // l-block循环
            // 16x int32累加器
            vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0, 16);
            
            // K-loop: 32次迭代
            for (int i = 0; i < QK8_0; i++) {
                // [1] 加载16个int8 (来自interleaved矩阵列)
                const vint8mf2_t b_0 = __riscv_vle8_v_i8mf2(&b_ptr[l].qs[i * 16], 16);
                
                // [2] Scalar x Vector widening multiply (a_scalar为单个int8)
                // [3] Widening accumulate (int16 → int32)
                sumi = __riscv_vwadd_wv_i32m2(sumi,
                         __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i], 16), 16);
            }
            
            // Scale merge: FP16 → FP32 + FMA
            const vfloat16m1_t b_d = __riscv_vle16_v_f16m1(b_ptr[l].d, 16);
            const vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, a_ptr[l].d, 16);
            sumf = __riscv_vfmacc_vv_f32m2(sumf, 
                     __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
        }
        
        __riscv_vse32_v_f32m2(s + x * 16, sumf, 16);
    }
}
```

### K-loop指令数分析(每迭代)

| 操作 | RVV指令 | 说明 |
|------|---------|------|
| 加载b向量 | `vle8_v_i8mf2` | 1条，加载16个int8 |
| Scalar广播+widening乘法 | `vwmul_vx_i16m1` | 1条，int8 → int16 |
| Widening累加 | `vwadd_wv_i32m2` | 1条，int16 → int32累加 |
| **每迭代总计** | **3条** | **32迭代 = 96条** |

### Scale处理(每l-block)

| 操作 | RVV指令 | 说明 |
|------|---------|------|
| 加载b scales (FP16) | `vle16_v_f16m1` | 1条 |
| FP16 × FP16 widening乘法 | `vfwmul_vf_f32m2` | 1条 |
| int32 → FP32转换 | `vfcvt_f_x_v_f32m2` | 1条 |
| FMA累加 | `vfmacc_vv_f32m2` | 1条 |
| **每l-block Scale处理** | **4条** | |

### 总指令数(单个x-block)

- K-loop: 32 × 3 = 96条
- Scale处理: 4条 (每l-block)
- **总计**: 100条/l-block (K-loop + Scale)

---

## 各平台对比分析

### 1. x86 AVX-512 VNNI

**核心特点**:
- 寄存器宽度: 512-bit = 64×int8 或 16×int32
- VNNI扩展提供int8 × int8 → int32直接点积
- 无lane-indexed variant，需vector × vector

**高价值指令**:
| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `VDPBSSD` (VNNI) | signed int8 × signed int8 → int32 dot (4元素累加) | **缺失** - Zvdot4a8i格式不兼容 |
| `VDPBUSD` (VNNI) | unsigned × signed dot product | **缺失** |
| `VPMADDUBSW` | unsigned × signed pairwise MAC (i16输出) | **缺失** |

**收益分析**:
```asm
; AVX-512 VNNI实现 (处理64个int8 → 16个int32)
; 归一化到VLEN=512
vdpbssd zmm_acc, zmm_b, zmm_a   ; 单条: 16×(4×int8×int8) → 16×int32
; 等效RVV: 32×(vwmul+vwadd) = 64条 → 1条
```

**归一化对比** (处理16列 × 32元素):

| 实现方式 | K-loop核心指令数 | l-block总计 |
|---------|-----------------|-------------|
| RVV当前(vwmul+vwadd) | 96条(32×3) | 100条 |
| x86 VNNI(vdpbssd) | 8条(16列需多次vdpbssd或vdpbusd) | ~12条 |
| **减少** | **约87.5%** (核心计算) | |

**注**: x86实现需数据重排以匹配16列并行布局，实际收益取决于数据组织效率。

---

### 2. ARM NEON (Dotprod Extension)

**核心特点**:
- 寄存器宽度: 128-bit = 16×int8 或 4×int32
- **关键优势**: `vdotq_laneq_s32` - lane-indexed dot product
- 归一化因子: 4 (相对于RVV VLEN=512)

**高价值指令**:
| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vdotq_laneq_s32` | lane-indexed dot product (从128-bit向量选择一个4-element lane作为乘数) | **缺失** - 无lane-indexed指令 |
| `vdotq_s32` | 4×int8 × int8 → int32 dot | Zvdot4a8i (格式不同) |

**Lane-indexed Dot语义详解**:
```c
// ARM NEON: vdotq_laneq_s32(acc, b, a, lane_index)
// acc: 4×int32累加器
// b: 16×int8向量
// a: 16×int8向量 (选择lane_index对应的4个元素)
// 语义: acc[j] += b[4j..4j+3] · a[4*lane_index..4*lane_index+3]
// 单条指令完成: 4×(4×int8×int8) → 4×int32
// 关键: a的4个元素被广播到b的每个4-element group
```

**ARM NEON实现示例** (类似场景，参考repack.cpp:474-485):
```asm
; ARM NEON处理4列并行
; 每次处理4×int8向量(b)与单个int8标量(a)
; 8条vdotq_laneq完成32×4=128次int8乘加
int32x4_t sumi = vdupq_n_s32(0);
sumi = vdotq_laneq_s32(sumi, b_0_lo, a_0, 0);  ; b_0[0..3] × a_0[0..3]
sumi = vdotq_laneq_s32(sumi, b_0_hi, a_1, 0);  ; b_0[4..7] × a_1[0..3]
sumi = vdotq_laneq_s32(sumi, b_1_lo, a_0, 1);  ; b_1[0..3] × a_0[4..7]
sumi = vdotq_laneq_s32(sumi, b_1_hi, a_1, 1);  ; ...
; 共8条vdot完成32元素×4列的int8点积
```

**归一化对比** (处理16列 × 32元素):

| 实现方式 | 指令数 | 说明 |
|---------|--------|------|
| RVV当前(K-loop) | 96条(32×3) | 每迭代: load + vwmul.vx + vwadd.wv |
| ARM NEON(lane-indexed) | 8条×4(归一化) = 32条等效 | 归一化因子4(128-bit vs 512-bit) |
| **减少** | **约67%** | |

**关键差距**: ARM的`vdotq_laneq_s32`在单条指令中完成:
1. 从a向量选择4-element lane (替代RVV的scalar broadcast)
2. 4×int8 × int8 dot → int32 (替代RVV的vwmul+vwadd两步)

---

### 3. LoongArch LASX

**核心特点**:
- 寄存器宽度: 256-bit = 32×int8 或 8×int32
- 归一化因子: 2 (相对于RVV VLEN=512)

**高价值指令**:
| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `xvmaddwev.h.b` | widening MAC even positions (int8→int16) | **缺失** |
| `xvmaddwod.h.b` | widening MAC odd positions (int8→int16) | **缺失** |

**Even/Odd MAC语义**:
```asm
; LoongArch LASX
; xvmaddwev.h.b vd, vs2, vs1
; 对于每个j: vd.h[j] += vs2.b[2j] × vs1.b[2j]  (even positions)
; xvmaddwod.h.b vd, vs2, vs1
; 对于每个j: vd.h[j] += vs2.b[2j+1] × vs1.b[2j+1]  (odd positions)
; 两条指令完成pairwise MAC，减少依赖链
```

**收益分析**:
- 可用于减少RVV `vwmul + vwadd` 的两步操作
- 但LoongArch输出为int16，需后续 widening add 到 int32

---

### 4. Power VSX (POWER10)

**核心特点**:
- 寄存器宽度: 128-bit (VSX)
- MMA扩展: 4×4矩阵乘辅助

**高价值指令**:
| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `xvf32gerpp` (MMA) | 4×4 FP32矩阵外积累加 | **缺失** - 无矩阵级指令 |

**收益分析**:
- MMA适用于GEMM而非GEMV
- 对gemv_q8_0_16x1场景收益有限

---

### 5. WASM SIMD

**核心特点**:
- 寄存器宽度: 128-bit = 16×int8
- 归一化因子: 4

**高价值指令**:
| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `i8x16.dot_i16x8_s` | 16×int8 × int8 → 8×int16 dot | **缺失** |

**语义**:
```asm
; WASM SIMD
; i8x16.dot_i16x8_s(a, b)
; 输出: 8×int16，每个为pairwise sum
; result[j] = a[2j]×b[2j] + a[2j+1]×b[2j+1]
```

**收益分析**:
- 输出int16而非int32，需后续累加
- 单条完成16×int8 pairwise MAC

---

### 6. S390X Z-Vector

**核心特点**:
- 寄存器宽度: 128-bit
- 归一化因子: 4

**高价值指令**:
| 指令 | 功能 | RVV现状 |
|------|------|---------|
| `vpkd` | pack/decompress | 用于量化解压 |
| `vsumg` | sum across vector | 归约辅助 |

**收益分析**:
- 对gemv_q8_0场景收益有限

---

## RVV现有扩展分析: Zvdot4a8i

### Zvdot4a8i扩展概述

Zvdot4a8i扩展定义了`vdot4a`指令族，提供4×int8点积能力:

| 指令 | 输入类型 | 操作 |
|------|----------|------|
| `vdot4au` | vuint32 (packed 4×uint8) | unsigned × unsigned → uint32 |
| `vdot4asu` | vuint32 (packed 4×uint8) | packed unsigned × packed unsigned → signed int32 accumulator |

### Zvdot4a8i与Q8_0工作负载的不兼容性

**问题1: 输入格式不匹配**
```
; Zvdot4a8i vdot4a要求:
; vs2 = vuint32向量 (每个uint32 packed 4×int8)
; 示例: vs2.w[i] = {int8_0, int8_1, int8_2, int8_3} as uint32
; 
; Q8_0实际数据格式:
; b_ptr[l].qs[i*16] = 16×int8向量 (直接int8存储)
; 需额外打包操作才能使用vdot4a
```

**问题2: 无signed×signed variant**
```
; Q8_0使用signed int8 × signed int8
; Zvdot4a8i仅支持:
;   vdot4au: unsigned × unsigned
;   vdot4asu: signed(被解释为unsigned) × unsigned
; 无pure signed × signed dot product
```

**问题3: 无lane-indexed版本**
```
; ARM NEON vdotq_laneq_s32可从a向量选择特定lane
; Zvdot4a8i无此功能，需预先广播或重排
```

### 结论

Zvdot4a8i扩展不适合Q8_0 × Q8_0工作负载，需要新的扩展指令。

---

## RVV扩展指令建议详细说明

### [P0] vdot4ss.vv - Signed int8 × Signed int8 4元素点积

**指令定义**:
```
vdot4ss.vv vd, vs2, vs1, vm
语义: 对于每个i (0 <= i < VL/4):
  vd.w[i] += vs2.b[4i]×vs1.b[4i] + vs2.b[4i+1]×vs1.b[4i+1] +
            vs2.b[4i+2]×vs1.b[4i+2] + vs2.b[4i+3]×vs1.b[4i+3]
输入: vs2, vs1 = signed int8向量
输出: vd = signed int32向量 (widening + horizontal sum)
编码约束:
  - SEW_src = 8 (源元素宽度)
  - SEW_dest = 32 (widening ratio 4:1)
  - LMUL_dest = 4 × LMUL_src
```

**与Zvdot4a8i关键差异**:
| 特性 | Zvdot4a8i vdot4a | 本文建议vdot4ss.vv |
|------|-------------------|-------------------|
| 输入类型 | vuint32 (packed) | 直接vint8向量 |
| signed×signed | 不支持 | 支持 |
| 数据准备 | 需打包为uint32 | 无额外开销 |

**应用场景**: Q8_0 × Q8_0量化推理int8点积，消除i16中间步骤。

**性能对比**:

| 实现方式 | K-loop指令数 (VLEN=512, 16列×32元素) |
|---------|--------------------------------------|
| RVV当前(vwmul.vx+vwadd.wv) | 96条(32×3) + 4条scale = 100条 |
| RVV+vdot4ss.vv (需数据重排) | ~16条(假设每条处理64元素) + 4条 = ~20条 |
| **减少** | **约80%** (假设理想数据布局) |

**注**: 实际收益取决于数据重排开销。若b矩阵已预先packed为4×int8 groups，可接近理想收益。

---

### [P0] vdot_lane.vx - Lane-indexed Dot Product

**指令定义**:
```
vdot_lane.vx vd, vs2, vs1, lane_idx, vm
语义: 对于每个i (0 <= i < VL/4):
  ; 从vs1选择lane_idx对应的4个int8元素
  ; lane_idx指定vs1中的哪个4-element group
  ; 设 lane_base = lane_idx × 4
  vd.w[i] += vs2.b[4i]×vs1.b[lane_base] + 
            vs2.b[4i+1]×vs1.b[lane_base+1] +
            vs2.b[4i+2]×vs1.b[lane_base+2] + 
            vs2.b[4i+3]×vs1.b[lane_base+3]
输入: 
  vs2 = signed int8向量 (完整向量)
  vs1 = signed int8向量 (提供4-element lane)
  lane_idx = 立即数(0..VL/4-1)
输出: vd = signed int32向量
```

**ARM NEON对应**: `vdotq_laneq_s32(acc, b, a, lane_index)`

**应用场景**: gemv_q8_0_16x1中的scalar广播替代。

**性能对比** (K-loop单迭代):

| 实现方式 | 指令数 | 说明 |
|---------|--------|------|
| RVV当前 | 3条 (vle8 + vwmul.vx + vwadd.wv) | scalar广播+widening乘+累加 |
| RVV+vdot_lane.vx | 2条 (vle8 + vdot_lane.vx) | 加载+lane-indexed dot |
| **减少** | **33%** | K-loop BB内 |

**详细收益计算**:

当前RVV K-loop(32迭代):
- 每迭代: 1 load + 1 vwmul.vx + 1 vwadd.wv = 3条
- 32迭代: 96条
- Scale处理: 4条
- **总计**: 100条

使用vdot_lane.vx:
- 每迭代: 1 load + 1 vdot_lane.vx = 2条
- 32迭代: 64条
- Scale处理: 4条 (不变)
- **总计**: 68条
- **减少**: (100-68)/100 = 32%

---

### [P1] vwmacc_lane.vx - Lane-indexed Widening MAC

**指令定义**:
```
vwmacc_lane.vx vd, vs2, vs1, lane_idx, vm
语义: 对于每个i (0 <= i < VL):
  ; 从vs1选择lane_idx对应元素(或2元素pair)
  ; widening multiply + accumulate
  vd.w[i] += (int32)vs2.h[i] × (int32)vs1.b[lane_idx]
; 或对于int8→int16 variant:
vwmacc_lane.h.vx vd, vs2, vs1, lane_idx, vm
  vd.h[i] += (int16)vs2.b[i] × (int16)vs1.b[lane_idx]
```

**应用场景**: 辅助vdot_lane.vx，用于部分widening MAC场景。

---

### [P2] vwmaccwev/wod.vv - Even/Odd Widening MAC

**指令定义**:
```
vwmaccwev.vv vd, vs2, vs1, vm  ; even positions
语义: 对于每个j (0 <= j < VL/2):
  vd.h[j] += vs2.b[2j] × vs1.b[2j]
  
vwmaccwod.vv vd, vs2, vs1, vm  ; odd positions
语义: 对于每个j (0 <= j < VL/2):
  vd.h[j] += vs2.b[2j+1] × vs1.b[2j+1]
```

**LoongArch对应**: `xvmaddwev.h.b` / `xvmaddwod.h.b`

**应用场景**: 减少依赖链，even/odd分离计算可并行执行。

**收益**: 两条指令完成pairwise MAC，但输出int16需后续处理。

---

## 归一化收益汇总 (VLEN=512)

### K-loop BB内收益

| 扩展指令组合 | K-loop指令减少 | Scale处理 | 总减少比例(K-loop BB) |
|-------------|---------------|-----------|----------------------|
| 仅vdot_lane.vx | 33% (96→64条) | 不变(4条) | **32%** |
| 仅vdot4ss.vv | ~80% (假设理想布局) | 不变 | ~76% |
| vdot_lane.vx + vdot4ss.vv | 取最大 | 不变 | ~32% (受限于数据布局) |

### 整体收益估算限制

**无BBV数据**: 无法计算整体收益，需以下信息:
1. gemv_q8_0_16x1函数在推理中的调用频率
2. K-loop BB在各BB中的执行占比
3. l-block循环的执行次数分布

建议通过 `./tools/profile_to_dfg.sh` 获取BBV数据后重新估算。

---

## 附录

### Dot Product指令对比表

| 平台 | 指令 | 操作 | 输入宽度 | 输出宽度 | Lane-indexed | RVV现状 |
|------|------|------|---------|---------|--------------|---------|
| x86 VNNI | VDPBSSD | i8×i8→i32(4-sum) | 512-bit | 16×i32 | No | Zvdot4a8i(格式不兼容) |
| ARM NEON | vdotq_s32 | i8×i8→i32(4-sum) | 128-bit | 4×i32 | No | Zvdot4a8i |
| ARM NEON | vdotq_laneq_s32 | i8×i8→i32(4-sum) | 128-bit | 4×i32 | **Yes** | **缺失** |
| LoongArch LASX | xvmaddwev/wod | i8×i8→i16(pair) | 256-bit | 16×i16 | No | **缺失** |
| WASM SIMD | i8x16.dot_i16x8_s | i8×i8→i16(pair) | 128-bit | 8×i16 | No | **缺失** |
| RVV | vwmul.vx + vwadd.wv | i8×i8→i32(两步) | 可变 | 可变 | No | 已有 |
| RVV(Zvdot) | vdot4a/vdot4au/vdot4asu | 4×i8→i32 | packed u32 | 可变 | No | 可选扩展 |

### Widening MAC指令对比表

| 平台 | 指令 | 操作 | 输入 | 输出 | RVV现状 |
|------|------|------|------|------|---------|
| RVV | vwmacc.vx | scalar×vector wid. MAC | i8×i8 | i16 | 已有 |
| RVV | vwmacc.vv | vector×vector wid. MAC | i16×i16 | i32 | 已有 |
| ARM SVE2 | indexed MAC | lane-indexed wid. MAC | 可变 | 可变 | **缺失** |
| LoongArch | xvmaddwev/wod | even/odd wid. MAC | i8×i8 | i16 | **缺失** |

---

## 结论

### 关键差距总结

1. **P0级差距 - Lane-indexed Dot Product**: ARM NEON的`vdotq_laneq_s32`在单条指令中完成lane选择+4×int8 dot+int32累加。RVV需3条指令(load + vwmul.vx + vwadd.wv)。建议新增`vdot_lane.vx`，K-loop BB内收益约33%。

2. **P0级差距 - Signed×Signed int8→int32单步点积**: x86 VNNI的`VDPBSSD`一次完成4×int8×int8→int32点积。RVV现有Zvdot4a8i扩展格式不兼容(需packed uint32)，且不支持signed×signed。建议新增`vdot4ss.vv`，直接使用int8向量输入。

3. **P1级差距 - Lane-indexed Widening MAC**: ARM/SVE2提供indexed MAC指令，RVV的`vwmacc`仅支持scalar或完整vector操作数。建议新增`vwmacc_lane.vx`作为辅助。

4. **P2级差距 - Even/Odd Widening MAC**: LoongArch LASX的`xvmaddwev/wod`可减少依赖链。RVV无类似指令。

### Zvdot4a8i适用性分析

Zvdot4a8i扩展不适合Q8_0 × Q8_0工作负载:
- **输入格式**: 要求packed uint32，Q8_0为直接int8存储
- **signed×signed**: 不支持，仅unsigned variants
- **lane-indexed**: 无，需额外广播/重排

建议设计新的`vdot4ss.vv`和`vdot_lane.vx`指令，直接支持signed int8向量输入和lane索引。

### 优先级建议

| 优先级 | 扩展指令 | 预估收益(K-loop BB) | 实现复杂度 |
|--------|----------|---------------------|-----------|
| P0 | vdot_lane.vx | 最高(K-loop减少33%) | 高(需lane-indexed设计) |
| P0 | vdot4ss.vv | 高(核心计算减少~80%) | 中(类似Zvdot但格式调整) |
| P1 | vwmacc_lane.vx | 中(辅助优化) | 中 |
| P2 | vwmaccwev/wod.vv | 低(减少依赖链) | 中 |

### 后续工作建议

1. **获取BBV profiling数据**: 运行`./tools/profile_to_dfg.sh`获取gemv_q8_0_16x1的实际执行热点分布，计算整体收益。

2. **数据布局优化**: 若b矩阵可预先packed为4×int8 groups，`vdot4ss.vv`收益可最大化。

3. **指令编码设计**: 为`vdot_lane.vx`和`vdot4ss.vv`设计具体的指令编码，提交RISC-V国际社区审议。

---

## 审查日志

| 轮次 | 发现问题数 | 已修复 | 剩余 |
|------|-----------|--------|------|
| R1 | 3 | 3 | 0 |
| R2 | 0 | - | 0 |

最终审查结论：审查通过，所有问题已修复。

### R1修复详情

1. **[MAJOR]** Zvdot4a8i表格`vdot4au`行: 输出类型从`int32`修正为`uint32`
2. **[MINOR]** Zvdot4a8i表格`vdot4asu`行: 澄清语义描述为`packed unsigned × packed unsigned → signed int32 accumulator`
3. **[MINOR]** 附录Dot Product对比表: RVV(Zvdot)行指令名称从`vdot4a`更新为`vdot4a/vdot4au/vdot4asu`以展示多个变体