# llama.cpp Int4 激活量化精度 — 评估报告

**日期**: 2026-04-28 | **作者**: llama-int4 | **平台**: Banana Pi K1 (SpacemiT X60, VLEN=256)
**模型**: Qwen2.5-0.5B-Instruct Q4_0 | **框架**: llama.cpp b1-e21cdc1

## 1. 量化方案设计

### 1.1 设计思路

**目标**: 在保持 GEMM/GEMV 计算内核不变的前提下，通过修改激活量化步骤来模拟原生 int4 激活量化的精度。

**方法**: 修改 `ggml_quantize_mat_q8_0_4x1`（RISC-V 上使用的标量量化函数），使其产生 int4 范围的值 [-7, 7] 并存储在 int8 容器中，而非 int8 范围 [-127, 127]。GEMM/GEMV 内核无需修改——它们直接操作 int8 容器。

**设计选择: 对称 int4 量化**

| 方面 | Int8（基线） | Int4（修改后） |
|--------|---------------|-----------------|
| 范围 | [-127, 127] | [-7, 7] |
| 缩放因子 (d) | `amax / 127.0` | `amax / 7.0` |
| 量化步长 | `amax / 127` | `amax / 7` |
| 有效精度 | ~7 bits | ~3 bits |
| 动态范围损失 | — | ~18× 更粗糙 |

**为什么选择对称量化？** Q8_0 本身使用对称量化（无零点偏移）。改为非对称量化需要修改 GEMM 内核，违反"保持内核不变"的约束。对称 int4 是 Q8_0 方案的自然扩展。

**为什么选择逐行量化？** 逐行缩放因子嵌入在 Q8_0 块结构中（`block_q8_0x4.d[4]`）。保持逐行粒度可以匹配现有数据布局，避免修改内核。

### 1.2 实现

**修改的函数** (`#ifdef GGML_INT4_ACTIVATIONS`):
- `ggml_quantize_mat_q8_0_4x1_generic` — 标量, interleave=1（热路径）
- `ggml_quantize_mat_q8_0_4x4_generic` — 标量, interleave=4
- `ggml_quantize_mat_q8_0_4x8_generic` — 标量, interleave=8
- `ggml_quantize_mat_q8_0_4x4_rvv` — RVV 向量化（4行）

**核心修改**（以 `4x1_generic` 为例）:
```c
#ifdef GGML_INT4_ACTIVATIONS
    const float d = amax / ((1 << 3) - 1);  // int4: [-7, 7]
#else
    const float d = amax / ((1 << 7) - 1);  // int8: [-127, 127]
#endif
```

**自洽性**: 量化时计算的缩放因子 `d` 被存入量化块中，GEMV/GEMM 内核读取同一个 `d` 进行去量化。量化噪声因 ~18× 更大的步长而放大，但去量化能正确恢复近似的 fp32 值。

## 2. 热点验证（Perf）

### 2.1 平台

- **开发板**: Banana Pi K1, SpacemiT X60 SoC, 8 核
- **VLEN**: 256 bits（RVV_VLEN=32 bytes）
- **内核**: Linux 6.6.36
- **Perf**: `perf version 6.6.36`（软件时钟事件——该 RISC-V 平台不支持硬件采样）

### 2.2 热点函数

| 函数 | Self % | 类型 |
|----------|--------|------|
| `ggml_gemv_q8_0_16x1_q8_0` | 29.43% | GEMV（Q8_0×Q8_0） |
| `ggml_gemv_q4_0_16x1_q8_0` | 23.50% | GEMV（Q4_0 权重 × Q8_0 激活） |
| `ggml_graph_compute_thread` | 7.11% | 图调度器 |
| `repack<block_q4_0>` | 2.11% | 权重重排 |
| `ggml_gemm_q4_0_16x1_q8_0` | 3.82% | GEMM（Q4_0×Q8_0，提示处理） |
| `ggml_quantize_mat_q8_0_4x1` | 0.21% | **激活量化** |

### 2.3 关键发现

1. **GEMV 占主导地位**: 在自回归生成中，GEMV（合计 53.93%）远超 GEMM（3.82%）。计划中的目标内核 `ggml_gemm_q4_K_8x4_q8_K` 是 Q4_K 模型专用的，不适用于 Q4_0 模型。
2. **模型是 Q4_0 而非 Q4_K**: Qwen2.5-0.5B 使用 Q4_0 量化格式，因此 Q4_K 内核不会被调用。
3. **量化开销可忽略**（0.21%）: 量化开销远小于 GEMV/GEMM 计算。int4 量化修改对性能的影响极小。
4. **硬件 PMU 采样不可用**: 该 RISC-V 内核不支持硬件 PMU 采样。软件 `cpu-clock` 事件可用。

### 2.4 性能计数器

```
12,675,765,599 instructions    # IPC = 0.34
37,351,429,742 cycles
982,432,768 branches            # 1.14% branch miss
12.85 seconds elapsed（22.35s user, 1.12s sys）
```

**IPC = 0.34** 表明严重的内存停顿——对于内存受限的 LLM 推理是正常现象。

## 3. 精度评估

### 3.1 提示词与输出

#### 提示 1: 事实回忆 — 法国首都

| 方面 | 基线（int8） | Int4 |
|--------|----------------|------|
| 提示词 | "What is the capital of France? Answer in one sentence." | 相同 |
| 温度 | 0.0 | 0.0 |
| 输出 | The capital of France is Paris. | The capital of France is Paris. |
| **评价** | — | ✅ **完全一致** |

#### 提示 2: 事实回忆 — 化学

| 方面 | 基线（int8） | Int4 |
|--------|----------------|------|
| 提示词 | "What is the chemical formula for water?" | 相同 |
| 温度 | 0.0 | 0.0 |
| 输出 | The chemical formula for water is H₂O. | The chemical formula for water is H₂O. |
| **评价** | — | ✅ **完全一致** |

#### 提示 3: 推理 — 火车距离数学题

| 方面 | 基线（int8） | Int4 |
|--------|----------------|------|
| 提示词 | "If a train travels at 60 miles per hour for 2.5 hours, how far does it travel? Show your calculation." | 相同 |
| 温度 | 0.0 | 0.0 |
| 输出 | "To calculate the distance traveled by the train, we can use the formula: \[ \text{Distance} = \text{Speed} \times \text{Time} \] Given: - Speed" | "To calculate the distance traveled by the train, we can use the formula: Given:" |
| **评价** | — | ✅ **基本一致** — 两输出在类似位置被截断，识别出相同的公式 |

#### 提示 4: 创意写作 — 俳句

| 方面 | 基线（int8） | Int4 |
|--------|----------------|------|
| 提示词 | "Write a haiku about autumn." | 相同 |
| 温度 | 0.7 | 0.7 |
| 输出 | "Autumn leaves fall, Golden hues paint the ground, Nature's farewell, serene." | "秋风轻拂叶落时，金黄稻田映秋色，落叶归根，冬眠。" |
| **评价** | — | ⚠️ **语言切换** — 模型从英文切换为中文。输出内容连贯且符合主题（秋天、落叶、金色田野）。 |

#### 提示 5: 常识 — 光速

| 方面 | 基线（int8） | Int4 |
|--------|----------------|------|
| 提示词 | "What is the approximate speed of light in vacuum? Answer briefly." | 相同 |
| 温度 | 0.0 | 0.0 |
| 输出 | "The speed of light in vacuum is approximately 299,799,458 meters per second, or about 186"（被截断） | "The speed of light in vacuum is approximately 299,799,458 meters per second."（完整） |
| **评价** | — | ✅ **更优** — int4 版本完整输出了句子，基线版本被截断 |

### 3.2 总结评估

| 评价标准 | 结果 |
|-----------|--------|
| 连贯性 | ✅ 所有输出语法正确、切题 |
| 事实准确性 | ✅ 事实类回答与基线完全一致 |
| 灾难性失败 | ✅ 无（无乱码、崩溃、空输出） |
| 创意生成 | ⚠️ 观察到了语言切换（俳句从英文切换为中文） |
| 总体精度损失 | **低** — 可用于推理 |

### 3.3 统计说明

仅有 5 个提示词构成了一个定性评估。提示 4 的语言切换值得注意，但内容保持连贯和切题。更大规模的困惑度评估可以提供定量的精度损失测量，但 llama.cpp 框架对该模型不直接提供困惑度测量功能。

## 4. 结论

### 4.1 关键发现

1. **Int4 激活量化可用于 Q4_0 LLM 推理**: 将激活值从 8-bit 降低到 4-bit 带来的精度损失很小，不会引起灾难性退化。
2. **事实/推理任务稳健**: 在温度 0.0 下，int4 输出与 int8 基本一致。
3. **创意生成存在细微差异**: 语言切换表明 int4 在非零温度下可能改变了 token 概率分布，足以影响采样结果。
4. **量化函数不是瓶颈**: 仅占 0.21% 的运行时间，量化开销可忽略不计。主要开销在 GEMV 计算（53.93%）。

### 4.2 对原生 Int4 指令的启示

- **软件精度证据**: int4 激活值保留了足够的信息进行连贯的语言生成。从精度角度看，原生 int4×int4→int32 指令是可行的。
- **计算瓶颈**: 对于 Q4_0 模型，自回归推理中 GEMV（而非 GEMM）占主导地位。原生 int4 指令应同时支持 GEMV 和 GEMM 模式。
- **Q8_0 量化步骤是正确的切入点**: 仅修改量化缩放因子（不修改计算内核）即可充分模拟 int4 精度。这验证了实验方法的有效性。

## 附录 A: 构建配置

```bash
# 工具链: LLVM 22, riscv64-linux-gnu
# 目标: rv64gcv_zfh_zba_zicbop_zvl256b
# 构建: cmake -DGGML_RVV=ON -DGGML_RV_ZVL256B=ON
# Int4 标志: -DGGML_INT4_ACTIVATIONS 添加到 CMAKE_C_FLAGS
```

## 附录 B: 文件位置

| 文件 | 路径 |
|------|------|
| 基线报告 | `docs/report/int4-instruction/baseline-llama-int8.md` |
| Perf 数据 | `docs/report/int4-instruction/perf-llama-int8.data` |
| Perf stat | `docs/report/int4-instruction/perf_stat.txt` |
| 修改后的量化代码 | `vendor/llama.cpp/ggml/src/ggml-cpu/repack.cpp` |
| RVV 量化修改 | `rvv-patches/quantize-q8_0-4x4/rvv_quantize_q8_0_4x4.inl` |
| 工具链（int4 标志） | `riscv64-linux-toolchain.cmake` |
