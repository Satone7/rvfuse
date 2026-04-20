# llama.cpp Q4_0 向量化推理性能分析报告

## 基本信息

- **数据来源**: `applications/llama.cpp/temp/perf_q4_v.txt`
- **模型**: Q4_0 量化格式
- **采样数**: 108K samples (cpu-clock事件)
- **事件计数**: ~27.09 billion
- **配置**: 开启向量化支持

---

## 函数分类与原始数据

### 一、核心计算函数 (推理计算)

#### 1. GEMV/GEMM 矩阵运算 (核心热点)

| 函数 | Children | Self | 说明 |
|------|----------|------|------|
| `ggml_gemv_q4_0_16x1_q8_0` | 27.82% | 27.73% | Q4_0量化矩阵向量乘法 (主要热点) |
| `ggml_gemv_q8_0_16x1_q8_0` | 16.44% | 16.39% | Q8_0量化矩阵向量乘法 |
| `ggml_gemm_q4_0_16x1_q8_0` | 5.56% | 5.49% | Q4_0量化矩阵乘法 |
| `void ggml::cpu::repack::gemv<block_q4_0, 1l, 16l>` | 0.05% | 0.05% | repack模板函数 |

**GEMV/GEMM Self总计**: **49.61%**

#### 2. 数据重打包函数 (Repack)

| 函数 | Children | Self | 说明 |
|------|----------|------|------|
| `repack_q4_0_to_q4_0_16_bl` | 8.82% | 8.01% | Q4_0数据重打包为16块布局 |
| `repack_q8_0_to_q8_0_16_bl` | 5.21% | 4.61% | Q8_0数据重打包为16块布局 |
| `ggml::cpu::repack::tensor_traits<block_q4_0>::compute_forward` | 0.99% | 0.71% | repack计算前向传播 |

**Repack Self总计**: **13.33%**

#### 3. 量化函数

| 函数 | Children | Self | 说明 |
|------|----------|------|------|
| `ggml_quantize_mat_q8_0_4x1` | 1.57% | 0.58% | Q8_0矩阵量化 |
| `roundf32` | 0.92% | 0.92% | 浮点round (量化辅助) |
| `quantize_row_q8_0` | 0.20% | 0.20% | 行量化 |

**量化 Self总计**: **1.70%**

#### 4. Flash Attention

| 函数 | Children | Self | 说明 |
|------|----------|------|------|
| `ggml_compute_forward_flash_attn_ext_f16_one_chunk` | 0.92% | 0.76% | Flash Attention核心计算 |
| `ggml_compute_forward_flash_attn_ext` | 0.11% | 0.04% | Flash Attention调度 |

**Flash Attention Self总计**: **0.80%**

#### 5. 其他计算前向传播函数

| 函数 | Self | 说明 |
|------|------|------|
| `ggml_graph_compute_thread.isra.0` | 1.44% | 计算图线程调度 |
| `ggml_compute_forward_add_non_quantized` | 0.36% | 非量化加法 |
| `ggml_vec_swiglu_f32` | 0.31% | SwiGLU向量操作 |
| `ggml_vec_dot_f16` | 0.21% | F16向量点积 |
| `rope_yarn` | 0.20% | RoPE位置编码计算 |
| `ggml_compute_forward_rms_norm` | 0.18% | RMS Norm归一化 |
| `rotate_pairs<float>` | 0.17% | RoPE旋转计算 |
| `void ggml_compute_forward_rope_flt<float>` | 0.09% | RoPE前向传播 |
| `ggml_compute_forward_mul` | 0.12% | 乘法运算 |
| `ggml_compute_forward_glu` | 0.09% | GLU门控单元 |
| `ggml_compute_forward_set_rows` | 0.03% | 设置行操作 |
| `ggml_compute_forward_add` | 0.03% | 加法运算 |

**其他计算函数 Self总计**: **~2.82%**

#### 6. 数学运算函数

| 函数 | Self | 说明 |
|------|------|------|
| `expf` | 0.15% | 指数函数 (softmax) |
| `sincosf32` | 0.14% | 三角函数 (RoPE) |
| `powf` | 0.08% | 幂函数 |
| `logf` | 0.06% | 对数函数 |

**数学函数 Self总计**: **0.43%**

---

### 二、初始化/加载相关函数 (忽略)

| 函数 | Self | 说明 |
|------|------|------|
| `llama_vocab::impl::load` | 2.22% | 词汇表加载 |
| `llama_kv_cache::llama_kv_cache` | 0.00% | KV缓存构造 |
| `llama_vocab::~llama_vocab` | 0.16% | 词汇表析构 |
| `gguf_reader::read` | 0.05% | GGUF文件读取 |
| `llama_mmap::impl::impl` | 0.00% | 内存映射初始化 |
| `llama_model::load_hparams` | 0.00% | 模型参数加载 |
| `llama_model_loader`构造函数 | 0.00% | 模型加载器初始化 |

**内存管理 (加载相关)**:
| 函数 | Self |
|------|------|
| `memset` | 1.21% |
| `malloc` | 0.67% |
| `free` | 0.19% |
| `operator new` | 0.08% |
| `memcpy` | 0.70% |
| `__mmap` | 0.00% |

**词汇表Hashtable操作 (加载相关)**:
| 函数 | Self |
|------|------|
| 各种 `std::_Hashtable::*` 操作 | ~2.5% |

**系统调用 (加载/初始化)**:
| 函数 | Self |
|------|------|
| `syscall` | 0.37% |
| `read` | 0.01% |
| `_IO_fread` | 0.40% |

---

## 三、等比放大后的计算函数占比

### 计算公式

假设忽略初始化/加载函数后，计算函数的总占比为100%。

**计算函数 Self 总计**:
- GEMV/GEMM: 49.61%
- Repack: 13.33%
- Quantize: 1.70%
- Flash Attention: 0.80%
- 其他计算: 2.82%
- 数学函数: 0.43%

**计算函数总和**: **68.69%**

**放大系数**: `100 / 68.69 = 1.46`

### 等比放大后的分布

| 类别 | 原始Self占比 | 放大后占比 |
|------|-------------|-----------|
| **GEMV/GEMM核心运算** | 49.61% | **72.58%** |
| **Repack数据重打包** | 13.33% | **19.48%** |
| **Quantize量化** | 1.70% | **2.48%** |
| **Flash Attention** | 0.80% | **1.17%** |
| **其他计算函数** | 2.82% | **4.11%** |
| **数学运算** | 0.43% | **0.63%** |

### 详细函数放大占比表

| 函数 | 原始Self | 放大后Self | 占比 |
|------|----------|-----------|------|
| `ggml_gemv_q4_0_16x1_q8_0` | 27.73% | **40.68%** | **40.68%** |
| `ggml_gemv_q8_0_16x1_q8_0` | 16.39% | **24.06%** | **24.06%** |
| `ggml_gemm_q4_0_16x1_q8_0` | 5.49% | **8.05%** | **8.05%** |
| `repack_q4_0_to_q4_0_16_bl` | 8.01% | **11.73%** | **11.73%** |
| `repack_q8_0_to_q8_0_16_bl` | 4.61% | **6.73%** | **6.73%** |
| `ggml::cpu::repack::tensor_traits<>::compute_forward` | 0.71% | **1.04%** | **1.04%** |
| `ggml_graph_compute_thread` | 1.44% | **2.11%** | **2.11%** |
| `ggml_quantize_mat_q8_0_4x1` | 0.58% | **0.85%** | **0.85%** |
| `roundf32` | 0.92% | **1.35%** | **1.35%** |
| `ggml_compute_forward_flash_attn_ext_f16_one_chunk` | 0.76% | **1.11%** | **1.11%** |
| `ggml_compute_forward_add_non_quantized` | 0.36% | **0.53%** | **0.53%** |
| `ggml_vec_swiglu_f32` | 0.31% | **0.45%** | **0.45%** |
| `ggml_vec_dot_f16` | 0.21% | **0.31%** | **0.31%** |
| `rope_yarn` | 0.20% | **0.29%** | **0.29%** |
| `ggml_compute_forward_rms_norm` | 0.18% | **0.26%** | **0.26%** |
| `rotate_pairs<float>` | 0.17% | **0.25%** | **0.25%** |
| `sincosf32` | 0.14% | **0.21%** | **0.21%** |
| `expf` | 0.15% | **0.22%** | **0.22%** |
| `ggml_compute_forward_mul` | 0.12% | **0.18%** | **0.18%** |
| `ggml_compute_forward_rope_flt<float>` | 0.09% | **0.13%** | **0.13%** |
| `ggml_compute_forward_glu` | 0.09% | **0.13%** | **0.13%** |
| `powf` | 0.08% | **0.12%** | **0.12%** |

---

## 四、关键洞察

### 1. 热点分布特征

**核心发现**: 在纯推理计算阶段，约 **72.58%** 的时间用于矩阵运算 (GEMV/GEMM):

| 操作类型 | 放大占比 | 热点级别 |
|----------|----------|---------|
| Q4_0 GEMV | 40.68% | 🔴 **极高热点** |
| Q8_0 GEMV | 24.06% | 🔴 **极高热点** |
| Q4_0 GEMM | 8.05% | 🟠 **高热点** |
| Q4_0 Repack | 11.73% | 🟠 **高热点** |
| Q8_0 Repack | 6.73% | 🟡 **中热点** |

### 2. 数据重打包开销显著

**Repack函数总占比: 19.48%**

这表明数据从原始Q4_0/Q8_0布局转换为16块布局存在较大开销。这可能是:
- 预推理时的权重重排
- 或推理时的动态重排

**潜在优化方向**: 如果repack是预推理的静态操作，可忽略；如果是动态操作，值得优化。

### 3. 量化开销相对较小

量化相关函数仅占 **2.48%**，表明量化计算本身开销不大，主要开销在矩阵运算。

### 4. Flash Attention占比低

Flash Attention仅占 **1.17%**，可能因为:
- 模型较小，attention head数量有限
- 或向量化实现已足够高效

### 5. 其他算子开销

| 算子类型 | 放大占比 |
|----------|----------|
| RMS Norm | 0.26% |
| RoPE位置编码 | ~0.42% (rope_yarn + rotate_pairs + rope_flt) |
| SwiGLU | 0.45% |
| Add/Mul | ~0.71% |
| Vec Dot | 0.31% |

这些算子开销相对较小，合计约 **2.1%**。

---

## 五、优化建议

### 优先级排序

1. **最高优先级 - GEMV优化** (72.58%)
   - `ggml_gemv_q4_0_16x1_q8_0` 占 40.68%
   - `ggml_gemv_q8_0_16x1_q8_0` 占 24.06%
   - 这是融合指令优化最关键的目标

2. **次高优先级 - Repack优化** (19.48%)
   - 评估是否可以预处理避免动态repack
   - 或优化repack算法本身

3. **中优先级 - GEMM优化** (8.05%)
   - 相比GEMV，GEMM占比较小但仍值得关注

4. **低优先级 - 其他算子** (~4%)
   - Flash Attention、量化、RoPE等开销较小

---

## 六、融合候选分析

基于热点分布，最值得关注的融合候选:

### 候选1: GEMV内部指令序列
`ggml_gemv_q4_0_16x1_q8_0` 函数内部可能包含:
- 反量化操作 (Q4_0 → FP32)
- 向量乘法累加
- 结果缩放

**建议**: 分析该函数的反汇编，识别可融合的指令序列。

### 候选2: Repack操作内部
`repack_q4_0_to_q4_0_16_bl` 可能包含:
- 内存拷贝/重排
- 块对齐操作

**建议**: 如果是动态操作，分析是否可融合内存操作。

### 候选3: RoPE位置编码
`rope_yarn` + `rotate_pairs` + `sincosf32` 合计约 0.63%，包含:
- 三角函数计算
- 复数旋转乘法

**建议**: 可考虑融合 sin/cos + rotate 操作。

---

## 附录: 内核函数分析

报告中约 **9.20%** 的时间在 `[unknown]` 内核地址，主要是:
- `0xffffffff80f457fe` 等内核入口
- 系统调度和内存管理相关

这部分在放大计算中已排除。