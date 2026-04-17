# llama.cpp 性能热点分析报告 — Qwen2.5-0.5B Q4_K_M

## 测试环境

- **平台**: 香蕉派 K1 (Spacemit K1-X, rv64imafdcv, 8核, 内核 6.6.36)
- **模型**: Qwen2.5-0.5B-Instruct-Q4_K_M (469MB)
- **编译**: 动态链接标量版 (`-march=rv64gc`)
- **工具**: perf (cpu-clock 采样, 3,624,834 samples)
- **Prompt**: "hi", 生成 128 tokens

---

## 总览

推理阶段 **98% 的时间花在矩阵乘法**，几乎全部是量化格式的向量点积（vec_dot）运算。

```
ggml_graph_compute (98.04%)
  └─ ggml_compute_forward (97.89%)
       └─ ggml_compute_forward_mul_mat (96.82%)
            └─ mul_mat_one_chunk (96.64%)
                 ├─ vec_dot_q5_0_q8_0_generic   55.45%
                 ├─ vec_dot_q6_K_q8_K_generic    18.22%
                 ├─ vec_dot_q4_K_q8_K_generic    15.92%
                 ├─ vec_dot_q8_0_q8_0_generic     5.59%
                 └─ flash_attn_ext                 0.80%
```

---

## 热点函数排行

| 排名 | 函数 | Self% | 说明 |
|------|------|-------|------|
| 1 | `ggml_vec_dot_q5_0_q8_0_generic` | **55.45%** | Q5_0 权重与 Q8_0 激活值的向量点积（反量化 + 乘加） |
| 2 | `ggml_vec_dot_q6_K_q8_K_generic` | **18.22%** | Q6_K 权重与 Q8_K 激活值的向量点积 |
| 3 | `ggml_vec_dot_q4_K_q8_K_generic` | **15.92%** | Q4_K 权重与 Q8_K 激活值的向量点积 |
| 4 | `ggml_vec_dot_q8_0_q8_0_generic` | **5.59%** | Q8_0 权重与 Q8_0 激活值的向量点积 |
| 5 | `ggml_compute_forward_flash_attn_ext` | 0.80% | Flash Attention 扩展 |
| 6 | `memcpy` | 0.52% | 数据搬运 |

---

## 热点分类

### 矩阵乘法（96.82%）

推理的绝对瓶颈。每个 Token 生成都需要对所有权重矩阵做一次矩阵乘法。在标量实现下，这些 `_generic` 后缀的函数是纯 C 循环，没有向量化或 SIMD 加速。

### Flash Attention（0.80%）

Attention 计算占比很小，因为 Qwen2.5-0.5B 是小模型（只有 24 层，注意力头数少）。

### 其他（~2.4%）

包括内存拷贝、归一化、激活函数、采样等，占比都很小。

---

## 为何 Q4 模型的最大热点是 Q5_0？

Q4_K_M 是**混合精度**量化方案，不同层根据张量重要性使用不同 bit 宽度：

| 量化格式 | 占比 | 用于哪些层 |
|---------|------|-----------|
| Q5_0 | 55.45% | FFN 的 up/down 投影矩阵（层最大，占计算主体） |
| Q6_K | 18.22% | Attention 的 K/V 矩阵（精度敏感） |
| Q4_K | 15.92% | 其余权重层 |
| Q8_0 | 5.59% | 首层 embedding / 末层 output（保持精度） |

"Q4_K_M" 的 Q4 指平均每权重约 4.xx bit，不是每层都是 4bit。越大的矩阵分配越多 bit（Q5、Q6），因为它们对精度影响更大。

---

## RVV 向量化优化建议

按收益排序：

### 优先级 1：Q5_0 × Q8_0 向量点积（预期收益 55%）

当前热点函数 `ggml_vec_dot_q5_0_q8_0_generic` 是纯标量 C 实现。优化方向：

- 用 RVV 向量加载 (vle32.v/vle64.v) 替代标量逐元素加载
- 用 vredsum 做向量求和
- 5-bit 反量化可以用 vnsrl + vmul 做向量级缩放

### 优先级 2：Q6_K × Q8_K 向量点积（预期收益 18%）

K-quant 格式的 super-block 结构更复杂，需要先做块级反量化再乘加。

### 优先级 3：Q4_K × Q8_K 向量点积（预期收益 16%）

4-bit 反量化用 vnsrl 移位提取每个 4-bit 值，然后用 vmul 做向量乘法。

### 优先级 4：Q8_0 × Q8_0 向量点积（预期收益 6%）

最简单的格式，8-bit 直接加载和乘加。

---

## 结论

在 RISC-V 标量 CPU 上运行 Qwen2.5-0.5B Q4_K_M 模型：

1. **97% 的计算时间在矩阵乘法**，其中 55% 集中在 Q5_0 的向量点积
2. 所有热点函数都是 `_generic` 标量实现，RVV 向量化空间巨大
3. 优先实现 Q5_0 的 RVV 向量点积 kernel，单这一个优化预期可覆盖 55% 的计算开销
4. 完成全部 4 个量化格式的 RVV kernel 后，理论上可覆盖 95% 以上的推理计算

---

*报告生成时间: 2026-04-17*
*工具: perf 6.6.36, 采样事件 cpu-clock, 3.6M samples*
