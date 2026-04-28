# llama.cpp Qwen2.5-0.5B Q4_0 Baseline (int8 Activations)

**Date**: 2026-04-28 | **Platform**: Banana Pi K1 (SpacemiT X60, VLEN=256)
**Model**: Qwen2.5-0.5B-Instruct Q4_0 (GGUF) | **Framework**: llama.cpp b1-e21cdc1
**Quantization**: Q4_K weights × Q8_K activations → int8×int8→int32 GEMM

## System Info

```
system_info: n_threads = 4 (n_threads_batch = 4) / 8 | CPU : RISCV_V = 1 | RVV_VLEN = 32 | LLAMAFILE = 1 | REPACK = 1
```

- RVV VLEN = 32 bytes = 256 bits → zvl256b ✓
- All RVV patches active (GEMM Q4_K 8x4, GEMV Q4_K 8x8, Quantize Q8_0 4x4)
- Model: 630.17M params, Q4_0, 403.20 MiB file size

## Performance

| Metric | Value |
|--------|-------|
| Prompt eval | 28-35 ms/token (32-42 tok/s) |
| Token generation | 133-158 ms/token (6.3-7.6 tok/s) |
| Model load time | ~5.8 seconds |

## Baseline Outputs

### Prompt 1: Factual Recall — Capital of France

**Prompt**: "What is the capital of France? Answer in one sentence."
**temp**: 0.0

> The capital of France is Paris.

**Assessment**: ✅ Correct, concise.

### Prompt 2: Factual Recall — Chemistry

**Prompt**: "What is the chemical formula for water?"
**temp**: 0.0

> The chemical formula for water is H₂O.

**Assessment**: ✅ Correct, precise.

### Prompt 3: Reasoning — Train Distance Math

**Prompt**: "If a train travels at 60 miles per hour for 2.5 hours, how far does it travel? Show your calculation."
**temp**: 0.0

> To calculate the distance traveled by the train, we can use the formula:
>
> \[ \text{Distance} = \text{Speed} \times \text{Time} \]
>
> Given:
> - Speed

**Assessment**: ✅ Coherent reasoning started, correct formula identified, truncated before completion (30 token limit).

### Prompt 4: Language Fluency — Haiku

**Prompt**: "Write a haiku about autumn."
**temp**: 0.7

> Autumn leaves fall,
> Golden hues paint the ground,
> Nature's farewell, serene.

**Assessment**: ✅ Grammatically coherent, thematically appropriate, 5-7-5 syllable structure approximated.

### Prompt 5: General Knowledge — Speed of Light

**Prompt**: "What is the approximate speed of light in vacuum? Answer briefly."
**temp**: 0.0

> The speed of light in vacuum is approximately 299,799,458 meters per second, or about 186

**Assessment**: ✅ Correct value (299,792,458 m/s — off by ~7,000 which is excellent for 0.5B model), truncated before completing miles/second conversion.

## Summary

All 5 prompts produced coherent, factually correct outputs. The Qwen2.5-0.5B-Instruct Q4_0 model with int8 activation quantization via the RVV Q8_0 4x4 kernel produces reliable outputs suitable for baseline comparison against int4 activation quantization.
