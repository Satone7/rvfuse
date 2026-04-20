# Report Index

This directory contains analysis reports generated during RVFuse development.

## Inline Format Test

This section validates that all common Markdown inline formats render correctly in PDF output:

| Format | Syntax | Rendered |
|--------|--------|----------|
| Bold | `**bold text**` | **bold text** |
| Italic | `*italic text*` | *italic text* |
| Bold Italic | `***bold italic***` | ***bold italic*** |
| Inline Code | `` `code` `` | `code` |
| Strikethrough | `~~strikethrough~~` | ~~strikethrough~~ |
| Link | `[link](url)` | [link](url) |
| Combined | `**bold** *italic* ~~strike~~` | **bold** *italic* ~~strike~~ |

## Directory Structure

```
docs/report/
├── llama.cpp/          # llama.cpp RVV optimization reports
│   ├── benchmark/data/ # Raw performance data
│   └── *.md / *.pdf    # Gap analysis and optimization reports
├── yolo/               # YOLO/ONNX Runtime analysis reports
│   └── *.md / *.pdf    # Quantization, vector extension reports
└── *.md / *.pdf        # General RVV extension proposals
```

## llama.cpp Reports

| Report | Description |
|--------|-------------|
| `rvv-gap-analysis-quantize-mat-q8-0-4x4-*` | Q8_0 quantization matrix gap analysis |
| `rvv-gap-analysis-gemm-q4_K-8x4-q8_K-*` | GEMM Q4_K×Q8_K kernel analysis |
| `rvv-gap-analysis-ggml_gemv_q4_K_8x8_q8_K-*` | GEMV Q4_K×Q8_K kernel analysis |
| `rvv-gap-analysis-gemv_q8_0_16x1_q8_0-*` | GEMV Q8_0 kernel analysis |
| `rvv-gap-analysis-ggml_gemv_q4_0_16x1_q8_0-*` | GEMV Q4_0×Q8_0 kernel analysis |
| `rvv-gap-analysis-gemv-integrated-*` | Integrated GEMV optimization analysis |
| `rvv-gap-analysis-gemv-benefit-estimation-*` | Performance benefit estimation |
| `rvv-gap-analysis-vec-dot-q5_0_q8_0-zh-*` | Vec-dot Q5_0×Q8_0 analysis (Chinese) |
| `perf_q4_hotspot_analysis-*` | Q4 kernel hotspot profiling analysis |
| `llama_cpp_perf_q4_v_analysis-*` | Q4_V quantization performance analysis |

## YOLO Reports

| Report | Description |
|--------|-------------|
| `multi-platform-vector-comparison-2026-04-16` | Multi-platform vector extension comparison |
| `onnxruntime_quantization_support_report-2026-04-17` | ONNX Runtime quantization support analysis |
| `rvv-extension-analysis-report-2026-04-17` | RVV extension requirements for YOLO |
| `latency_weighted_benefit_analysis-2026-04-17` | Latency-weighted benefit analysis |
| `latency_weighted_summary-2026-04-17` | Summary of latency-weighted findings |

## General RVV Reports

| Report | Description |
|--------|-------------|
| `rvv-extension-proposal-*` | RVV extension proposal for fusion |
| `rvv-extension-comprehensive-analysis-*` | Comprehensive RVV extension analysis |

## Raw Data

Performance benchmark raw data is stored in:
- `applications/llama.cpp/benchmark/data/`
  - `perf_q4.txt` — Q4 kernel profiling (55MB)
  - `perf_q4_v.txt` — Q4_V kernel profiling

## PDF Generation

Use `tools/generate_report_pdfs.py` to generate PDFs from markdown reports:

```bash
# Generate missing PDFs
python3 tools/generate_report_pdfs.py

# Force regenerate all PDFs
python3 tools/generate_report_pdfs.py --force
```

The script uses the project's `md2pdf` skill (`.claude/skills/md2pdf/md2pdf.py`) which:
- Uses reportlab for professional table rendering
- Inline TOC with clickable links (no separate page)
- GitHub Light theme
- No forced page breaks (content flows naturally)