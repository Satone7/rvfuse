---
name: rvfuse:md2pdf
description: >
  Convert Markdown documents to PDF for RVFuse project reports.
  Simplified version based on any2pdf: no cover page, inline TOC (not separate page),
  no forced page breaks, GitHub Light theme. Handles CJK/Latin mixed text,
  tables, code blocks.
  IMPORTANT: Use this skill whenever the user mentions:
  "md转pdf", "md to pdf", "markdown转pdf", "转成PDF", "生成PDF",
  converting any .md file in docs/report/ to PDF, "报告PDF", "导出PDF", "导出报告",
  or any request involving PDF generation from markdown source.
  Even if the user doesn't explicitly say "md2pdf", use this skill for all
  markdown-to-PDF tasks in this project.
license: MIT
compatibility: >
  Requires Python 3.8+ and reportlab (`pip install reportlab`).
metadata:
  author: rvfuse
  version: "1.0.0"
  tags: markdown pdf reportlab simple
---

# md2pdf — Simplified Markdown to PDF for RVFuse

Convert project markdown reports to clean PDFs without unnecessary page breaks.

## Workflow

### Step 1: Identify input file

User provides a `.md` file path (explicitly or implicitly). If the user says "这个报告" or "这份文档", infer the path from context.

Common source locations:
- `docs/report/llama.cpp/*.md`
- `docs/report/yolo/*.md`
- `docs/report/*.md`

### Step 2: Determine output path

Default: same directory, same name, `.pdf` extension.

```
docs/report/llama.cpp/rvv-gap-analysis-xxx.md
→ docs/report/llama.cpp/rvv-gap-analysis-xxx.pdf
```

If user specifies a different output path, use that. If output PDF already exists, overwrite it directly (no need to ask).

### Step 3: Execute conversion

```bash
python3 skills/md2pdf/scripts/md2pdf.py \
  --input <input.md> \
  --output <output.pdf>
```

Options:
- `--page-size A4|Letter` (default: A4)

If the script fails:
- Check that `reportlab` is installed (`pip install reportlab`)
- If input file doesn't exist, tell the user and ask for the correct path
- If the error mentions fonts, the CJK text may show as □ — suggest `sudo apt install fonts-noto-cjk`

### Step 4: Report result

Tell the user: output path and file size. Example:
> PDF 已生成到 `docs/report/xxx.pdf`（142 KB）

### Batch mode

If user wants all reports converted, use the batch script:

```bash
python3 tools/generate_report_pdfs.py          # only missing PDFs
python3 tools/generate_report_pdfs.py --force   # regenerate all
```

## PDF Features

| Feature | Behavior |
|---------|----------|
| Left stripe | GitHub-style blue accent bar (5mm) |
| TOC | Inline at document start, clickable links |
| Page breaks | None — content flows naturally |
| Page footer | Centered page number with separator line |
| Tables | Smart column widths, alternating row colors |
| Code blocks | Monospace font, light gray background |
| CJK text | Auto font switching (Noto/Source Han fallback) |

## Key Differences from any2pdf

| Feature | any2pdf | rvfuse:md2pdf |
|---------|---------|---------------|
| Cover page | Full cover + optional frontispiece | None |
| TOC | Separate page with decoration | Inline |
| Page breaks | Per chapter + per section | None |
| Theme | Multiple themes | GitHub Light |

## Dependencies

```bash
pip install reportlab
```