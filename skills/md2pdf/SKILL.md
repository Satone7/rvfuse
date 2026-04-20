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

The output PDF has these exact specifications:

| Feature | Specification |
|---------|--------------|
| Left stripe | Blue accent bar, 5mm wide, color `#0969DA` |
| TOC | Inline at document start, H1/H2 entries, clickable PDF links |
| Page breaks | None — content flows naturally, no blank pages |
| Page footer | Centered page number, separator line above it |
| Margins | 20mm on all sides |
| Body text | Sans font, 10pt, justified, 15pt leading |
| Headings | H1: 22pt bold, H2: 16pt bold, H3: 12pt accent color |
| Tables | Blue header (`#0969DA`), smart column widths (min 18mm), alternating rows |
| Code blocks | Monospace 8pt, gray background (`#F6F8FA`) |
| CJK text | Auto font switching (Noto/Source Han/WQY ZenHei fallback) |
| Page size | A4 (595.3 × 841.9pt), white background |

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