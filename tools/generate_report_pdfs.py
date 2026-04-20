#!/usr/bin/env python3
"""Generate PDFs for markdown reports using md2pdf skill.

Uses the simplified md2pdf script (reportlab) instead of WeasyPrint:
- No cover page
- Inline TOC (not separate page)
- No forced page breaks
- GitHub Light theme
- Better table rendering
"""

import os
import sys
import glob
import subprocess

BASE_DIR = '/home/pren/wsp/cx/rvfuse/docs/report'
MD2PDF_SCRIPT = '/home/pren/wsp/cx/rvfuse/skills/md2pdf/scripts/md2pdf.py'

def md_to_pdf(md_path: str, force: bool = False) -> str:
    """Convert markdown file to PDF using md2pdf."""
    pdf_path = os.path.splitext(md_path)[0] + '.pdf'

    if os.path.exists(pdf_path) and not force:
        print(f"  SKIP: {os.path.basename(pdf_path)} already exists")
        return pdf_path

    result = subprocess.run(
        [sys.executable, MD2PDF_SCRIPT, '--input', md_path, '--output', pdf_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None

    print(f"  GEN: {os.path.basename(pdf_path)}")
    return pdf_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate PDFs for markdown reports')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Regenerate PDFs even if they exist')
    args = parser.parse_args()

    print("Generating PDFs using md2pdf skill...")

    # Find all MD files
    md_files = glob.glob(f'{BASE_DIR}/**/*.md', recursive=True)

    for md_path in sorted(md_files):
        md_to_pdf(md_path, force=args.force)

    print("Done.")

if __name__ == '__main__':
    main()