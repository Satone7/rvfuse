#!/usr/bin/env python3
"""
md2pdf — Simplified Markdown to PDF for RVFuse project reports.

Based on any2pdf skill but simplified:
- No cover page (直接开始正文)
- TOC inline (不单独分页)
- No forced page breaks (内容自然流动)
- GitHub Light theme by default

Usage:
  python md2pdf.py --input report.md --output report.pdf

Dependencies:
  pip install reportlab
"""

import re, os, sys, argparse
from datetime import date
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import platform as _platform
_PLAT = _platform.system()

# ═══════════════════════════════════════════════════════════════════════
# FONTS — cross-platform
# ═══════════════════════════════════════════════════════════════════════
def _find_font(candidates):
    for c in candidates:
        path = c[0] if isinstance(c, tuple) else c
        if os.path.exists(path):
            return c
    return None

_FONT_CANDIDATES = {
    "Sans": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ],
    "SansBold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ],
    "CJK": [
        ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
        "C:/Windows/Fonts/simsun.ttc",
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ],
    "Mono": [
        ("/System/Library/Fonts/Menlo.ttc", 0),
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    ],
    "MonoBold": [
        ("/System/Library/Fonts/Menlo.ttc", 1),
        "C:/Windows/Fonts/consolab.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ],
}

def register_fonts():
    missing = []
    for name, candidates in _FONT_CANDIDATES.items():
        spec = _find_font(candidates)
        if spec is None:
            missing.append(name)
            continue
        try:
            if isinstance(spec, tuple):
                pdfmetrics.registerFont(TTFont(name, spec[0], subfontIndex=spec[1]))
            else:
                pdfmetrics.registerFont(TTFont(name, spec))
        except Exception as e:
            missing.append(name)
    if missing:
        print(f"Warning: Missing fonts: {', '.join(missing)}", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════
# THEME — GitHub Light (simplified)
# ═══════════════════════════════════════════════════════════════════════
THEME = {
    "canvas": HexColor("#FFFFFF"),
    "canvas_sec": HexColor("#F6F8FA"),
    "ink": HexColor("#1F2328"),
    "ink_faded": HexColor("#656D76"),
    "accent": HexColor("#0969DA"),
    "border": HexColor("#D0D7DE"),
}

# ═══════════════════════════════════════════════════════════════════════
# CJK SUPPORT
# ═══════════════════════════════════════════════════════════════════════
_CJK_RANGES = [
    (0x4E00,0x9FFF),(0x3400,0x4DBF),(0xF900,0xFAFF),(0x3000,0x303F),
    (0xFF00,0xFFEF),(0x2E80,0x2EFF),(0x2F00,0x2FDF),(0xFE30,0xFE4F),
]

def _is_cjk(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)

def _font_wrap(text):
    out, buf, in_cjk = [], [], False
    for ch in text:
        c = _is_cjk(ch)
        if c != in_cjk and buf:
            seg = ''.join(buf)
            out.append(f"<font name='CJK'>{seg}</font>" if in_cjk else seg)
            buf = []
        buf.append(ch); in_cjk = c
    if buf:
        seg = ''.join(buf)
        out.append(f"<font name='CJK'>{seg}</font>" if in_cjk else seg)
    return ''.join(out)

# ═══════════════════════════════════════════════════════════════════════
# MARKDOWN ESCAPE + INLINE
# ═══════════════════════════════════════════════════════════════════════
def esc(text):
    return text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def esc_code(text):
    out = []
    for line in text.split('\n'):
        e = esc(line)
        stripped = e.lstrip(' ')
        indent = len(e) - len(stripped)
        out.append('&nbsp;' * indent + stripped)
    return '<br/>'.join(out)

def md_inline(text, accent_hex="#0969DA"):
    text = esc(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`',
        rf"<font name='Mono' size='8' color='{accent_hex}'>\1</font>", text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'<u>\1</u>', text)
    return _font_wrap(text)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM FLOWABLE — Bookmark anchor (no visual)
# ═══════════════════════════════════════════════════════════════════════
_anchor_counter = [0]

class AnchorMark(Flowable):
    width = height = 0
    def __init__(self, title, level=0):
        Flowable.__init__(self)
        self.title = title; self.level = level
        _anchor_counter[0] += 1
        self.key = f"anchor_{_anchor_counter[0]}"
    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level)

# ═══════════════════════════════════════════════════════════════════════
# PDF BUILDER — Simplified
# ═══════════════════════════════════════════════════════════════════════
class SimplePDFBuilder:
    def __init__(self, config):
        self.cfg = config
        self.page_w, self.page_h = config["page_size"]
        self.lm, self.rm, self.tm, self.bm = 20*mm, 20*mm, 20*mm, 20*mm
        self.body_w = self.page_w - self.lm - self.rm
        self.body_h = self.page_h - self.tm - self.bm
        self.accent_hex = "#0969DA"
        self.ST = self._build_styles()

    def _build_styles(self):
        s = {}
        # Headings — left aligned, no decoration
        s['h1'] = ParagraphStyle('H1', fontName="SansBold", fontSize=22,
            leading=28, textColor=THEME["ink"], alignment=TA_LEFT,
            spaceBefore=12, spaceAfter=8)
        s['h2'] = ParagraphStyle('H2', fontName="SansBold", fontSize=16,
            leading=22, textColor=THEME["ink"], alignment=TA_LEFT,
            spaceBefore=10, spaceAfter=6)
        s['h3'] = ParagraphStyle('H3', fontName="SansBold", fontSize=12,
            leading=16, textColor=THEME["accent"], alignment=TA_LEFT,
            spaceBefore=8, spaceAfter=4)
        # Body
        s['body'] = ParagraphStyle('Body', fontName="Sans", fontSize=10,
            leading=15, textColor=THEME["ink"], alignment=TA_JUSTIFY,
            spaceBefore=2, spaceAfter=4, wordWrap='CJK')
        s['bullet'] = ParagraphStyle('Bul', fontName="Sans", fontSize=10,
            leading=15, textColor=THEME["ink"], alignment=TA_LEFT,
            spaceBefore=1, spaceAfter=1, leftIndent=18, bulletIndent=6,
            wordWrap='CJK')
        # Code block
        s['code'] = ParagraphStyle('Code', fontName="Mono", fontSize=8,
            leading=11, textColor=THEME["ink"], alignment=TA_LEFT,
            spaceBefore=4, spaceAfter=4, leftIndent=8, rightIndent=8,
            backColor=THEME["canvas_sec"], borderPadding=6)
        # TOC
        s['toc1'] = ParagraphStyle('T1', fontName="SansBold", fontSize=12,
            leading=18, textColor=THEME["ink"], spaceBefore=4, spaceAfter=2)
        s['toc2'] = ParagraphStyle('T2', fontName="Sans", fontSize=10,
            leading=14, textColor=THEME["ink_faded"], leftIndent=14,
            spaceBefore=1, spaceAfter=1)
        # Table
        s['th'] = ParagraphStyle('TH', fontName="SansBold", fontSize=9,
            leading=12, textColor=white, alignment=TA_CENTER)
        s['tc'] = ParagraphStyle('TC', fontName="Sans", fontSize=8,
            leading=11, textColor=THEME["ink"], alignment=TA_LEFT)
        return s

    def _normal_page(self, c, doc):
        # Background
        c.setFillColor(THEME["canvas"])
        c.rect(0, 0, self.page_w, self.page_h, fill=1, stroke=0)

        # Left accent stripe (GitHub style)
        c.setFillColor(THEME["accent"])
        c.rect(0, 0, 5*mm, self.page_h, fill=1, stroke=0)

        # Footer line above page number
        c.setStrokeColor(THEME["border"])
        c.setLineWidth(0.5)
        c.line(self.lm, self.bm - 6*mm, self.page_w - self.rm, self.bm - 6*mm)

        # Footer — page number
        pg = c.getPageNumber()
        c.setFillColor(THEME["ink_faded"])
        c.setFont("Sans", 8)
        c.drawCentredString(self.page_w/2, self.bm - 10*mm, str(pg))

    def parse_table(self, lines):
        rows = []
        for l in lines:
            l = l.strip().strip('|')
            rows.append([c.strip() for c in l.split('|')])
        if len(rows) < 2: return None
        header = rows[0]
        data = [r for r in rows[1:] if not all(set(c.strip()) <= set('-: ') for c in r)]
        if not data: return None
        nc = len(header)
        ST = self.ST
        td = [[Paragraph(md_inline(h, self.accent_hex), ST['th']) for h in header]]
        for r in data:
            while len(r) < nc: r.append("")
            td.append([Paragraph(md_inline(c, self.accent_hex), ST['tc']) for c in r[:nc]])
        # Smart column widths
        avail = self.body_w - 4*mm
        max_lens = [max((len(r[ci]) if ci < len(r) else 0) for r in [header]+data) for ci in range(nc)]
        max_lens = [max(m, 2) for m in max_lens]
        total = sum(max_lens)
        cw = [avail * m / total for m in max_lens]
        min_w = 18*mm
        for ci in range(nc):
            if cw[ci] < min_w:
                deficit = min_w - cw[ci]; cw[ci] = min_w
                widest = sorted(range(nc), key=lambda x: -cw[x])
                for oi in widest:
                    if oi != ci: cw[oi] -= deficit; break
        t = Table(td, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0), THEME["accent"]),
            ('TEXTCOLOR',(0,0),(-1,0), white),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [white, THEME["canvas_sec"]]),
            ('GRID',(0,0),(-1,-1), 0.5, THEME["border"]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
        ]))
        return t

    def parse_md(self, md):
        story, toc = [], []
        lines = md.split('\n')
        i, in_code, code_buf = 0, False, []
        ST = self.ST; ah = self.accent_hex

        while i < len(lines):
            line = lines[i]; stripped = line.strip()
            # Code blocks
            if stripped.startswith('```'):
                if in_code:
                    ct = '\n'.join(code_buf)
                    if ct.strip():
                        para = Paragraph(_font_wrap(esc_code(ct)), ST['code'])
                        story.append(para)
                    code_buf = []; in_code = False
                else:
                    in_code = True; code_buf = []
                i += 1; continue
            if in_code:
                code_buf.append(line)
                i += 1; continue

            # Skip empty / metadata
            if stripped in ('---', '') or stripped.startswith(('title:', 'subtitle:', 'author:', 'date:')):
                i += 1; continue

            # H1 — with bookmark
            if stripped.startswith('# ') and not stripped.startswith('## '):
                title = stripped[2:].strip()
                cm = AnchorMark(title, level=0)
                story.append(cm)
                story.append(Paragraph(md_inline(title, ah), ST['h1']))
                toc.append(('h1', title, cm.key))
                i += 1; continue

            # H2 — with bookmark
            if stripped.startswith('## '):
                title = stripped[3:].strip()
                cm = AnchorMark(title, level=1)
                story.append(cm)
                story.append(Paragraph(md_inline(title, ah), ST['h2']))
                toc.append(('h2', title, cm.key))
                i += 1; continue

            # H3
            if stripped.startswith('### '):
                story.append(Paragraph(md_inline(stripped[4:].strip(), ah), ST['h3']))
                i += 1; continue

            # H4+ — render as bold body text (no special style)
            if re.match(r'^#{4,}\s', stripped):
                title = re.sub(r'^#{4,}\s+', '', stripped)
                story.append(Paragraph(f"<b>{md_inline(title, ah)}</b>", ST['h3']))
                i += 1; continue

            # Tables
            if stripped.startswith('|'):
                tl = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    tl.append(lines[i]); i += 1
                t = self.parse_table(tl)
                if t:
                    story.append(Spacer(1, 2*mm))
                    story.append(t)
                    story.append(Spacer(1, 2*mm))
                continue

            # Bullets
            if stripped.startswith('- ') or stripped.startswith('* '):
                story.append(Paragraph(f"\u2022  {md_inline(stripped[2:].strip(), ah)}", ST['bullet']))
                i += 1; continue

            # Numbered list
            m = re.match(r'^(\d+)\.\s+(.+)', stripped)
            if m:
                story.append(Paragraph(f"{m.group(1)}.  {md_inline(m.group(2), ah)}", ST['bullet']))
                i += 1; continue

            # Blockquote
            if stripped.startswith('> '):
                bq_style = ParagraphStyle('BQ', parent=ST['body'],
                    leftIndent=14, textColor=THEME["ink_faded"])
                story.append(Paragraph(md_inline(stripped[2:].strip(), ah), bq_style))
                i += 1; continue

            # Paragraph — join consecutive lines
            plines = []
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith('#') or l.startswith('```') or l.startswith('|') or \
                   l.startswith('- ') or l.startswith('* ') or l.startswith('> ') or re.match(r'^\d+\.\s', l):
                    break
                plines.append(l); i += 1
            if plines:
                merged = plines[0]
                for pl in plines[1:]:
                    if merged and pl and _is_cjk(merged[-1]) and _is_cjk(pl[0]):
                        merged += pl
                    else:
                        merged += ' ' + pl
                story.append(Paragraph(md_inline(merged, ah), ST['body']))
            continue

        return story, toc

    def build_toc_inline(self, toc):
        """Build TOC as inline content (no separate page)."""
        s = []
        s.append(Paragraph(md_inline("目录", self.accent_hex), self.ST['h1']))
        s.append(Spacer(1, 6*mm))
        ink = THEME["ink"]
        ink_hex = f"#{int(ink.red*255):02x}{int(ink.green*255):02x}{int(ink.blue*255):02x}" if hasattr(ink, 'red') else "#1F2328"
        for etype, title, key in toc:
            style = self.ST['toc1'] if etype == 'h1' else self.ST['toc2']
            linked = f"<a href=\"#{key}\" color=\"{ink_hex}\">{_font_wrap(esc(title))}</a>"
            s.append(Paragraph(linked, style))
        s.append(Spacer(1, 10*mm))
        return s

    def build(self, md_text, output_path):
        register_fonts()
        print("Parsing markdown...")
        story_content, toc = self.parse_md(md_text)
        print(f"  {len(story_content)} elements, {len(toc)} TOC entries")

        body_frame = Frame(self.lm, self.bm, self.body_w, self.body_h, id='body')

        doc = BaseDocTemplate(output_path, pagesize=(self.page_w, self.page_h),
                              leftMargin=self.lm, rightMargin=self.rm,
                              topMargin=self.tm, bottomMargin=self.bm,
                              title=self.cfg.get("title", ""))

        templates = [
            PageTemplate(id='normal', frames=[body_frame], onPage=self._normal_page),
        ]
        doc.addPageTemplates(templates)

        # Build story: TOC inline + content
        story = []
        if toc:
            story.extend(self.build_toc_inline(toc))
        story.extend(story_content)

        print("Building PDF...")
        doc.build(story)
        size = os.path.getsize(output_path)
        print(f"Done! {output_path} ({size/1024:.1f} KB)")

# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="md2pdf — Simplified Markdown to PDF for RVFuse")
    parser.add_argument("--input", "-i", required=True, help="Input markdown file")
    parser.add_argument("--output", "-o", default="output.pdf", help="Output PDF path")
    parser.add_argument("--page-size", default="A4", choices=["A4","Letter"])
    args = parser.parse_args()

    with open(args.input, encoding='utf-8') as f:
        md_text = f.read()

    # Extract title from first H1
    m = re.search(r'^# (.+)$', md_text, re.MULTILINE)
    title = m.group(1).strip() if m else "Document"

    config = {
        "title": title,
        "page_size": A4 if args.page_size == "A4" else LETTER,
    }

    builder = SimplePDFBuilder(config)
    builder.build(md_text, args.output)

if __name__ == "__main__":
    main()