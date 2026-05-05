"""
convert_to_plaintext.py
-----------------------
Converts MinerU/OCR-style markdown/HTML invoice files to clean plain text.
 
Fixes applied v2:
  - Strips markdown image tags ![alt](path)
  - Splits squashed totals lines (Subtotal/GST/Total/Payment/Balance) into
    separate right-aligned rows
  - <br> tags inside HTML table cells preserved as newlines
  - Nested <table> inside <div> detected and rendered
  - Windows CRLF line endings normalised
 
Upstream limitations (not fixable here):
  - Label+value merged in header cells (e.g. "Effective Date2/27/2026")
    — this happens before the file reaches the converter; the <strong> tag
    and adjacent text node are already concatenated by the OCR pipeline.
  - <br> inside cells that were already plain-text before reaching us
    — once the br is gone from the source it cannot be recovered.
 
Usage:
    python markdown_converter.py <input_file> [output_file]
 
Dependencies:
    pip install beautifulsoup4
"""
 
import sys
import re
from pathlib import Path
from bs4 import BeautifulSoup
 
 
# ---------------------------------------------------------------------------
# Table renderer  (br-aware)
# ---------------------------------------------------------------------------
 
def cell_text(td) -> str:
    """Extract text from a <td>/<th>, converting <br> to newlines."""
    for br in td.find_all("br"):
        br.replace_with("\n")
    return td.get_text(strip=False).strip()
 
 
def render_table(soup_table) -> str:
    rows = []
    for tr in soup_table.find_all("tr"):
        cells = [cell_text(td) for td in tr.find_all(["th", "td"])]
        rows.append(cells)
 
    if not rows:
        return ""
 
    max_cols = max(len(r) for r in rows)
 
    # Pad short rows to max_cols (left-pad so value lands in last column)
    padded = []
    for row in rows:
        if len(row) < max_cols:
            row = [""] * (max_cols - len(row)) + row
        padded.append(row)
 
    # Column widths — account for multi-line cells
    col_widths = [0] * max_cols
    for row in padded:
        for i, cell in enumerate(row):
            for sub in cell.split("\n"):
                col_widths[i] = max(col_widths[i], len(sub))
 
    sep = "  ".join("-" * w for w in col_widths)
 
    lines = []
    for idx, row in enumerate(padded):
        cell_lines = [cell.split("\n") for cell in row]
        max_sub = max(len(cl) for cl in cell_lines)
        cell_lines = [cl + [""] * (max_sub - len(cl)) for cl in cell_lines]
        for sub_idx in range(max_sub):
            line = "  ".join(
                cell_lines[i][sub_idx].ljust(col_widths[i])
                for i in range(max_cols)
            )
            lines.append(line.rstrip())
            
        # Only draw the separator if the first row actually looks like a header
        if idx == 0:
            is_header = True
            for cell in row:
                # If a cell has a currency symbol or is just a number, it's data, not a header
                if re.search(r'[\$£€]', cell) or re.match(r'^\s*[\d.,]+\s*$', cell):
                    is_header = False
                    break
            if is_header:
                lines.append(sep)
            

 
    return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# Block renderer
# ---------------------------------------------------------------------------
 
def heading_level(tag_name: str) -> int:
    m = re.match(r"h([1-6])", tag_name.lower())
    return int(m.group(1)) if m else 0
 
 
def render_block(tag) -> str:
    name = tag.name.lower() if tag.name else ""
 
    if name == "table":
        return render_table(tag) + "\n"
 
    if name == "hr":
        return "-" * 80 + "\n"
 
    # div containing a table — render the table
    if name == "div":
        inner = tag.find("table")
        if inner:
            return render_table(inner) + "\n"
 
    level = heading_level(name)
    for br in tag.find_all("br"):
        br.replace_with("\n")
    text = tag.get_text(separator=" ", strip=True)
    text = re.sub(r" {2,}", " ", text).strip()
 
    if not text:
        return ""
 
    if level == 1:
        return f"\n{text.upper()}\n" + "=" * len(text) + "\n"
    if level == 2:
        return f"\n{text}\n" + "-" * len(text) + "\n"
    if level in (3, 4):
        return f"\n{text}\n"
 
    return text + "\n"
 
 
# ---------------------------------------------------------------------------
# Totals line splitter
# ---------------------------------------------------------------------------
 
# Matches: "Subtotal : ₹0.00" or "GST : 0% ₹0.00" etc.
TOTALS_TOKEN_RE = re.compile(
    r"((?:Subtotal|GST\s*[:\d%]+|Total|Payment|Balance)\s*[:\s][\s\S]*?)(?=(?:Subtotal|GST|Total|Payment|Balance)|$)"
)
 
def split_totals_line(line: str) -> str:
    """Split a squashed totals line into separate right-aligned rows."""
    tokens = re.findall(
        r"((?:Subtotal|GST)[^₹\d]*[₹\d][^\s]*|(?:Total|Payment|Balance)\s*:\s*[₹\d]\S*)",
        line
    )
    if len(tokens) < 2:
        return line
    width = 55
    return "\n".join(t.strip().rjust(width) for t in tokens)
 
 
# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
 
def preprocess(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>\n?", "", raw) # strip OCR reference tags
    raw = re.sub(r"!\[.*?\]\(.*?\)", "", raw)        # strip image tags
    raw = re.sub(r'\s*style="[^"]*"', "", raw)        # strip style attrs
    
    # DeepSeek-OCR-2 encodes $ signs as LaTeX inline math delimiters 
    raw = raw.replace("\\(", "$").replace("\\)", "$")
    raw = raw.replace("\\[", "").replace("\\]", "")  # strip LaTeX escaped brackets
    
    raw = re.sub(r"[─━]{4,}", "----", raw)            # normalise separators
    return raw
 
 
# ---------------------------------------------------------------------------
# Page splitter / chunk converter
# ---------------------------------------------------------------------------
 
PAGE_RE = re.compile(r"\[Page\s+(\d+)\]")
 
 
def split_pages(raw_text: str):
    parts = PAGE_RE.split(raw_text)
    pages = []
    i = 1
    while i < len(parts) - 1:
        pages.append((parts[i].strip(), parts[i + 1]))
        i += 2
    return pages
 
 
def convert_chunk(chunk: str) -> str:
    lines_out = []
    sections = re.split(r"\n[-─]{4,}\n", chunk)
 
    for section in sections:
        section = section.strip()
        if not section:
            continue
 
        soup = BeautifulSoup(f"<div>{section}</div>", "html.parser")
        root = soup.find("div")
 
        for child in root.children:
            if hasattr(child, "name") and child.name:
                rendered = render_block(child)
                if rendered.strip():
                    lines_out.append(rendered)
            else:
                for line in str(child).splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    # Strip image tags
                    s = re.sub(r"!\[.*?\]\(.*?\)", "", s).strip()
                    if not s:
                        continue
                    # Bold → uppercase
                    s = re.sub(r"\*\*(.*?)\*\*", lambda m: m.group(1).upper(), s)
                    # Markdown headings
                    m = re.match(r"^(#{1,3})\s+(.*)", s)
                    if m:
                        lvl = len(m.group(1))
                        val = m.group(2).strip()
                        if lvl == 1:
                            lines_out.append(f"\n{val.upper()}\n" + "=" * len(val))
                        elif lvl == 2:
                            lines_out.append(f"\n{val}\n" + "-" * len(val))
                        else:
                            lines_out.append(f"\n{val}")
                    # Squashed totals line
                    elif re.search(r"(Subtotal|Total|Payment|Balance).*:", s):
                        lines_out.append(split_totals_line(s))
                    else:
                        lines_out.append(s)
 
    return "\n".join(lines_out)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def convert_file(input_path: Path, output_path: Path):
    raw = preprocess(input_path.read_text(encoding="utf-8"))
    pages = split_pages(raw) or [("1", raw)]
 
    parts = []
    for page_num, chunk in pages:
        body = convert_chunk(chunk)
        parts.append(f"[Page {page_num}]\n\n{body.strip()}")
 
    output_path.write_text(("\n\n" + "-" * 80 + "\n\n").join(parts) + "\n",
                            encoding="utf-8")
    print(f"Converted: {input_path.name}  ->  {output_path.name}")
    print(f"Pages processed: {len(pages)}")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_plaintext.py <input_file> [output_file]")
        sys.exit(1)
    inp = Path(sys.argv[1])
    if not inp.exists():
        print(f"Error: file not found — {inp}")
        sys.exit(1)
    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else \
          inp.with_stem(inp.stem + "_plain").with_suffix(".txt")
    convert_file(inp, out)