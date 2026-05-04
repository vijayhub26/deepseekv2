# Session Report — April 27, 2026

## Summary

Today's session focused on **hardening the DeepSeek-OCR-2 pipeline** — integrating the plaintext converter, stripping OCR artifacts from output, designing a format-aware evaluation metric, and adding benchmark instrumentation.

---

## 1. Stripping OCR Reference Tags

**Problem:** The OCR output contained DeepSeek-specific layout annotation tags that were leaking into the final plaintext:
```
<|ref|>table<|/ref|><|det|>[[118, 160, 881, 428]]<|/det|>
<|ref|>sub_title<|/ref|><|det|>[[128, 471, 338, 485]]<|/det|>
```

**Fix:** Added a regex to [markdown_converter.py](file:///c:/projects/test3/markdown_converter.py) `preprocess()`:
```python
raw = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>\n?", "", raw)
```

---

## 2. Stripping LaTeX Escaped Parentheses

**Problem:** DeepSeek OCR uses LaTeX math delimiters `\(` and `\)` around numeric values, causing "colour boxes" in the output:
```
3M SJ3519FR Scotchmate Fast HK   6   \(84   \)504
```

**Fix:** Added to [markdown_converter.py](file:///c:/projects/test3/markdown_converter.py) `preprocess()`:
```python
raw = raw.replace("\\(", "").replace("\\)", "")
raw = raw.replace("\\[", "").replace("\\]", "")
```

---

## 3. Integrated Markdown Converter into Main Pipeline

**Problem:** Users had to run `main.py` (produces `.md`) and then manually run `markdown_converter.py` to get clean `.txt`.

**Fix:** Modified [main.py](file:///c:/projects/test3/main.py) to import and call `convert_file()` automatically:

- Pipeline now runs: `PDF -> page images -> OCR (markdown) -> plaintext (.txt)`
- Steps changed from 3/3 to 4/4 (added plaintext conversion step)
- Intermediate `.md` is deleted by default
- Added `--keep-md` flag to retain it

```bash
# Standard — final output is .txt only
python main.py document.pdf -o ./output

# Keep intermediate .md alongside .txt
python main.py document.pdf -o ./output --keep-md
```

---

## 4. CER Evaluation Script (`cer_test.py`)

**Problem:** Ground truth and OCR output have fundamentally different formats:
- **Ground truth**: Multi-line table cells, form-feed page separators (`<<<\f`), irregular whitespace
- **OCR output**: Single-line table rows, `[Page N]` markers, dashed separators, underline headings

A naive CER comparison gave **63% CER** — wildly inflated by formatting differences, not actual content errors.

**Solution:** Rewrote [cer_test.py](file:///c:/projects/test3/cer_test.py) with a **format-aware normalization pipeline** using `jiwer`:

| Normalization Step | What it strips |
|---|---|
| Form feeds, `<<<` | Ground truth page separators |
| `[Page N]`, `PAGE N` | Page markers from both formats |
| Lines of `---`, `===`, `___` | Decorative separator lines |
| Inline `--`, `-----` tokens | Table header dashes |
| `\(`, `\)`, `\[`, `\]` | LaTeX escape sequences |
| Smart quotes | Unicode quote variants |
| `$` signs | Currency symbols (inconsistent between formats) |
| Whitespace collapse | All whitespace -> single space |
| `'` after digits -> `"` | Dimension quote normalization (`1000'` vs `1000"`) |
| Period+space before lowercase | Abbreviation normalization (`Prot. Tp.` -> `Prot.Tp.`) |

**Metrics computed:**

| Metric | Purpose |
|---|---|
| **CER** (Character Error Rate) | Standard character-level accuracy |
| **WER** (Word Error Rate) | Standard word-level accuracy |
| **Numeric Accuracy** | % of numbers correctly extracted (critical for invoices) |
| **Word F1** (order-insensitive) | Content completeness regardless of word ordering |

**Results on batch2:**

| Metric | Value |
|---|---|
| CER | 29.13% (inflated by word-order differences) |
| WER | 37.43% (same — order-sensitive) |
| Numeric Accuracy | **100%** |
| Word F1 | **100%** |

> [!IMPORTANT]
> The CER/WER are penalized by **word ordering** (ground truth has multi-line cells; OCR output has single-line rows). The **Word F1 and Numeric Accuracy are the most meaningful metrics** for this invoice OCR use case.

**Usage:**
```bash
python cer_test.py ground_truth.txt ocr_output.txt --ignore-case
python cer_test.py ground_truth.txt ocr_output.txt --ignore-case --per-page
python cer_test.py ground_truth.txt ocr_output.txt --show-diff
python cer_test.py ground_truth.txt ocr_output.txt --show-normalized
```

---

## 5. Benchmark Timing Instrumentation

**Added to** [main.py](file:///c:/projects/test3/main.py):

- Per-page timing data collected during OCR: PDF name, page number, OCR seconds, output char count, DPI, timestamp
- **Appends to `output/benchmark.csv`** after each run (accumulates over time)
- Prints a **timing summary** at the end of each run

**Sample benchmark.csv:**

| pdf | page | total_pages | ocr_seconds | output_chars | dpi |
|---|---|---|---|---|---|
| batch3-0999.pdf | 1 | 3 | 185.4 | 1819 | 300 |
| batch3-0999.pdf | 2 | 3 | 158.0 | 1506 | 300 |
| batch3-0999.pdf | 3 | 3 | 146.0 | 1252 | 300 |

**Console output:**
```
  Timing Summary (3 pages)
  Avg/page:        163.1s  (2.7 min)
  Min page:        146.0s  (2.4 min)
  Max page:        185.4s  (3.1 min)
  Total:           489.7s  (8.2 min)
```

---

## Files Modified

| File | Changes |
|---|---|
| [markdown_converter.py](file:///c:/projects/test3/markdown_converter.py) | Added tag stripping (`<\|ref\|>`, `\(`, `\)`) in `preprocess()` |
| [main.py](file:///c:/projects/test3/main.py) | Integrated converter, added `--keep-md`, added CSV benchmark logging |
| [cer_test.py](file:///c:/projects/test3/cer_test.py) | Full rewrite with format-aware normalization and 4 metrics |

---

## Per-Page Timing Observations

Based on all runs today, per-page OCR time on the **GTX 1650 (4-bit NF4)** ranges from **~2.4 to ~7.7 min/page**, with an average around **3–5 min/page** depending on content density. Pages with more text generate longer token sequences, which directly increases inference time.
