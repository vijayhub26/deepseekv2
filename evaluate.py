"""
evaluate.py
-----------
Layout-agnostic OCR evaluation — compares OCR output against ground truth
by stripping all layout/whitespace differences and measuring only
character-level and word-level accuracy.

Metrics:
  - CER  (Character Error Rate)   = (S + D + I) / N_ref_chars
  - WER  (Word Error Rate)        = (S + D + I) / N_ref_words
  - Character Accuracy            = 1 - CER
  - Word Accuracy                 = 1 - WER
  - Error breakdown (substitutions, deletions, insertions, hits)
    at both character and word level

Usage:
    python evaluate.py <ocr_output> <ground_truth>
    python evaluate.py <ocr_output> <ground_truth> --per-page
    python evaluate.py <ocr_output> <ground_truth> --csv results.csv
"""

import argparse
import csv
import collections
import re
import sys
from pathlib import Path
from typing import List, Tuple


def safe_print(text: str) -> None:
    """Print text safely on Windows terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(
            sys.stdout.encoding or 'ascii', errors='replace'))


# ---------------------------------------------------------------------------
# Text normalisation — strip layout, collapse whitespace
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Normalise text for layout-agnostic comparison.

    Strips:
      - Page markers ([Page N], <<<)
      - Separator lines (----, ====, table borders)
      - All extra whitespace (tabs, multiple spaces, blank lines)
      - Leading/trailing whitespace per line
    Preserves:
      - Actual textual content
      - Punctuation, numbers, case
    """
    # Remove common page/section markers
    text = re.sub(r"\[Page\s*\d+\]", " ", text)
    text = re.sub(r"<<<", " ", text)

    # Remove separator lines (dashes, equals, table borders)
    text = re.sub(r"^[\s\-=─━|+:]+$", " ", text, flags=re.MULTILINE)

    # Remove markdown image tags
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove OCR reference/detection tags
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text)

    # Collapse all whitespace (spaces, tabs, newlines) into single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def to_chars(text: str) -> List[str]:
    """Convert normalised text to a list of characters."""
    return list(text)


def to_words(text: str) -> List[str]:
    """Convert normalised text to a list of words."""
    return text.split()


# ---------------------------------------------------------------------------
# Edit distance with backtrace — returns (S, D, I, H) counts
# ---------------------------------------------------------------------------

def edit_distance_breakdown(
    ref: List[str], hyp: List[str]
) -> Tuple[int, int, int, int]:
    """Compute edit distance and return (substitutions, deletions, insertions, hits).

    Uses standard dynamic programming with backtrace.

    Args:
        ref: Reference (ground truth) token sequence.
        hyp: Hypothesis (OCR output) token sequence.

    Returns:
        (substitutions, deletions, insertions, hits)
        - substitution: ref token replaced by different hyp token
        - deletion:     ref token missing from hyp
        - insertion:    extra hyp token not in ref
        - hit:          ref token correctly matched in hyp
    """
    n = len(ref)
    m = len(hyp)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i  # all deletions
    for j in range(m + 1):
        dp[0][j] = j  # all insertions

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # match
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                )

    # Backtrace to get S, D, I, H
    subs = 0
    dels = 0
    ins = 0
    hits = 0
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            # Fallback (shouldn't happen, but safety)
            if i > 0 and j > 0:
                subs += 1
                i -= 1
                j -= 1
            elif i > 0:
                dels += 1
                i -= 1
            else:
                ins += 1
                j -= 1

    return subs, dels, ins, hits


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(ref_text: str, hyp_text: str) -> dict:
    """Compute all evaluation metrics between reference and hypothesis text.

    Both texts are normalised (layout/whitespace stripped) before comparison.

    Returns dict with:
        ref_chars, hyp_chars, ref_words, hyp_words,
        char_{subs, dels, ins, hits, errors}, cer, char_accuracy,
        word_{subs, dels, ins, hits, errors}, wer, word_accuracy
    """
    ref_norm = normalise(ref_text)
    hyp_norm = normalise(hyp_text)

    # For character-level: strip ALL whitespace so only actual text
    # characters are compared (layout/spacing fully ignored)
    ref_chars_text = re.sub(r"\s+", "", ref_norm)
    hyp_chars_text = re.sub(r"\s+", "", hyp_norm)

    # Sort both sequences so reading order is eliminated —
    # only actual OCR errors (wrong/missing/extra chars/words) are measured
    ref_chars = sorted(to_chars(ref_chars_text))
    hyp_chars = sorted(to_chars(hyp_chars_text))
    ref_words = sorted(to_words(ref_norm))
    hyp_words = sorted(to_words(hyp_norm))

    # Character-level (whitespace-free, order-agnostic)
    c_sub, c_del, c_ins, c_hit = edit_distance_breakdown(ref_chars, hyp_chars)
    c_errors = c_sub + c_del + c_ins
    n_ref_c = len(ref_chars)
    cer = c_errors / n_ref_c if n_ref_c > 0 else 0.0

    # Word-level (order-agnostic)
    w_sub, w_del, w_ins, w_hit = edit_distance_breakdown(ref_words, hyp_words)
    w_errors = w_sub + w_del + w_ins
    n_ref_w = len(ref_words)
    wer = w_errors / n_ref_w if n_ref_w > 0 else 0.0

    # Exact word differences
    ref_word_counter = collections.Counter(ref_words)
    hyp_word_counter = collections.Counter(hyp_words)
    missing_words = list((ref_word_counter - hyp_word_counter).elements())
    extra_words = list((hyp_word_counter - ref_word_counter).elements())

    return {
        # Counts
        "ref_chars": n_ref_c,
        "hyp_chars": len(hyp_chars),
        "ref_words": n_ref_w,
        "hyp_words": len(hyp_words),
        # Character breakdown
        "char_hits": c_hit,
        "char_subs": c_sub,
        "char_dels": c_del,
        "char_ins": c_ins,
        "char_errors": c_errors,
        "cer": cer,
        "char_accuracy": 1.0 - cer,
        # Word breakdown
        "word_hits": w_hit,
        "word_subs": w_sub,
        "word_dels": w_del,
        "word_ins": w_ins,
        "word_errors": w_errors,
        "wer": wer,
        "word_accuracy": 1.0 - wer,
        "missing_words": missing_words,
        "extra_words": extra_words
    }


# ---------------------------------------------------------------------------
# Page splitting (optional per-page evaluation)
# ---------------------------------------------------------------------------

def split_pages_ocr(text: str) -> List[str]:
    """Split OCR output by [Page N] markers."""
    parts = re.split(r"\[Page\s*\d+\]", text)
    return [p.strip() for p in parts if p.strip()]


def split_pages_gt(text: str) -> List[str]:
    """Split ground truth by <<< markers."""
    parts = text.split("<<<")
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_report(metrics: dict, label: str = "Overall") -> str:
    """Format metrics into a human-readable report."""
    lines = []
    lines.append(f"\n{'=' * 64}")
    lines.append(f"  {label}")
    lines.append(f"{'=' * 64}")

    lines.append(f"\n  Reference: {metrics['ref_chars']:,} chars, {metrics['ref_words']:,} words")
    lines.append(f"  Hypothesis: {metrics['hyp_chars']:,} chars, {metrics['hyp_words']:,} words")

    lines.append(f"\n  {'-' * 60}")
    lines.append(f"  CHARACTER-LEVEL BREAKDOWN")
    lines.append(f"  {'-' * 60}")
    lines.append(f"  {'Hits (correct):':<22} {metrics['char_hits']:>8,}")
    lines.append(f"  {'Substitutions:':<22} {metrics['char_subs']:>8,}")
    lines.append(f"  {'Deletions:':<22} {metrics['char_dels']:>8,}")
    lines.append(f"  {'Insertions:':<22} {metrics['char_ins']:>8,}")
    lines.append(f"  {'Total errors:':<22} {metrics['char_errors']:>8,}")
    lines.append(f"")
    lines.append(f"  CER:                {metrics['cer']:>8.2%}")
    lines.append(f"  Char Accuracy:      {metrics['char_accuracy']:>8.2%}")

    lines.append(f"\n  {'-' * 60}")
    lines.append(f"  WORD-LEVEL BREAKDOWN")
    lines.append(f"  {'-' * 60}")
    lines.append(f"  {'Hits (correct):':<22} {metrics['word_hits']:>8,}")
    lines.append(f"  {'Substitutions:':<22} {metrics['word_subs']:>8,}")
    lines.append(f"  {'Deletions:':<22} {metrics['word_dels']:>8,}")
    lines.append(f"  {'Insertions:':<22} {metrics['word_ins']:>8,}")
    lines.append(f"  {'Total errors:':<22} {metrics['word_errors']:>8,}")
    lines.append(f"")
    lines.append(f"  WER:                {metrics['wer']:>8.2%}")
    lines.append(f"  Word Accuracy:      {metrics['word_accuracy']:>8.2%}")

    if metrics.get("missing_words") or metrics.get("extra_words"):
        lines.append(f"\n  {'-' * 60}")
        lines.append(f"  ERROR DETAILS (Word-Level)")
        lines.append(f"  {'-' * 60}")
        if metrics.get("missing_words"):
            lines.append(f"  Missing from OCR (Deletions): {metrics['missing_words']}")
        if metrics.get("extra_words"):
            lines.append(f"  Extra in OCR (Insertions):     {metrics['extra_words']}")

    lines.append(f"\n{'=' * 64}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "label",
    "ref_chars", "hyp_chars", "ref_words", "hyp_words",
    "char_hits", "char_subs", "char_dels", "char_ins", "char_errors",
    "cer", "char_accuracy",
    "word_hits", "word_subs", "word_dels", "word_ins", "word_errors",
    "wer", "word_accuracy",
]


def write_csv(rows: List[dict], csv_path: Path) -> None:
    """Write evaluation results to CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layout-agnostic OCR evaluation: CER, WER, and error breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py output/batch1.txt ground_truth/batch1_gt.txt
  python evaluate.py output/batch1.txt ground_truth/batch1_gt.txt --per-page
  python evaluate.py output/batch1.txt ground_truth/batch1_gt.txt --csv eval.csv
        """,
    )
    parser.add_argument(
        "ocr_output",
        help="Path to the OCR output text file.",
    )
    parser.add_argument(
        "ground_truth",
        help="Path to the ground truth text file.",
    )
    parser.add_argument(
        "--per-page",
        action="store_true",
        help="Also report per-page metrics (requires page markers in both files).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Save results to a CSV file.",
    )

    args = parser.parse_args()

    ocr_path = Path(args.ocr_output)
    gt_path = Path(args.ground_truth)

    if not ocr_path.exists():
        print(f"Error: OCR output file not found — {ocr_path}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"Error: Ground truth file not found — {gt_path}")
        sys.exit(1)

    ocr_text = ocr_path.read_text(encoding="utf-8")
    gt_text = gt_path.read_text(encoding="utf-8")

    csv_rows = []

    # Overall metrics
    overall = compute_metrics(gt_text, ocr_text)
    safe_print(format_report(overall, label=f"Overall -- {ocr_path.name} vs {gt_path.name}"))
    csv_rows.append({"label": "overall", **overall})

    # Per-page metrics
    if args.per_page:
        ocr_pages = split_pages_ocr(ocr_text)
        gt_pages = split_pages_gt(gt_text)

        if len(ocr_pages) != len(gt_pages):
            print(f"\n  ⚠  Page count mismatch: OCR has {len(ocr_pages)} pages, "
                  f"GT has {len(gt_pages)} pages.")
            print(f"     Evaluating min({len(ocr_pages)}, {len(gt_pages)}) pages.\n")

        n_pages = min(len(ocr_pages), len(gt_pages))
        for i in range(n_pages):
            page_metrics = compute_metrics(gt_pages[i], ocr_pages[i])
            safe_print(format_report(page_metrics, label=f"Page {i + 1}"))
            csv_rows.append({"label": f"page_{i + 1}", **page_metrics})

    # CSV output
    if args.csv:
        write_csv(csv_rows, Path(args.csv))


if __name__ == "__main__":
    main()
