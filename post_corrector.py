"""
post_corrector.py
-----------------
Dictionary-based post-correction for common OCR misrecognitions.

The corrections are loaded from a JSON file (default: corrections.json)
so users can add new entries without touching code.

Features:
  - Word-level corrections (case-preserving)
  - Phrase/substring corrections
  - Regex pattern corrections
  - Detailed diff report showing every change made

Usage (standalone):
    python post_corrector.py input.txt [output.txt]
    python post_corrector.py input.txt output.txt --diff   # show detailed changes

Usage (as module):
    from post_corrector import PostCorrector
    pc = PostCorrector()                       # loads corrections.json
    fixed_text, changes = pc.apply(ocr_text)
"""

import json
import re
import sys
from pathlib import Path

# Default corrections file lives next to this script
DEFAULT_CORRECTIONS_FILE = Path(__file__).parent / "corrections.json"


def safe_print(text):
    """Print text safely on Windows terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(
            sys.stdout.encoding or 'ascii', errors='replace'))


class PostCorrector:
    """Apply dictionary-based corrections to OCR output text."""

    def __init__(self, corrections_path: Path = None):
        self.corrections_path = corrections_path or DEFAULT_CORRECTIONS_FILE
        self.word_corrections: dict = {}     # exact word replacements
        self.phrase_corrections: dict = {}   # multi-word / substring replacements
        self.regex_corrections: list = []    # regex pattern replacements
        self._load()

    def _load(self):
        """Load corrections from JSON file."""
        if not self.corrections_path.exists():
            return

        with open(self.corrections_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.word_corrections = {k: v for k, v in data.get("words", {}).items()
                                  if not k.startswith("_")}
        self.phrase_corrections = {k: v for k, v in data.get("phrases", {}).items()
                                    if not k.startswith("_")}
        self.regex_corrections = data.get("regex", [])

    def apply(self, text: str) -> tuple:
        """Apply all corrections to the given text.
        
        Returns: (corrected_text, list_of_changes)
        Each change is a dict: {type, pattern, original, replacement, line}
        """
        changes = []

        # 1. Phrase corrections (substring replacements)
        for wrong, right in self.phrase_corrections.items():
            new_text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
            if new_text != text:
                # Count how many replacements
                count = len(re.findall(re.escape(wrong), text, flags=re.IGNORECASE))
                changes.append({
                    "type": "phrase",
                    "pattern": wrong,
                    "replacement": right,
                    "count": count
                })
                text = new_text

        # 2. Word corrections (whole-word, case-preserving)
        for wrong, right in self.word_corrections.items():
            new_text = self._replace_word(text, wrong, right)
            if new_text != text:
                count = len(re.findall(r'\b' + re.escape(wrong) + r'\b', text, re.IGNORECASE))
                changes.append({
                    "type": "word",
                    "pattern": wrong,
                    "replacement": right,
                    "count": count
                })
                text = new_text

        # 3. Regex corrections
        for entry in self.regex_corrections:
            pattern = entry.get("pattern", "")
            replacement = entry.get("replacement", "")
            comment = entry.get("_comment", pattern)
            flags = re.IGNORECASE if entry.get("ignore_case", True) else 0
            if pattern:
                new_text = re.sub(pattern, replacement, text, flags=flags)
                if new_text != text:
                    count = len(re.findall(pattern, text, flags=flags))
                    changes.append({
                        "type": "regex",
                        "pattern": comment,
                        "replacement": repr(replacement) if replacement else "(removed)",
                        "count": count
                    })
                    text = new_text

        return text, changes

    @staticmethod
    def _replace_word(text: str, wrong: str, right: str) -> str:
        """Replace whole-word occurrences, preserving original case pattern."""
        def _match_case(original: str, replacement: str) -> str:
            if original.isupper():
                return replacement.upper()
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)

        def replacer(m):
            return _match_case(m.group(0), right)

        return pattern.sub(replacer, text)

    def stats(self) -> dict:
        """Return count of loaded corrections."""
        return {
            "words": len(self.word_corrections),
            "phrases": len(self.phrase_corrections),
            "regex": len(self.regex_corrections),
            "total": (len(self.word_corrections)
                      + len(self.phrase_corrections)
                      + len(self.regex_corrections)),
        }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python post_corrector.py <input.txt> [output.txt] [--diff]")
        sys.exit(1)

    show_diff = "--diff" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    inp = Path(args[0])
    if not inp.exists():
        print(f"Error: file not found - {inp}")
        sys.exit(1)

    pc = PostCorrector()
    s = pc.stats()
    print(f"Loaded {s['total']} corrections "
          f"({s['words']} words, {s['phrases']} phrases, {s['regex']} regex)")

    text = inp.read_text(encoding="utf-8")
    fixed, changes = pc.apply(text)

    out = Path(args[1]) if len(args) >= 2 else inp
    out.write_text(fixed, encoding="utf-8")

    if changes:
        print(f"\n  Applied {len(changes)} correction(s) -> {out.name}")
        if show_diff or True:  # always show for now
            print(f"  {'~' * 55}")
            for ch in changes:
                tag = ch['type'].upper()[:5]
                safe_print(f"  [{tag}] {ch['pattern']}")
                safe_print(f"         -> {ch['replacement']}  (x{ch['count']})")
            print(f"  {'~' * 55}")
    else:
        print("  No corrections applied.")
