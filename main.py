"""
OCR Pipeline — CLI Entry Point

Converts multi-page PDFs into clean plaintext files using DeepSeek-OCR-2
(4-bit quantized local inference).

The pipeline runs: PDF → page images → OCR (markdown) → plaintext (.txt)

Usage:
    python main.py input.pdf -o ./output
    python main.py ./pdf_folder/ -o ./output --dpi 200
    python main.py input.pdf --dpi 300
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from core.pdf_processor import pdf_to_images, get_pdf_info
from markdown_converter import convert_file
from post_corrector import PostCorrector


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def find_pdfs(input_path: str) -> List[Path]:
    """Find all PDF files from the given input path.

    Args:
        input_path: Path to a single PDF or a directory of PDFs.

    Returns:
        List of resolved Path objects to PDF files.
    """
    path = Path(input_path).resolve()

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            return [path]
        else:
            print(f"Error: {path} is not a PDF file.")
            sys.exit(1)

    elif path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            print(f"Error: No PDF files found in {path}")
            sys.exit(1)
        return pdfs

    else:
        print(f"Error: {path} does not exist.")
        sys.exit(1)


def load_ocr_engine(args):
    """Load the DeepSeek-OCR-2 engine.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Loaded DeepSeekOCR instance with .load(), .ocr_pil_image(),
        and .unload() methods.
    """
    logger = logging.getLogger("pipeline")
    from core.deepseek_ocr import DeepSeekOCR

    logger.info("=" * 60)
    logger.info("Loading DeepSeek-OCR-2 model...")
    logger.info(f"  Quantization: {'4-bit NF4' if not args.full_precision else 'Full (bfloat16)'}")
    logger.info("  Attention: eager (GTX 1650 compatible)")
    logger.info("  Device mapping: auto (GPU + CPU offload)")
    logger.info("=" * 60)

    ocr_engine = DeepSeekOCR(
        use_4bit=not args.full_precision,
    )

    ocr_engine.load()
    return ocr_engine


def process_pdf(
    pdf_path: Path,
    ocr_engine,
    output_dir: Path,
    dpi: int = 200,
    image_size: int = 768,
    base_size: int = 1024,
    preserve_layout: bool = True,
    keep_md: bool = False,
    corrector: PostCorrector = None,
) -> tuple:
    """Process a single PDF through the full pipeline.

    Args:
        pdf_path: Path to the PDF file.
        ocr_engine: Loaded DeepSeekOCR engine instance.
        output_dir: Directory to write the output text file.
        dpi: Render resolution for PDF pages.
        image_size: Model input image size.
        base_size: Model input base size for tiles.
        preserve_layout: Whether to preserve document layout.

    Returns:
        Tuple of (output_path, page_timings) where page_timings is a list
        of dicts with per-page benchmark data.
    """
    logger = logging.getLogger("pipeline")
    logger.info(f"Processing: {pdf_path.name}")

    # Step 1: PDF → Page Images
    logger.info("Step 1/3: Splitting PDF into page images...")
    temp_dir = str(output_dir / ".temp")
    page_images = pdf_to_images(str(pdf_path), dpi=dpi)
    total_pages = len(page_images)
    logger.info(f"  → {total_pages} pages extracted")

    # Step 2: OCR each page
    logger.info("Step 2/3: Running OCR on each page...")
    page_markdowns: List[str] = []
    page_timings: list = []

    for page_num, image in page_images:
        page_start = time.time()
        logger.info(f"  OCR page {page_num}/{total_pages}...")

        markdown = ocr_engine.ocr_pil_image(
            image=image,
            page_num=page_num,
            temp_dir=temp_dir,
            preserve_layout=preserve_layout,
            image_size=image_size,
            base_size=base_size,
        )
        page_markdowns.append(markdown)

        elapsed = time.time() - page_start
        logger.info(f"  -> Page {page_num} done ({elapsed:.1f}s)")

        page_timings.append({
            "pdf": pdf_path.name,
            "page": page_num,
            "total_pages": total_pages,
            "ocr_seconds": round(elapsed, 1),
            "output_chars": len(markdown) if markdown else 0,
            "dpi": dpi,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    # Step 3: Save intermediate markdown
    logger.info("Step 3/5: Saving intermediate markdown...")
    
    parts = []
    for i, md in enumerate(page_markdowns, start=1):
        text = md if md else ""
        parts.append(f"[Page {i}]\n\n{text}")
    
    final_md = "\n\n".join(parts) + "\n"

    md_file = output_dir / f"{pdf_path.stem}.md"
    md_file.write_text(final_md, encoding="utf-8")
    logger.info(f"  -> Markdown saved: {md_file}")

    # Step 4: Convert markdown -> plaintext
    logger.info("Step 4/5: Converting to plaintext...")
    txt_file = output_dir / f"{pdf_path.stem}.txt"
    convert_file(md_file, txt_file)
    logger.info(f"  -> Plaintext saved: {txt_file}")

    # Step 5: Post-correction
    if corrector and corrector.stats()["total"] > 0:
        logger.info("Step 5/5: Applying post-corrections...")
        raw = txt_file.read_text(encoding="utf-8")
        fixed, changes = corrector.apply(raw)
        txt_file.write_text(fixed, encoding="utf-8")
        if changes:
            logger.info(f"  -> {len(changes)} correction(s) applied:")
            for ch in changes:
                logger.info(f"     [{ch['type'].upper()[:5]}] {ch['pattern']} -> {ch['replacement']} (x{ch['count']})")
        else:
            logger.info("  -> No corrections needed")
    else:
        logger.info("Step 5/5: Post-correction skipped (no corrections loaded)")

    # Remove intermediate .md unless --keep-md
    if not keep_md:
        md_file.unlink(missing_ok=True)
        logger.info("  → Intermediate .md removed (use --keep-md to retain)")

    # Cleanup temp directory
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return txt_file, page_timings


def main():
    parser = argparse.ArgumentParser(
        description="OCR Pipeline: Convert PDFs to layout-preserved text files using DeepSeek-OCR-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf
  python main.py document.pdf -o ./results
  python main.py document.pdf --dpi 300
  python main.py ./pdf_folder/ -o ./results --dpi 200
  python main.py document.pdf --no-layout
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a PDF file or directory containing PDFs.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.json",
        help="Path to JSON configuration file (default: config.json).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Output directory for text files (default: ./output).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render DPI for PDF pages (default: 200).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Model input image size (default: 768).",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Base resolution for the model's tile processing (default: 1024).",
    )
    parser.add_argument(
        "--no-layout",
        action="store_true",
        help="Use free OCR mode (no layout preservation).",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Load model in full precision (requires >=16GB VRAM).",
    )
    parser.add_argument(
        "--keep-md",
        action="store_true",
        help="Keep the intermediate markdown file alongside the .txt output.",
    )
    parser.add_argument(
        "--no-correct",
        action="store_true",
        help="Skip post-correction dictionary step.",
    )
    parser.add_argument(
        "--corrections",
        type=str,
        default=None,
        help="Path to custom corrections JSON file (default: corrections.json).",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Path to ground truth text file or directory for evaluation (generates _eval.txt report).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    # First parse to get the config file path if specified
    args, remaining = parser.parse_known_args()

    config_path = Path(args.config)
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                parser.set_defaults(**config_data)
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {e}")

    # Final parse to override config defaults with any other CLI args
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger("pipeline")
    output_dir = Path(args.output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    # Find PDFs
    pdf_files = find_pdfs(args.input)
    logger.info(f"Found {len(pdf_files)} PDF(s) to process")

    # Print PDF info
    for pdf in pdf_files:
        info = get_pdf_info(str(pdf))
        logger.info(f"  {info['name']}: {info['pages']} pages")

    # Load model
    ocr_engine = load_ocr_engine(args)

    # Load post-corrector
    corrector = None
    if not args.no_correct:
        corrections_path = Path(args.corrections) if args.corrections else None
        corrector = PostCorrector(corrections_path)
        s = corrector.stats()
        if s["total"] > 0:
            logger.info(f"Post-correction loaded: {s['total']} rules "
                        f"({s['words']} words, {s['phrases']} phrases, {s['regex']} regex)")
        else:
            logger.info("No corrections file found — post-correction disabled")

    # Process each PDF
    total_start = time.time()
    output_files: List[Path] = []
    all_timings: list = []

    for idx, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PDF {idx}/{len(pdf_files)}: {pdf_path.name}")
        logger.info(f"{'=' * 60}")

        output_file, page_timings = process_pdf(
            pdf_path=pdf_path,
            ocr_engine=ocr_engine,
            output_dir=output_dir,
            dpi=args.dpi,
            image_size=args.image_size,
            base_size=args.base_size,
            preserve_layout=not args.no_layout,
            keep_md=args.keep_md,
            corrector=corrector,
        )
        output_files.append(output_file)
        all_timings.extend(page_timings)

    # Summary
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"COMPLETE -- {len(output_files)} file(s) processed in {total_elapsed:.1f}s")
    for f in output_files:
        logger.info(f"  -> {f}")
    logger.info(f"{'=' * 60}")

    # Write benchmark CSV
    if all_timings:
        csv_path = output_dir / "benchmark.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "pdf", "page", "total_pages", "ocr_seconds",
                "output_chars", "dpi", "timestamp",
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_timings)
        logger.info(f"Benchmark data appended to: {csv_path}")

        # Print timing summary table
        times = [t["ocr_seconds"] for t in all_timings]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_pages = len(times)
        logger.info(f"")
        logger.info(f"  Timing Summary ({total_pages} pages)")
        logger.info(f"  {'Avg/page:':<14} {avg_time:>7.1f}s  ({avg_time/60:.1f} min)")
        logger.info(f"  {'Min page:':<14} {min_time:>7.1f}s  ({min_time/60:.1f} min)")
        logger.info(f"  {'Max page:':<14} {max_time:>7.1f}s  ({max_time/60:.1f} min)")
        logger.info(f"  {'Total:':<14} {total_elapsed:>7.1f}s  ({total_elapsed/60:.1f} min)")

    # Evaluation
    if args.evaluate:
        from evaluate import compute_metrics, format_report
        gt_base_path = Path(args.evaluate)
        
        for out_file in output_files:
            if gt_base_path.is_file():
                current_gt = gt_base_path
            else:
                current_gt = gt_base_path / out_file.name
                
            if current_gt.exists():
                logger.info(f"Evaluating {out_file.name} against {current_gt.name}...")
                with open(current_gt, "r", encoding="utf-8") as f:
                    gt_text = f.read()
                with open(out_file, "r", encoding="utf-8") as f:
                    ocr_text = f.read()
                    
                metrics = compute_metrics(gt_text, ocr_text)
                report = format_report(metrics, label=f"Evaluation: {out_file.name} vs {current_gt.name}")
                
                eval_path = output_dir / f"{out_file.stem}_eval.txt"
                with open(eval_path, "w", encoding="utf-8") as f:
                    f.write(report)
                logger.info(f"  -> Report saved: {eval_path.name}")
            else:
                logger.warning(f"Could not find ground truth for evaluation: {current_gt}")

    # Cleanup
    ocr_engine.unload()


if __name__ == "__main__":
    main()
