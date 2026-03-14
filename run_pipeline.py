"""
run_pipeline.py — Full end-to-end pipeline runner.

Steps:
    1. Discover all 30 folders in test_data/
    2. For each folder: pass the .md file through the LLM → save to output/
    3. Run batch evaluation (test.py logic) → save output/evaluation_report.json
    4. Generate error heat-map PNG
    5. Write report.md

Usage:
    python run_pipeline.py [--skip-llm] [--test-data test_data] [--output-dir output]

Flags:
    --skip-llm     Skip LLM extraction (use existing JSON files in output/)
    --test-data    Path to test_data directory (default: test_data)
    --output-dir   Path to output directory (default: output)
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluator import evaluate
from src.heatmap import generate_heatmap_png, generate_report_md
from src.llm_extractor import extract_entities_from_file
from src.utils import discover_test_data, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="HiLabs Workshop — Full Evaluation Pipeline"
    )
    parser.add_argument("--skip-llm",   action="store_true",
                        help="Skip LLM extraction (use existing output/ files)")
    parser.add_argument("--test-data",  default=os.getenv("TEST_DATA_DIR", "drive"),
                        help="Path to test_data directory (default: drive)")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "output"),
                        help="Path to output directory (default: output)")
    return parser.parse_args()


def step_banner(n: int, title: str):
    print()
    print(f"{'═' * 55}")
    print(f"  STEP {n}: {title}")
    print(f"{'═' * 55}")


def main():
    args = parse_args()
    test_data_dir = args.test_data
    output_dir    = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print()
    print("  ██╗  ██╗██╗██╗      █████╗ ██████╗ ███████╗")
    print("  ██║  ██║██║██║     ██╔══██╗██╔══██╗██╔════╝")
    print("  ███████║██║██║     ███████║██████╔╝███████╗")
    print("  ██╔══██║██║██║     ██╔══██║██╔══██╗╚════██║")
    print("  ██║  ██║██║███████╗██║  ██║██████╔╝███████║")
    print("  ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝")
    print()
    print("  HiLabs Workshop — GenAI Evaluation Framework")
    print(f"  test_data: {test_data_dir}  |  output: {output_dir}")
    print()

    # ── Step 1: Discover dataset ─────────────────────────────────────────────
    step_banner(1, "Discovering dataset")
    try:
        records = discover_test_data(test_data_dir)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print(f"   Please download the dataset and place it in '{test_data_dir}/'")
        sys.exit(1)

    print(f"  Found {len(records)} document(s).")
    if len(records) == 0:
        print("  ⚠️  No documents found. Exiting.")
        sys.exit(0)

    # ── Step 2: LLM extraction ───────────────────────────────────────────────
    step_banner(2, "LLM Entity Extraction")

    if args.skip_llm:
        print("  --skip-llm flag set — skipping LLM extraction.\n")
    else:
        provider = os.getenv("LLM_PROVIDER", "gemini")
        print(f"  Provider: {provider.upper()}\n")

        failed = []
        for rec in tqdm(records, desc="  Extracting", unit="doc"):
            out_path = os.path.join(output_dir, f"{rec['name']}.json")
            if os.path.exists(out_path):
                tqdm.write(f"  ↩  {rec['name']} — already extracted, skipping.")
                continue
            try:
                entities = extract_entities_from_file(rec["md_path"])
                save_json(entities, out_path)
                tqdm.write(f"  ✓  {rec['name']} — {len(entities)} entities extracted.")
            except Exception as e:
                tqdm.write(f"  ✗  {rec['name']} — FAILED: {e}")
                failed.append(rec["name"])
                # Save empty list so evaluation can still proceed
                save_json([], out_path)
            time.sleep(0.5)  # Be polite to rate limits

        if failed:
            print(f"\n  ⚠️  {len(failed)} document(s) failed extraction: {failed}")

    # ── Step 3: Evaluate ─────────────────────────────────────────────────────
    step_banner(3, "Running Evaluation")

    batch_input = {
        "mode":          "batch",
        "test_data_dir": test_data_dir,
        "output_dir":    output_dir,
    }
    eval_result = evaluate(batch_input)
    summary     = eval_result.get("summary", {})
    el          = summary.get("entity_level", {})

    report_json_path = os.path.join(output_dir, "evaluation_report.json")
    save_json(eval_result, report_json_path)
    print(f"  ✓ Evaluation report saved → {report_json_path}")

    print()
    print(f"  Precision : {el.get('precision', 'N/A')}")
    print(f"  Recall    : {el.get('recall', 'N/A')}")
    print(f"  F1        : {el.get('f1', 'N/A')}")

    # ── Step 4: Heat-map ─────────────────────────────────────────────────────
    step_banner(4, "Generating Error Heat-map")

    heatmap_path = os.path.join(output_dir, "error_heatmap.png")
    try:
        generate_heatmap_png(eval_result.get("heatmap_data", {}), heatmap_path)
        print(f"  ✓ Heat-map saved → {heatmap_path}")
    except Exception as e:
        print(f"  ⚠️  Could not generate heat-map PNG: {e}")
        heatmap_path = None

    # ── Step 5: Report ───────────────────────────────────────────────────────
    step_banner(5, "Writing report.md")

    report_path = "report.md"
    try:
        generate_report_md(
            eval_result,
            heatmap_path=heatmap_path or "output/error_heatmap.png",
            output_path=report_path,
        )
        print(f"  ✓ Report saved → {report_path}")
    except Exception as e:
        print(f"  ⚠️  Could not generate report.md: {e}")

    # ── Done ─────────────────────────────────────────────────────────────────
    print()
    print("═" * 55)
    print("  ✅  Pipeline complete!")
    print(f"      output/              → LLM entity JSONs")
    print(f"      {report_json_path}   → Full eval data")
    print(f"      report.md            → Human-readable report")
    if heatmap_path:
        print(f"      {heatmap_path} → Error heat-map PNG")
    print("═" * 55)
    print()


if __name__ == "__main__":
    main()
