"""
test.py — Submission entry-point for the HiLabs Workshop evaluation framework.

Usage:
    python test.py input.json output.json

input.json — Single document mode:
    {
        "ground_truth_path": "test_data/folder_1/folder_1.json",
        "prediction_path":   "output/folder_1.json"
    }

input.json — Batch mode (all 30 documents):
    {
        "mode":          "batch",
        "test_data_dir": "test_data",
        "output_dir":    "output"
    }

output.json — Full evaluation report (JSON)
"""

import json
import sys
import os
from pathlib import Path

# Make sure src/ is on the import path when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluator import evaluate


def main():
    if len(sys.argv) != 3:
        print("Usage: python test.py input.json output.json")
        print()
        print("  input.json  — evaluation config (single doc or batch)")
        print("  output.json — path to write the evaluation report")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    # ── Load input ────────────────────────────────────────────────────────────
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    mode = input_data.get("mode", "single")
    print(f"🔍 Running evaluation in [{mode}] mode …")

    # ── Run evaluation ────────────────────────────────────────────────────────
    try:
        result = evaluate(input_data)
    except FileNotFoundError as e:
        print(f"❌ File not found during evaluation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Print quick summary ───────────────────────────────────────────────────
    print()
    if mode == "batch":
        s = result.get("summary", {})
        el = s.get("entity_level", {})
        print("┌─────────────────────────────────────────────────┐")
        print("│           EVALUATION SUMMARY (BATCH)            │")
        print("├─────────────────────────────────────────────────┤")
        print(f"│  Documents:      {s.get('total_documents', 'N/A'):<32}│")
        print(f"│  GT Entities:    {s.get('total_gt_entities', 'N/A'):<32}│")
        print(f"│  Predicted:      {s.get('total_pred_entities', 'N/A'):<32}│")
        print(f"│  Matched:        {s.get('matched', 'N/A'):<32}│")
        print(f"│  False Negatives:{s.get('false_negatives', 'N/A'):<32}│")
        print(f"│  False Positives:{s.get('false_positives', 'N/A'):<32}│")
        print(f"│  Precision:      {el.get('precision', 'N/A'):<32}│")
        print(f"│  Recall:         {el.get('recall', 'N/A'):<32}│")
        print(f"│  F1 Score:       {el.get('f1', 'N/A'):<32}│")
        print("└─────────────────────────────────────────────────┘")

        fa = result.get("field_accuracy", {})
        if fa:
            print()
            print("  Field-level accuracy:")
            for field, acc in fa.items():
                bar = "█" * round((acc or 0) * 20)
                print(f"    {field:<15}: {bar:<20} {acc * 100:.1f}%")
    else:
        print(f"  GT entities:   {result.get('total_gt', 'N/A')}")
        print(f"  Predicted:     {result.get('total_pred', 'N/A')}")
        print(f"  Matched:       {result.get('matched', 'N/A')}")
        print(f"  False Neg:     {len(result.get('false_negatives', []))}")
        print(f"  False Pos:     {len(result.get('false_positives', []))}")

    print()
    print(f"✅ Report saved → {output_path}")


if __name__ == "__main__":
    main()
