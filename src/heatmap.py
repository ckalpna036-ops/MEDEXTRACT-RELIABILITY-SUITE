"""
heatmap.py — Error heat-map generation and Markdown report writing.

Generates:
  1. A matplotlib PNG heat-map (entity_type × field → accuracy)
  2. A formatted report.md with all evaluation metrics
"""

import os
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import CATEGORICAL_FIELDS, ENTITY_TYPES, METADATA_FIELDS


# ---------------------------------------------------------------------------
# Heat-map PNG
# ---------------------------------------------------------------------------

def generate_heatmap_png(heatmap_data: dict, output_path: str = "output/error_heatmap.png") -> str:
    """
    Build a seaborn heat-map:
        rows   = entity_type (10 types)
        cols   = categorical fields
        cells  = field-level accuracy (0.0–1.0), NaN if no data
    """
    fields  = list(CATEGORICAL_FIELDS.keys())
    et_list = ENTITY_TYPES

    # Build matrix
    matrix = []
    for et in et_list:
        row = []
        for field in fields:
            val = heatmap_data.get(et, {}).get(field)
            row.append(val if val is not None else np.nan)
        matrix.append(row)

    df = pd.DataFrame(matrix, index=et_list, columns=fields)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df,
        ax=ax,
        annot=True,
        fmt=".0%",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="#dddddd",
        cbar_kws={"label": "Field Accuracy"},
    )
    ax.set_title("Error Heat-map: Field Accuracy by Entity Type", fontsize=14, pad=14)
    ax.set_xlabel("Field", fontsize=11)
    ax.set_ylabel("Entity Type", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


def _bar(v, width: int = 20) -> str:
    if v is None:
        return "░" * width + " N/A"
    filled = round(v * width)
    color  = "🟢" if v >= 0.8 else ("🟡" if v >= 0.5 else "🔴")
    return color + " " + ("█" * filled).ljust(width) + f" {_pct(v)}"


def generate_report_md(
    eval_result: dict,
    heatmap_path: str = "output/error_heatmap.png",
    output_path: str  = "report.md",
) -> str:
    """
    Write a human-readable report.md from the batch evaluation result.
    """
    summary  = eval_result.get("summary", {})
    entity_m = summary.get("entity_level", {})
    fa       = eval_result.get("field_accuracy", {})
    et_rec   = eval_result.get("entity_type_recall", {})
    meta_sc  = eval_result.get("metadata_scores", {})
    reliability = eval_result.get("reliability_checks", {})
    workshop_examples = eval_result.get("workshop_error_examples", [])
    per_doc  = eval_result.get("per_document", {})

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# 📊 HiLabs Workshop — Evaluation Report",
        "",
        f"> Generated: {now}  ",
        "> Pipeline: OCR → Clinical NLP → Entity Extraction",
        "",
        "---",
        "",
    ]

    # ── 0. Problem Understanding ────────────────────────────────────────────
    lines += [
        "## 0. Problem Understanding & Pipeline",
        "",
        "This system evaluates a healthcare AI extraction pipeline and its reliability layer.",
        "",
        "**Workflow:**",
        "1. Medical PDF document",
        "2. OCR conversion",
        "3. Text normalization",
        "4. Clinical NLP entity extraction",
        "5. Structured JSON output",
        "",
        "The evaluator compares extracted structured entities against reference labels and reports quantitative + qualitative failure modes.",
        "",
    ]

    # ── 1. Quantitative Summary ──────────────────────────────────────────────
    lines += [
        "## 1. Quantitative Evaluation Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Documents evaluated | {summary.get('total_documents', 'N/A')} |",
        f"| Ground truth entities | {summary.get('total_gt_entities', 'N/A')} |",
        f"| Predicted entities | {summary.get('total_pred_entities', 'N/A')} |",
        f"| Matched (≥{os.getenv('FUZZY_MATCH_THRESHOLD','80')}% similarity) | {summary.get('matched', 'N/A')} |",
        f"| False Negatives (missed) | {summary.get('false_negatives', 'N/A')} |",
        f"| False Positives (hallucinated) | {summary.get('false_positives', 'N/A')} |",
        f"| **Entity Precision** | **{_pct(entity_m.get('precision'))}** |",
        f"| **Entity Recall** | **{_pct(entity_m.get('recall'))}** |",
        f"| **Entity F1** | **{_pct(entity_m.get('f1'))}** |",
        "",
        "### Field-Level Accuracy (on matched entities)",
        "",
    ]
    for field, acc in fa.items():
        lines.append(f"- **{field}**: {_bar(acc)}")
    lines.append("")

    # ── 2. Error Heat-map ────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 2. Error Heat-map",
        "",
        "Rows = entity types, Columns = categorical fields. Cell = accuracy.",
        "",
        f"![Error Heat-map]({heatmap_path})",
        "",
    ]

    # ── 2B. Input vs Output Examples ───────────────────────────────────────
    lines += [
        "## 2B. Input vs Output Analysis (Examples)",
        "",
        "The table below compares source text snippets with AI-generated structured output and expected correction.",
        "",
        "| Document | Source Text | AI Output | Error Type | Explanation | Correct Output |",
        "|---|---|---|---|---|---|",
    ]

    if workshop_examples:
        for ex in workshop_examples[:10]:
            source_text = str(ex.get("source_text", "")).replace("|", " ").strip()
            ai_output = str(ex.get("ai_output", "")).replace("|", " ").strip()
            correct = str(ex.get("correct_output", "")).replace("|", " ").strip()
            lines.append(
                f"| {ex.get('document', 'N/A')} | {source_text[:180]} | {ai_output[:180]} | {ex.get('error_type', 'N/A')} | {ex.get('explanation', 'N/A')} | {correct[:180]} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | No qualitative examples generated | N/A |")

    lines.append("")

    # ── 3. Entity-Type Recall ────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 3. Entity Type Recall",
        "",
        "| Entity Type | Recall |",
        "|---|---|",
    ]
    for et, rec in sorted(et_rec.items(), key=lambda x: (x[1] or 0)):
        lines.append(f"| {et} | {_pct(rec)} |")
    lines.append("")

    # ── 4. Metadata Field Accuracy ───────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 4. Metadata Field Scores",
        "",
        "| Field | Correct | Wrong | Missing | Accuracy |",
        "|---|---|---|---|---|",
    ]
    for field in METADATA_FIELDS:
        counts = meta_sc.get(field, {})
        correct  = counts.get("correct",  0)
        wrong    = counts.get("wrong",    0)
        missing  = counts.get("missing",  0)
        total    = correct + wrong + missing
        acc      = round(correct / total, 4) if total else None
        lines.append(f"| `{field}` | {correct} | {wrong} | {missing} | {_pct(acc)} |")
    lines.append("")

    # ── 5. Top Systemic Weaknesses ───────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 5. Top Systemic Weaknesses",
        "",
    ]

    # Find worst field accuracies
    worst_fields = sorted(
        [(f, a) for f, a in fa.items() if a is not None],
        key=lambda x: x[1],
    )[:3]

    # Find worst entity types by recall
    worst_et = sorted(
        [(et, r) for et, r in et_rec.items() if r is not None],
        key=lambda x: x[1],
    )[:3]

    # FP / FN ratio
    fn = summary.get("false_negatives", 0)
    fp = summary.get("false_positives", 0)
    total_gt = summary.get("total_gt_entities", 1)
    total_pred = summary.get("total_pred_entities", 0)

    lines.append(f"### 5.1 Entity Detection")
    lines.append(f"- **False Negative rate**: {_pct(fn / total_gt if total_gt else None)} — the pipeline *misses* {fn} ground-truth entities.")
    lines.append(f"- **False Positive rate**: {_pct(fp / total_pred if total_pred else None)} — the pipeline *hallucinates* {fp} entities not in the text.")
    lines.append("")

    # ── 5B. Error Classification Quality ────────────────────────────────────
    lines += [
        "## 5B. Error Classification",
        "",
        "Errors are categorized into extraction misses, hallucinations, and field-level misclassifications across:",
        "- Entity extraction",
        "- Temporality",
        "- Subject attribution",
        "- Negation/assertion",
        "",
        "Each qualitative row is recorded with: source text, AI output, error type, explanation, and corrected output.",
        "",
    ]
    lines.append("### 5.2 Worst-Performing Fields")
    for field, acc in worst_fields:
        lines.append(f"- `{field}` accuracy: **{_pct(acc)}** — high misclassification rate.")
    lines.append("")
    lines.append("### 5.3 Lowest-Recall Entity Types")
    for et, rec in worst_et:
        lines.append(f"- `{et}`: recall **{_pct(rec)}** — frequently missed by the LLM.")
    lines.append("")

    # ── 6. Proposed Guardrails ───────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 6. Proposed Guardrails for Reliability",
        "",
        "### 6.1 Confidence Scoring",
        "- Attach a confidence score (0–1) to each extracted entity.",
        "- Flag entities below threshold (e.g., < 0.7) for human review.",
        "- Especially important for `entity_type` and `assertion` fields.",
        "",
        "### 6.2 Schema Validation",
        "- Validate every LLM output against the JSON schema before saving.",
        "- Reject any entity with missing required fields or invalid enum values.",
        "- Auto-correct obvious normalisation issues (e.g. lowercase → UPPERCASE enums).",
        "",
        "### 6.3 Rule-Based Post-Processing",
        "- Apply medication-specific rules: if `entity_type == MEDICINE`,",
        "  require at least one of `DOSE`, `ROUTE`, or `FREQUENCY` in metadata.",
        "- Cross-reference extracted dates against document date context.",
        "- Flag `FAMILY_MEMBER` subject entities for secondary review.",
        "",
        "### 6.4 Ensemble / Voting",
        "- Run two LLMs and take the intersection as high-confidence output.",
        "- Entities only in one model's output are flagged as uncertain.",
        "",
        "### 6.5 Hallucination Detection",
        "- After extraction, verify each entity string appears (approximately) in the source text.",
        "- If fuzzy match score to source < 60%, flag as potential hallucination.",
        "",
        "### 6.6 Section-Aware Extraction",
        "- Pre-segment documents by heading before LLM extraction.",
        "- This constrains the `heading` field and reduces cross-section confusion.",
        "",
        "### 6.7 Implemented Reliability Checks (Run-Time)",
        "| Check | Flagged | Passed | Flag Rate |",
        "|---|---|---|---|",
    ]

    checks = reliability.get("checks", {}) if isinstance(reliability, dict) else {}
    for check_name in [
        "source_consistency_check",
        "hallucination_detection",
        "negation_validation",
        "temporal_validation",
        "subject_attribution_validation",
    ]:
        c = checks.get(check_name, {})
        lines.append(
            f"| {check_name} | {c.get('flagged', 0)} | {c.get('passed', 0)} | {_pct(c.get('flag_rate'))} |"
        )

    lines += [
        "",
        "### 6.8 Reliability Check Examples",
    ]

    r_examples = reliability.get("examples", []) if isinstance(reliability, dict) else []
    if r_examples:
        for ex in r_examples[:5]:
            issue_names = ", ".join(i.get("check", "") for i in ex.get("issues", []))
            context = str(ex.get("context", "")).replace("|", " ")[:200]
            lines.append(f"- **{ex.get('document', 'N/A')} / {ex.get('entity', 'N/A')}** → {issue_names} | context: {context}")
    else:
        lines.append("- No reliability flags triggered in this run.")

    lines += [
        "",
        "---",
        "",
        "## 7. Per-Document Summary",
        "",
        "| Document | GT | Pred | Matched | FN | FP | Recall | Precision |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for doc_name, doc_res in sorted(per_doc.items()):
        gt   = doc_res.get("total_gt", 0)
        pred = doc_res.get("total_pred", 0)
        m    = doc_res.get("matched", 0)
        fn   = len(doc_res.get("false_negatives", []))
        fp   = len(doc_res.get("false_positives", []))
        rec  = _pct(m / gt   if gt   else None)
        prec = _pct(m / pred if pred else None)
        lines.append(f"| {doc_name} | {gt} | {pred} | {m} | {fn} | {fp} | {rec} | {prec} |")

    lines += ["", "---", "", "*Report auto-generated by `run_pipeline.py`*", ""]

    report_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return output_path
