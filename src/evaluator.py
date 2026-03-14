"""
evaluator.py — Core entity comparison and scoring engine.

Strategy
--------
1. For each entity in the PREDICTION, find the best-matching entity in the
   GROUND TRUTH using fuzzy string matching on the `entity` field.
2. Score the matched pair on every categorical field (entity_type, assertion,
   temporality, subject) and every metadata field present in the ground truth.
3. Unmatched ground truth entities = False Negatives (missed).
   Unmatched prediction entities   = False Positives (hallucinated).
4. Aggregate results per document and across all documents for the batch report.
"""

import os
import re
from collections import defaultdict
from typing import Any

from rapidfuzz import fuzz

from src.utils import (
    CATEGORICAL_FIELDS,
    ENTITY_TYPES,
    METADATA_FIELDS,
    extract_metadata_flat,
    load_json,
    normalise_entity_list,
    load_text,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", "80"))

NEGATION_CUES = (
    "denies", "deny", "denied", "no ", "not ", "without", "negative for"
)

FUTURE_CUES = (
    "plan", "planned", "schedule", "scheduled", "follow-up", "follow up", "next week", "upcoming"
)

FAMILY_CUES = (
    "family history", "mother", "father", "sister", "brother", "maternal", "paternal"
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _entity_string(e: dict) -> str:
    return str(e.get("entity", "")).strip().lower()


def _best_match(pred_entity: dict, candidates: list[dict]) -> tuple[int, float]:
    """
    Return (index_of_best_candidate, similarity_score).
    Returns (-1, 0.0) if no candidate meets the threshold.
    """
    pred_str = _entity_string(pred_entity)
    best_idx, best_score = -1, 0.0
    for i, cand in enumerate(candidates):
        score = fuzz.token_set_ratio(pred_str, _entity_string(cand))
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score >= FUZZY_THRESHOLD:
        return best_idx, best_score
    return -1, 0.0


def _compare_metadata(pred: dict, gt: dict) -> dict:
    """
    Compare metadata_from_qa fields using the REAL relations-list format.
    Converts both pred and gt metadata to flat {FIELD: value} dicts first.
    Returns per-field match/mismatch counts.
    """
    pred_meta = extract_metadata_flat(pred.get("metadata_from_qa") or {})
    gt_meta   = extract_metadata_flat(gt.get("metadata_from_qa")   or {})

    results = {}
    for field, gt_val in gt_meta.items():
        pred_val = pred_meta.get(field)
        if pred_val is None:
            results[field] = "missing"
        elif pred_val.strip().lower() == gt_val.strip().lower():
            results[field] = "correct"
        else:
            results[field] = "wrong"
    return results


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _entity_appears_in_text(entity: str, source_text: str) -> bool:
    ent = _normalise_whitespace(entity)
    src = _normalise_whitespace(source_text)
    if not ent or not src:
        return False
    return ent in src


def _run_reliability_checks_for_entity(entity: dict, source_text: str) -> list[dict]:
    """
    Lightweight rule checks used for reliability-layer reporting.
    Returns list of triggered issues for this predicted entity.
    """
    issues = []
    ent = str(entity.get("entity", ""))
    assertion = str(entity.get("assertion", "")).upper()
    temporality = str(entity.get("temporality", "")).upper()
    subject = str(entity.get("subject", "")).upper()
    context = _normalise_whitespace(str(entity.get("text", "")))
    heading = _normalise_whitespace(str(entity.get("heading", "")))
    src = _normalise_whitespace(source_text)

    if not _entity_appears_in_text(ent, src):
        issues.append(
            {
                "check": "hallucination_detection",
                "severity": "high",
                "message": "Extracted entity does not appear in source text.",
            }
        )

    if assertion in {"POSITIVE", "NEGATIVE"}:
        has_negation_cue = any(cue in context for cue in NEGATION_CUES)
        if assertion == "POSITIVE" and has_negation_cue:
            issues.append(
                {
                    "check": "negation_validation",
                    "severity": "high",
                    "message": "Positive assertion with negation cues in context.",
                }
            )
        if assertion == "NEGATIVE" and not has_negation_cue:
            issues.append(
                {
                    "check": "negation_validation",
                    "severity": "medium",
                    "message": "Negative assertion but no explicit negation cue in context.",
                }
            )

    if temporality == "UPCOMING":
        has_future_cue = any(cue in context for cue in FUTURE_CUES)
        if not has_future_cue:
            issues.append(
                {
                    "check": "temporal_validation",
                    "severity": "medium",
                    "message": "Upcoming temporality without future/planning cue.",
                }
            )

    has_family_cue = any(cue in context or cue in heading for cue in FAMILY_CUES)
    if subject == "PATIENT" and has_family_cue:
        issues.append(
            {
                "check": "subject_attribution_validation",
                "severity": "high",
                "message": "Patient subject may conflict with family-history cues.",
            }
        )
    if subject == "FAMILY_MEMBER" and not has_family_cue:
        issues.append(
            {
                "check": "subject_attribution_validation",
                "severity": "medium",
                "message": "Family-member subject without family-history cues.",
            }
        )

    return issues


def _aggregate_reliability_checks(records: list[dict], per_doc_results: dict) -> dict:
    """
    Compute lightweight reliability metrics and examples from predictions vs source text.
    """
    check_counts: dict[str, dict[str, int]] = {
        "hallucination_detection": defaultdict(int),
        "negation_validation": defaultdict(int),
        "temporal_validation": defaultdict(int),
        "subject_attribution_validation": defaultdict(int),
        "source_consistency_check": defaultdict(int),
    }
    examples = []
    total_pred_entities = 0

    for rec in records:
        name = rec["name"]
        source_text = load_text(rec["md_path"]) if rec.get("md_path") and os.path.exists(rec["md_path"]) else ""
        try:
            pred_entities = normalise_entity_list(load_json(rec["pred_path"]))
        except FileNotFoundError:
            pred_entities = []

        total_pred_entities += len(pred_entities)

        # Source consistency from FP counts
        doc_res = per_doc_results.get(name, {})
        fp_count = len(doc_res.get("false_positives", []))
        matched = doc_res.get("matched", 0)
        check_counts["source_consistency_check"]["flagged"] += fp_count
        check_counts["source_consistency_check"]["passed"] += matched

        for ent in pred_entities:
            issues = _run_reliability_checks_for_entity(ent, source_text)
            if not issues:
                for check in check_counts:
                    if check != "source_consistency_check":
                        check_counts[check]["passed"] += 1
            else:
                triggered = {i["check"] for i in issues}
                for check in check_counts:
                    if check == "source_consistency_check":
                        continue
                    if check in triggered:
                        check_counts[check]["flagged"] += 1
                    else:
                        check_counts[check]["passed"] += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "document": name,
                            "entity": ent.get("entity", ""),
                            "context": ent.get("text", ""),
                            "issues": issues,
                        }
                    )

    # Normalize output with rates
    checks_out = {}
    for check, counts in check_counts.items():
        flagged = counts.get("flagged", 0)
        passed = counts.get("passed", 0)
        total = flagged + passed
        checks_out[check] = {
            "flagged": flagged,
            "passed": passed,
            "flag_rate": round(flagged / total, 4) if total else 0.0,
            "coverage_entities": total_pred_entities if check != "source_consistency_check" else None,
        }

    return {
        "checks": checks_out,
        "examples": examples,
    }


def _build_workshop_error_examples(records: list[dict], per_doc_results: dict, max_examples: int = 12) -> list[dict]:
    """
    Build workshop-required error table rows:
    Source text | AI output | Error type | Explanation | Correct output
    """
    rows = []

    for rec in records:
        name = rec["name"]
        source_text = load_text(rec["md_path"]) if rec.get("md_path") and os.path.exists(rec["md_path"]) else ""
        doc_res = per_doc_results.get(name, {})

        for fp in doc_res.get("false_positives", []):
            pred_ent = fp.get("entity", {})
            rows.append(
                {
                    "document": name,
                    "source_text": pred_ent.get("text") or (source_text[:200] + "..." if source_text else ""),
                    "ai_output": {
                        "entity": pred_ent.get("entity"),
                        "entity_type": pred_ent.get("entity_type"),
                        "assertion": pred_ent.get("assertion"),
                        "temporality": pred_ent.get("temporality"),
                        "subject": pred_ent.get("subject"),
                    },
                    "error_type": "Hallucination",
                    "explanation": "Predicted entity has no matched ground-truth entity.",
                    "correct_output": "Remove this entity from output.",
                }
            )
            if len(rows) >= max_examples:
                return rows

        for fn in doc_res.get("false_negatives", []):
            gt_ent = fn.get("entity", {})
            rows.append(
                {
                    "document": name,
                    "source_text": gt_ent.get("text") or "",
                    "ai_output": "Missing entity",
                    "error_type": "Entity Extraction Error",
                    "explanation": "Ground-truth entity was not extracted.",
                    "correct_output": {
                        "entity": gt_ent.get("entity"),
                        "entity_type": gt_ent.get("entity_type"),
                        "assertion": gt_ent.get("assertion"),
                        "temporality": gt_ent.get("temporality"),
                        "subject": gt_ent.get("subject"),
                    },
                }
            )
            if len(rows) >= max_examples:
                return rows

    return rows


# ---------------------------------------------------------------------------
# Per-document evaluation
# ---------------------------------------------------------------------------

def evaluate_document(pred_entities: list[dict], gt_entities: list[dict]) -> dict:
    """
    Compare prediction vs. ground truth for a single document.

    Returns
    -------
    {
        "total_gt": int,
        "total_pred": int,
        "matched": int,
        "false_negatives": list[dict],   # entities missed by the LLM
        "false_positives": list[dict],   # entities hallucinated by the LLM
        "field_scores": {
            "entity_type":  {"correct": n, "wrong": n, ...},
            "assertion":    {...},
            "temporality":  {...},
            "subject":      {...},
        },
        "entity_type_breakdown": {
            "MEDICINE": {
                "gt_count": n,
                "matched": n,
                "field_scores": {...}
            },
            ...
        },
        "metadata_scores": {
            "DOSE": {"correct": n, "wrong": n, "missing": n},
            ...
        },
        "matched_pairs": [...],   # for detailed inspection
    }
    """
    remaining_gt = list(gt_entities)  # pool of unmatched GT entities
    false_positives = []
    matched_pairs = []

    # Accumulators
    field_scores: dict[str, dict[str, int]] = {
        f: defaultdict(int) for f in CATEGORICAL_FIELDS
    }
    metadata_scores: dict[str, dict[str, int]] = {
        f: defaultdict(int) for f in METADATA_FIELDS
    }
    entity_type_breakdown: dict[str, dict] = {
        et: {"gt_count": 0, "matched": 0, "field_scores": {f: defaultdict(int) for f in CATEGORICAL_FIELDS}}
        for et in ENTITY_TYPES
    }

    # Count GT entities per type
    for gt_e in gt_entities:
        et = str(gt_e.get("entity_type", "")).upper()
        if et in entity_type_breakdown:
            entity_type_breakdown[et]["gt_count"] += 1

    # Match predictions → ground truth
    for pred_e in pred_entities:
        best_idx, score = _best_match(pred_e, remaining_gt)

        if best_idx == -1:
            false_positives.append({"entity": pred_e, "reason": "no_matching_gt"})
            continue

        gt_e = remaining_gt.pop(best_idx)
        pair = {"pred": pred_e, "gt": gt_e, "similarity": score, "field_results": {}}

        # Score categorical fields
        for field in CATEGORICAL_FIELDS:
            pred_val = str(pred_e.get(field, "")).strip().upper()
            gt_val = str(gt_e.get(field, "")).strip().upper()
            result = "correct" if pred_val == gt_val else "wrong"
            field_scores[field][result] += 1
            pair["field_results"][field] = result

            # Per entity_type breakdown
            gt_et = str(gt_e.get("entity_type", "")).upper()
            if gt_et in entity_type_breakdown:
                entity_type_breakdown[gt_et]["matched"] += 1 if field == "entity_type" else 0
                entity_type_breakdown[gt_et]["field_scores"][field][result] += 1

        # Score metadata fields
        meta_results = _compare_metadata(pred_e, gt_e)
        for field, result in meta_results.items():
            metadata_scores[field][result] += 1
        pair["metadata_results"] = meta_results

        matched_pairs.append(pair)

    false_negatives = [{"entity": gt_e, "reason": "not_extracted"} for gt_e in remaining_gt]

    # Normalise defaultdicts -> regular dicts for JSON serialisation
    def _norm(d):
        return {k: dict(v) for k, v in d.items()}

    def _norm_breakdown(bd):
        return {
            et: {
                "gt_count": v["gt_count"],
                "matched": v["matched"],
                "recall": round(v["matched"] / v["gt_count"], 4) if v["gt_count"] else None,
                "field_scores": {f: dict(fs) for f, fs in v["field_scores"].items()},
            }
            for et, v in bd.items()
        }

    return {
        "total_gt": len(gt_entities),
        "total_pred": len(pred_entities),
        "matched": len(matched_pairs),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "field_scores": _norm(field_scores),
        "entity_type_breakdown": _norm_breakdown(entity_type_breakdown),
        "metadata_scores": _norm(metadata_scores),
        "matched_pairs": matched_pairs,
    }


# ---------------------------------------------------------------------------
# Batch evaluation across all documents
# ---------------------------------------------------------------------------

def evaluate_batch(records: list[dict]) -> dict:
    """
    Evaluate all documents.

    Each record: { "name": str, "gt_path": str, "pred_path": str }
    Returns aggregated report.
    """
    per_doc_results = {}
    agg_field_scores: dict[str, dict[str, int]] = {f: defaultdict(int) for f in CATEGORICAL_FIELDS}
    agg_meta_scores: dict[str, dict[str, int]] = {f: defaultdict(int) for f in METADATA_FIELDS}
    agg_et_breakdown: dict[str, dict] = {
        et: {"gt_count": 0, "matched": 0} for et in ENTITY_TYPES
    }
    total_gt = total_pred = total_matched = total_fn = total_fp = 0

    for rec in records:
        name = rec["name"]
        gt_entities = normalise_entity_list(load_json(rec["gt_path"]))
        try:
            pred_entities = normalise_entity_list(load_json(rec["pred_path"]))
        except FileNotFoundError:
            pred_entities = []

        doc_result = evaluate_document(pred_entities, gt_entities)
        # Keep matched_pairs for downstream qualitative error classification in reports.
        per_doc_results[name] = doc_result

        total_gt      += doc_result["total_gt"]
        total_pred    += doc_result["total_pred"]
        total_matched += doc_result["matched"]
        total_fn      += len(doc_result["false_negatives"])
        total_fp      += len(doc_result["false_positives"])

        for field, counts in doc_result["field_scores"].items():
            for result, n in counts.items():
                agg_field_scores[field][result] += n

        for field, counts in doc_result["metadata_scores"].items():
            for result, n in counts.items():
                agg_meta_scores[field][result] += n

        for et, data in doc_result["entity_type_breakdown"].items():
            agg_et_breakdown[et]["gt_count"] += data["gt_count"]
            agg_et_breakdown[et]["matched"]  += data["matched"]

    # Compute precision / recall / F1 at entity level
    precision = round(total_matched / total_pred, 4) if total_pred else 0.0
    recall    = round(total_matched / total_gt,   4) if total_gt   else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    # Per field accuracy
    field_accuracy = {}
    for field, counts in agg_field_scores.items():
        correct = counts.get("correct", 0)
        total   = correct + counts.get("wrong", 0)
        field_accuracy[field] = round(correct / total, 4) if total else None

    # Per entity-type recall
    et_recall = {
        et: round(v["matched"] / v["gt_count"], 4) if v["gt_count"] else None
        for et, v in agg_et_breakdown.items()
    }

    # Heat-map data: entity_type × field → accuracy
    heatmap_data = _build_heatmap_data(per_doc_results)
    reliability = _aggregate_reliability_checks(records, per_doc_results)
    workshop_error_examples = _build_workshop_error_examples(records, per_doc_results)

    def _nd(d):
        return {k: dict(v) for k, v in d.items()}

    return {
        "summary": {
            "total_documents": len(records),
            "total_gt_entities": total_gt,
            "total_pred_entities": total_pred,
            "matched": total_matched,
            "false_negatives": total_fn,
            "false_positives": total_fp,
            "entity_level": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        },
        "field_accuracy": field_accuracy,
        "entity_type_recall": et_recall,
        "metadata_scores": _nd(agg_meta_scores),
        "heatmap_data": heatmap_data,
        "reliability_checks": reliability,
        "workshop_error_examples": workshop_error_examples,
        "per_document": per_doc_results,
    }


def _build_heatmap_data(per_doc_results: dict) -> dict:
    """
    Build a heatmap matrix: rows = entity_type, cols = categorical fields.
    Cell = accuracy of that field for entities of that type.
    """
    # entity_type → field → {correct, wrong}
    matrix: dict[str, dict[str, dict[str, int]]] = {
        et: {f: defaultdict(int) for f in CATEGORICAL_FIELDS} for et in ENTITY_TYPES
    }

    for doc_name, doc_res in per_doc_results.items():
        for et, et_data in doc_res.get("entity_type_breakdown", {}).items():
            for field, counts in et_data.get("field_scores", {}).items():
                for result, n in counts.items():
                    matrix[et][field][result] += n

    heatmap = {}
    for et in ENTITY_TYPES:
        heatmap[et] = {}
        for field in CATEGORICAL_FIELDS:
            counts = matrix[et][field]
            correct = counts.get("correct", 0)
            total   = correct + counts.get("wrong", 0)
            heatmap[et][field] = round(correct / total, 4) if total else None

    return heatmap


# ---------------------------------------------------------------------------
# Public evaluate() function called by test.py
# ---------------------------------------------------------------------------

def evaluate(input_data: dict) -> dict:
    """
    Entry point called by test.py.

    Accepts input_data in two modes:
    ---
    Mode 1 — Single document:
        {
            "ground_truth_path": "test_data/folder_1/folder_1.json",
            "prediction_path":   "output/folder_1.json"
        }

    Mode 2 — Batch:
        {
            "mode": "batch",
            "test_data_dir": "test_data",
            "output_dir":    "output"
        }
    """
    if input_data.get("mode") == "batch":
        from src.utils import discover_test_data
        test_data_dir = input_data.get("test_data_dir", "test_data")
        output_dir    = input_data.get("output_dir",    "output")
        raw_records   = discover_test_data(test_data_dir)
        records = [
            {
                "name":    r["name"],
                "md_path": r["md_path"],
                "gt_path": r["json_path"],
                "pred_path": os.path.join(output_dir, r["name"] + ".json"),
            }
            for r in raw_records
        ]
        return evaluate_batch(records)

    else:
        gt_path   = input_data["ground_truth_path"]
        pred_path = input_data["prediction_path"]
        gt_entities   = normalise_entity_list(load_json(gt_path))
        pred_entities = normalise_entity_list(load_json(pred_path))
        result = evaluate_document(pred_entities, gt_entities)
        # Clean up bulky matched_pairs for default output
        result.pop("matched_pairs", None)
        return result
