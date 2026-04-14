"""
fraud_engine.py
───────────────
Step 4C of the Fraud Detection Pipeline — The Fraud Engine.

PURPOSE:
    Combines rule-based results (from rule_checks.py) and LLM results
    (from llm_checker.py) into ONE final fraud verdict per claim.

    This is the master decision layer. It loads both result sets,
    merges them using a clear priority logic, and produces a single
    clean output file per claim with the final verdict.

DECISION LOGIC:
    1. If rule engine flagged the claim → FRAUD (rules are 100% accurate)
    2. If LLM flagged the claim with HIGH confidence → FRAUD
    3. If LLM flagged the claim with MEDIUM confidence → FRAUD
       (we accept medium confidence — better to flag for review than miss)
    4. If LLM flagged with LOW confidence → LEGITIMATE
       (low confidence = not enough signal — don't flag)
    5. If neither flagged → LEGITIMATE

WHY RULES TAKE PRIORITY:
    Rule-based checks (NCCI table, duplicate billing) are deterministic
    and always correct. If a rule fires, we don't need LLM confirmation.
    The LLM is only consulted for claims that passed all rules.

INPUT:
    data/rule_results/<claim_id>_rules.json  — rule engine results
    data/llm_results/<claim_id>_llm.json     — LLM results
    data/edi_parsed/<claim_id>.json          — original claim data

OUTPUT:
    data/final_results/<claim_id>_final.json — one final verdict per claim
    data/final_results/all_results.json      — all 1,300 verdicts in one file
    data/final_results/summary.json          — overall statistics

HOW TO RUN:
    python3 src/fraud_engine.py
"""

import json    # for reading/writing JSON
import sys     # for terminal output
from pathlib import Path   # for clean file path handling
from collections import defaultdict  # for counting results

# ── File paths ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
PARSED_DIR    = BASE_DIR / "data" / "edi_parsed"      # original claim data
RULE_DIR      = BASE_DIR / "data" / "rule_results"    # rule check results
LLM_DIR       = BASE_DIR / "data" / "llm_results"     # LLM check results
FINAL_DIR     = BASE_DIR / "data" / "final_results"   # combined final results
CLAIMS_JSON   = BASE_DIR / "data" / "raw_claims" / "claims.json"  # ground truth
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth(claims_json_path):
    """
    Loads the ground truth labels from our synthetic dataset.
    Used for evaluation — to compare our predictions against the real labels.

    Returns:
        A dict mapping claim_id → {fraud_type, fraud_indicator}
    """
    if not Path(claims_json_path).exists():
        return {}
    with open(claims_json_path) as f:
        claims = json.load(f)
    return {
        c["claim_id"]: {
            "true_fraud_type":      c.get("fraud_type", "Legitimate"),
            "true_fraud_indicator": c.get("fraud_indicator", False),
        }
        for c in claims
    }


def merge_results(claim_id, rule_result, llm_result, ground_truth):
    """
    Merges rule and LLM results into one final verdict for a claim.

    Decision priority:
    1. Rule engine fired → FRAUD (deterministic, always correct)
    2. LLM high/medium confidence → FRAUD
    3. LLM low confidence → LEGITIMATE
    4. Neither fired → LEGITIMATE

    Parameters:
        claim_id     — the claim ID string
        rule_result  — dict from rule_checks.py (or None if not found)
        llm_result   — dict from llm_checker.py (or None if not found)
        ground_truth — dict with true labels for this claim

    Returns:
        A final verdict dict with all relevant fields.
    """
    # ── Determine final fraud decision ────────────────────────────────────────

    rule_fraud = rule_result.get("fraud_detected", False) if rule_result else False
    rule_type  = rule_result.get("fraud_type")            if rule_result else None
    rule_expl  = rule_result.get("explanation", "")       if rule_result else ""

    llm_fraud  = llm_result.get("fraud_detected", False)  if llm_result else False
    llm_type   = llm_result.get("fraud_type")             if llm_result else None
    llm_conf   = llm_result.get("confidence", "low")      if llm_result else "low"
    llm_expl   = llm_result.get("explanation", "")        if llm_result else ""
    llm_skip   = llm_result.get("skipped", False)         if llm_result else False

    # Priority 1: Rule engine fired — always trust rules
    if rule_fraud and not llm_skip:
        final_fraud = True
        final_type  = rule_type
        final_expl  = rule_expl
        decision_by = "rule_engine"
        confidence  = "high"

    # Priority 1b: LLM was skipped because rule already flagged
    elif rule_fraud and llm_skip:
        final_fraud = True
        final_type  = rule_type
        final_expl  = rule_expl
        decision_by = "rule_engine"
        confidence  = "high"

    # Priority 2: LLM flagged with high or medium confidence
    elif llm_fraud and llm_conf in ["high", "medium"]:
        final_fraud = True
        final_type  = llm_type
        final_expl  = llm_expl
        decision_by = "llm_claude"
        confidence  = llm_conf

    # Priority 3: LLM flagged with low confidence — don't trust it
    elif llm_fraud and llm_conf == "low":
        final_fraud = False
        final_type  = None
        final_expl  = f"LLM flagged as {llm_type} but with low confidence — treating as legitimate."
        decision_by = "llm_claude"
        confidence  = "low"

    # Priority 4: Nothing flagged — legitimate
    else:
        final_fraud = False
        final_type  = None
        final_expl  = "All checks passed — no fraud detected."
        decision_by = "both"
        confidence  = "high"

    # ── Get ground truth for this claim ────────────────────────────────────────
    gt = ground_truth.get(claim_id, {})
    true_fraud = gt.get("true_fraud_indicator", None)  # True/False/None
    true_type  = gt.get("true_fraud_type", "Unknown")

    # ── Determine if our prediction was correct ────────────────────────────────
    # correct = True if our fraud_detected matches the ground truth label
    if true_fraud is not None:
        correct = (final_fraud == true_fraud)
        # Also check if fraud TYPE is correct (only matters if both are fraud)
        if final_fraud and true_fraud:
            type_correct = (final_type == true_type)
        else:
            type_correct = True  # N/A if one side is legitimate
    else:
        correct      = None  # unknown — no ground truth
        type_correct = None

    return {
        # ── Core verdict ──────────────────────────────────────────────────────
        "claim_id":          claim_id,
        "fraud_detected":    final_fraud,
        "fraud_type":        final_type,
        "confidence":        confidence,
        "explanation":       final_expl,
        "decision_by":       decision_by,  # which engine made the final call

        # ── Ground truth comparison ───────────────────────────────────────────
        "true_fraud":        true_fraud,   # actual label from our dataset
        "true_fraud_type":   true_type,    # actual fraud type
        "prediction_correct":correct,      # did we get it right?
        "type_correct":      type_correct, # was the fraud TYPE also right?

        # ── Supporting evidence from both engines ─────────────────────────────
        "rule_result": {
            "fraud_detected": rule_fraud,
            "fraud_type":     rule_type,
            "explanation":    rule_expl,
        },
        "llm_result": {
            "fraud_detected": llm_fraud,
            "fraud_type":     llm_type,
            "confidence":     llm_conf,
            "explanation":    llm_expl,
            "skipped":        llm_skip,
        },
    }


def compute_metrics(results):
    """
    Computes precision, recall, and F1-score for the overall system
    and per fraud type.

    Definitions (in fraud detection context):
    - True Positive (TP):  We said FRAUD, ground truth is FRAUD
    - False Positive (FP): We said FRAUD, ground truth is LEGITIMATE
    - False Negative (FN): We said LEGITIMATE, ground truth is FRAUD
    - True Negative (TN):  We said LEGITIMATE, ground truth is LEGITIMATE

    - Precision = TP / (TP + FP)  — of all fraud flags, how many were right?
    - Recall    = TP / (TP + FN)  — of all real fraud, how many did we catch?
    - F1        = 2 × (P × R) / (P + R)  — harmonic mean of precision + recall

    For a fraud detection system, RECALL is most important — missing fraud
    is worse than a false alarm. We target F1 ≥ 0.75 overall.
    """
    # Filter to only claims with ground truth
    labeled = [r for r in results if r["true_fraud"] is not None]

    if not labeled:
        return {"error": "No ground truth labels found"}

    # ── Binary metrics (fraud vs legitimate) ──────────────────────────────────
    tp = sum(1 for r in labeled if r["fraud_detected"] and r["true_fraud"])
    fp = sum(1 for r in labeled if r["fraud_detected"] and not r["true_fraud"])
    fn = sum(1 for r in labeled if not r["fraud_detected"] and r["true_fraud"])
    tn = sum(1 for r in labeled if not r["fraud_detected"] and not r["true_fraud"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(labeled) if labeled else 0.0

    # ── Per fraud type metrics ─────────────────────────────────────────────────
    # For each fraud type, compute how well we detected that specific type
    fraud_types = set(
        r["true_fraud_type"] for r in labeled
        if r["true_fraud"] and r["true_fraud_type"] != "Legitimate"
    )

    per_type = {}
    for ft in sorted(fraud_types):
        # Claims where this fraud type is the true label
        ft_labeled = [r for r in labeled if r["true_fraud_type"] == ft]
        ft_tp = sum(1 for r in ft_labeled if r["fraud_detected"])
        ft_fn = sum(1 for r in ft_labeled if not r["fraud_detected"])
        # False positives: we said this type but true type is different
        ft_fp = sum(1 for r in labeled
                   if r["fraud_detected"] and r["fraud_type"] == ft
                   and r["true_fraud_type"] != ft)

        ft_prec = ft_tp / (ft_tp + ft_fp) if (ft_tp + ft_fp) > 0 else 0.0
        ft_rec  = ft_tp / (ft_tp + ft_fn) if (ft_tp + ft_fn) > 0 else 0.0
        ft_f1   = 2 * ft_prec * ft_rec / (ft_prec + ft_rec) if (ft_prec + ft_rec) > 0 else 0.0

        per_type[ft] = {
            "true_count": len(ft_labeled),
            "detected":   ft_tp,
            "missed":     ft_fn,
            "precision":  round(ft_prec, 3),
            "recall":     round(ft_rec, 3),
            "f1":         round(ft_f1, 3),
        }

    return {
        "total_claims":    len(labeled),
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives":  tn,
        "precision":       round(precision, 3),
        "recall":          round(recall, 3),
        "f1_score":        round(f1, 3),
        "accuracy":        round(accuracy, 3),
        "per_fraud_type":  per_type,
        "target_met":      f1 >= 0.75,  # our project target
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nFraud Engine — Combining Rule + LLM Results")
    print("=" * 55)

    # ── Load ground truth labels ───────────────────────────────────────────────
    print(f"\nLoading ground truth from {CLAIMS_JSON.name}...")
    ground_truth = load_ground_truth(CLAIMS_JSON)
    print(f"  Loaded {len(ground_truth):,} labeled claims")

    # ── Load all parsed claim IDs ──────────────────────────────────────────────
    claim_files  = sorted(PARSED_DIR.glob("*.json"))
    claim_ids    = [f.stem for f in claim_files]
    print(f"  Found {len(claim_ids):,} claims to process\n")

    # ── Process each claim ─────────────────────────────────────────────────────
    all_results  = []
    errors       = 0

    for i, claim_id in enumerate(claim_ids):
        try:
            # Load rule result for this claim
            rule_path = RULE_DIR / f"{claim_id}_rules.json"
            rule_result = None
            if rule_path.exists():
                with open(rule_path) as f:
                    rule_result = json.load(f)

            # Load LLM result for this claim
            llm_path = LLM_DIR / f"{claim_id}_llm.json"
            llm_result = None
            if llm_path.exists():
                with open(llm_path) as f:
                    llm_result = json.load(f)

            # Merge into final verdict
            final = merge_results(claim_id, rule_result, llm_result, ground_truth)
            all_results.append(final)

            # Save individual final result
            out_path = FINAL_DIR / f"{claim_id}_final.json"
            with open(out_path, "w") as f:
                json.dump(final, f, indent=2)

            # Progress every 200 claims
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(claim_ids)} processed...")
                sys.stdout.flush()

        except Exception as e:
            errors += 1
            print(f"[ERROR] {claim_id}: {e}")

    # ── Compute evaluation metrics ─────────────────────────────────────────────
    print(f"\nComputing evaluation metrics...")
    metrics = compute_metrics(all_results)

    # ── Save all results in one file ───────────────────────────────────────────
    with open(FINAL_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(FINAL_DIR / "summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Print results ──────────────────────────────────────────────────────────
    fraud_count  = sum(1 for r in all_results if r["fraud_detected"])
    legit_count  = sum(1 for r in all_results if not r["fraud_detected"])
    by_engine    = defaultdict(int)
    for r in all_results:
        if r["fraud_detected"]:
            by_engine[r["decision_by"]] += 1

    print(f"\n{'='*55}")
    print(f"  FINAL FRAUD ENGINE RESULTS")
    print(f"{'='*55}")
    print(f"  Total claims     : {len(all_results)}")
    print(f"  Fraud detected   : {fraud_count}")
    print(f"  Legitimate       : {legit_count}")
    print(f"  Errors           : {errors}")
    print(f"\n  Detections by engine:")
    for engine, count in sorted(by_engine.items()):
        print(f"    {engine:<25} {count:>5}")

    print(f"\n{'─'*55}")
    print(f"  EVALUATION METRICS (vs ground truth)")
    print(f"{'─'*55}")
    print(f"  Precision   : {metrics['precision']:.3f}  "
          f"(of all fraud flags, {metrics['precision']*100:.1f}% were correct)")
    print(f"  Recall      : {metrics['recall']:.3f}  "
          f"(caught {metrics['recall']*100:.1f}% of all real fraud)")
    print(f"  F1 Score    : {metrics['f1_score']:.3f}  "
          f"{'✅ TARGET MET (≥0.75)' if metrics['target_met'] else '⚠️ Below target (0.75)'}")
    print(f"  Accuracy    : {metrics['accuracy']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives  : {metrics['true_positives']:>5}  (real fraud, correctly flagged)")
    print(f"    False Positives : {metrics['false_positives']:>5}  (legitimate, wrongly flagged)")
    print(f"    True Negatives  : {metrics['true_negatives']:>5}  (legitimate, correctly cleared)")
    print(f"    False Negatives : {metrics['false_negatives']:>5}  (real fraud, missed)")

    print(f"\n{'─'*55}")
    print(f"  PER FRAUD TYPE BREAKDOWN")
    print(f"{'─'*55}")
    print(f"  {'Fraud Type':<28} {'Count':>6} {'Det':>5} {'Miss':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'─'*28} {'─'*6} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*6}")
    for ft, m in sorted(metrics["per_fraud_type"].items()):
        f1_flag = "✅" if m['f1'] >= 0.75 else "⚠️ "
        print(f"  {ft:<28} {m['true_count']:>6} {m['detected']:>5} "
              f"{m['missed']:>5} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {f1_flag}")

    print(f"\n{'='*55}")
    print(f"  Results saved to: {FINAL_DIR}")
    print(f"  Next step: run src/evaluate.py for detailed analysis")
    print(f"{'='*55}\n")
