"""
rule_checks.py
──────────────
Step 4A of the Fraud Detection Pipeline — Rule-Based Fraud Checks.

PURPOSE:
    Applies deterministic rule-based checks to a normalized claim object
    to detect 4 fraud types that follow fixed, provable patterns.
    No AI needed — these rules are 100% accurate when patterns match.

THE 4 FRAUD TYPES DETECTED HERE:
    1. Duplicate Billing     — same CPT code billed twice for same visit
    2. Unbundling            — component codes billed separately instead
                               of one comprehensive code (uses NCCI table)
    3. Modifier Abuse (-59)  — modifier -59 used on NCCI-bundled code pair
    4. Screening Code Abuse  — screening CPT billed without required ICD-10

WHY RULE-BASED FOR THESE 4:
    These fraud types have clear, fixed, provable patterns that do not
    require clinical judgment. A rule that says "if the same CPT code
    appears twice on the same claim, flag it" is always correct.
    Rules are also instant (microseconds) and cost nothing to run.

INPUT:
    A normalized claim object (dict) — from either parse_edi.py or
    extract_fields.py. Example:
    {
        "claim_id":        "CLM01093",
        "procedure_codes": ["85025", "36415"],
        "diagnosis_codes": ["Z00.00"],
        "modifiers":       ["-59"],
        "total_charge":    150.00,
        ...
    }

OUTPUT:
    A result dict:
    {
        "claim_id":          "CLM01093",
        "fraud_detected":    True,
        "fraud_type":        "Unbundling",
        "rule_triggered":    "ncci_bundle",
        "explanation":       "CPT 85025 and 36415 are NCCI-bundled...",
        "confidence":        "high",
        "checked_by":        "rule_engine"
    }

HOW TO USE:
    from src.rule_checks import RuleEngine
    engine = RuleEngine()
    result = engine.check(claim)

HOW TO RUN STANDALONE (tests on all EDI-parsed claims):
    python3 src/rule_checks.py
"""

import json      # for reading claim files
import sys       # for terminal output
import time      # for timing performance
import openpyxl  # for reading the NCCI Excel file
from pathlib import Path  # for clean file path handling
from collections import defaultdict  # for grouping results

# ── File paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
NCCI_PATH   = BASE_DIR / "data" / "ncci_edits.xlsx"
PARSED_DIR  = BASE_DIR / "data" / "edi_parsed"   # parsed EDI claims
RESULTS_DIR = BASE_DIR / "data" / "rule_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── CMS 2026 conversion factor ─────────────────────────────────────────────────
# Medicare payment = Total RVU × conversion factor
# Used to calculate expected price ranges per CPT code
CMS_CONVERSION_FACTOR = 32.35  # CY 2026 physician fee schedule conversion factor

# ── Screening code requirements ────────────────────────────────────────────────
# These CPT screening codes REQUIRE a specific ICD-10 Z-code diagnosis.
# If the required diagnosis is absent, it's screening code abuse.
# Source: CMS Local Coverage Determinations (LCDs)
SCREENING_REQUIREMENTS = {
    # CPT code: [list of valid ICD-10 codes that justify it]
    "G0103": ["Z00.00", "Z03.89", "Z12.5"],           # PSA screening — needs routine exam
    "G0101": ["Z12.31", "Z00.00", "Z01.419"],          # Cervical cancer screening
    "82270": ["Z12.11", "K92.1", "K57.30"],            # FOBT — needs colorectal context
    "G0107": ["Z12.11"],                                # Colorectal FOBT — needs Z12.11
    "80061": ["Z13.6", "I10", "I25.10", "E78.00"],    # Lipid panel — cardiovascular context
    "82465": ["Z13.6", "I10", "I25.10", "E78.00"],    # Cholesterol — cardiovascular context
    "83036": ["E11.9", "E11.65", "Z13.1", "E10.9"],   # HbA1c — needs diabetes diagnosis
    "G0108": ["Z13.1", "E11.9", "E10.9"],              # Diabetes self-management training
    "77067": ["Z12.31", "Z12.39"],                     # Screening mammography
    "G0202": ["Z12.31", "Z12.39"],                     # Digital mammography screening
}

# ── Known bundled code pairs (subset for fast lookup before NCCI table) ────────
# These are the most common unbundling pairs we expect in our dataset.
# The full NCCI table is used for comprehensive checks.
KNOWN_BUNDLES = {
    # (comprehensive_code, component_code): explanation
    ("80061", "82465"): "Lipid panel (80061) already includes cholesterol (82465) — billing both is unbundling",
    ("80061", "82470"): "Lipid panel (80061) already includes HDL cholesterol (82470)",
    ("80053", "82947"): "Comprehensive metabolic panel (80053) includes glucose (82947)",
    ("80053", "82565"): "Comprehensive metabolic panel (80053) includes creatinine (82565)",
    ("85025", "36415"): "CBC (85025) includes the venipuncture (36415) — billing both is unbundling",
    ("93306", "93000"): "Complete echocardiogram (93306) includes ECG (93000)",
    ("93306", "93005"): "Complete echocardiogram (93306) includes ECG tracing (93005)",
    ("93000", "93010"): "Complete ECG (93000) already includes the interpretation and report (93010)",
    ("93000", "93005"): "Complete ECG (93000) already includes the tracing-only service (93005)",
    ("99213", "99212"): "Cannot bill two E&M visit levels for the same patient on the same day",
    ("99214", "99213"): "Cannot bill two E&M visit levels for the same patient on the same day",
    ("99215", "99214"): "Cannot bill two E&M visit levels for the same patient on the same day",
}

# ── Panel composition table ──────────────────────────────────────────────────
# Classic unbundling: billing the individual COMPONENT tests of a lab panel or
# comprehensive procedure separately, WITHOUT billing the panel itself.
# NCCI PTP edits cannot catch this pattern — PTP pairs are (comprehensive,
# component), so if the comprehensive code is absent from the claim there is
# nothing to pair against. This table lets us detect the components directly.
#
# Rule: if a claim bills 2+ components of the same panel and does NOT bill
# the panel code itself, flag as Unbundling.
#
# Sources: CPT organ/disease panel definitions (80047-80076) and CMS lab
# panel billing policy.
PANEL_COMPONENTS = {
    # Lipid panel (80061): cholesterol, HDL, triglycerides
    "80061": {"82465", "82470", "83718", "84478"},
    # Comprehensive metabolic panel (80053): 14 chemistry components
    "80053": {"82040", "82247", "82310", "82374", "82435", "82565",
              "82947", "84075", "84132", "84155", "84295", "84450",
              "84460", "84520"},
    # Basic metabolic panel (80048)
    "80048": {"82310", "82374", "82435", "82565", "82947", "84132",
              "84295", "84520"},
    # CBC with differential (85025): hct, hgb, RBC, WBC, platelet counts
    "85025": {"85014", "85018", "85041", "85048", "85049"},
    # Complete transthoracic echocardiogram (93306): limited/component echo
    # codes and ECG tracing
    "93306": {"93303", "93304", "93307", "93308", "93005"},
}

# Office/outpatient E&M visit levels — only ONE level may be billed per
# patient per day. Billing two different levels is visit-splitting.
EM_VISIT_LEVELS = {"99211", "99212", "99213", "99214", "99215"}


# ══════════════════════════════════════════════════════════════════════════════
# NCCI TABLE LOADER
# Loads the full NCCI Procedure-to-Procedure edit table from Excel
# ══════════════════════════════════════════════════════════════════════════════

def load_ncci_table(ncci_path):
    """
    Loads the NCCI PTP edit table into a Python dictionary for fast lookup.

    The NCCI table has 675,000+ rows. Each row defines a pair of CPT codes
    that cannot be billed together (or can only be billed together with
    a specific modifier).

    Structure of each row:
        Column 1 (index 0): Column 1 code (comprehensive / primary code)
        Column 2 (index 1): Column 2 code (component / secondary code)
        Modifier (index 5): 0 = never allowed, 1 = allowed with modifier, 9 = N/A

    We build a dict: {(col1_code, col2_code): modifier_indicator}
    This lets us check any code pair in O(1) constant time.

    Returns:
        A dict mapping (code1, code2) tuples to modifier indicator ('0' or '1')
        Returns empty dict if file not found (graceful degradation)
    """
    ncci_dict = {}

    if not Path(ncci_path).exists():
        print(f"  ⚠️  NCCI file not found at {ncci_path}")
        print(f"  Using built-in known bundles only.")
        return ncci_dict

    print(f"  Loading NCCI edit table from {Path(ncci_path).name}...")
    start = time.time()

    try:
        wb = openpyxl.load_workbook(str(ncci_path), read_only=True)
        ws = wb.active

        count = 0
        for row in ws.iter_rows(values_only=True):
            col1     = row[0]  # Column 1 (comprehensive code)
            col2     = row[1]  # Column 2 (component code)
            modifier = row[5]  # Modifier indicator: 0, 1, or 9

            # Skip header rows and empty rows
            if not col1 or not col2:
                continue
            if not str(col1)[0].isdigit() and not str(col1)[0].isalpha():
                continue

            col1_str = str(col1).strip().zfill(5)  # pad to 5 chars
            col2_str = str(col2).strip().zfill(5)
            mod_str  = str(modifier).strip() if modifier else "9"

            # Store both directions — (A,B) and (B,A) — for bidirectional lookup
            ncci_dict[(col1_str, col2_str)] = mod_str
            ncci_dict[(col2_str, col1_str)] = mod_str
            count += 1

        elapsed = time.time() - start
        print(f"  ✅ Loaded {count:,} NCCI edit pairs in {elapsed:.1f}s")

    except Exception as e:
        print(f"  ⚠️  Could not load NCCI table: {e}")
        print(f"  Using built-in known bundles only.")

    return ncci_dict


# ══════════════════════════════════════════════════════════════════════════════
# RULE ENGINE CLASS
# Contains all 4 rule-based checks as methods
# ══════════════════════════════════════════════════════════════════════════════

class RuleEngine:
    """
    The rule-based fraud detection engine.

    On initialization, it loads the NCCI edit table once into memory.
    Then it can check any number of claims quickly using the loaded data.

    Usage:
        engine = RuleEngine()           # loads NCCI table
        result = engine.check(claim)    # check one claim
    """

    def __init__(self, ncci_path=NCCI_PATH):
        """
        Initializes the rule engine by loading the NCCI edit table.
        This takes a few seconds once, then all subsequent checks are instant.
        """
        print("\nInitializing Rule Engine...")
        # Load the NCCI bundling table
        self.ncci = load_ncci_table(ncci_path)
        # Add our known bundles as a fallback / supplement
        self.known_bundles = KNOWN_BUNDLES
        print("  Rule engine ready.\n")

    # ── CHECK 1: DUPLICATE BILLING ─────────────────────────────────────────────

    def check_duplicate_billing(self, claim):
        """
        Detects duplicate billing: the same CPT code billed more than once
        on the same claim for the same visit date.

        This is the simplest fraud type to detect — if the same code appears
        twice in the procedure_codes list, it's a duplicate.

        Example of fraud:
            procedure_codes: ["85025", "85025"]
            → Same CBC blood test billed twice for one visit

        Returns:
            A fraud result dict if duplicate found, None otherwise.
        """
        proc_codes = claim.get("procedure_codes", [])

        # Count occurrences of each code
        code_counts = {}
        for code in proc_codes:
            code_counts[code] = code_counts.get(code, 0) + 1

        # Find any code that appears more than once
        duplicates = {code: count for code, count in code_counts.items() if count > 1}

        if duplicates:
            # Build a readable list of the duplicate codes
            dup_list = ", ".join([f"{code} (×{count})" for code, count in duplicates.items()])
            return {
                "fraud_detected":  True,
                "fraud_type":      "Duplicate Billing",
                "rule_triggered":  "duplicate_cpt_code",
                "explanation":     f"Procedure code(s) billed more than once on the same claim: {dup_list}. "
                                   f"Each procedure should be billed exactly once per visit.",
                "confidence":      "high",   # rule-based = always high confidence
                "flagged_codes":   list(duplicates.keys()),
            }
        return None  # no duplicate found

    # ── CHECK 2: UNBUNDLING ────────────────────────────────────────────────────

    def check_unbundling(self, claim):
        """
        Detects unbundling: billing component procedures separately instead
        of using one comprehensive code that covers all of them.

        Detection strategy:
        1. First check our known_bundles dict (fast, covers common cases)
        2. Then check the full NCCI edit table (comprehensive, 675K+ pairs)

        The NCCI table uses modifier indicators:
            '0' = these codes can NEVER be billed together (hard violation)
            '1' = can only be billed together with a valid modifier like -59
            '9' = not applicable

        Example of fraud:
            procedure_codes: ["80061", "82465"]
            → Lipid panel billed PLUS cholesterol billed separately
            → 82465 (cholesterol) is already included in 80061 (lipid panel)

        Returns:
            A fraud result dict if unbundling found, None otherwise.
        """
        proc_codes = claim.get("procedure_codes", [])
        modifiers  = claim.get("modifiers", [])
        code_set   = {str(c).strip() for c in proc_codes}

        # ── Check 0: Panel-component split ────────────────────────────────
        # 2+ components of a panel billed WITHOUT the panel code itself.
        # NCCI PTP lookups cannot catch this (see PANEL_COMPONENTS note).
        for panel_code, components in PANEL_COMPONENTS.items():
            billed_components = code_set & components
            if panel_code not in code_set and len(billed_components) >= 2:
                comp_list = ", ".join(sorted(billed_components))
                return {
                    "fraud_detected": True,
                    "fraud_type":     "Unbundling",
                    "rule_triggered": "panel_component_split",
                    "explanation":    f"Components of panel {panel_code} billed individually "
                                      f"({comp_list}) instead of the single comprehensive "
                                      f"panel code. Billing components separately inflates "
                                      f"reimbursement versus the bundled panel rate.",
                    "confidence":     "high",
                    "flagged_codes":  sorted(billed_components),
                }

        # ── Check 0b: E&M visit-level splitting ───────────────────────────
        # Two different office-visit levels on one claim = one visit split
        # into multiple E&M codes.
        em_billed = code_set & EM_VISIT_LEVELS
        if len(em_billed) >= 2:
            em_list = ", ".join(sorted(em_billed))
            return {
                "fraud_detected": True,
                "fraud_type":     "Unbundling",
                "rule_triggered": "em_visit_split",
                "explanation":    f"Multiple E&M office-visit levels billed for the same "
                                  f"patient on the same claim ({em_list}). Only one visit "
                                  f"level may be billed per patient per day.",
                "confidence":     "high",
                "flagged_codes":  sorted(em_billed),
            }

        # Generate all possible pairs from the procedure codes on this claim
        # e.g. ["80061","82465","85025"] → pairs: (80061,82465),(80061,85025),(82465,85025)
        for i in range(len(proc_codes)):
            for j in range(i + 1, len(proc_codes)):
                code_a = str(proc_codes[i]).strip()
                code_b = str(proc_codes[j]).strip()

                # ── Check 1: Known bundles (our curated list) ────────────────
                pair_forward  = (code_a, code_b)
                pair_backward = (code_b, code_a)

                explanation = (self.known_bundles.get(pair_forward) or
                               self.known_bundles.get(pair_backward))

                if explanation:
                    return {
                        "fraud_detected": True,
                        "fraud_type":     "Unbundling",
                        "rule_triggered": "known_bundle_violation",
                        "explanation":    explanation,
                        "confidence":     "high",
                        "flagged_codes":  [code_a, code_b],
                    }

                # ── Check 2: NCCI table lookup ───────────────────────────────
                # Pad codes to 5 characters for NCCI table matching
                code_a_padded = code_a.zfill(5)
                code_b_padded = code_b.zfill(5)

                modifier_indicator = (
                    self.ncci.get((code_a_padded, code_b_padded)) or
                    self.ncci.get((code_b_padded, code_a_padded))
                )

                if modifier_indicator == "0":
                    # '0' = NEVER allowed together — hard violation
                    return {
                        "fraud_detected": True,
                        "fraud_type":     "Unbundling",
                        "rule_triggered": "ncci_ptp_modifier_0",
                        "explanation":    f"CPT {code_a} and CPT {code_b} cannot be billed together "
                                         f"according to the NCCI Procedure-to-Procedure edit table "
                                         f"(modifier indicator = 0, never allowed). "
                                         f"One code is a component of the other.",
                        "confidence":     "high",
                        "flagged_codes":  [code_a, code_b],
                    }

                elif modifier_indicator == "1" and "-59" not in modifiers:
                    # '1' = allowed ONLY with a valid modifier — but no modifier present
                    # This is a softer violation — flag as medium confidence
                    return {
                        "fraud_detected": True,
                        "fraud_type":     "Unbundling",
                        "rule_triggered": "ncci_ptp_modifier_1_missing",
                        "explanation":    f"CPT {code_a} and CPT {code_b} require modifier -59 or "
                                         f"an X-modifier to be billed on the same claim "
                                         f"(NCCI modifier indicator = 1), but no valid modifier is present.",
                        "confidence":     "medium",
                        "flagged_codes":  [code_a, code_b],
                    }

        return None  # no unbundling found

    # ── CHECK 3: MODIFIER ABUSE (-59) ─────────────────────────────────────────

    def check_modifier_abuse(self, claim):
        """
        Detects modifier abuse: using modifier -59 on an NCCI-bundled code
        pair to make them appear as separate procedures.

        Modifier -59 is legitimate when two genuinely distinct procedures
        happen to appear together on a claim. It's fraud when it's used
        specifically to bypass NCCI bundling rules — billing component codes
        separately with -59 to avoid the comprehensive code.

        Detection logic:
        - Modifier -59 is present on the claim
        - AND at least one code pair on the claim appears in the NCCI table
          with modifier indicator '0' (never allowed) or '1' (restricted)
        - This combination = modifier -59 being used to bypass bundling

        Example of fraud:
            procedure_codes: ["93000", "93306"]  ← ECG + Echo billed separately
            modifiers:       ["-59"]              ← -59 applied to bypass bundling
            → Echo (93306) includes ECG (93000). The -59 is being used to
              fraudulently unbundle them.

        Returns:
            A fraud result dict if modifier abuse found, None otherwise.
        """
        modifiers  = claim.get("modifiers", [])
        proc_codes = claim.get("procedure_codes", [])

        # Only check if modifier -59 is present
        has_59 = any("-59" in str(m) or str(m).strip() == "59" for m in modifiers)
        if not has_59:
            return None  # no -59 modifier — this check doesn't apply

        # Check every code pair to see if any are NCCI-bundled
        for i in range(len(proc_codes)):
            for j in range(i + 1, len(proc_codes)):
                code_a = str(proc_codes[i]).strip().zfill(5)
                code_b = str(proc_codes[j]).strip().zfill(5)

                # Check known bundles first
                pair_fwd = (code_a.lstrip("0") or code_a, code_b.lstrip("0") or code_b)
                pair_bwd = (code_b.lstrip("0") or code_b, code_a.lstrip("0") or code_a)

                in_known = (pair_fwd in self.known_bundles or
                            pair_bwd in self.known_bundles)

                # Check NCCI table
                ncci_indicator = (self.ncci.get((code_a, code_b)) or
                                  self.ncci.get((code_b, code_a)))

                if in_known or ncci_indicator in ["0", "1"]:
                    clean_a = code_a.lstrip("0") or code_a
                    clean_b = code_b.lstrip("0") or code_b
                    return {
                        "fraud_detected": True,
                        "fraud_type":     "Modifier Abuse (-59)",
                        "rule_triggered": "modifier_59_on_bundled_pair",
                        "explanation":    f"Modifier -59 applied to NCCI-bundled code pair "
                                         f"CPT {clean_a} and CPT {clean_b}. "
                                         f"These codes cannot be billed separately — "
                                         f"modifier -59 is being used to bypass NCCI bundling rules.",
                        "confidence":     "high",
                        "flagged_codes":  [clean_a, clean_b],
                    }

        # -59 is present but no bundled pairs found — may be legitimate use
        return None

    # ── CHECK 4: SCREENING CODE ABUSE ─────────────────────────────────────────

    def check_screening_code_abuse(self, claim):
        """
        Detects screening code abuse: billing a preventive screening test
        without the required supporting diagnosis code.

        Many screening CPT codes are only covered by insurance when the
        patient has a specific diagnosis code (usually a Z-code for
        preventive care or screening encounter). Billing the screening code
        with an unrelated diagnosis is abuse.

        Example of fraud:
            procedure_codes: ["G0103"]   ← PSA screening
            diagnosis_codes: ["I10"]     ← essential hypertension
            → PSA screening requires Z00.00 (routine exam) or Z03.89
            → I10 (hypertension) does not justify a PSA screening

        Returns:
            A fraud result dict if screening abuse found, None otherwise.
        """
        proc_codes  = claim.get("procedure_codes", [])
        diag_codes  = claim.get("diagnosis_codes", [])

        # Convert diagnosis codes to uppercase set for fast membership testing
        diag_set = {str(d).upper().strip() for d in diag_codes}

        for code in proc_codes:
            code_upper = str(code).upper().strip()

            # Check if this CPT is a screening code with requirements
            if code_upper in SCREENING_REQUIREMENTS:
                required_icds = SCREENING_REQUIREMENTS[code_upper]

                # Check if ANY of the required ICD-10 codes are present
                has_required = any(
                    req.upper() in diag_set
                    for req in required_icds
                )

                if not has_required:
                    # None of the required diagnosis codes are on this claim
                    req_list = ", ".join(required_icds[:3])  # show first 3 for brevity
                    actual   = ", ".join(list(diag_set)[:3]) if diag_set else "None"
                    return {
                        "fraud_detected": True,
                        "fraud_type":     "Screening Code Abuse",
                        "rule_triggered": "missing_required_diagnosis",
                        "explanation":    f"Screening code {code_upper} requires one of these "
                                         f"diagnosis codes to justify coverage: [{req_list}]. "
                                         f"Diagnoses on this claim: [{actual}]. "
                                         f"None of the required diagnosis codes are present.",
                        "confidence":     "high",
                        "flagged_codes":  [code_upper],
                        "required_icd10": required_icds,
                        "actual_icd10":   list(diag_set),
                    }

        return None  # no screening abuse found

    # ── MASTER CHECK FUNCTION ─────────────────────────────────────────────────

    def check(self, claim):
        """
        Runs all 4 rule-based checks on a single claim.

        Checks are run in order of severity — if a high-severity fraud type
        is found, we return immediately (a claim only needs one fraud type).
        This is the function called by the fraud engine (fraud_engine.py).

        Parameters:
            claim — a normalized claim dict from parse_edi.py or extract_fields.py

        Returns:
            A result dict with fraud verdict and explanation.
            fraud_detected = True means the claim was flagged.
            fraud_detected = False means all rule checks passed.
        """
        claim_id = claim.get("claim_id", "UNKNOWN")

        # Run checks in order: highest severity first
        # Each check returns a result dict if fraud found, or None if clean

        # Check 1: Duplicate billing (simplest, fastest)
        result = self.check_duplicate_billing(claim)
        if result:
            result["claim_id"]    = claim_id
            result["checked_by"]  = "rule_engine"
            return result

        # Check 2: Modifier abuse (-59) — run BEFORE unbundling
        # because modifier abuse IS unbundling but with an explicit -59 flag
        # catching it here gives more specific fraud type labelling
        result = self.check_modifier_abuse(claim)
        if result:
            result["claim_id"]    = claim_id
            result["checked_by"]  = "rule_engine"
            return result

        # Check 3: Unbundling (uses NCCI table)
        result = self.check_unbundling(claim)
        if result:
            result["claim_id"]    = claim_id
            result["checked_by"]  = "rule_engine"
            return result

        # Check 4: Screening code abuse
        result = self.check_screening_code_abuse(claim)
        if result:
            result["claim_id"]    = claim_id
            result["checked_by"]  = "rule_engine"
            return result

        # All checks passed — claim appears legitimate (by rules)
        # Note: LLM checks (llm_checker.py) will run next for contextual fraud
        return {
            "claim_id":        claim_id,
            "fraud_detected":  False,
            "fraud_type":      None,
            "rule_triggered":  None,
            "explanation":     "All rule-based checks passed. "
                               "No duplicate billing, unbundling, modifier abuse, "
                               "or screening code abuse detected.",
            "confidence":      "high",
            "checked_by":      "rule_engine",
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — runs all EDI-parsed claims through the rule engine
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize the rule engine (loads NCCI table)
    engine = RuleEngine()

    # Load all parsed EDI claim files
    claim_files = sorted(PARSED_DIR.glob("*.json"))

    if not claim_files:
        print(f"No parsed claims found in {PARSED_DIR}")
        print("Run src/parse_edi.py first.")
        sys.exit(1)

    print(f"Running rule checks on {len(claim_files)} claims...\n")

    # Track results
    results       = []
    fraud_counts  = defaultdict(int)
    errors        = 0
    start_time    = time.time()

    for i, claim_path in enumerate(claim_files):
        try:
            # Load the parsed claim
            with open(claim_path) as f:
                claim = json.load(f)

            # Run all rule checks
            result = engine.check(claim)
            results.append(result)

            # Track fraud type counts
            if result["fraud_detected"]:
                fraud_counts[result["fraud_type"]] += 1
            else:
                fraud_counts["Legitimate (rules passed)"] += 1

            # Save individual result
            out_path = RESULTS_DIR / f"{claim_path.stem}_rules.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            # Progress every 200 claims
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(claim_files)} done...")
                sys.stdout.flush()

        except Exception as e:
            errors += 1
            print(f"[ERROR] {claim_path.name}: {e}")

    # ── Print summary ──────────────────────────────────────────────────────────
    elapsed     = time.time() - start_time
    total_fraud = sum(v for k, v in fraud_counts.items() if k != "Legitimate (rules passed)")
    total_clean = fraud_counts.get("Legitimate (rules passed)", 0)

    print(f"\n{'═'*55}")
    print(f"  RULE CHECK RESULTS — {len(claim_files)} claims in {elapsed:.1f}s")
    print(f"{'═'*55}")
    print(f"  Fraud flagged by rules : {total_fraud}")
    print(f"  Passed all rule checks : {total_clean}")
    print(f"  Errors                 : {errors}")
    print(f"\n  Breakdown by fraud type:")
    for fraud_type, count in sorted(fraud_counts.items()):
        icon = "✅" if "Legitimate" in fraud_type else "🚨"
        print(f"  {icon}  {fraud_type:<35} {count:>5}")
    print(f"{'═'*55}")
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"\n  Next step: run src/llm_checker.py for contextual fraud checks")

    # ── Show 3 sample fraud detections ────────────────────────────────────────
    fraud_examples = [r for r in results if r["fraud_detected"]][:3]
    if fraud_examples:
        print(f"\n{'─'*55}")
        print(f"  SAMPLE DETECTIONS:")
        print(f"{'─'*55}")
        for ex in fraud_examples:
            print(f"\n  Claim     : {ex['claim_id']}")
            print(f"  Fraud Type: {ex['fraud_type']}")
            print(f"  Rule      : {ex['rule_triggered']}")
            print(f"  Confidence: {ex['confidence']}")
            print(f"  Why       : {ex['explanation'][:120]}...")