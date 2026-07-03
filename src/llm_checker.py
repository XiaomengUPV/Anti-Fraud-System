"""
llm_checker.py
──────────────
Step 4B — LLM Reasoning Layer.

Detects 5 fraud types requiring clinical judgment:
  1. Upcoding          — too expensive for the diagnosis
  2. Code Padding      — unrelated codes added to inflate total
  3. Phantom Billing   — clinically impossible given diagnosis
  4. Diagnosis Mismatch— CPT and ICD-10 have no valid relationship
  5. Code Substitution — non-covered code swapped for covered one

Model: claude-haiku-4-5 (dev/cheap) or claude-sonnet-4-6 (final/better)
Cost:  ~$1.50 all 1300 claims with Haiku / ~$5.00 with Sonnet

SETUP:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-your-key-here"

RUN:
    python3 src/llm_checker.py
"""

import json, os, sys, time
from pathlib import Path
from collections import defaultdict

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

BASE_DIR    = Path(__file__).resolve().parent.parent
PARSED_DIR  = BASE_DIR / "data" / "edi_parsed"
RULE_DIR    = BASE_DIR / "data" / "rule_results"
LLM_DIR     = BASE_DIR / "data" / "llm_results"
FEE_PATH    = BASE_DIR / "data" / "cms_fee_schedule.xlsx"
LLM_DIR.mkdir(parents=True, exist_ok=True)

# ── Change MODEL_FINAL for your final evaluation run ──────────────────────────
MODEL_DEV   = "claude-haiku-4-5"    # cheap: ~$1.50 for 1300 claims
MODEL_FINAL = "claude-sonnet-4-6"   # better: ~$5.00 for 1300 claims
MODEL       = MODEL_DEV             # ← change to MODEL_FINAL when ready

RATE_LIMIT_DELAY = 0.1  # seconds between API calls to avoid rate limits


def load_fee_schedule(path):
    """
    Loads CMS 2026 Medicare fee schedule into a dict for price context.
    Maps CPT code → {description, medicare_rate} using RVU × $32.35 conversion.
    Returns empty dict if file not found — the LLM still works without it.
    """
    fee_dict = {}
    CF = 32.35  # CY2026 Medicare conversion factor

    if not Path(path).exists():
        print(f"  ⚠️  Fee schedule not found — proceeding without price data")
        return fee_dict
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True)
        ws = wb.active
        for row in ws.iter_rows(values_only=True):
            code, desc, rvu = row[0], row[4], row[9]
            if not code or not desc:
                continue
            code_str = str(code).strip()
            if isinstance(rvu, (int, float)) and rvu > 0 and code_str not in fee_dict:
                fee_dict[code_str] = {
                    "description":   str(desc).strip(),
                    "medicare_rate": round(rvu * CF, 2),
                }
        print(f"  ✅ Fee schedule loaded: {len(fee_dict):,} CPT codes")
    except Exception as e:
        print(f"  ⚠️  Fee schedule error: {e}")
    return fee_dict


def build_fraud_prompt(claim, fee_schedule):
    """
    Builds the prompt sent to Claude for each claim.

    Contains:
    - Task description and the 5 fraud types to check for
    - Claim data: CPT codes, ICD-10 codes, charges, provider
    - CMS price context: what each procedure normally costs
    - Strict JSON output format for reliable parsing

    WHY JSON OUTPUT: Parsing structured JSON is far more reliable than
    extracting information from free text — Claude follows this format
    very consistently.
    """
    cpt_codes    = claim.get("procedure_codes", [])
    icd10_codes  = claim.get("diagnosis_codes", [])
    total_charge = claim.get("total_charge", 0)
    modifiers    = claim.get("modifiers", [])
    provider     = claim.get("provider_name", "Unknown")
    date         = claim.get("date", "Unknown")

    # Build price context lines for each CPT code
    price_lines = []
    for code in cpt_codes:
        if code in fee_schedule:
            info = fee_schedule[code]
            price_lines.append(
                f"    {code} ({info['description']}): "
                f"Medicare rate = ${info['medicare_rate']:.2f}"
            )
        else:
            price_lines.append(f"    {code}: not in fee schedule")
    price_context = "\n".join(price_lines) if price_lines else "    No fee schedule data"

    # Build diagnosis context (code + description when available)
    diag_descs = claim.get("diagnosis_descs", [])
    diag_lines = []
    for i, icd in enumerate(icd10_codes):
        desc = diag_descs[i] if i < len(diag_descs) else ""
        diag_lines.append(f"    {icd}" + (f" \u2014 {desc}" if desc else ""))
    diag_context = "\n".join(diag_lines) if diag_lines else "    None"

    return f"""You are a medical billing fraud analyst reviewing a single CMS-1500 claim in isolation.

Fraud types to consider (pick the ONE that fits best if fraud is present):
1. PHANTOM BILLING \u2014 the billed procedure could not plausibly have been performed for this
   patient encounter (e.g. highly specialized diagnostics with no supporting diagnosis at all).
2. DIAGNOSIS MISMATCH \u2014 a real, plausible procedure, but no valid clinical relationship
   between it and ANY diagnosis on the claim.
3. UPCODING \u2014 a more complex/expensive CODE LEVEL than the documented condition justifies
   (e.g. highest-level office visit for a minor stable condition). Judged on code choice,
   NOT on the dollar amount billed.
4. CODE PADDING \u2014 a plausible base service plus added unrelated high-value codes.
5. CODE SUBSTITUTION \u2014 the code pattern suggests a non-covered service (cosmetic,
   experimental, screening beyond frequency limits) was rebilled under a covered code,
   e.g. the diagnosis fits a non-covered service while the billed code is a covered look-alike.

How to distinguish: PHANTOM = service almost certainly never happened; MISMATCH = service
plausibly happened but doesn't fit the diagnosis; UPCODING = right service family, inflated level.

CLAIM:
- CPT/HCPCS Codes: {", ".join(cpt_codes) if cpt_codes else "None"}
- ICD-10 Diagnoses:
{diag_context}
- Modifiers: {", ".join(modifiers) if modifiers else "None"}
- Total Billed: ${total_charge:,.2f}

CMS FEE SCHEDULE CONTEXT:
{price_context}

PRICING GUIDANCE: Billed charges routinely run 100\u2013400% of the Medicare rate for
commercial payers. A high markup ALONE is not fraud and must not be flagged. Price is
only a SUPPORTING signal when there is already a clinical inconsistency between the
procedure and the diagnoses.

EVIDENCE STANDARD:
- fraud_detected=true with confidence "high" ONLY when the procedure/diagnosis combination
  is clinically indefensible (e.g. invasive cardiac catheterization for a common cold).
- confidence "medium" when the combination is unusual and hard to justify, but a rare
  legitimate scenario exists.
- If the procedure is a reasonable match for at least one diagnosis on the claim, the claim
  is LEGITIMATE regardless of price. Do not invent doubts. A single routine service with a
  consistent diagnosis is legitimate.

Respond ONLY with valid JSON, no other text:
{{
  "fraud_detected": true or false,
  "fraud_type": "Phantom Billing" | "Diagnosis Mismatch" | "Upcoding" | "Code Padding" | "Code Substitution" | null,
  "confidence": "high" | "medium" | "low",
  "explanation": "One clear sentence explaining the fraud or why the claim is legitimate."
}}"""


class LLMChecker:
    """
    LLM-based fraud detection using the Claude API.

    Loads the API key from ANTHROPIC_API_KEY environment variable.
    Loads the CMS fee schedule for price context.
    Sends each claim to Claude and parses the JSON response.

    Usage:
        checker = LLMChecker()
        result  = checker.check(claim)
    """

    def __init__(self, model=MODEL):
        print("\nInitializing LLM Checker...")

        # Get API key from environment — never hardcode keys in source files
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set.\n"
                "Run: export ANTHROPIC_API_KEY='sk-ant-your-key-here'"
            )

        # Initialize the Anthropic SDK client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model  = model

        # Load fee schedule for price context in prompts
        self.fee_schedule = load_fee_schedule(FEE_PATH)

        # Track API usage for cost estimation
        self.input_tokens  = 0
        self.output_tokens = 0
        self.api_calls     = 0

        print(f"  Model : {self.model}")
        print(f"  Ready.\n")

    def check(self, claim):
        """
        Sends one claim to Claude and returns a structured fraud verdict.

        Steps:
        1. Build prompt with claim data + fee schedule context
        2. Call Claude API (max_tokens=300 — enough for our JSON response)
        3. Parse the JSON response
        4. Return fraud result dict

        Handles rate limit errors with automatic retry.
        Returns error result if Claude response cannot be parsed.
        """
        claim_id = claim.get("claim_id", "UNKNOWN")
        try:
            # Build and send the prompt
            prompt   = build_fraud_prompt(claim, self.fee_schedule)
            response = self.client.messages.create(
                model      = self.model,
                max_tokens = 300,
                messages   = [{"role": "user", "content": prompt}]
            )

            # Track token usage for cost estimation
            self.input_tokens  += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
            self.api_calls     += 1

            # Get the raw text response
            raw = response.content[0].text.strip()

            # Strip markdown code fences if Claude wrapped the JSON
            # e.g. ```json { ... } ``` → { ... }
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Parse JSON response
            parsed = json.loads(raw)

            return {
                "claim_id":       claim_id,
                "fraud_detected": bool(parsed.get("fraud_detected", False)),
                "fraud_type":     parsed.get("fraud_type"),
                "confidence":     parsed.get("confidence", "medium"),
                "explanation":    parsed.get("explanation", ""),
                "checked_by":     "llm_claude",
                "model_used":     self.model,
            }

        except json.JSONDecodeError as e:
            # Claude returned non-JSON — treat as no fraud detected
            return {
                "claim_id":       claim_id,
                "fraud_detected": False,
                "fraud_type":     None,
                "confidence":     "low",
                "explanation":    f"Response parse error: {str(e)[:80]}",
                "checked_by":     "llm_claude",
                "model_used":     self.model,
                "error":          True,
            }

        except Exception as e:
            # Rate limit — wait 5 seconds and retry once
            if "rate" in str(e).lower():
                time.sleep(5)
                return self.check(claim)
            return {
                "claim_id":       claim_id,
                "fraud_detected": False,
                "fraud_type":     None,
                "confidence":     "low",
                "explanation":    f"API error: {str(e)[:80]}",
                "checked_by":     "llm_claude",
                "model_used":     self.model,
                "error":          True,
            }

    def cost_summary(self):
        """Returns estimated API cost based on tokens used so far."""
        # Pricing per million tokens
        rates = {"haiku": (1.00, 5.00), "sonnet": (3.00, 15.00)}
        key   = "haiku" if "haiku" in self.model.lower() else "sonnet"
        ir, or_ = rates[key]
        cost  = (self.input_tokens / 1e6) * ir + (self.output_tokens / 1e6) * or_
        return {
            "api_calls":     self.api_calls,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd":      round(cost, 4),
            "model":         self.model,
        }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not ANTHROPIC_AVAILABLE:
        print("Install anthropic: pip install anthropic")
        sys.exit(1)

    try:
        checker = LLMChecker()
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # Load all parsed EDI claims
    claim_files = sorted(PARSED_DIR.glob("*.json"))
    if not claim_files:
        print(f"No claims found in {PARSED_DIR}. Run parse_edi.py first.")
        sys.exit(1)

    # Load which claims were already flagged by rules
    # We skip those to save API cost — they're already caught
    rule_flagged = set()
    if RULE_DIR.exists():
        for rf in RULE_DIR.glob("*.json"):
            try:
                with open(rf) as f:
                    rr = json.load(f)
                if rr.get("fraud_detected"):
                    claim_id = rf.stem.replace("_rules", "")
                    rule_flagged.add(claim_id)
            except Exception:
                pass

    to_check = len(claim_files) - len(rule_flagged)
    print(f"Total claims      : {len(claim_files)}")
    print(f"Skipping (rules)  : {len(rule_flagged)}")
    print(f"Sending to LLM    : {to_check}")
    print(f"Model             : {checker.model}\n")

    results      = []
    fraud_counts = defaultdict(int)
    errors       = 0
    start        = time.time()

    for i, claim_path in enumerate(claim_files):
        try:
            with open(claim_path) as f:
                claim = json.load(f)
            claim_id = claim.get("claim_id", claim_path.stem)

            # Skip claims already flagged by rules — saves API cost
            if claim_id in rule_flagged:
                result = {
                    "claim_id":       claim_id,
                    "fraud_detected": True,
                    "fraud_type":     "Flagged by Rule Engine",
                    "confidence":     "high",
                    "explanation":    "Already flagged by rule engine — LLM skipped.",
                    "checked_by":     "rule_engine",
                    "skipped":        True,
                }
            else:
                # Send to Claude API
                result = checker.check(claim)
                time.sleep(RATE_LIMIT_DELAY)  # gentle rate limiting

            results.append(result)

            # Count by fraud type
            if result["fraud_detected"]:
                fraud_counts[result["fraud_type"] or "Unknown"] += 1
            else:
                fraud_counts["Legitimate (LLM passed)"] += 1

            # Save individual result file
            with open(LLM_DIR / f"{claim_id}_llm.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            # Progress + live cost every 100 claims
            if (i + 1) % 100 == 0:
                cost = checker.cost_summary()
                print(f"  {i+1}/{len(claim_files)} done... "
                      f"API cost so far: ${cost['cost_usd']:.3f}")
                sys.stdout.flush()

        except Exception as e:
            errors += 1
            print(f"[ERROR] {claim_path.name}: {e}")

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed    = time.time() - start
    cost       = checker.cost_summary()
    total_fraud = sum(v for k,v in fraud_counts.items() if "Legitimate" not in k)
    total_clean = fraud_counts.get("Legitimate (LLM passed)", 0)

    print(f"\n{'='*55}")
    print(f"  LLM CHECK RESULTS — {len(claim_files)} claims in {elapsed:.0f}s")
    print(f"{'='*55}")
    print(f"  Fraud detected    : {total_fraud}")
    print(f"  Clean (LLM)       : {total_clean}")
    print(f"  Errors            : {errors}")
    print(f"\n  Breakdown by fraud type:")
    for ft, count in sorted(fraud_counts.items()):
        icon = "✅" if "Legitimate" in ft else "🚨"
        print(f"  {icon}  {ft:<35} {count:>5}")
    print(f"\n  API Usage:")
    print(f"  Calls made        : {cost['api_calls']}")
    print(f"  Input tokens      : {cost['input_tokens']:,}")
    print(f"  Output tokens     : {cost['output_tokens']:,}")
    print(f"  Estimated cost    : ${cost['cost_usd']:.4f}")
    print(f"{'='*55}")
    print(f"\n  Results saved to  : {LLM_DIR}")
    print(f"\n  Next step: run src/fraud_engine.py")

    # Show 3 sample LLM-only detections
    llm_only = [r for r in results
                if r["fraud_detected"] and not r.get("skipped") and not r.get("error")][:3]
    if llm_only:
        print(f"\n{'─'*55}")
        print(f"  SAMPLE LLM DETECTIONS:")
        print(f"{'─'*55}")
        for ex in llm_only:
            print(f"\n  Claim      : {ex['claim_id']}")
            print(f"  Fraud Type : {ex['fraud_type']}")
            print(f"  Confidence : {ex['confidence']}")
            print(f"  Why        : {ex['explanation'][:150]}")
