"""
parse_edi.py
────────────
Step 3C of the Fraud Detection Pipeline — EDI 837 Parser.

PURPOSE:
    Reads each EDI 837P file and extracts the structured fields
    needed for fraud detection — CPT codes, ICD-10 diagnosis codes,
    charges, modifiers, patient info, provider info.

WHY THIS MATTERS:
    ~95% of real insurance claims arrive as EDI 837 files.
    Parsing EDI is faster and more accurate than OCR because
    the data is already structured text — no image recognition needed.

OUTPUT FORMAT — Normalized Claim Object:
    Produces the EXACT same structure as extract_fields.py (OCR path):
    {
        "claim_id":         "CLM01093",
        "date":             "2025-09-21",
        "patient_name":     "Frank Jackson",
        "provider_name":    "Emily Nguyen",
        "facility":         "Sunrise Health Associates",
        "insurer":          "Cigna Health",
        "policy_no":        "POL346961",
        "procedure_codes":  ["87491"],
        "diagnosis_codes":  ["Z03.89"],
        "modifiers":        [],
        "line_charges":     [107.89],
        "total_charge":     107.89,
        "source":           "edi_837"
    }

    The fraud engine only ever sees this normalized object —
    it never knows if data came from PDF/OCR or EDI.

INPUT:  data/edi/<claim_id>.edi
OUTPUT: data/edi_parsed/<claim_id>.json

HOW TO RUN:
    python3 src/parse_edi.py
"""

import json    # for saving parsed results as JSON
import sys     # for terminal output flushing
import re      # for regex pattern matching
from pathlib import Path  # for clean file path handling

# ── File paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
EDI_DIR    = BASE_DIR / "data" / "edi"           # folder with .edi files
PARSED_DIR = BASE_DIR / "data" / "edi_parsed"    # output folder for parsed JSON
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# EDI PARSING FUNDAMENTALS
#
# EDI files use a simple structure:
#   - Each line is a SEGMENT ending with ~
#   - Within each segment, fields are separated by *
#   - The first field of every segment is the SEGMENT ID (e.g. NM1, SV1, HI)
#   - Example: SV1*HC:99213*92.20*UN*1***A~
#              ↑    ↑       ↑     ↑  ↑
#              ID   CPT     $     UN qty
#
# Key segments we care about for fraud detection:
#   NM1  — Name (patient, provider, insurer)
#   CLM  — Claim information (claim ID, total charge)
#   DTP  — Date/Time (service date)
#   HI   — Health Information (ICD-10 diagnosis codes)   ← KEY for fraud
#   SV1  — Professional Service (CPT code, charge)       ← KEY for fraud
#   LX   — Service line counter
# ══════════════════════════════════════════════════════════════════════════════

def parse_edi_file(edi_path):
    """
    Parses a complete EDI 837P file and returns a normalized claim dict.

    The parsing strategy:
    1. Split the file into segments (split on ~)
    2. Split each segment into fields (split on *)
    3. Identify each segment by its first field (segment ID)
    4. Extract specific fields from the segments we care about

    Parameters:
        edi_path — Path object pointing to the .edi file

    Returns:
        A normalized claim dict, or None if parsing fails.
    """
    try:
        with open(edi_path) as f:
            content = f.read()
    except Exception as e:
        print(f"  [READ ERROR] {edi_path.name}: {e}")
        return None

    # ── Step 1: Split into segments ────────────────────────────────────────────
    # Segments are separated by ~ in EDI files
    # We strip whitespace and filter out empty lines
    raw_segments = content.split("~")
    segments = [s.strip() for s in raw_segments if s.strip()]

    # ── Step 2: Parse each segment into a list of fields ──────────────────────
    # Each segment: "NM1*85*2*SUNRISE HEALTH*****XX*5344083406"
    # Becomes:      ["NM1", "85", "2", "SUNRISE HEALTH", "", "", "", "", "XX", "5344083406"]
    parsed_segments = []
    for seg in segments:
        fields = seg.split("*")
        parsed_segments.append(fields)

    # ── Step 3: Build a lookup dict organized by segment ID ───────────────────
    # This lets us quickly find all NM1 segments, all SV1 segments, etc.
    seg_map = {}
    for fields in parsed_segments:
        seg_id = fields[0].upper()
        if seg_id not in seg_map:
            seg_map[seg_id] = []
        seg_map[seg_id].append(fields)

    # ── Step 4: Extract each field we need ────────────────────────────────────

    # --- CLAIM ID ---
    # From CLM segment: CLM*CLM01093*107.89***11:B:1*Y*A*Y*I
    # Field index 1 = claim ID
    claim_id = edi_path.stem  # fallback: use filename
    if "CLM" in seg_map:
        clm = seg_map["CLM"][0]  # take first CLM segment
        if len(clm) > 1 and clm[1]:
            claim_id = clm[1]

    # --- TOTAL CHARGE ---
    # From CLM segment: field index 2 = total charge
    total_charge = 0.0
    if "CLM" in seg_map:
        clm = seg_map["CLM"][0]
        if len(clm) > 2 and clm[2]:
            try:
                total_charge = float(clm[2])
            except ValueError:
                pass  # keep 0.0 if conversion fails

    # --- SERVICE DATE ---
    # From DTP segment: DTP*434*D8*20250921
    # Field index 3 = date in YYYYMMDD format
    # We convert back to YYYY-MM-DD for consistency with our JSON format
    service_date = None
    if "DTP" in seg_map:
        for dtp in seg_map["DTP"]:
            # DTP qualifier 434 = statement dates, 472 = service date
            if len(dtp) > 3 and dtp[1] in ["434", "472"]:
                raw_date = dtp[3]
                # Convert YYYYMMDD → YYYY-MM-DD
                if len(raw_date) == 8 and raw_date.isdigit():
                    service_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                    break

    # --- DIAGNOSIS CODES (ICD-10) ---
    # From HI segment: HI*ABK:Z03.89*ABF:I10
    # Each field after the segment ID has format: QUALIFIER:CODE
    # ABK = principal diagnosis, ABF = additional diagnosis
    # The codes are what we cross-reference against CPT codes for fraud detection
    diagnosis_codes = []
    if "HI" in seg_map:
        for hi in seg_map["HI"]:
            # Fields 1 onwards contain diagnosis code pairs
            for field in hi[1:]:
                if ":" in field:
                    parts = field.split(":")
                    qualifier = parts[0].upper()
                    code      = parts[1] if len(parts) > 1 else ""
                    # ABK = principal, ABF = additional diagnosis
                    if qualifier in ["ABK", "ABF"] and code:
                        diagnosis_codes.append(code.strip())

    # --- PROCEDURE CODES AND CHARGES ---
    # From SV1 segment: SV1*HC:99213*92.20*UN*1***A
    # Field 1 = service ID (HC:CPTcode or HC:CPTcode:modifier)
    # Field 2 = charge amount
    # Field 4 = units
    # Field 7 = diagnosis pointer (A, B, C... maps to HI diagnosis list)
    # The CPT codes here are the KEY fraud detection signals
    procedure_codes = []
    line_charges    = []
    modifiers       = []

    if "SV1" in seg_map:
        for sv1 in seg_map["SV1"]:
            if len(sv1) < 2:
                continue

            # Parse the composite service ID: HC:CPTcode or HC:CPTcode:modifier
            service_id = sv1[1]  # e.g. "HC:99213" or "HC:99213:59"
            parts = service_id.split(":")

            # parts[0] = "HC" (qualifier), parts[1] = CPT code
            if len(parts) >= 2:
                cpt_code = parts[1].strip()
                procedure_codes.append(cpt_code)

                # parts[2] = modifier (if present)
                if len(parts) >= 3 and parts[2].strip():
                    modifiers.append(f"-{parts[2].strip()}")

            # Extract the charge amount (field 2)
            if len(sv1) > 2 and sv1[2]:
                try:
                    line_charges.append(float(sv1[2]))
                except ValueError:
                    line_charges.append(0.0)

    # --- PATIENT NAME ---
    # From NM1 segment with qualifier QC (patient)
    # NM1*QC*1*JACKSON*FRANK****MI*P6372
    # Field 3 = last name, Field 4 = first name
    patient_name = "Unknown"
    if "NM1" in seg_map:
        for nm1 in seg_map["NM1"]:
            if len(nm1) > 1 and nm1[1] == "QC":  # QC = patient
                last  = nm1[3] if len(nm1) > 3 else ""
                first = nm1[4] if len(nm1) > 4 else ""
                if last or first:
                    patient_name = f"{first} {last}".strip()
                break

    # --- PROVIDER NAME ---
    # From NM1 segment with qualifier 82 (rendering provider)
    # NM1*82*1*NGUYEN*EMILY****XX*5344083406
    provider_name = "Unknown"
    if "NM1" in seg_map:
        for nm1 in seg_map["NM1"]:
            if len(nm1) > 1 and nm1[1] == "82":  # 82 = rendering provider
                last  = nm1[3] if len(nm1) > 3 else ""
                first = nm1[4] if len(nm1) > 4 else ""
                if last or first:
                    provider_name = f"Dr. {first} {last}".strip()
                break

    # --- FACILITY / BILLING PROVIDER ---
    # From NM1 segment with qualifier 85 (billing provider)
    # NM1*85*2*SUNRISE HEALTH ASSOCIATES*****XX*5344083406
    facility = "Unknown"
    if "NM1" in seg_map:
        for nm1 in seg_map["NM1"]:
            if len(nm1) > 1 and nm1[1] == "85":  # 85 = billing provider
                if len(nm1) > 3 and nm1[3]:
                    facility = nm1[3]
                break

    # --- INSURER ---
    # From NM1 segment with qualifier PR (payer)
    # NM1*PR*2*CIGNA HEALTH*****PI*POL346961
    insurer = "Unknown"
    if "NM1" in seg_map:
        for nm1 in seg_map["NM1"]:
            if len(nm1) > 1 and nm1[1] == "PR":  # PR = payer
                if len(nm1) > 3 and nm1[3]:
                    insurer = nm1[3]
                break

    # --- POLICY NUMBER ---
    # From NM1 PR segment, field 9 = member ID
    policy_no = ""
    if "NM1" in seg_map:
        for nm1 in seg_map["NM1"]:
            if len(nm1) > 1 and nm1[1] == "PR":
                if len(nm1) > 9 and nm1[9]:
                    policy_no = nm1[9]
                break

    # ── Step 5: Assemble normalized claim object ───────────────────────────────
    # This is identical in structure to what extract_fields.py (OCR) produces
    # The fraud engine doesn't know or care which path generated this object
    return {
        "claim_id":        claim_id,
        "date":            service_date,
        "patient_name":    patient_name,
        "provider_name":   provider_name,
        "facility":        facility,
        "insurer":         insurer,
        "policy_no":       policy_no,
        "procedure_codes": procedure_codes,   # KEY — CPT codes for fraud detection
        "diagnosis_codes": diagnosis_codes,   # KEY — ICD-10 codes for fraud detection
        "modifiers":       modifiers,         # KEY — for modifier abuse detection
        "line_charges":    line_charges,      # per-line charges for upcoding check
        "total_charge":    total_charge,      # total for upcoding check
        "source":          "edi_837",         # tells fraud engine this came from EDI
    }


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION FUNCTION
# Checks that the parsed output matches what we expect from the ground truth
# ══════════════════════════════════════════════════════════════════════════════

def verify_sample(parsed, claim_id):
    """
    Prints a formatted summary of one parsed claim for verification.
    Lets us visually confirm that parsing extracted the right fields.

    Parameters:
        parsed   — the normalized claim dict returned by parse_edi_file
        claim_id — the claim ID string for display purposes
    """
    print(f"\n{'─'*55}")
    print(f"  PARSED CLAIM: {claim_id}")
    print(f"{'─'*55}")
    print(f"  Date          : {parsed.get('date', 'N/A')}")
    print(f"  Patient       : {parsed.get('patient_name', 'N/A')}")
    print(f"  Provider      : {parsed.get('provider_name', 'N/A')}")
    print(f"  Facility      : {parsed.get('facility', 'N/A')}")
    print(f"  Insurer       : {parsed.get('insurer', 'N/A')}")
    print(f"  Policy No.    : {parsed.get('policy_no', 'N/A')}")
    print(f"  CPT Codes     : {parsed.get('procedure_codes', [])}")
    print(f"  ICD-10 Codes  : {parsed.get('diagnosis_codes', [])}")
    print(f"  Modifiers     : {parsed.get('modifiers', [])}")
    print(f"  Line Charges  : {parsed.get('line_charges', [])}")
    print(f"  Total Charge  : ${parsed.get('total_charge', 0):.2f}")
    print(f"  Source        : {parsed.get('source', 'N/A')}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Find all EDI files in the edi directory
    edi_files = sorted(EDI_DIR.glob("*.edi"))

    if not edi_files:
        print(f"\nNo EDI files found in {EDI_DIR}")
        print("Run src/generate_edi.py first.")
        import sys; sys.exit(1)

    print(f"\nParsing {len(edi_files)} EDI 837P files...")
    print(f"Output folder: {PARSED_DIR}\n")

    success = errors = 0

    for i, edi_path in enumerate(edi_files):
        try:
            # Parse the EDI file into a normalized claim dict
            claim = parse_edi_file(edi_path)

            if claim:
                # Save as JSON in the edi_parsed folder
                out_path = PARSED_DIR / f"{edi_path.stem}.json"
                with open(out_path, "w") as f:
                    json.dump(claim, f, indent=2)
                success += 1
            else:
                errors += 1

            # Print progress every 100 files
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(edi_files)} done...")
                sys.stdout.flush()

        except Exception as e:
            errors += 1
            print(f"[ERROR] {edi_path.name}: {e}")

    print(f"\n✅  Done!")
    print(f"   Successfully parsed : {success}")
    print(f"   Errors              : {errors}")
    print(f"   Output folder       : {PARSED_DIR}")

    # ── Verify 3 sample claims so we can visually check the output ─────────────
    # Print the parsed fields for one legitimate and one fraud claim
    print(f"\n{'═'*55}")
    print(f"  SAMPLE VERIFICATION — checking 3 claims")
    print(f"{'═'*55}")

    # Load and display 3 sample parsed files
    sample_files = sorted(PARSED_DIR.glob("*.json"))[:3]
    for sample in sample_files:
        with open(sample) as f:
            parsed = json.load(f)
        verify_sample(parsed, sample.stem)

    print(f"\nNext step: run src/rule_checks.py (fraud detection engine)")
