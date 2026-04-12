"""
generate_edi.py
───────────────
Step 3A of the Fraud Detection Pipeline — EDI 837 Generator.

PURPOSE:
    Converts each synthetic claim (stored as JSON) into a real EDI 837P file.
    EDI 837P is the electronic standard used by ~95% of US insurance claims.
    This is how claims actually arrive at insurance companies in production.

WHY EDI 837 MATTERS:
    Our system supports TWO input paths:
      Path A — PDF  → OCR  → structured fields → fraud engine
      Path B — EDI  → parser → structured fields → fraud engine
    Both paths feed the same fraud detection engine.
    EDI is faster, more accurate, and more realistic than OCR.

EDI 837 FORMAT:
    EDI files are plain text with fields separated by asterisks (*)
    and segments separated by tildes (~). Example:
        ST*837*0001~          ← transaction set header
        NM1*85*2*CLINIC~      ← billing provider name
        SV1*HC:99213*92.20~   ← service line (CPT code + charge)

INPUT:
    data/raw_claims/claims.json

OUTPUT:
    data/edi/<claim_id>.edi   — one EDI file per claim (1,300 total)

HOW TO RUN:
    python3 src/generate_edi.py
"""

import json      # for reading the claims JSON
import sys       # for flushing terminal output
import re        # for cleaning text to be EDI-safe
from pathlib import Path  # for clean file path handling
from datetime import datetime  # for generating timestamps

# ── File paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CLAIMS_JSON = BASE_DIR / "data" / "raw_claims" / "claims.json"
EDI_DIR     = BASE_DIR / "data" / "edi"
EDI_DIR.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist


# ── EDI formatting helpers ────────────────────────────────────────────────────

def edi_safe(text, max_len=35):
    """
    Cleans a string so it's safe to include in an EDI file.
    EDI uses * as a field delimiter and ~ as a segment terminator,
    so we strip those characters from any field values.
    Also strips leading/trailing whitespace and truncates to max_len.
    """
    if not text:
        return ""
    # Remove EDI control characters and truncate
    cleaned = str(text).replace("*", "").replace("~", "").replace("\n", " ").strip()
    return cleaned[:max_len]


def format_edi_date(date_str):
    """
    Converts a date from YYYY-MM-DD (our format) to YYYYMMDD (EDI format).
    Example: '2025-09-21' → '20250921'
    """
    return date_str.replace("-", "") if date_str else ""


def format_edi_amount(amount):
    """
    Formats a dollar amount for EDI — two decimal places, no dollar sign.
    Example: 107.89 → '107.89'
    """
    return f"{float(amount):.2f}" if amount else "0.00"


def clean_npi(npi):
    """
    Ensures NPI is exactly 10 digits (EDI requirement).
    If shorter, pads with zeros on the left.
    """
    npi_str = str(npi).strip()[:10]
    return npi_str.zfill(10)  # pad with leading zeros if needed


# ── Main EDI generator ─────────────────────────────────────────────────────────

def generate_edi_837(claim):
    """
    Generates a complete EDI 837P (Professional) transaction from a claim dict.

    The EDI 837P structure follows the X12 standard and contains these segments:
      ISA  — Interchange Control Header (EDI envelope, always required)
      GS   — Functional Group Header (groups related transactions)
      ST   — Transaction Set Header (starts each individual claim)
      BPR  — Beginning of Payment Order (payment info)
      NM1  — Name segments (used for payer, provider, patient)
      CLM  — Claim information (claim ID, total charge, facility type)
      DTP  — Date/Time (service dates)
      HI   — Health Information (diagnosis codes)
      LX   — Service line counter
      SV1  — Professional Service (CPT code, charge, units per line)
      SE   — Transaction Set Trailer (ends the claim)
      GE   — Functional Group Trailer
      IEA  — Interchange Control Trailer

    Returns:
        A string containing the complete EDI 837P file content.
    """

    # Extract commonly used fields from the claim dict for convenience
    patient   = claim["patient"]
    provider  = claim["provider"]
    date_str  = format_edi_date(claim["date"])     # service date in YYYYMMDD
    now       = datetime.now().strftime("%Y%m%d")  # today's date for envelope

    # Unique control numbers (required by EDI standard for tracking)
    # In production these would be assigned by the clearinghouse
    ctrl_num  = claim["claim_id"].replace("CLM", "")  # e.g. "01093"
    isa_ctrl  = ctrl_num.zfill(9)[:9]    # 9-digit interchange control number
    gs_ctrl   = ctrl_num[:9]             # functional group control number
    st_ctrl   = "0001"                   # transaction set control number

    # Split patient name into last and first for EDI format
    name_parts = patient["name"].split()
    last_name  = edi_safe(name_parts[-1])  if len(name_parts) > 0 else "UNKNOWN"
    first_name = edi_safe(name_parts[0])   if len(name_parts) > 1 else "UNKNOWN"

    # Split provider name — remove "Dr." prefix if present
    prov_parts    = provider["name"].replace("Dr. ", "").split()
    prov_last     = edi_safe(prov_parts[-1])  if len(prov_parts) > 0 else "PROVIDER"
    prov_first    = edi_safe(prov_parts[0])   if len(prov_parts) > 1 else ""

    # Format NPI to exactly 10 digits
    provider_npi  = clean_npi(provider["npi"])

    # Format total charge
    total_charge  = format_edi_amount(claim.get("total_charge", 0))

    # Build EDI segments as a list, then join with ~ and newline
    # Each segment starts with a segment ID and fields separated by *
    segments = []

    # ── ISA: Interchange Control Header ────────────────────────────────────────
    # This is the outer envelope of the EDI file — like an address on an envelope
    # ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       *date*time*^*00501*ctrl*0*P*:
    segments.append(
        f"ISA*00*          *00*          *ZZ*FRAUDDETECT    *ZZ*INSURER        "
        f"*{now[:8]}*{datetime.now().strftime('%H%M')}*^*00501*{isa_ctrl}*0*P*:"
    )

    # ── GS: Functional Group Header ────────────────────────────────────────────
    # Groups related claim transactions together
    # HC = Health Care Claim, 005010X222A1 = EDI 837P version
    segments.append(f"GS*HC*FRAUDDETECT*INSURER*{now}*{datetime.now().strftime('%H%M')}*1*X*005010X222A1")

    # ── ST: Transaction Set Header ──────────────────────────────────────────────
    # 837 = Health Care Claim, st_ctrl = this transaction's unique control number
    segments.append(f"ST*837*{st_ctrl}*005010X222A1")

    # ── BHT: Beginning of Hierarchical Transaction ──────────────────────────────
    # 0019 = original claim, 00 = original, claim_id, date, time, CH = chargeable
    segments.append(f"BHT*0019*00*{claim['claim_id']}*{now}*{datetime.now().strftime('%H%M%S')}*CH")

    # ── NM1: Submitter Name ─────────────────────────────────────────────────────
    # 41 = submitter, 2 = non-person (organization), XX = NPI qualifier
    facility_safe = edi_safe(provider["facility"])
    segments.append(f"NM1*41*2*{facility_safe}*****46*{provider['tax_id'].replace('-','')[:9]}")

    # ── PER: Submitter Contact ──────────────────────────────────────────────────
    # IC = information contact, TE = telephone
    segments.append(f"PER*IC*BILLING DEPT*TE*5555551234")

    # ── NM1: Receiver (Insurance company) ──────────────────────────────────────
    # 40 = receiver, 2 = non-person
    insurer_safe = edi_safe(patient["insurer"])
    segments.append(f"NM1*40*2*{insurer_safe}*****46*999999999")

    # ── HL: Hierarchical Level — Billing Provider ──────────────────────────────
    # 1 = first level, 20 = billing provider
    segments.append(f"HL*1**20*1")

    # ── PRV: Provider Specialty ─────────────────────────────────────────────────
    # BI = billing, PXC = provider taxonomy, 207Q00000X = general practice
    segments.append(f"PRV*BI*PXC*207Q00000X")

    # ── NM1: Billing Provider Name ──────────────────────────────────────────────
    # 85 = billing provider, 2 = non-person entity, XX = NPI
    segments.append(f"NM1*85*2*{facility_safe}*****XX*{provider_npi}")

    # ── N3/N4: Billing Provider Address ────────────────────────────────────────
    segments.append(f"N3*123 MEDICAL CENTER BLVD")
    segments.append(f"N4*NEW YORK*NY*10001")

    # ── REF: Tax ID ──────────────────────────────────────────────────────────────
    # EI = employer identification number (federal tax ID)
    tax_id_clean = provider["tax_id"].replace("-", "")[:9]
    segments.append(f"REF*EI*{tax_id_clean}")

    # ── HL: Hierarchical Level — Subscriber (insured person) ──────────────────
    segments.append(f"HL*2*1*22*1")

    # ── SBR: Subscriber Information ─────────────────────────────────────────────
    # P = primary, 18 = self, CI = commercial insurance
    segments.append(f"SBR*P*18*{edi_safe(patient['group_no'])}**CI****")

    # ── NM1: Payer (Insurance company) ──────────────────────────────────────────
    # PR = payer, 2 = non-person, PI = payer ID
    segments.append(f"NM1*PR*2*{insurer_safe}*****PI*{patient['policy_no'][:9]}")

    # ── HL: Hierarchical Level — Patient ────────────────────────────────────────
    segments.append(f"HL*3*2*23*0")

    # ── PAT: Patient Information ─────────────────────────────────────────────────
    # 19 = child (using as default for simplicity)
    segments.append(f"PAT*19")

    # ── NM1: Patient Name ────────────────────────────────────────────────────────
    # QC = patient, 1 = person, MI = member ID
    segments.append(
        f"NM1*QC*1*{last_name}*{first_name}****MI*{edi_safe(patient['id'] if 'id' in patient else patient.get('patient_id','P0000'))}"
    )

    # ── N3/N4: Patient Address ────────────────────────────────────────────────────
    # Parse address — split on comma to get street vs city/state
    addr_parts = patient["address"].split(",")
    street = edi_safe(addr_parts[0]) if addr_parts else "123 MAIN ST"
    city_state = addr_parts[1].strip() if len(addr_parts) > 1 else "NEW YORK NY 10001"
    # Try to split city state zip
    cs_parts = city_state.split()
    city = edi_safe(cs_parts[0]) if cs_parts else "NEW YORK"
    state = cs_parts[-2] if len(cs_parts) >= 2 else "NY"
    zipcode = cs_parts[-1] if len(cs_parts) >= 1 else "10001"

    segments.append(f"N3*{street}")
    segments.append(f"N4*{city}*{state}*{zipcode[:5]}")

    # ── DMG: Patient Demographics ─────────────────────────────────────────────────
    # D8 = date format YYYYMMDD, M/F = sex
    dob_edi = format_edi_date(patient.get("dob", "1980-01-01"))
    sex_code = patient.get("sex", "U")  # M, F, or U
    segments.append(f"DMG*D8*{dob_edi}*{sex_code}")

    # ── CLM: Claim Information ────────────────────────────────────────────────────
    # claim_id, total_charge, 11 = office (place of service), B = provider signature on file
    segments.append(
        f"CLM*{claim['claim_id']}*{total_charge}***11:B:1*Y*A*Y*I"
    )

    # ── REF: Prior Authorization (if applicable) ──────────────────────────────────
    # G1 = prior authorization number
    segments.append(f"REF*G1*AUTH{ctrl_num}")

    # ── DTP: Statement Dates (service date) ──────────────────────────────────────
    # 434 = statement dates, RD8 = range of dates, D8 = single date
    segments.append(f"DTP*434*D8*{date_str}")

    # ── HI: Diagnosis Codes (ICD-10) ─────────────────────────────────────────────
    # This is a KEY field — our fraud engine checks if CPT codes are consistent
    # with these diagnosis codes. ABK = principal diagnosis in ICD-10-CM
    diag_codes = claim.get("diagnosis_codes", ["Z00.00"])
    # Format: ABK:primary_diag*ABF:secondary_diag*...
    hi_parts = []
    for idx, code in enumerate(diag_codes):
        qualifier = "ABK" if idx == 0 else "ABF"  # ABK=principal, ABF=additional
        hi_parts.append(f"{qualifier}:{edi_safe(code, 10)}")
    segments.append(f"HI*{'*'.join(hi_parts)}")

    # ── NM1: Rendering Provider ────────────────────────────────────────────────────
    # 82 = rendering provider, 1 = person, XX = NPI
    segments.append(
        f"NM1*82*1*{prov_last}*{prov_first}****XX*{provider_npi}"
    )

    # ── Service Lines — one LX/SV1/DTP block per procedure ────────────────────────
    # Each procedure line has:
    #   LX  — line counter (1, 2, 3...)
    #   SV1 — professional service (CPT code, charge, units)
    #   DTP — date of service for this line

    proc_codes   = claim.get("procedure_codes",  [])
    line_charges = claim.get("line_charges",     [])
    modifiers    = claim.get("modifiers",         [])
    n_lines      = min(len(proc_codes), 6)  # max 6 lines on a CMS-1500

    for li in range(n_lines):
        # LX: service line counter (starts at 1)
        segments.append(f"LX*{li + 1}")

        # SV1: professional service
        # Format: HC:cpt_code[:modifier], charge, UN (units), qty, , , diagnosis_pointer
        code      = edi_safe(proc_codes[li], 10)
        charge    = format_edi_amount(line_charges[li] if li < len(line_charges) else 0)
        modifier  = edi_safe(modifiers[li], 2) if li < len(modifiers) and modifiers[li] else ""

        # Build the composite service ID: HC:CPTcode or HC:CPTcode:modifier
        if modifier:
            service_id = f"HC:{code}:{modifier.replace('-','')}"
        else:
            service_id = f"HC:{code}"

        # A = first diagnosis pointer (links this procedure to diagnosis A above)
        segments.append(f"SV1*{service_id}*{charge}*UN*1***A")

        # DTP: date of service for this specific line
        segments.append(f"DTP*472*D8*{date_str}")

    # ── SE: Transaction Set Trailer ────────────────────────────────────────────────
    # Count the number of segments in this transaction (from ST to SE inclusive)
    # We add 2 to account for ST and SE themselves
    segment_count = len(segments) - 3 + 2  # subtract ISA, GS, add ST+SE
    segments.append(f"SE*{segment_count}*{st_ctrl}")

    # ── GE: Functional Group Trailer ──────────────────────────────────────────────
    # 1 = number of transaction sets in this group
    segments.append(f"GE*1*1")

    # ── IEA: Interchange Control Trailer ──────────────────────────────────────────
    # 1 = number of functional groups, isa_ctrl = matches the ISA control number
    segments.append(f"IEA*1*{isa_ctrl}")

    # Join all segments with tilde + newline
    # The ~ is the official EDI segment terminator
    return "~\n".join(segments) + "~\n"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load all 1,300 claims from JSON
    with open(CLAIMS_JSON) as f:
        claims = json.load(f)

    print(f"\nGenerating {len(claims)} EDI 837P files...")
    print(f"Output folder: {EDI_DIR}\n")

    errors = 0
    for i, claim in enumerate(claims):
        try:
            # Generate the EDI content for this claim
            edi_content = generate_edi_837(claim)

            # Write to file: e.g. data/edi/CLM01093.edi
            out_path = EDI_DIR / f"{claim['claim_id']}.edi"
            with open(out_path, "w") as f:
                f.write(edi_content)

            # Print progress every 100 claims
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(claims)} done...")
                sys.stdout.flush()

        except Exception as e:
            errors += 1
            print(f"[ERROR] {claim['claim_id']}: {e}")

    print(f"\n✅  Done!")
    print(f"   EDI files generated : {len(claims) - errors}")
    print(f"   Errors              : {errors}")
    print(f"   Output folder       : {EDI_DIR}")
    print(f"\nNext step: run src/parse_edi.py to verify parsing works.")
