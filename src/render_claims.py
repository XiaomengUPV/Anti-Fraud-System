"""
render_claims.py
────────────────
Step 2 of the Fraud Detection Pipeline.

PURPOSE:
    Converts each synthetic claim (stored as JSON) into a realistic
    CMS-1500 health insurance claim form PDF.

WHY WE NEED THIS:
    In the real world, insurance companies receive claims as PDF documents.
    Our fraud detection system must work on raw PDF files — not just
    structured data. These PDFs are what the OCR pipeline will read in Step 3.

INPUT:
    data/raw_claims/claims.json  — 1,300 synthetic claims generated in Step 1

OUTPUT:
    data/pdfs/<claim_id>.pdf     — one PDF per claim (1,300 total)

HOW TO RUN:
    python3 src/render_claims.py
"""

import json   # for reading the claims JSON file
import sys    # for flushing progress output to terminal
from pathlib import Path  # for clean file path handling

# ReportLab is a Python library for generating PDFs programmatically
from reportlab.pdfgen import canvas          # the main PDF drawing engine
from reportlab.lib.pagesizes import letter   # US Letter = 8.5 x 11 inches
from reportlab.lib import colors             # color constants and HexColor
from reportlab.lib.units import inch         # lets us write 0.5*inch instead of raw points

# ── File paths ─────────────────────────────────────────────────────────────────
# __file__ is this script's location; .parent.parent goes up two levels to project root
BASE_DIR    = Path(__file__).resolve().parent.parent
CLAIMS_JSON = BASE_DIR / "data" / "raw_claims" / "claims.json"  # input
PDF_DIR     = BASE_DIR / "data" / "pdfs"                         # output folder
PDF_DIR.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

# ── Page setup ─────────────────────────────────────────────────────────────────
W, H = letter   # W=612 points, H=792 points (1 point = 1/72 inch)

# ── Color palette (matching the CMS-1500 form style) ──────────────────────────
NAVY      = colors.HexColor("#1A3A5C")   # dark blue — header background, section lines
LGRAY     = colors.HexColor("#F4F4F4")   # light gray — alternating row backgrounds
MGRAY     = colors.HexColor("#CCCCCC")   # medium gray — box borders and dividers
DGRAY     = colors.HexColor("#555555")   # dark gray — small field labels
FRAUD_RED = colors.HexColor("#C0392B")   # red — fraud banner and flagged text
LEGIT_GRN = colors.HexColor("#1E8449")   # green — legitimate claim banner
WARN_BG   = colors.HexColor("#FADBD8")   # light pink — "why flagged" explanation box

# ── Layout margins and constants ───────────────────────────────────────────────
M_LEFT  = 0.35 * inch   # left margin
M_RIGHT = W - 0.35 * inch  # right margin
CW      = M_RIGHT - M_LEFT  # total usable content width

LSIZE = 5.5   # font size for field labels (small text at top of each box)
VSIZE = 8.0   # font size for field values (main readable content)
ROW_H = 0.40 * inch  # standard row height — tall enough for label + value without overlap


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION: labeled_box
# Draws a single labeled field box — the building block of the form
# ══════════════════════════════════════════════════════════════════════════════

def labeled_box(c, x, y, w, h, label, value,
                vs=VSIZE, bold=False, bg=None, vc=colors.black, truncate=True):
    """
    Draws a rectangular field with:
      - A small gray label pinned to the TOP of the box  (field name)
      - A larger value pinned to the BOTTOM of the box   (field content)

    This two-level layout ensures the label and value NEVER overlap,
    because the label is at y-8 and the value is at y-h+8, with h=0.40in (~29pt).
    Combined label+value height is ~14pt, leaving 15pt of breathing room.

    Parameters:
        c        — the ReportLab canvas to draw on
        x, y     — top-left corner of the box (y is TOP edge in ReportLab)
        w, h     — width and height of the box
        label    — small text at top (e.g. "2. PATIENT NAME")
        value    — main content at bottom (e.g. "Frank Jackson")
        vs       — value font size (default 8)
        bold     — whether the value is bold
        bg       — background color (None = white/transparent)
        vc       — value text color
        truncate — auto-shorten text so it never overflows the box width
    """
    # Draw the box border (and fill background if specified)
    c.setStrokeColor(MGRAY)
    c.setLineWidth(0.4)
    if bg:
        c.setFillColor(bg)
        c.rect(x, y - h, w, h, fill=1, stroke=1)
    else:
        c.rect(x, y - h, w, h, fill=0, stroke=1)

    # Draw the field label — pinned 3pt from top edge and 3pt from left edge
    c.setFillColor(DGRAY)
    c.setFont("Helvetica", LSIZE)
    c.drawString(x + 3, y - LSIZE - 3, label)

    # Draw the field value — pinned 8pt from bottom edge
    if value:
        c.setFillColor(vc)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", vs)
        val_str = str(value)
        if truncate:
            # Calculate max characters that fit in this column width
            # 0.58 is an approximation for Helvetica character width ratio
            max_chars = max(1, int((w - 8) / (vs * 0.58)))
            val_str = val_str[:max_chars]
        c.drawString(x + 4, y - h + 8, val_str)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION: hline
# Draws a horizontal divider line between sections
# ══════════════════════════════════════════════════════════════════════════════

def hline(c, x1, x2, y, w=0.5, col=MGRAY):
    """Draws a horizontal line from (x1,y) to (x2,y)."""
    c.setStrokeColor(col)
    c.setLineWidth(w)
    c.line(x1, y, x2, y)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: render_claim
# Takes one claim dict and draws the full CMS-1500 form as a PDF
# ══════════════════════════════════════════════════════════════════════════════

def render_claim(claim, out_path):
    """
    Renders a single claim as a CMS-1500 PDF.

    The CMS-1500 form has these sections (top to bottom):
      1. Header banner          — form title and claim ID
      2. Insurance info         — insurance type + insured ID
      3. Patient info           — name, DOB, sex, address
      4. Insurance plan info    — plan name, group, policy number
      5. Provider info          — physician name, facility, NPI
      6. Diagnosis section      — ICD-10 codes and descriptions
      7. Procedure table        — up to 6 lines of CPT codes + charges
      8. Totals row             — total charge, amount paid
      9. Signature row          — physician signature + billing info
     10. Fraud verdict banner   — green (legitimate) or red (fraud)
     11. Why flagged box        — explanation of the fraud (fraud claims only)
     12. Footer                 — form ID and attribution

    Parameters:
        claim    — a dict from claims.json with all claim fields
        out_path — where to save the PDF file
    """
    # Create a new PDF canvas — this is the drawing surface
    c = canvas.Canvas(str(out_path), pagesize=letter)

    # y tracks our current vertical position as we draw downward
    # In ReportLab, y=0 is at the BOTTOM of the page, so we start near the top
    y = H - 0.2 * inch

    # ── SECTION 1: HEADER BANNER ───────────────────────────────────────────────
    # Dark navy rectangle spanning the full width at the top
    HDR_H = 0.50 * inch
    c.setFillColor(NAVY)
    c.rect(M_LEFT, y - HDR_H, CW, HDR_H, fill=1, stroke=0)

    # White bold title text
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(M_LEFT + 8, y - 18, "HEALTH INSURANCE CLAIM FORM")

    # Subtitle in light blue
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#AED6F1"))
    c.drawString(M_LEFT + 8, y - 30,
        "APPROVED BY NATIONAL UNIFORM CLAIM COMMITTEE (NUCC)  |  CMS-1500 (02-12)")

    # Claim ID and date in top-right corner
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(M_RIGHT - 6, y - 18, f"CLAIM ID: {claim['claim_id']}")
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#AED6F1"))
    c.drawRightString(M_RIGHT - 6, y - 30, f"DATE: {claim['date']}")

    y -= HDR_H + 0.06 * inch  # move cursor below the header

    # ── SECTION 2: INSURANCE TYPE + INSURED ID ────────────────────────────────
    # Two boxes side by side — insurance type on left (55%), insured ID on right (45%)
    w1 = CW * 0.55
    labeled_box(c, M_LEFT,    y, w1,    ROW_H,
                "1. INSURANCE TYPE", "☑ OTHER / PRIVATE INSURANCE", bg=LGRAY)
    labeled_box(c, M_LEFT+w1, y, CW-w1, ROW_H,
                "1a. INSURED'S I.D. NUMBER", claim["patient"]["policy_no"], bold=True)
    y -= ROW_H

    # ── SECTION 3: PATIENT INFO ROW ───────────────────────────────────────────
    # Four boxes: Name (38%) | DOB (17%) | Sex (9%) | Insured Name (36%)
    a, b, cc = CW*0.38, CW*0.17, CW*0.09
    d = CW - a - b - cc  # remaining width for insured name
    labeled_box(c, M_LEFT,        y, a,  ROW_H,
                "2. PATIENT NAME (Last, First, MI)", claim["patient"]["name"], bold=True)
    labeled_box(c, M_LEFT+a,      y, b,  ROW_H,
                "3. DATE OF BIRTH", claim["patient"]["dob"])
    labeled_box(c, M_LEFT+a+b,    y, cc, ROW_H,
                "SEX", claim["patient"]["sex"], bold=True)
    labeled_box(c, M_LEFT+a+b+cc, y, d,  ROW_H,
                "4. INSURED'S NAME", claim["patient"]["name"])
    y -= ROW_H

    # ── SECTION 3b: ADDRESS ROW ───────────────────────────────────────────────
    # Two boxes: patient address (55%) | insured address (45%)
    w1 = CW * 0.55
    labeled_box(c, M_LEFT,    y, w1,    ROW_H,
                "5. PATIENT ADDRESS", claim["patient"]["address"][:52])
    labeled_box(c, M_LEFT+w1, y, CW-w1, ROW_H,
                "7. INSURED ADDRESS", claim["patient"]["address"][:38])
    y -= ROW_H

    # ── SECTION 4: INSURANCE PLAN INFO ────────────────────────────────────────
    # Three boxes: Insurance plan name (50%) | Group no. (28%) | Policy no. (22%)
    p1, p2, p3 = CW*0.50, CW*0.28, CW*0.22
    labeled_box(c, M_LEFT,       y, p1, ROW_H,
                "11c. INSURANCE PLAN OR PROGRAM NAME",
                claim["patient"]["insurer"], bold=True, bg=LGRAY)
    labeled_box(c, M_LEFT+p1,    y, p2, ROW_H,
                "11. POLICY GROUP NUMBER", claim["patient"]["group_no"], bg=LGRAY)
    labeled_box(c, M_LEFT+p1+p2, y, p3, ROW_H,
                "POLICY NO.", claim["patient"]["policy_no"], bg=LGRAY)
    y -= ROW_H + 0.10 * inch

    # ── SECTION 5: PROVIDER INFO ──────────────────────────────────────────────
    # Navy divider line to visually separate patient info from provider info
    hline(c, M_LEFT, M_RIGHT, y + 0.04*inch, w=1.2, col=NAVY)

    # Three boxes: Physician name (46%) | Facility name (35%) | NPI number (19%)
    pr1, pr2, pr3 = CW*0.46, CW*0.35, CW*0.19
    labeled_box(c, M_LEFT,         y, pr1, ROW_H,
                "17. TREATING PHYSICIAN", claim["provider"]["name"], bold=True)
    labeled_box(c, M_LEFT+pr1,     y, pr2, ROW_H,
                "32. SERVICE FACILITY", claim["provider"]["facility"])
    labeled_box(c, M_LEFT+pr1+pr2, y, pr3, ROW_H,
                "NPI", claim["provider"]["npi"][:12])  # NPI is 10 digits, truncate for safety
    y -= ROW_H + 0.10 * inch

    # ── SECTION 6: DIAGNOSIS CODES ────────────────────────────────────────────
    # This is the KEY section for fraud detection:
    # ICD-10 codes tell us WHY the patient was seen.
    # If the CPT procedure codes don't match these diagnoses → fraud signal.
    hline(c, M_LEFT, M_RIGHT, y + 0.04*inch, w=1.2, col=NAVY)

    DG_H = 0.52 * inch  # taller box to fit label + code + description on 3 separate lines
    c.setFillColor(LGRAY)
    c.setStrokeColor(MGRAY)
    c.setLineWidth(0.4)
    c.rect(M_LEFT, y - DG_H, CW, DG_H, fill=1, stroke=1)

    # Field label at top of box
    c.setFillColor(DGRAY)
    c.setFont("Helvetica", LSIZE)
    c.drawString(M_LEFT + 4, y - LSIZE - 3,
        "21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY (ICD-10-CM)")

    # ICD-10 code(s) on second line — bold navy
    diag_codes = claim.get("diagnosis_codes", [])
    diag_descs = claim.get("diagnosis_descs", [])
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(M_LEFT + 4, y - LSIZE - 16, "  |  ".join(diag_codes))

    # Diagnosis description(s) on third line — smaller gray text
    c.setFillColor(DGRAY)
    c.setFont("Helvetica", 7.5)
    desc_str = "  |  ".join([d[:55] for d in diag_descs])
    c.drawString(M_LEFT + 4, y - LSIZE - 27, desc_str)

    y -= DG_H + 0.10 * inch

    # ── SECTION 7: PROCEDURE TABLE ────────────────────────────────────────────
    # This is the CORE of the claim — each line is one billed procedure.
    # CPT codes here are cross-referenced against the ICD-10 codes above
    # to detect fraud patterns like diagnosis mismatch, phantom billing, etc.

    # Define table columns: (name, fraction of total width)
    COLS = [
        ("DATE",        0.11),   # date of service
        ("POS",         0.05),   # place of service (11 = office)
        ("CPT/HCPCS",   0.13),   # procedure code — THE KEY FRAUD DETECTION FIELD
        ("DESCRIPTION", 0.27),   # human-readable procedure name
        ("MODIFIER",    0.08),   # billing modifier (e.g. -59 = separate procedure)
        ("DIAG PTR",    0.08),   # which diagnosis (A,B,C...) justifies this procedure
        ("$ CHARGES",   0.13),   # amount billed — used to detect upcoding
        ("UNITS",       0.06),   # how many times the procedure was performed
        ("NPI",         0.09),   # rendering provider NPI number
    ]
    col_pts = [(n, f * CW) for n, f in COLS]  # convert fractions to actual points

    # Draw the table header row (dark navy background, white text)
    HDR2_H = 0.22 * inch
    c.setFillColor(NAVY)
    c.rect(M_LEFT, y - HDR2_H, CW, HDR2_H, fill=1, stroke=0)
    cx = M_LEFT
    for name, cw in col_pts:
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 5.5)
        c.drawString(cx + 3, y - HDR2_H + 7, name)
        # Draw vertical divider between columns
        if cx + cw < M_RIGHT - 1:
            c.setStrokeColor(colors.HexColor("#4A6A8A"))
            c.setLineWidth(0.3)
            c.line(cx + cw, y, cx + cw, y - HDR2_H)
        cx += cw
    y -= HDR2_H

    # Pull procedure data from the claim
    proc_codes   = claim.get("procedure_codes", [])   # list of CPT codes
    proc_descs   = claim.get("procedure_descs", [])   # list of descriptions
    line_charges = claim.get("line_charges",   [])    # list of dollar amounts
    modifiers    = claim.get("modifiers",      [])    # list of modifiers (e.g. "-59")
    n_lines      = min(len(proc_codes), 6)             # CMS-1500 supports max 6 lines
    LINE_H       = 0.30 * inch

    # Draw 6 rows (filled with data if available, empty if not)
    for li in range(6):
        bg = LGRAY if li % 2 == 0 else colors.white  # alternating row colors
        c.setFillColor(bg)
        c.setStrokeColor(MGRAY)
        c.setLineWidth(0.3)
        c.rect(M_LEFT, y - LINE_H, CW, LINE_H, fill=1, stroke=1)

        if li < n_lines:
            # Build the row values for this procedure line
            npi_short = claim["provider"]["npi"][:11]  # truncate NPI to fit column
            row_vals = [
                claim["date"],                                          # DATE
                "11",                                                   # POS (office)
                proc_codes[li] if li < len(proc_codes) else "",        # CPT CODE
                proc_descs[li][:36] if li < len(proc_descs) else "",   # DESCRIPTION
                modifiers[li] if li < len(modifiers) else "",          # MODIFIER
                "A",                                                    # DIAG POINTER
                f"${line_charges[li]:,.2f}" if li < len(line_charges) else "",  # CHARGES
                "1",                                                    # UNITS
                npi_short,                                              # NPI
            ]

            cx = M_LEFT
            for vi, (val, (col_name, cw)) in enumerate(zip(row_vals, col_pts)):
                # Highlight CPT codes in navy blue and charges in red for visual emphasis
                fc = NAVY if vi==2 else (FRAUD_RED if vi==6 else colors.black)
                fb = vi in [2, 6]  # bold for CPT code and charges columns
                c.setFillColor(fc)
                c.setFont("Helvetica-Bold" if fb else "Helvetica", 7.5)
                # Truncate value to fit within column width
                max_c = max(1, int((cw - 6) / (7.5 * 0.58)))
                c.drawString(cx + 3, y - LINE_H + 10, str(val)[:max_c])
                # Vertical divider between columns
                if cx + cw < M_RIGHT - 1:
                    c.setStrokeColor(MGRAY)
                    c.setLineWidth(0.3)
                    c.line(cx + cw, y, cx + cw, y - LINE_H)
                cx += cw
        y -= LINE_H

    # Heavy navy line below the procedure table
    y -= 0.08 * inch
    hline(c, M_LEFT, M_RIGHT, y, w=1.2, col=NAVY)
    y -= 0.08 * inch

    # ── SECTION 8: TOTALS ROW ─────────────────────────────────────────────────
    # Shows the total charge — used in our upcoding detection to compare
    # against the CMS Medicare fee schedule (is this amount reasonable?)
    TOT_H = 0.42 * inch  # tall enough for label on top + value on bottom
    c.setFillColor(LGRAY)
    c.setStrokeColor(MGRAY)
    c.setLineWidth(0.4)
    c.rect(M_LEFT, y - TOT_H, CW, TOT_H, fill=1, stroke=1)

    total = claim.get("total_charge", 0)

    # Tax ID — left section
    c.setFillColor(DGRAY); c.setFont("Helvetica", LSIZE)
    c.drawString(M_LEFT + 4, y - LSIZE - 3, "25. FEDERAL TAX I.D. NUMBER")
    c.setFillColor(colors.black); c.setFont("Helvetica", 8)
    c.drawString(M_LEFT + 4, y - TOT_H + 9, claim["provider"]["tax_id"])

    # Total charge — middle section (large, navy blue for emphasis)
    tc_x = M_LEFT + CW * 0.48
    c.setFillColor(DGRAY); c.setFont("Helvetica", LSIZE)
    c.drawString(tc_x, y - LSIZE - 3, "28. TOTAL CHARGE")
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 13)
    c.drawString(tc_x, y - TOT_H + 9, f"${total:,.2f}")

    # Amount paid — right section
    ap_x = M_LEFT + CW * 0.74
    c.setFillColor(DGRAY); c.setFont("Helvetica", LSIZE)
    c.drawString(ap_x, y - LSIZE - 3, "29. AMOUNT PAID")
    c.setFillColor(colors.black); c.setFont("Helvetica", 8)
    c.drawString(ap_x, y - TOT_H + 9, "$0.00")

    y -= TOT_H

    # ── SECTION 9: SIGNATURE ROW ──────────────────────────────────────────────
    # Fields 31 (physician signature) and 33 (billing provider info)
    SIG_H = 0.38 * inch  # tall enough for label + value without overlap
    s1 = CW * 0.60       # field 31 takes 60% of the width

    c.setFillColor(colors.white); c.setStrokeColor(MGRAY); c.setLineWidth(0.4)
    c.rect(M_LEFT,      y - SIG_H, s1,    SIG_H, fill=1, stroke=1)  # field 31 box
    c.rect(M_LEFT + s1, y - SIG_H, CW-s1, SIG_H, fill=1, stroke=1)  # field 33 box

    # Field 31 — physician signature
    c.setFillColor(DGRAY); c.setFont("Helvetica", LSIZE)
    c.drawString(M_LEFT + 4, y - LSIZE - 3,
                 "31. SIGNATURE OF PHYSICIAN OR SUPPLIER")
    c.setFillColor(colors.black); c.setFont("Helvetica", 7.5)
    c.drawString(M_LEFT + 4, y - SIG_H + 9,
        f"Signed: {claim['provider']['name']}    Date: {claim['date']}")

    # Field 33 — billing provider
    c.setFillColor(DGRAY); c.setFont("Helvetica", LSIZE)
    c.drawString(M_LEFT + s1 + 4, y - LSIZE - 3,
                 "33. BILLING PROVIDER INFO & PH #")
    c.setFillColor(colors.black); c.setFont("Helvetica", 7.5)
    c.drawString(M_LEFT + s1 + 4, y - SIG_H + 9,
        claim["provider"]["facility"])

    y -= SIG_H + 0.12 * inch

    # ── SECTION 10: FRAUD VERDICT BANNER ──────────────────────────────────────
    # This shows the fraud detection result:
    #   Green banner = legitimate claim (no fraud detected)
    #   Red banner   = fraud detected, with fraud type shown
    # In the real pipeline, this banner is generated AFTER the fraud engine runs.
    # In our synthetic PDFs, we pre-fill it from the ground truth labels.
    is_fraud   = claim.get("fraud_indicator", False)
    fraud_type = claim.get("fraud_type", "Legitimate")
    BNR_H = 0.32 * inch

    # Rounded rectangle in green or red
    bc = FRAUD_RED if is_fraud else LEGIT_GRN
    c.setFillColor(bc)
    c.roundRect(M_LEFT, y - BNR_H, CW, BNR_H, 5, fill=1, stroke=0)

    # Banner text
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 10)
    icon  = "⚠" if is_fraud else "✓"
    label = f"  FRAUD DETECTED — {fraud_type.upper()}" if is_fraud \
            else "  LEGITIMATE CLAIM — No Fraud Detected"
    c.drawString(M_LEFT + 10, y - BNR_H + 10, f"{icon}{label}")
    y -= BNR_H

    # ── SECTION 11: WHY FLAGGED BOX (fraud claims only) ───────────────────────
    # Shows a plain-English explanation of exactly what fraud was detected.
    # This is what makes our system actionable — not just "fraud/not fraud"
    # but a specific reason an insurance reviewer can act on immediately.
    if is_fraud and claim.get("fraud_explanation"):
        y -= 0.06 * inch

        # Split long explanations across two lines so text never overflows the box
        exp_full = claim["fraud_explanation"]
        max_line = 105  # approx max characters per line at font size 7.5

        if len(exp_full) <= max_line:
            lines = [exp_full]  # short explanation — fits on one line
        else:
            # Find the last space before the line limit to break cleanly
            split_at = exp_full.rfind(" ", 0, max_line)
            if split_at == -1:
                split_at = max_line  # no space found — hard cut
            lines = [exp_full[:split_at], exp_full[split_at+1:max_line*2]]

        # Box height scales dynamically: base height + 0.14in per line of text
        # The 0.18in base gives top padding for the label + bottom breathing room
        EXP_H = (0.18 + 0.14 * len(lines)) * inch

        # Light pink box with red border
        c.setFillColor(WARN_BG); c.setStrokeColor(FRAUD_RED); c.setLineWidth(0.6)
        c.rect(M_LEFT, y - EXP_H, CW, EXP_H, fill=1, stroke=1)

        # "WHY FLAGGED:" label at top of box
        c.setFillColor(DGRAY); c.setFont("Helvetica-Bold", LSIZE)
        c.drawString(M_LEFT + 4, y - LSIZE - 2, "WHY FLAGGED:")

        # Explanation text — each line gets its own y position
        c.setFillColor(FRAUD_RED); c.setFont("Helvetica", 7.5)
        for li, line in enumerate(lines):
            # First line starts just below the label; subsequent lines 10pt lower
            line_y = y - LSIZE - 2 - 10 - (li * 10)
            c.drawString(M_LEFT + 4, line_y, line)

        y -= EXP_H

    # ── SECTION 12: FOOTER ────────────────────────────────────────────────────
    # Standard CMS-1500 footer with form reference and project attribution
    fy = 0.18 * inch
    hline(c, M_LEFT, M_RIGHT, fy + 0.14*inch, w=0.5, col=NAVY)
    c.setFillColor(DGRAY); c.setFont("Helvetica", 5.5)
    c.drawString(M_LEFT, fy,
        "NUCC Instruction Manual: www.nucc.org  |  Cornell University — Fraud Detection System  |  "
        "APPROVED OMB-0938-1197 FORM 1500 (02-12)  |  CONFIDENTIAL")

    # Save the completed PDF to disk
    c.save()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — runs when you execute: python3 src/render_claims.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load all 1,300 claims from the JSON file generated in Step 1
    with open(CLAIMS_JSON) as f:
        claims = json.load(f)

    print(f"\nRendering {len(claims)} claims to PDF...")
    print(f"Output folder: {PDF_DIR}\n")

    errors = 0
    for i, claim in enumerate(claims):
        try:
            # Build output path: e.g. data/pdfs/CLM01093.pdf
            out_path = PDF_DIR / f"{claim['claim_id']}.pdf"
            render_claim(claim, out_path)

            # Print progress every 100 claims
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(claims)} done...")
                sys.stdout.flush()  # ensure output appears immediately in terminal

        except Exception as e:
            errors += 1
            print(f"[ERROR] {claim['claim_id']}: {e}")

    print(f"\n✅  Done!")
    print(f"   PDFs generated : {len(claims) - errors}")
    print(f"   Errors         : {errors}")
    print(f"   Output folder  : {PDF_DIR}")
    print(f"\nNext step: run src/extract_fields.py (OCR pipeline)")
