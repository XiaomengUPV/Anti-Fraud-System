"""
extract_fields.py
─────────────────
Step 3B of the Fraud Detection Pipeline — OCR Field Extractor.

PURPOSE:
    Reads each CMS-1500 PDF using OCR (Optical Character Recognition)
    and extracts the key fields needed for fraud detection:
    CPT codes, ICD-10 diagnosis codes, charges, modifiers, patient info.

WHY OCR:
    Some claims arrive as scanned PDFs from small clinics or paper forms.
    OCR converts the PDF image into text, then we parse out the specific
    fields. This is slower than EDI parsing but handles any PDF document.

OUTPUT FORMAT — Normalized Claim Object:
    Both this OCR path AND the EDI parser produce the same structure:
    {
        "claim_id":         "CLM01093",
        "date":             "2025-09-21",
        "patient_name":     "Frank Jackson",
        "provider_name":    "Dr. Emily Nguyen",
        "facility":         "Sunrise Health Associates",
        "insurer":          "Cigna Health",
        "policy_no":        "POL346961",
        "procedure_codes":  ["87491"],
        "diagnosis_codes":  ["Z03.89"],
        "modifiers":        [],
        "line_charges":     [107.89],
        "total_charge":     107.89,
        "source":           "pdf_ocr"
    }

INPUT:  data/pdfs/<claim_id>.pdf
OUTPUT: data/ocr_output/<claim_id>.json

REQUIREMENTS:
    pip install pytesseract Pillow pdf2image
    brew install tesseract poppler

HOW TO RUN:
    python3 src/extract_fields.py
"""

import json, re, sys, os
from pathlib import Path

# Conditional imports — require installation
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not installed. Run:")
    print("  pip install pytesseract Pillow pdf2image")
    print("  brew install tesseract poppler")

BASE_DIR    = Path(__file__).resolve().parent.parent
PDF_DIR     = BASE_DIR / "data" / "pdfs"
OCR_DIR     = BASE_DIR / "data" / "ocr_output"
OCR_DIR.mkdir(parents=True, exist_ok=True)


def pdf_to_image(pdf_path):
    """
    Converts the first page of a PDF to a PIL Image at 300 DPI.
    300 DPI gives the best balance of OCR accuracy vs speed.
    Higher DPI = better accuracy but slower.
    """
    try:
        images = convert_from_path(str(pdf_path), dpi=300)
        return images[0] if images else None
    except Exception as e:
        print(f"  [PDF→IMAGE ERROR] {pdf_path.name}: {e}")
        return None


def image_to_text(image):
    """
    Runs Tesseract OCR on a PIL image and returns the raw extracted text.
    --psm 6 = treat image as a single uniform block of text (good for forms).
    --oem 3 = use both legacy and LSTM neural net engines.
    """
    if image is None:
        return ""
    try:
        custom_config = r"--oem 3 --psm 6"
        return pytesseract.image_to_string(image, config=custom_config)
    except Exception as e:
        print(f"  [OCR ERROR]: {e}")
        return ""


def extract_cpt_codes(text):
    """
    Extracts all CPT/HCPCS procedure codes from the OCR text.
    These are the KEY fraud detection signals.

    Patterns matched:
      - 5-digit numeric CPT:  99213, 85025, 71046
      - HCPCS G-codes:        G0103, G0101
      - M-codes (lab):        0006M, 0019M
      - U-codes (lab):        0026U, 0048U
    """
    codes = []

    # Standard 5-digit CPT codes
    codes.extend(re.findall(r"\b(\d{5})\b", text))

    # HCPCS G-codes
    codes.extend([c.upper() for c in re.findall(r"\b(G\d{4})\b", text, re.IGNORECASE)])

    # M-codes (multianalyte lab assays)
    codes.extend([c.upper() for c in re.findall(r"\b(\d{4}M)\b", text, re.IGNORECASE)])

    # U-codes (proprietary lab analyses)
    codes.extend([c.upper() for c in re.findall(r"\b(\d{4}U)\b", text, re.IGNORECASE)])

    # Deduplicate while preserving order, filter out year-like numbers
    seen = set()
    result = []
    for code in codes:
        if code.isdigit() and 1900 <= int(code) <= 2099:
            continue  # skip years (2025, 2026 etc.)
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def extract_icd10_codes(text):
    """
    Extracts ICD-10-CM diagnosis codes from the OCR text.
    ICD-10 format: one letter + 2 digits + optional .suffix
    Examples: I10, E11.9, Z00.00, J06.9, M54.5
    """
    # Pattern: letter + 2 digits + optional decimal + up to 4 chars
    codes = re.findall(r"\b([A-Z]\d{2}(?:\.\w{1,4})?)\b", text, re.IGNORECASE)

    # Deduplicate and normalize to uppercase
    seen = set()
    result = []
    for code in codes:
        code_upper = code.upper()
        if code_upper[0].isalpha() and code_upper not in seen:
            seen.add(code_upper)
            result.append(code_upper)
    return result


def extract_total_charge(text):
    """
    Finds the total charge amount — used in upcoding detection
    to compare against the CMS Medicare fee schedule.
    """
    # Look for 'TOTAL CHARGE' label followed by a dollar amount
    match = re.search(r"TOTAL\s+CHARGE[:\s]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))

    # Fallback: find all dollar amounts and return the largest
    amounts = re.findall(r"\$([\d,]+\.\d{2})", text)
    if amounts:
        try:
            return max(float(a.replace(",", "")) for a in amounts)
        except ValueError:
            pass
    return 0.0


def extract_modifiers(text):
    """
    Extracts billing modifiers — critical for modifier abuse detection.
    Modifier -59 is the most important fraud signal (unbundling).
    """
    modifiers = []
    if re.search(r"-59\b|modifier.*59|\b59\b.*modifier", text, re.IGNORECASE):
        modifiers.append("-59")
    return modifiers


def extract_date(text):
    """Finds the service date in YYYY-MM-DD format."""
    match = re.search(r"DATE[:\s]+(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    return match.group(1) if match else None


def build_normalized_claim(text, pdf_path):
    """
    Assembles all extracted fields into the normalized claim object.
    This is the SAME structure produced by the EDI parser — so the fraud
    engine works identically regardless of whether input was PDF or EDI.
    """
    return {
        "claim_id":        pdf_path.stem,              # from filename e.g. CLM01093
        "date":            extract_date(text),
        "procedure_codes": extract_cpt_codes(text),    # KEY fraud detection field
        "diagnosis_codes": extract_icd10_codes(text),  # KEY fraud detection field
        "modifiers":       extract_modifiers(text),    # KEY for modifier abuse
        "total_charge":    extract_total_charge(text), # KEY for upcoding detection
        "line_charges":    [],                         # harder to OCR reliably
        "raw_ocr_text":    text[:500],                 # first 500 chars for debugging
        "source":          "pdf_ocr",                  # tells fraud engine input type
    }


def process_pdf(pdf_path):
    """
    Full pipeline for one PDF:
    PDF → Image → OCR text → Structured fields → Normalized claim dict
    """
    image = pdf_to_image(pdf_path)
    if image is None:
        return None
    text = image_to_text(image)
    if not text.strip():
        return None
    return build_normalized_claim(text, pdf_path)


if __name__ == "__main__":
    if not OCR_AVAILABLE:
        print("\nCannot run — install dependencies first:")
        print("  pip install pytesseract Pillow pdf2image")
        print("  brew install tesseract poppler")
        sys.exit(1)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}. Run render_claims.py first.")
        sys.exit(1)

    print(f"\nRunning OCR on {len(pdf_files)} PDFs...")
    print(f"Expected time: ~{len(pdf_files)*2//60} minutes\n")

    success = errors = 0
    for i, pdf_path in enumerate(pdf_files):
        try:
            claim = process_pdf(pdf_path)
            if claim:
                out_path = OCR_DIR / f"{pdf_path.stem}.json"
                with open(out_path, "w") as f:
                    json.dump(claim, f, indent=2)
                success += 1
            else:
                errors += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pdf_files)} done...")
                sys.stdout.flush()
        except Exception as e:
            errors += 1
            print(f"[ERROR] {pdf_path.name}: {e}")

    print(f"\nDone! {success} extracted, {errors} errors.")
    print(f"Output: {OCR_DIR}")
    print(f"Next: run src/parse_edi.py")
