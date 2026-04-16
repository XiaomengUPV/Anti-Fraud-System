"""
extract_fields.py  — VERSION 2 (OCR Accuracy Improvements)
──────────────────────────────────────────────────────────
Step 3B of the Fraud Detection Pipeline — OCR Field Extractor.

IMPROVEMENTS OVER V1:
─────────────────────
This version addresses the OCR accuracy gap identified in Session 4 testing.
The original OCR pipeline achieved F1 = 0.550 vs EDI pipeline F1 = 0.763,
a gap of 0.213 caused entirely by OCR errors in field extraction.

Three targeted improvements were made:

IMPROVEMENT A — Higher DPI (300 → 400):
    Tesseract accuracy improves significantly with higher resolution input.
    At 300 DPI, narrow table columns in CMS-1500 forms produce blurry
    character images that Tesseract misreads. At 400 DPI the same characters
    are sharper and more distinct.
    Expected improvement: +5 to +8 F1 points.

IMPROVEMENT B — Better Tesseract configuration:
    Original: --psm 6 (uniform block of text — treats entire page as one block)
    New:      --psm 6 primary, with --psm 4 fallback (single column of text)
              Also added --oem 1 to force the LSTM neural net engine which is
              more accurate on printed forms than the legacy engine.
    Expected improvement: +3 to +5 F1 points.

IMPROVEMENT C — Image pre-processing before OCR:
    CMS-1500 forms use light grey borders and small fonts that Tesseract
    struggles with at normal contrast. We apply:
    - Sharpness enhancement (2.0x) — makes character edges crisper
    - Contrast enhancement (1.5x) — darkens text relative to background
    - Convert to greyscale — removes any colour noise
    This is standard practice in production OCR pipelines for medical forms.
    Expected improvement: +5 to +10 F1 points.

IMPROVEMENTS CARRIED FORWARD FROM V1:
    - Zip code filter (e.g. IL 35405 not treated as CPT code)
    - ICD-10 I-code OCR fix (I25.10 ← 125.10)
    - ICD-10 O-code OCR fix (O09.90 ← 009.90)
    - Dollar amount protection before ICD fixes

TOTAL EXPECTED IMPROVEMENT: F1 0.550 → ~0.65–0.70

INPUT:  data/pdfs/<claim_id>.pdf
OUTPUT: data/ocr_output/<claim_id>.json

HOW TO RUN:
    python3 src/extract_fields.py
"""

import json, re, sys
from pathlib import Path

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not installed. Run:")
    print("  pip install pytesseract Pillow pdf2image")
    print("  brew install tesseract poppler")

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR  = BASE_DIR / "data" / "pdfs"
OCR_DIR  = BASE_DIR / "data" / "ocr_output"
OCR_DIR.mkdir(parents=True, exist_ok=True)


def pdf_to_image(pdf_path):
    """
    Converts first page of PDF to PIL Image.

    IMPROVEMENT A: DPI increased from 300 to 400.
    At 400 DPI, each character in a small-font table cell is rendered
    with ~33% more pixels, giving Tesseract significantly more detail
    to work with. The tradeoff is ~40% longer OCR time per page, which
    is acceptable given the accuracy gains.
    """
    try:
        images = convert_from_path(str(pdf_path), dpi=400)   # ← was 300
        return images[0] if images else None
    except Exception as e:
        print(f"  [PDF→IMAGE ERROR] {pdf_path.name}: {e}")
        return None


def preprocess_image(image):
    """
    IMPROVEMENT C: Image pre-processing pipeline before OCR.

    Step 1 — Greyscale conversion:
        Colour information is irrelevant for text extraction and can
        introduce noise. Greyscale reduces the image to pure intensity
        values that Tesseract handles more reliably.

    Step 2 — Sharpness enhancement (2.0x):
        CMS-1500 forms rendered from PDF often have slightly soft edges
        on characters due to anti-aliasing. Sharpening at 2.0x restores
        crisp character boundaries that Tesseract uses to distinguish
        similar-looking characters (I vs 1, O vs 0, S vs 5).

    Step 3 — Contrast enhancement (1.5x):
        The form uses light grey for field borders and column dividers.
        Enhancing contrast at 1.5x darkens text relative to the background,
        making it easier for Tesseract to segment characters from the page.
        We use 1.5x rather than a higher value to avoid over-darkening the
        grey table borders into black, which would confuse the layout parser.
    """
    # Step 1: Convert to greyscale
    image = image.convert("L")

    # Step 2: Sharpen edges
    image = ImageEnhance.Sharpness(image).enhance(2.0)

    # Step 3: Increase contrast
    image = ImageEnhance.Contrast(image).enhance(1.5)

    return image


def image_to_text(image):
    """
    Runs Tesseract OCR on a pre-processed PIL image.

    IMPROVEMENT B: Better Tesseract configuration.

    --oem 1: Force LSTM neural net engine only.
        The default --oem 3 uses both legacy and LSTM engines and picks
        the best result. For printed medical forms, the LSTM engine
        consistently outperforms the legacy engine, so we force it.

    --psm 6: Uniform block of text (primary).
        Tells Tesseract to treat the page as a single uniform block.
        This is the best starting config for full-page form extraction.

    We try --psm 6 first. If it returns fewer than 50 characters
    (suggesting it failed to read the page), we fall back to --psm 4
    (single column of text) which sometimes works better on forms
    with a clear left-to-right reading order.
    """
    if image is None:
        return ""
    try:
        # Primary attempt: LSTM engine, uniform block
        config_primary  = "--oem 1 --psm 6"
        text = pytesseract.image_to_string(image, config=config_primary)

        # Fallback: if primary got almost nothing, try column mode
        if len(text.strip()) < 50:
            config_fallback = "--oem 1 --psm 4"
            text_fallback   = pytesseract.image_to_string(image, config=config_fallback)
            if len(text_fallback.strip()) > len(text.strip()):
                text = text_fallback

        return text
    except Exception as e:
        print(f"  [OCR ERROR]: {e}")
        return ""


def fix_ocr_errors(text):
    """
    Corrects common Tesseract misreads in medical billing codes.

    CARRIED FORWARD FROM V1 — no changes here.

    Protect dollar amounts first, then fix character substitutions:
    - Capital I → digit 1 in ICD-10 codes (I25.10 ← 125.10)
    - Capital O → digit 0 in ICD-10 codes (O09.90 ← 009.90)
    - Capital S → digit 5 in injury codes (S72.001A ← 572.001A)
    """
    # Protect dollar amounts before character substitutions
    dollar_amounts = re.findall(r'\$[\d,]+\.\d{2}', text)
    for i, amt in enumerate(dollar_amounts):
        text = text.replace(amt, f'DOLLAR{i}PLACEHOLDER', 1)

    # Fix ICD-10 I-codes: 125.10 → I25.10, 110 → I10
    text = re.sub(r'\b1(\d{2}\.\w{1,4})\b', r'I\1', text)
    text = re.sub(r'(?<!\d)1(\d{2})(?!\d)', r'I\1', text)

    # Fix ICD-10 O-codes: 009.90 → O09.90
    text = re.sub(r'\b0(\d{2}\.\w{1,4})\b', r'O\1', text)

    # Fix S-codes with letter suffix: 572.001A → S72.001A
    text = re.sub(r'\b5(\d{2}\.\w{1,3}[A-Z])\b', r'S\1', text)

    # Restore dollar amounts
    for i, amt in enumerate(dollar_amounts):
        text = text.replace(f'DOLLAR{i}PLACEHOLDER', amt)

    return text


def extract_cpt_codes(text):
    """
    Extracts CPT/HCPCS procedure codes.

    ZIP CODE FIX (carried forward from V1):
    Removes state+zip patterns (e.g. "IL 35405") before extraction
    so zip codes are not treated as CPT codes.
    """
    # Remove zip codes: 2-letter state + 5-digit zip
    cleaned = re.sub(r'\b[A-Z]{2}\s+(\d{5})\b', 'STATE ZIPREMOVED', text)

    codes = []
    codes.extend(re.findall(r"\b(\d{5})\b", cleaned))
    codes.extend([c.upper() for c in re.findall(r"\b(G\d{4})\b", cleaned, re.IGNORECASE)])
    codes.extend([c.upper() for c in re.findall(r"\b(\d{4}M)\b", cleaned, re.IGNORECASE)])
    codes.extend([c.upper() for c in re.findall(r"\b(\d{4}U)\b", cleaned, re.IGNORECASE)])

    # REPLACE with this:
    result = []
    for code in codes:
        if code.isdigit() and 1900 <= int(code) <= 2099:
            continue
        result.append(code)
    return result


def extract_icd10_codes(text):
    """Extracts ICD-10-CM diagnosis codes. Call fix_ocr_errors() first."""
    codes = re.findall(r"\b([A-Z]\d{2}(?:\.\w{1,4})?)\b", text, re.IGNORECASE)
    seen = set()
    result = []
    for code in codes:
        code_upper = code.upper()
        if code_upper[0].isalpha() and code_upper not in seen:
            seen.add(code_upper)
            result.append(code_upper)
    return result


def extract_total_charge(text):
    """Finds the total charge amount — used for upcoding detection."""
    match = re.search(r"TOTAL\s+CHARGE[:\s]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    amounts = re.findall(r"\$([\d,]+\.\d{2})", text)
    if amounts:
        try:
            return max(float(a.replace(",", "")) for a in amounts)
        except ValueError:
            pass
    return 0.0


def extract_modifiers(text):
    """Extracts billing modifiers — critical for modifier abuse detection."""
    modifiers = []
    if re.search(r"-59\b|modifier.*59|\b59\b.*modifier", text, re.IGNORECASE):
        modifiers.append("-59")
    return modifiers


def extract_date(text):
    """Finds the service date."""
    match = re.search(r"DATE[:\s]+(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    return match.group(1) if match else None


def build_normalized_claim(text, pdf_path):
    """
    Applies fix_ocr_errors() first, then extracts all fields.
    Same normalized structure as EDI parser output.
    """
    corrected = fix_ocr_errors(text)
    return {
        "claim_id":        pdf_path.stem,
        "date":            extract_date(corrected),
        "procedure_codes": extract_cpt_codes(corrected),
        "diagnosis_codes": extract_icd10_codes(corrected),
        "modifiers":       extract_modifiers(corrected),
        "total_charge":    extract_total_charge(corrected),
        "line_charges":    [],
        "raw_ocr_text":    text[:500],
        "source":          "pdf_ocr_v2",
    }


def process_pdf(pdf_path):
    """
    Full v2 pipeline for one PDF:
    PDF → Image (400 DPI) → Preprocess (sharpen + contrast) →
    OCR (LSTM, PSM 6/4 fallback) → Fix errors → Extract fields
    """
    # Step 1: Render at higher DPI
    image = pdf_to_image(pdf_path)
    if image is None:
        return None

    # Step 2: Pre-process image (NEW in v2)
    image = preprocess_image(image)

    # Step 3: OCR with improved config (NEW in v2)
    text = image_to_text(image)
    if not text.strip():
        return None

    # Step 4: Fix OCR errors and extract fields
    return build_normalized_claim(text, Path(pdf_path))


if __name__ == "__main__":
    if not OCR_AVAILABLE:
        print("\nCannot run — install dependencies first:")
        print("  pip install pytesseract Pillow pdf2image")
        print("  brew install tesseract poppler")
        sys.exit(1)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}.")
        sys.exit(1)

    print(f"\nRunning OCR v2 on {len(pdf_files)} PDFs...")
    print(f"Estimated time: ~{len(pdf_files) * 3 // 60} minutes "
          f"(longer than v1 due to higher DPI + preprocessing)\n")
    print("Improvements active:")
    print("  ✓ DPI: 400 (was 300) — sharper character rendering")
    print("  ✓ LSTM engine forced (--oem 1) — more accurate than legacy")
    print("  ✓ PSM 4 fallback — handles column-format pages")
    print("  ✓ Image preprocessing — sharpen 2.0x, contrast 1.5x, greyscale")
    print("  ✓ Zip code filter — no zip codes extracted as CPT codes")
    print("  ✓ ICD-10 I/O/S-code fixes — capital letters not misread as digits")
    print("  ✓ Dollar amount protection — $131.20 never becomes I31.20\n")

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
    print(f"\nNext: run python3 src/run_ocr_pipeline.py to get updated F1 scores")
