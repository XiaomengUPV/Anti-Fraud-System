"""
extract_fields.py  — VERSION 3 (Real CMS-1500 Form Support)
─────────────────────────────────────────────────────────────
Step 3B of the Fraud Detection Pipeline — OCR Field Extractor.

IMPROVEMENTS OVER V2:
─────────────────────
V1: F1 = 0.550 (300 DPI, raw Tesseract, no preprocessing)
V2: F1 = 0.566 (400 DPI, LSTM engine, sharpen + contrast)
V3: Targets real CMS-1500 forms specifically

V2 was designed for our generated PDFs (clean layouts, black text).
V3 is designed for REAL CMS-1500 forms as submitted by providers —
red printed grid, tiny 8pt typewriter font, scanner noise, slight skew.

NEW IMPROVEMENT A — Red channel extraction (replaces greyscale):
    Real CMS-1500 forms print the box grid and field labels in RED INK.
    The data filled in by the provider is in BLACK.
    By extracting only the GREEN channel of the image, red ink disappears
    (red has high R, low G) while black text remains (black has low G).
    This removes all the preprinted "21. DIAGNOSIS OR NATURE OF ILLNESS"
    labels that were confusing Tesseract and producing false CPT codes.
    Expected improvement: +0.05 to +0.10 F1

NEW IMPROVEMENT B — Crop to procedure table (rows 24A-24J):
    Procedure codes live in a fixed position on every CMS-1500 form.
    Rows 24A-24J (date, place of service, CPT, modifier, diagnosis pointer,
    charges) are always in the bottom third of the form between y=55% and y=85%.
    Cropping eliminates noise from: patient name, address, insurance details,
    provider NPI, signature blocks — none of which contain CPT codes.
    This also eliminates zip codes appearing near the procedure area.
    Expected improvement: +0.08 to +0.12 F1

NEW IMPROVEMENT C — Deskew before OCR:
    Scanned CMS-1500 forms are rarely perfectly straight. Even 1-2 degree
    tilt causes Tesseract to misread characters in narrow columns. We detect
    and correct skew using OpenCV's minAreaRect on dark pixel coordinates.
    Falls back gracefully if OpenCV is not installed.
    Expected improvement: +0.03 to +0.05 F1

CARRIED FORWARD FROM V2:
    - DPI: 400 (sharper character rendering)
    - Tesseract LSTM engine (--oem 1) with PSM fallback
    - Sharpness 2.0x + Contrast 1.5x preprocessing
    - Zip code filter (state+zip not treated as CPT code)
    - ICD-10 I/O/S-code OCR misread correction
    - Dollar amount protection before character fixes
    - Duplicate CPT codes preserved (needed for duplicate billing detection)

TOTAL EXPECTED F1: ~0.65-0.72 on real CMS-1500 forms

REQUIREMENTS:
    pip install pytesseract Pillow pdf2image
    pip install opencv-python   # for deskew (optional but recommended)
    brew install tesseract poppler

HOW TO RUN:
    python3 src/extract_fields.py
"""

import json, re, sys
from pathlib import Path

try:
    import pytesseract
    from PIL import Image, ImageEnhance
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR  = BASE_DIR / "data" / "pdfs"
OCR_DIR  = BASE_DIR / "data" / "ocr_output"
OCR_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: PDF → high-res image ──────────────────────────────────────────────

def pdf_to_image(pdf_path):
    """
    Renders first page at 400 DPI — same as v2.
    Higher DPI = more pixels per character = better OCR accuracy.
    """
    try:
        images = convert_from_path(str(pdf_path), dpi=400)
        return images[0] if images else None
    except Exception as e:
        print(f"  [PDF→IMAGE ERROR] {pdf_path.name}: {e}")
        return None


# ── Step 2: Red channel extraction ────────────────────────────────────────────

def remove_red_grid(image):
    """
    IMPROVEMENT A: Extract green channel to eliminate red CMS-1500 grid.

    Real CMS-1500 forms are printed with the form structure (boxes, field
    labels, dividing lines) in RED ink. The provider's filled-in data is
    in BLACK. This is a deliberate design feature intended for exactly this
    purpose — software can strip the red layer to read only the data.

    How it works:
    - Red pixels have high R value, low G value
    - Black pixels have low R, low G, low B values
    - Green channel: red → white (disappears), black → dark (stays)

    Result: preprinted labels like "24D. PROCEDURES, SERVICES" vanish.
    Only the actual billing codes typed into the boxes remain.

    For PDFs that aren't real CMS-1500 scans (our generated ones),
    the image is purely black on white — extracting the green channel
    still works correctly because black has equal low values in all channels.
    """
    # Convert to RGB first if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Split into R, G, B channels
    r, g, b = image.split()

    # Use green channel — red ink disappears, black text stays
    return g


# ── Step 3: Crop to procedure table ──────────────────────────────────────────

def crop_to_procedure_table(image):
    """
    IMPROVEMENT B: Crop to rows 24A-24J only.

    On every CMS-1500 form, the procedure table occupies a fixed vertical
    position. The layout is standardized by NUCC (National Uniform Claim
    Committee) and has not changed since 2012.

    CMS-1500 vertical layout (approximate):
      0% - 15%:  Header (form title, carrier address)
     15% - 45%:  Patient & insured information (boxes 1-13)
     45% - 55%:  Condition/diagnosis info (boxes 14-23, including ICD-10)
     55% - 83%:  Procedure lines 24A-24J (CPT codes, dates, charges)
     83% - 100%: Provider & billing information (boxes 25-33)

    We crop to TWO regions and OCR them separately:
    1. Diagnosis region (45-57%): contains ICD-10 codes from box 21
    2. Procedure region (55-83%): contains CPT codes from box 24D

    Then combine the text from both regions for field extraction.
    This eliminates:
    - Patient name/address (often contains 5-digit zip codes)
    - Insurance policy numbers (often 5 digits, misread as CPT)
    - Provider NPI numbers (10 digits, but fragments look like CPT)
    - Signature blocks and certification text
    """
    width, height = image.size

    # Crop diagnosis section (box 21 ICD-10 codes)
    diag_top    = int(height * 0.44)
    diag_bottom = int(height * 0.58)
    diagnosis_region = image.crop((0, diag_top, width, diag_bottom))

    # Crop procedure table (box 24A-24J CPT codes)
    proc_top    = int(height * 0.55)
    proc_bottom = int(height * 0.85)
    procedure_region = image.crop((0, proc_top, width, proc_bottom))

    return diagnosis_region, procedure_region


# ── Step 4: Deskew ────────────────────────────────────────────────────────────

def deskew(image):
    """
    IMPROVEMENT C: Correct scan rotation before OCR.

    Scanned forms are rarely perfectly straight. Even 1-2 degree tilt
    causes Tesseract to misread characters, especially in narrow CPT
    code columns where a tilted column looks like blurred text.

    Algorithm:
    1. Convert to numpy array
    2. Find all dark pixels (text)
    3. Fit a minimum-area rectangle around all dark pixels
    4. Extract rotation angle from the rectangle
    5. Rotate image to correct the angle

    Falls back to original image if OpenCV is not available or if
    the detected angle is outside the expected range (±10 degrees).
    """
    if not CV2_AVAILABLE:
        return image  # Graceful fallback

    try:
        img_array = np.array(image)

        # Find dark pixels (text)
        # Threshold: pixels darker than 128 are "text"
        dark_pixels = np.column_stack(np.where(img_array < 128))

        if len(dark_pixels) < 100:
            return image  # Not enough text to detect angle

        # Fit minimum area rectangle to dark pixels
        rect = cv2.minAreaRect(dark_pixels.astype(np.float32))
        angle = rect[-1]

        # Normalize angle to -45 to +45 range
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct if tilt is significant but not extreme
        if abs(angle) < 0.5 or abs(angle) > 10:
            return image  # Skip tiny or extreme angles

        # Rotate to correct skew
        h, w = img_array.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img_array, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return Image.fromarray(rotated)

    except Exception:
        return image  # Fallback on any error


# ── Step 5: Image enhancement ─────────────────────────────────────────────────

def enhance_image(image):
    """
    Sharpness and contrast enhancement — carried forward from v2.
    Applied AFTER red channel extraction and deskewing.
    """
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return image


# ── Step 6: OCR ───────────────────────────────────────────────────────────────

def image_to_text(image):
    """
    LSTM engine with PSM fallback — same as v2.
    --oem 1: LSTM neural net only (more accurate than legacy)
    --psm 6: Uniform block of text (primary)
    --psm 4: Single column (fallback if primary fails)
    """
    if image is None:
        return ""
    try:
        config_primary = "--oem 1 --psm 6"
        text = pytesseract.image_to_string(image, config=config_primary)
        if len(text.strip()) < 30:
            text = pytesseract.image_to_string(image, config="--oem 1 --psm 4")
        return text
    except Exception as e:
        print(f"  [OCR ERROR]: {e}")
        return ""


# ── Step 7: OCR error correction ──────────────────────────────────────────────

def fix_ocr_errors(text):
    """
    Post-OCR character substitution fixes — carried forward from v2.
    Corrects Tesseract misreads specific to medical billing codes.
    """
    # Protect dollar amounts before character substitutions
    dollar_amounts = re.findall(r'\$[\d,]+\.\d{2}', text)
    for i, amt in enumerate(dollar_amounts):
        text = text.replace(amt, f'DOLLAR{i}PLACEHOLDER', 1)

    # Fix ICD-10 I-codes: 125.10 → I25.10, 110 → I10
    # Only fix when followed by decimal (e.g. 125.10) or at end of word
    # NOT when followed by space+letters (street addresses like "100 MAIN ST")
    text = re.sub(r'\b1(\d{2}\.\w{1,4})\b', r'I\1', text)
    # For standalone 3-digit I-codes (I10, I00), only fix when
    # preceded by non-address context (after punctuation/newline/letter-dot-space)
    text = re.sub(r'(?<=[A-L]\. )1(\d{2})(?!\d)(?! [A-Z])', r'I\1', text)
    text = re.sub(r'(?<=[\n,;\t])1(\d{2})(?!\d)(?! [A-Z]{2,})', r'I\1', text)

    # Fix ICD-10 O-codes: 009.90 → O09.90
    text = re.sub(r'\b0(\d{2}\.\w{1,4})\b', r'O\1', text)

    # Fix S-codes with letter suffix: 572.001A → S72.001A
    text = re.sub(r'\b5(\d{2}\.\w{1,3}[A-Z])\b', r'S\1', text)

    # Restore dollar amounts
    for i, amt in enumerate(dollar_amounts):
        text = text.replace(f'DOLLAR{i}PLACEHOLDER', amt)

    return text


# ── Step 8: Field extraction ──────────────────────────────────────────────────

def extract_cpt_codes(text):
    """
    Extracts CPT/HCPCS procedure codes.

    IMPORTANT: Preserves duplicate codes — if 99213 appears twice,
    both instances are kept. The rule engine needs ['99213', '99213']
    to detect duplicate billing. Do NOT deduplicate here.

    ZIP CODE FIX: Removes state+zip patterns before extraction.
    After cropping to procedure table (Improvement B), zip codes
    from the address section should already be gone. This is a
    belt-and-suspenders backup filter.
    """
    # Remove zip codes — 2-letter state abbreviation + 5-digit zip
    cleaned = re.sub(r'\b[A-Z]{2}\s+(\d{5})\b', 'STATE ZIPREMOVED', text)

    codes = []
    # 5-digit numeric CPT codes
    codes.extend(re.findall(r"\b(\d{5})\b", cleaned))
    # HCPCS alpha codes
    codes.extend([c.upper() for c in re.findall(r"\b(G\d{4})\b", cleaned, re.IGNORECASE)])
    codes.extend([c.upper() for c in re.findall(r"\b(\d{4}M)\b", cleaned, re.IGNORECASE)])
    codes.extend([c.upper() for c in re.findall(r"\b(\d{4}U)\b", cleaned, re.IGNORECASE)])

    # Filter years but KEEP duplicates — critical for duplicate billing detection
    # Also remove codes that appear as part of account/policy numbers
    # by checking their context in the original text
    result = []
    for code in codes:
        if code.isdigit() and 1900 <= int(code) <= 2099:
            continue  # Skip years
        # Skip codes that appear inside policy/account/ID numbers
        # e.g. GRP-AET-77342, PAT-38847, UHC-847291053, NPI 9876543210
        # Strategy: check if the code appears within 30 chars after known prefixes
        # OR if it's part of a hyphenated sequence (policy numbers like GRP-AET-77342)
        ctx_pattern = rf'(?:PAT|ACCT|GRP|UHC|NPI|SSN|EIN|FECA|TAX|AET|BCBS|CVS|HMO|PPO)[^\n]{{0,15}}{re.escape(code)}'
        if re.search(ctx_pattern, cleaned, re.IGNORECASE):
            continue
        # Also skip if the code is preceded by a hyphen (part of an ID like GRP-AET-77342)
        hyphen_pattern = rf'-{re.escape(code)}\b'
        if re.search(hyphen_pattern, cleaned):
            continue
        result.append(code)

    return result


def extract_icd10_codes(text):
    """
    Extracts ICD-10-CM diagnosis codes. Call fix_ocr_errors() first.
    Deduplicates ICD codes (same diagnosis appearing twice is not fraud).
    """
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
    """Finds total charge amount. Used for upcoding detection."""
    # First priority: find amount specifically labelled as TOTAL CHARGE
    match = re.search(r"TOTAL\s+CHARGE[^\d$]*\$?([\d,]+\.\d{2})", text, re.IGNORECASE)
    if match:
        val = float(match.group(1).replace(",", ""))
        if val > 0:
            return val
    # Second priority: largest dollar amount on the form (usually total charge)
    amounts = re.findall(r"\$([\d,]+\.\d{2})", text)
    if amounts:
        try:
            values = [float(a.replace(",", "")) for a in amounts if float(a.replace(",","")) > 0]
            if values:
                return max(values)
        except ValueError:
            pass
    return 0.0


def extract_modifiers(text):
    """Extracts billing modifiers — critical for modifier abuse detection."""
    modifiers = []
    if re.search(r"\b59\b|-59|modifier.*59", text, re.IGNORECASE):
        modifiers.append("-59")
    return modifiers


def extract_date(text):
    """Finds the service date."""
    match = re.search(r"DATE[:\s]+(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if match:
        return match.group(1)
    # Also look for MM DD YY format on real forms: "01 15 26"
    match = re.search(r"\b(\d{2})\s+(\d{2})\s+(\d{2})\b", text)
    if match:
        m, d, y = match.groups()
        year = f"20{y}" if int(y) < 50 else f"19{y}"
        return f"{year}-{m}-{d}"
    return None


# ── Full pipeline ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path):
    """
    Full v3 pipeline for one PDF:

    PDF → Image (400 DPI)
        → Remove red grid (green channel extraction)
        → Deskew (correct scan rotation)
        → Crop to diagnosis region → OCR → Fix errors → Extract ICD-10
        → Crop to procedure region → OCR → Fix errors → Extract CPT codes
        → Extract modifiers, charges, date from full text
        → Return normalized claim object
    """
    pdf_path = Path(pdf_path)

    # Step 1: Render at 400 DPI
    image = pdf_to_image(pdf_path)
    if image is None:
        return None

    # Step 2: Remove red CMS-1500 grid (NEW in v3)
    image = remove_red_grid(image)

    # Step 3: Deskew (NEW in v3)
    image = deskew(image)

    # Step 4: Enhance full image (carried from v2)
    image_enhanced = enhance_image(image)

    # Step 5: Crop to specific regions (NEW in v3)
    diag_region, proc_region = crop_to_procedure_table(image_enhanced)

    # Step 6: OCR each region separately
    # Use PSM 11 (sparse text) for small cropped regions — handles partial forms better
    def ocr_region(img):
        import pytesseract
        text = pytesseract.image_to_string(img, config="--oem 1 --psm 11")
        if len(text.strip()) < 10:
            text = pytesseract.image_to_string(img, config="--oem 1 --psm 6")
        return text

    diag_text = ocr_region(diag_region)
    proc_text = ocr_region(proc_region)
    full_text  = image_to_text(image_enhanced)  # full page for dates/totals

    # Step 7: Fix OCR errors in each region
    diag_text_fixed = fix_ocr_errors(diag_text)
    proc_text_fixed = fix_ocr_errors(proc_text)
    full_text_fixed = fix_ocr_errors(full_text)

    # Step 8: Extract fields — prefer cropped regions, fallback to full page
    cpt_codes = extract_cpt_codes(proc_text_fixed)
    icd_codes = extract_icd10_codes(diag_text_fixed)

    # Fallback: if cropped regions got nothing, use full page
    if not cpt_codes:
        cpt_codes = extract_cpt_codes(full_text_fixed)
    if not icd_codes:
        icd_codes = extract_icd10_codes(full_text_fixed)

    # Only return None if we got absolutely nothing
    if not full_text_fixed.strip():
        return None

    return {
        "claim_id":        pdf_path.stem,
        "date":            extract_date(full_text_fixed),
        "procedure_codes": cpt_codes,
        "diagnosis_codes": icd_codes,
        "modifiers":       extract_modifiers(proc_text_fixed + full_text_fixed),
        "total_charge":    extract_total_charge(full_text_fixed),
        "line_charges":    [],
        "raw_ocr_text":    full_text[:500],
        "source":          "pdf_ocr_v3",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not OCR_AVAILABLE:
        print("\nCannot run — install dependencies:")
        print("  pip install pytesseract Pillow pdf2image")
        print("  brew install tesseract poppler")
        sys.exit(1)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}.")
        sys.exit(1)

    print(f"\nRunning OCR v3 on {len(pdf_files)} PDFs...")
    print(f"Estimated time: ~{len(pdf_files) * 3 // 60} minutes\n")
    print("Improvements active:")
    print("  ✓ Red channel extraction — CMS-1500 red grid eliminated")
    print("  ✓ Crop to procedure table — noise from address/insurance removed")
    print("  ✓ Deskew — scan rotation corrected" + (" (OpenCV)" if CV2_AVAILABLE else " (skipped — install opencv-python)"))
    print("  ✓ LSTM engine + PSM fallback")
    print("  ✓ Sharpen 2.0x + Contrast 1.5x")
    print("  ✓ ICD-10 I/O/S-code misread correction")
    print("  ✓ Duplicate CPT codes preserved for duplicate billing detection")
    print()

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
