"""
generate_claims.py
──────────────────
Generates 1,000+ synthetic CMS-1500 medical claims as JSON.
Covers all 9 fraud types + legitimate claims.

Output: data/raw_claims/claims.json
        data/raw_claims/claims_summary.csv
"""

import json
import random
import csv
import os
from datetime import date, timedelta
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw_claims"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ── Real CPT codes from 2026 DHS list, organised by clinical category ─────────
# Each entry: (code, description, typical_price_min, typical_price_max, category)
CPT_BY_CATEGORY = {
    "office_visit_low": [
        ("99202", "Office visit new patient low complexity",         75,  150),
        ("99212", "Office visit established low complexity",         50,  120),
        ("99213", "Office visit established moderate complexity",   100,  200),
    ],
    "office_visit_high": [
        ("99205", "Office visit new patient high complexity",       250,  450),
        ("99215", "Office visit established high complexity",       200,  400),
        ("99214", "Office visit moderate-high complexity",          150,  300),
    ],
    "lab_basic": [
        ("85025", "Complete blood count with differential",          20,   60),
        ("80053", "Comprehensive metabolic panel",                   30,   80),
        ("80061", "Lipid panel",                                     25,   70),
        ("82465", "Cholesterol total",                               15,   40),
        ("36415", "Venipuncture routine",                             5,   25),
        ("82947", "Glucose quantitative",                            10,   30),
    ],
    "lab_advanced": [
        ("83036", "Hemoglobin A1c",                                  20,   60),
        ("87491", "Chlamydia/gonorrhea nucleic acid test",           50,  120),
        ("86592", "Syphilis test non-treponemal qualitative",        10,   35),
        ("87081", "Culture bacterial aerobic",                       25,   75),
    ],
    "lab_oncology": [
        ("0006M", "Onc hep gene risk classifier",                  800, 1500),
        ("0007M", "Onc gastro 51 gene nomogram",                  1000, 2000),
        ("0016M", "Onc bladder mrna 219 gen alg",                 1200, 2500),
        ("0017M", "Onc dlbcl mrna 20 genes alg",                  900, 1800),
        ("0026U", "Oncology thyroid DNA mRNA 112 genes",          2500, 4500),
        ("0036U", "Exome tumor and normal spec seq",              3000, 6000),
        ("0048U", "Oncology solid organ neo DNA 468 gene",        3500, 6500),
        ("0055U", "Cardiac heart transplant 96 DNA seq",          4000, 7000),
        ("0019M", "Cv ds plasma alys prtn bmrk",                  1500, 3000),
        ("0020M", "Onc cns alys 30000 dna loci",                  2000, 4000),
    ],
    "imaging_basic": [
        ("71046", "Chest X-ray 2 views",                           100,  250),
        ("73030", "Shoulder X-ray minimum 2 views",                 80,  200),
        ("72100", "X-ray spine lumbosacral minimum 2 views",       120,  280),
        ("73560", "X-ray knee 1 or 2 views",                        70,  180),
    ],
    "imaging_advanced": [
        ("71250", "CT thorax without contrast",                    700, 1800),
        ("70553", "MRI brain with and without contrast",          1200, 2800),
        ("72148", "MRI lumbar spine without contrast",             900, 2000),
        ("74178", "CT abdomen pelvis without/with contrast",      1400, 3000),
    ],
    "cardiac": [
        ("93000", "Electrocardiogram routine ECG",                  50,  150),
        ("93306", "Echo transthoracic complete",                   800, 1800),
        ("93458", "Left heart catheterization",                   3000, 6000),
        ("93010", "ECG interpretation and report",                  30,   90),
    ],
    "screening": [
        ("G0103", "PSA screening",                                  20,   60),
        ("G0101", "Cervical or vaginal cancer screening",            30,   80),
        ("82270", "Blood occult fecal hemoglobin",                  10,   35),
        ("G0107", "Colorectal cancer screening FOBT",               15,   40),
    ],
    "surgical": [
        ("27447", "Total knee arthroplasty",                      8000,15000),
        ("29881", "Knee arthroscopy with meniscectomy",           3000, 6000),
        ("43239", "EGD with biopsy",                              1200, 2500),
        ("49505", "Inguinal hernia repair",                       3500, 7000),
    ],
    "neuro_lab": [
        ("0062M", "AI neurology brain connectivity",              2000, 4000),
        ("0063M", "AI neurology CSF biomarker",                   1800, 3500),
        ("0016U", "Onc hmtlmf neo rna bcr/abl1",                 1500, 3000),
        ("0017U", "Onc brst ca erbb2 amp/nonamp",                1200, 2800),
        ("0023U", "Onc melanoma 11 gene expr",                   1000, 2500),
    ],
}

# Flatten for easy lookup
ALL_CPT = {}
for cat, codes in CPT_BY_CATEGORY.items():
    for code, desc, pmin, pmax in codes:
        ALL_CPT[code] = {"desc": desc, "price_min": pmin, "price_max": pmax, "category": cat}

# ── ICD-10 diagnosis codes ─────────────────────────────────────────────────────
ICD10 = {
    # Common / routine
    "Z00.00": "Routine adult health exam without abnormal findings",
    "Z13.6":  "Encounter for screening for cardiovascular disorders",
    "Z13.1":  "Encounter for screening for diabetes mellitus",
    "Z12.11": "Encounter for screening for colon cancer",
    "Z12.31": "Encounter for screening for cervical cancer",

    # Common conditions
    "I10":    "Essential hypertension",
    "E11.9":  "Type 2 diabetes mellitus without complications",
    "M54.5":  "Low back pain",
    "J06.9":  "Acute upper respiratory infection unspecified",
    "N39.0":  "Urinary tract infection unspecified",
    "J18.9":  "Pneumonia unspecified organism",
    "K21.0":  "Gastroesophageal reflux disease with esophagitis",
    "F32.9":  "Major depressive disorder single episode unspecified",
    "G43.909":"Migraine unspecified not intractable",
    "M25.511":"Pain in right shoulder",
    "M25.561":"Pain in right knee",
    "M17.11": "Unilateral primary osteoarthritis right knee",
    "M23.205":"Derangement of unspecified medial meniscus due to old tear right knee",
    "K40.90": "Unilateral inguinal hernia without obstruction or gangrene",
    "Z12.5":  "Encounter for screening for malignant neoplasm of prostate",

    # Cardiac
    "I25.10": "Atherosclerotic heart disease of native coronary artery",
    "I50.9":  "Heart failure unspecified",
    "I48.91": "Unspecified atrial fibrillation",
    "R00.0":  "Tachycardia unspecified",

    # Oncology
    "C50.911":"Malignant neoplasm of unspecified site of right female breast",
    "C18.9":  "Malignant neoplasm of colon unspecified",
    "C34.10": "Malignant neoplasm of upper lobe unspecified bronchus or lung",
    "C73":    "Malignant neoplasm of thyroid gland",

    # Screening that requires specific ICD
    "Z12.39": "Encounter for other screening for malignant neoplasm of breast",
    "Z03.89": "Encounter for observation for other suspected diseases ruled out",
}

# ── Legitimate CPT → valid ICD-10 mappings ────────────────────────────────────
VALID_CPT_ICD = {
    "99202": ["Z00.00","I10","E11.9","M54.5","J06.9","N39.0","J18.9","K21.0","F32.9","G43.909","M25.511"],
    "99212": ["Z00.00","I10","E11.9","M54.5","J06.9","N39.0","K21.0","F32.9","G43.909"],
    "99213": ["Z00.00","I10","E11.9","M54.5","J06.9","N39.0","K21.0","F32.9","G43.909","M25.511"],
    "99214": ["I10","E11.9","M54.5","J18.9","I25.10","I50.9","I48.91","C50.911","C18.9"],
    "99205": ["I25.10","I50.9","I48.91","C50.911","C18.9","C34.10","C73"],
    "99215": ["I25.10","I50.9","I48.91","C50.911","C18.9","C34.10","C73","J18.9"],
    "85025": ["Z00.00","I10","E11.9","Z13.6","J18.9","N39.0"],
    "80053": ["Z00.00","E11.9","I10","Z13.1","J18.9"],
    "80061": ["Z13.6","I10","E11.9","I25.10"],
    "82465": ["Z13.6","I10","I25.10"],
    "36415": ["Z00.00","Z13.6","Z13.1","I10","E11.9"],
    "82947": ["E11.9","Z13.1","Z00.00"],
    "83036": ["E11.9","Z13.1"],
    "87491": ["N39.0","Z03.89"],
    "86592": ["Z03.89","Z00.00"],
    "87081": ["J06.9","J18.9","N39.0"],
    "71046": ["J06.9","J18.9","R00.0"],
    "73030": ["M25.511"],
    "72100": ["M54.5"],
    "73560": ["M25.561","M17.11"],
    "71250": ["J18.9","C34.10","I25.10","J06.9"],
    "70553": ["G43.909","Z03.89","C50.911"],
    "72148": ["M54.5"],
    "74178": ["C18.9","K21.0","Z12.11"],
    "93000": ["I10","I48.91","R00.0","Z13.6","I25.10","I50.9"],
    "93306": ["I50.9","I48.91","I25.10","I10"],
    "93458": ["I25.10","I48.91","I50.9"],
    "93010": ["I10","I48.91","R00.0","I25.10"],
    "G0103": ["Z12.5"],
    "G0101": ["Z12.31","Z00.00"],
    "82270": ["Z12.11","K21.0"],
    "G0107": ["Z12.11"],
    "27447": ["M17.11","M25.561"],
    "29881": ["M23.205","M25.561"],
    "43239": ["K21.0","Z12.11"],
    "49505": ["K40.90"],
    # Oncology codes require cancer diagnosis
    "0006M": ["C18.9","C34.10","C73","C50.911"],
    "0007M": ["C18.9"],
    "0016M": ["C18.9","C50.911"],
    "0017M": ["C50.911","C18.9"],
    "0026U": ["C73","Z03.89"],
    "0036U": ["C50.911","C18.9","C34.10"],
    "0048U": ["C34.10","C18.9","C50.911"],
    "0055U": ["I50.9","I48.91"],
    "0019M": ["I25.10","I50.9"],
    "0020M": ["C50.911","C18.9","C34.10"],
    "0062M": ["G43.909","F32.9"],
    "0063M": ["G43.909","Z03.89"],
    "0016U": ["C50.911","C18.9"],
    "0017U": ["C50.911"],
    "0023U": ["C50.911","C18.9"],
}

# NCCI bundling pairs — these should NOT be billed together with -59
NCCI_BUNDLES = [
    ("99213", "99214"),  # Can't bill two visit levels same day
    ("99212", "99213"),
    ("80061", "82465"),  # Lipid panel includes cholesterol — unbundling
    ("85025", "36415"),  # CBC includes venipuncture
    ("93000", "93306"),  # ECG included in Echo
    ("93000", "93010"),  # Duplicate ECG components
    ("80053", "82947"),  # CMP includes glucose
]

# Screening codes that require specific ICD-10 Z-codes
SCREENING_REQUIREMENTS = {
    "G0103": ["Z00.00", "Z03.89"],       # PSA — requires routine exam
    "G0101": ["Z12.31", "Z00.00"],       # Cervical — requires cervical screening
    "82270": ["Z12.11", "K21.0"],        # FOBT — requires colorectal screening
    "G0107": ["Z12.11"],                 # Colorectal — requires Z12.11
    "80061": ["Z13.6", "I10", "I25.10"], # Lipid — needs cardiovascular context
    "82465": ["Z13.6", "I10", "I25.10"],
}

# ── People pools ───────────────────────────────────────────────────────────────
FIRST_NAMES = ["Alice","Brian","Carol","Daniel","Eva","Frank","Grace","Henry",
               "Iris","Jack","Karen","Leo","Mary","Nathan","Olivia","Paul",
               "Quinn","Rachel","Sam","Tina","Uma","Victor","Wendy","Xander",
               "Yara","Zoe","Adam","Beth","Chris","Diana","Eric","Fiona"]

LAST_NAMES  = ["Johnson","Williams","Smith","Brown","Martinez","Davis","Garcia",
               "Wilson","Anderson","Taylor","Thomas","Moore","Jackson","Martin",
               "Lee","Perez","Thompson","White","Harris","Sanchez","Clark",
               "Lewis","Robinson","Walker","Young","Allen","King","Wright","Scott"]

PROVIDERS   = ["Dr. Sarah Mitchell","Dr. James Patel","Dr. Linda Chen",
               "Dr. Robert Torres","Dr. Emily Nguyen","Dr. David Kim",
               "Dr. Angela Brooks","Dr. Michael Reed","Dr. Priya Sharma",
               "Dr. Thomas Wade","Dr. Carlos Mendez","Dr. Fatima Hassan"]

FACILITIES  = ["Greenfield Family Clinic","Maplewood Medical Center",
                "Sunrise Health Associates","Lakewood Primary Care",
                "Northside Medical Group","Riverview Outpatient Center"]

INSURERS    = ["BlueCross BlueShield","Aetna Health","UnitedHealthcare",
               "Cigna Health","Humana Inc.","Medicare Part B","Empire BCBS"]

STATES      = ["NY","CA","TX","FL","IL","PA","OH","GA","NC","MI"]


# ── Helper functions ───────────────────────────────────────────────────────────

def rand_patient():
    return {
        "id":   f"P{random.randint(1000,9999)}",
        "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "dob":  str(date(random.randint(1945,2000), random.randint(1,12), random.randint(1,28))),
        "sex":  random.choice(["M","F"]),
        "address": f"{random.randint(1,999)} {random.choice(LAST_NAMES)} St, "
                   f"{random.choice(['Albany','Syracuse','Rochester','Buffalo','Ithaca'])}, "
                   f"{random.choice(STATES)} {random.randint(10000,99999)}",
        "insurer": random.choice(INSURERS),
        "policy_no": f"POL{random.randint(100000,999999)}",
        "group_no":  f"GRP{random.randint(10000,99999)}",
    }

def rand_provider():
    return {
        "name":     random.choice(PROVIDERS),
        "facility": random.choice(FACILITIES),
        "npi":      str(random.randint(1000000000, 9999999999)),
        "tax_id":   f"{random.randint(10,99)}-{random.randint(1000000,9999999)}",
    }

def rand_date():
    start = date(2025, 1, 1)
    return str(start + timedelta(days=random.randint(0, 450)))

def rand_price(pmin, pmax):
    return round(random.uniform(pmin, pmax), 2)

def make_claim_id(idx):
    return f"CLM{idx:05d}"


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def make_legitimate(idx):
    """Generate a clean, internally consistent claim."""
    # Pick a random CPT from a non-oncology category
    safe_cats = ["office_visit_low","office_visit_high","lab_basic",
                 "lab_advanced","imaging_basic","cardiac","screening","surgical"]
    cat   = random.choice(safe_cats)
    entry = random.choice(CPT_BY_CATEGORY[cat])
    code, desc, pmin, pmax = entry

    # Pick a valid ICD-10 for this CPT
    valid_icds = VALID_CPT_ICD.get(code, ["Z00.00"])
    icd   = random.choice(valid_icds)
    price = rand_price(pmin, pmax)

    return {
        "claim_id":            make_claim_id(idx),
        "date":                rand_date(),
        "patient":             rand_patient(),
        "provider":            rand_provider(),
        "procedure_codes":     [code],
        "procedure_descs":     [desc],
        "diagnosis_codes":     [icd],
        "diagnosis_descs":     [ICD10[icd]],
        "modifiers":           [],
        "line_charges":        [price],
        "total_charge":        price,
        "fraud_type":          "Legitimate",
        "fraud_indicator":     False,
        "fraud_explanation":   None,
    }


def make_upcoding(idx):
    """Bill a more expensive code than the service warranted."""
    # Pair: (what should have been billed, what was actually billed)
    upcoding_pairs = [
        ("99213","Office visit established moderate complexity",  100,200,
         "99215","Office visit established high complexity",      200,400,
         "Patient had simple routine visit but high-complexity code billed"),
        ("71046","Chest X-ray 2 views",                          100,250,
         "71250","CT thorax without contrast",                    700,1800,
         "Simple chest X-ray justified — CT scan billed instead"),
        ("93000","Electrocardiogram routine ECG",                  50,150,
         "93306","Echo transthoracic complete",                   800,1800,
         "Routine ECG was performed — full echocardiogram billed"),
        ("85025","Complete blood count",                           20, 60,
         "0019M","Cv ds plasma alys prtn bmrk",                 1500,3000,
         "Simple CBC was performed — expensive cardiac protein panel billed"),
        ("80061","Lipid panel",                                    25, 70,
         "0006M","Onc hep gene risk classifier",                 800,1500,
         "Basic lipid panel warranted — oncology gene classifier billed"),
        ("99202","Office visit new patient low complexity",        75,150,
         "99205","Office visit new patient high complexity",      250,450,
         "New patient simple visit billed at highest complexity level"),
    ]
    pair = random.choice(upcoding_pairs)
    correct_code, correct_desc, _, _, fraud_code, fraud_desc, fraud_pmin, fraud_pmax, explanation = pair

    # Pick a valid ICD for the correct (lower) code
    valid_icds = VALID_CPT_ICD.get(correct_code, ["Z00.00"])
    icd   = random.choice(valid_icds)
    price = rand_price(fraud_pmin, fraud_pmax)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   [fraud_code],
        "procedure_descs":   [fraud_desc],
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [ICD10[icd]],
        "modifiers":         [],
        "line_charges":      [price],
        "total_charge":      price,
        "fraud_type":        "Upcoding",
        "fraud_indicator":   True,
        "fraud_explanation": explanation,
        "correct_code":      correct_code,
        "correct_desc":      correct_desc,
    }


def make_unbundling(idx):
    """Bill component codes separately instead of one comprehensive code."""
    unbundling_cases = [
        # (comprehensive_code, [components], icd, explanation)
        ("80061", ["82465","82470"], "Z13.6",
         "Lipid panel (80061) billed as individual cholesterol components"),
        ("85025", ["85014","85018","85041"], "Z00.00",
         "CBC (85025) unbundled into individual blood component tests"),
        ("93306", ["93303","93304","93005"], "I50.9",
         "Complete echo (93306) split into component services"),
        ("80053", ["82947","82565","84132","84295"], "E11.9",
         "Comprehensive metabolic panel (80053) billed as individual chemistry tests"),
        ("99214", ["99213","99212"], "I10",
         "Single office visit split into multiple visit codes"),
    ]
    case = random.choice(unbundling_cases)
    comprehensive, components, icd, explanation = case

    # Use the component codes as procedure codes
    comp_codes = components[:random.randint(2,len(components))]
    comp_descs = [ALL_CPT.get(c, {}).get("desc", f"Component service {c}") for c in comp_codes]
    comp_prices= [rand_price(
                    ALL_CPT.get(c, {}).get("price_min", 20),
                    ALL_CPT.get(c, {}).get("price_max", 100)
                  ) for c in comp_codes]
    total = round(sum(comp_prices), 2)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   comp_codes,
        "procedure_descs":   comp_descs,
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [ICD10[icd]],
        "modifiers":         [],
        "line_charges":      comp_prices,
        "total_charge":      total,
        "fraud_type":        "Unbundling",
        "fraud_indicator":   True,
        "fraud_explanation": explanation,
        "comprehensive_code": comprehensive,
    }


def make_code_padding(idx):
    """Add unrelated high-value codes to a legitimate claim."""
    # Start with a legitimate base claim
    base_cat   = random.choice(["office_visit_low","lab_basic","cardiac","screening"])
    base_entry = random.choice(CPT_BY_CATEGORY[base_cat])
    base_code, base_desc, pmin, pmax = base_entry

    valid_icds = VALID_CPT_ICD.get(base_code, ["Z00.00"])
    icd        = random.choice(valid_icds)
    base_price = rand_price(pmin, pmax)

    # Add 2-3 unrelated oncology/neuro codes
    padding_pool = [
        ("0006M","Onc hep gene risk classifier",         800,1500),
        ("0007M","Onc gastro 51 gene nomogram",         1000,2000),
        ("0020M","Onc cns alys 30000 dna loci",         2000,4000),
        ("0062M","AI neurology brain connectivity",      2000,4000),
        ("0063M","AI neurology CSF biomarker",           1800,3500),
        ("0019M","Cv ds plasma alys prtn bmrk",         1500,3000),
        ("0016U","Onc hmtlmf neo rna bcr/abl1",         1500,3000),
        ("0017U","Onc brst ca erbb2 amp/nonamp",        1200,2800),
        ("0023U","Onc melanoma 11 gene expr",            1000,2500),
        ("0026U","Oncology thyroid DNA mRNA 112 genes", 2500,4500),
    ]
    n_padding   = random.randint(2,3)
    padded      = random.sample(padding_pool, n_padding)
    pad_codes   = [p[0] for p in padded]
    pad_descs   = [p[1] for p in padded]
    pad_prices  = [rand_price(p[2], p[3]) for p in padded]

    all_codes  = [base_code]  + pad_codes
    all_descs  = [base_desc]  + pad_descs
    all_prices = [base_price] + pad_prices
    total      = round(sum(all_prices), 2)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   all_codes,
        "procedure_descs":   all_descs,
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [ICD10[icd]],
        "modifiers":         [],
        "line_charges":      all_prices,
        "total_charge":      total,
        "fraud_type":        "Code Padding",
        "fraud_indicator":   True,
        "fraud_explanation": f"Legitimate {base_code} ({base_desc}) padded with {n_padding} unrelated high-value codes unrelated to diagnosis {icd}",
    }


def make_phantom_billing(idx):
    """Bill for a procedure that was never performed — clinically impossible given diagnosis."""
    phantom_cases = [
        ("0026U","Oncology thyroid DNA mRNA 112 genes",  2500,4500,
         "Z00.00","Routine adult health exam without abnormal findings",
         "Complex thyroid oncology genomic test billed for routine wellness visit — never indicated"),
        ("0036U","Exome tumor and normal spec seq",       3000,6000,
         "Z13.1","Encounter for screening for diabetes mellitus",
         "Exome sequencing billed for diabetes screening — no cancer diagnosis to justify"),
        ("0048U","Oncology solid organ neo DNA 468 gene",3500,6500,
         "J06.9","Acute upper respiratory infection unspecified",
         "Complex oncology DNA panel billed for common cold visit — never performed"),
        ("0055U","Cardiac heart transplant 96 DNA seq",  4000,7000,
         "M54.5","Low back pain",
         "Cardiac transplant genomic test billed for back pain patient — impossible context"),
        ("93458","Left heart catheterization",           3000,6000,
         "Z00.00","Routine adult health exam without abnormal findings",
         "Invasive cardiac catheterization billed for routine health exam — never performed"),
        ("27447","Total knee arthroplasty",              8000,15000,
         "J06.9","Acute upper respiratory infection unspecified",
         "Total knee replacement billed for cold/flu visit — impossible scenario"),
        ("71250","CT thorax without contrast",            700,1800,
         "Z13.1","Encounter for screening for diabetes mellitus",
         "CT chest scan billed during diabetes screening — no respiratory indication"),
    ]
    case = random.choice(phantom_cases)
    code, desc, pmin, pmax, icd, icd_desc, explanation = case
    price = rand_price(pmin, pmax)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   [code],
        "procedure_descs":   [desc],
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [icd_desc],
        "modifiers":         [],
        "line_charges":      [price],
        "total_charge":      price,
        "fraud_type":        "Phantom Billing",
        "fraud_indicator":   True,
        "fraud_explanation": explanation,
    }


def make_diagnosis_mismatch(idx):
    """Pair a procedure with a wrong diagnosis to justify the cost."""
    mismatch_cases = [
        ("0006M","Onc hep gene risk classifier",  800,1500,
         "M54.5","Low back pain",
         "Oncology gene classifier requires cancer diagnosis — not back pain"),
        ("80061","Lipid panel",                    25, 70,
         "E11.9","Type 2 diabetes mellitus without complications",
         "Lipid panel requires cardiovascular screening diagnosis (Z13.6) not diabetes"),
        ("G0103","PSA screening",                  20, 60,
         "J06.9","Acute upper respiratory infection unspecified",
         "PSA screening billed during cold visit — diagnosis does not support screening"),
        ("82465","Cholesterol total",              15, 40,
         "Z00.00","Routine adult health exam without abnormal findings",
         "Cholesterol assay requires cardiovascular screening code Z13.6 not routine exam"),
        ("0019M","Cv ds plasma alys prtn bmrk",  1500,3000,
         "M54.5","Low back pain",
         "Cardiovascular protein panel billed for back pain — no cardiac diagnosis"),
        ("93458","Left heart catheterization",    3000,6000,
         "J06.9","Acute upper respiratory infection unspecified",
         "Invasive cardiac procedure billed with respiratory infection diagnosis — no indication"),
        ("70553","MRI brain with and without contrast",1200,2800,
         "N39.0","Urinary tract infection unspecified",
         "Brain MRI billed for UTI patient — completely unrelated diagnosis"),
        ("29881","Knee arthroscopy with meniscectomy",3000,6000,
         "I10","Essential hypertension",
         "Knee surgery billed with hypertension diagnosis — no orthopedic indication"),
    ]
    case = random.choice(mismatch_cases)
    code, desc, pmin, pmax, icd, icd_desc, explanation = case
    price = rand_price(pmin, pmax)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   [code],
        "procedure_descs":   [desc],
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [icd_desc],
        "modifiers":         [],
        "line_charges":      [price],
        "total_charge":      price,
        "fraud_type":        "Diagnosis Mismatch",
        "fraud_indicator":   True,
        "fraud_explanation": explanation,
    }


def make_code_substitution(idx):
    """Swap a non-covered code for a covered one that sounds similar."""
    # Prices are the TRUE non-covered service's charge (why fraudsters substitute):
    # the billed code is cheap/covered, the charge reflects the real service.
    substitution_cases = [
        ("99213","Office visit established moderate complexity",1300,2200,
         "Z00.00","Routine adult health exam without abnormal findings",
         "Cosmetic consultation coded as medical office visit — substituted to get coverage"),
        ("71046","Chest X-ray 2 views",                        750,1500,
         "J06.9","Acute upper respiratory infection unspecified",
         "Elective screening CT substituted with diagnostic chest X-ray code for coverage"),
        ("93000","Electrocardiogram routine ECG",              400,800,
         "Z13.6","Encounter for screening for cardiovascular disorders",
         "Non-covered executive wellness ECG substituted with diagnostic ECG code"),
        ("85025","Complete blood count with differential",     250,500,
         "Z00.00","Routine adult health exam without abnormal findings",
         "Non-covered wellness lab substituted with diagnostic CBC code for coverage"),
        ("G0101","Cervical or vaginal cancer screening",       500,900,
         "Z12.31","Encounter for screening for cervical cancer",
         "Elective procedure coded as covered preventive screening"),
        ("29881","Knee arthroscopy with meniscectomy",       12000,20000,
         "M25.511","Pain in right shoulder",
         "Non-covered cosmetic knee procedure substituted with therapeutic arthroscopy code"),
    ]
    case = random.choice(substitution_cases)
    code, desc, pmin, pmax, icd, icd_desc, explanation = case
    price = rand_price(pmin, pmax)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   [code],
        "procedure_descs":   [desc],
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [icd_desc],
        "modifiers":         [],
        "line_charges":      [price],
        "total_charge":      price,
        "fraud_type":        "Code Substitution",
        "fraud_indicator":   True,
        "fraud_explanation": explanation,
    }


def make_modifier_abuse(idx):
    """Misuse modifier -59 to bypass NCCI bundling rules."""
    bundle = random.choice(NCCI_BUNDLES)
    code_a, code_b = bundle

    info_a = ALL_CPT.get(code_a, {"desc": f"Service {code_a}", "price_min":50,"price_max":200})
    info_b = ALL_CPT.get(code_b, {"desc": f"Service {code_b}", "price_min":50,"price_max":200})

    price_a = rand_price(info_a["price_min"], info_a["price_max"])
    price_b = rand_price(info_b["price_min"], info_b["price_max"])
    total   = round(price_a + price_b, 2)

    # Pick a valid ICD for code_a
    valid_icds = VALID_CPT_ICD.get(code_a, ["I10"])
    icd = random.choice(valid_icds)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   [code_a, code_b],
        "procedure_descs":   [info_a["desc"], info_b["desc"]],
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [ICD10[icd]],
        "modifiers":         ["-59", "-59"],
        "line_charges":      [price_a, price_b],
        "total_charge":      total,
        "fraud_type":        "Modifier Abuse (-59)",
        "fraud_indicator":   True,
        "fraud_explanation": f"Modifier -59 applied to NCCI-bundled pair {code_a} + {code_b} to bypass bundling rules and bill both separately",
    }


def make_duplicate_billing(idx):
    """Submit the same code more than once for the same visit."""
    cats  = ["lab_basic","office_visit_low","cardiac","screening","imaging_basic"]
    cat   = random.choice(cats)
    entry = random.choice(CPT_BY_CATEGORY[cat])
    code, desc, pmin, pmax = entry

    valid_icds = VALID_CPT_ICD.get(code, ["Z00.00"])
    icd    = random.choice(valid_icds)
    price1 = rand_price(pmin, pmax)
    price2 = rand_price(pmin, pmax)
    total  = round(price1 + price2, 2)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   [code, code],
        "procedure_descs":   [desc, desc],
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [ICD10[icd]],
        "modifiers":         [],
        "line_charges":      [price1, price2],
        "total_charge":      total,
        "fraud_type":        "Duplicate Billing",
        "fraud_indicator":   True,
        "fraud_explanation": f"Procedure {code} ({desc}) billed twice on the same claim for the same visit date",
    }


def make_screening_code_abuse(idx):
    """Use a screening code without the required supporting diagnosis."""
    abuse_cases = [
        ("G0103","PSA screening",                      20,60,
         "I10","Essential hypertension",
         "PSA screening requires routine exam diagnosis (Z00.00) — hypertension does not qualify"),
        ("G0101","Cervical or vaginal cancer screening",30,80,
         "M54.5","Low back pain",
         "Cervical screening requires Z12.31 or Z00.00 — low back pain does not qualify"),
        ("82270","Blood occult fecal hemoglobin",      10,35,
         "I10","Essential hypertension",
         "FOBT requires colorectal screening diagnosis (Z12.11) — hypertension does not qualify"),
        ("G0107","Colorectal cancer screening FOBT",   15,40,
         "E11.9","Type 2 diabetes mellitus without complications",
         "Colorectal screening requires Z12.11 — diabetes diagnosis does not qualify"),
        ("80061","Lipid panel",                        25,70,
         "J06.9","Acute upper respiratory infection unspecified",
         "Lipid panel requires cardiovascular screening context (Z13.6) — URI does not qualify"),
    ]
    case = random.choice(abuse_cases)
    code, desc, pmin, pmax, icd, icd_desc, explanation = case
    price = rand_price(pmin, pmax)

    return {
        "claim_id":          make_claim_id(idx),
        "date":              rand_date(),
        "patient":           rand_patient(),
        "provider":          rand_provider(),
        "procedure_codes":   [code],
        "procedure_descs":   [desc],
        "diagnosis_codes":   [icd],
        "diagnosis_descs":   [icd_desc],
        "modifiers":         [],
        "line_charges":      [price],
        "total_charge":      price,
        "fraud_type":        "Screening Code Abuse",
        "fraud_indicator":   True,
        "fraud_explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — generate dataset
# ══════════════════════════════════════════════════════════════════════════════

FRAUD_GENERATORS = [
    ("Upcoding",             make_upcoding,           120),
    ("Unbundling",           make_unbundling,         100),
    ("Code Padding",         make_code_padding,       100),
    ("Phantom Billing",      make_phantom_billing,    100),
    ("Diagnosis Mismatch",   make_diagnosis_mismatch, 120),
    ("Code Substitution",    make_code_substitution,  100),
    ("Modifier Abuse (-59)", make_modifier_abuse,     100),
    ("Duplicate Billing",    make_duplicate_billing,   80),
    ("Screening Code Abuse", make_screening_code_abuse, 80),
]

N_LEGITIMATE = 400
TOTAL_FRAUD  = sum(n for _, _, n in FRAUD_GENERATORS)

print(f"Generating {N_LEGITIMATE} legitimate + {TOTAL_FRAUD} fraud = {N_LEGITIMATE+TOTAL_FRAUD} total claims...")

claims = []
idx    = 1000

# Legitimate
for _ in range(N_LEGITIMATE):
    claims.append(make_legitimate(idx))
    idx += 1

# Fraud
for fraud_name, gen_fn, count in FRAUD_GENERATORS:
    for _ in range(count):
        claims.append(gen_fn(idx))
        idx += 1

# Shuffle
random.shuffle(claims)

# Save JSON
json_path = OUTPUT_DIR / "claims.json"
with open(json_path, "w") as f:
    json.dump(claims, f, indent=2)

# Save CSV summary
csv_path = OUTPUT_DIR / "claims_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["claim_id","date","patient_name","provider_name",
                "procedure_codes","diagnosis_codes","modifiers",
                "total_charge","fraud_type","fraud_indicator","fraud_explanation"])
    for c in claims:
        w.writerow([
            c["claim_id"],
            c["date"],
            c["patient"]["name"],
            c["provider"]["name"],
            "; ".join(c["procedure_codes"]),
            "; ".join(c["diagnosis_codes"]),
            "; ".join(c["modifiers"]),
            c["total_charge"],
            c["fraud_type"],
            c["fraud_indicator"],
            c.get("fraud_explanation",""),
        ])

# Print summary
from collections import Counter
counts = Counter(c["fraud_type"] for c in claims)
print(f"\n{'─'*45}")
print(f"  Dataset generated: {len(claims)} total claims")
print(f"{'─'*45}")
for ft, n in sorted(counts.items(), key=lambda x: x[0]):
    flag = "✅" if ft == "Legitimate" else "🚨"
    print(f"  {flag}  {ft:<28} {n:>4}")
print(f"{'─'*45}")
print(f"  Saved to: {json_path}")
print(f"  CSV at:   {csv_path}")