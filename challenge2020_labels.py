"""
Static label metadata for the PhysioNet/CinC Challenge 2020 12-lead ECG dataset.

This module has **no project imports** (so `config.py` can import it without a
cycle). It defines:

- ``SCORED``            : the 27 officially scored SNOMED CT codes -> abbreviation
- ``EQUIVALENT``       : the 3 code pairs the Challenge scores as identical
- ``CLASSES_SCORED``   : the resulting **24** canonical multi-label classes
- ``DESCRIPTIONS``     : a short English phrase per class (for CLIP text prompts)
- ``SUPERCLASS_MAP``   : an *approximate* SNOMED -> PTB-XL superclass mapping
                         (NORM/MI/STTC/CD/HYP) for cross-dataset comparison
- helpers to turn a set of raw SNOMED codes into a label vector, and to build
  BiomedCoOp-style descriptive prompt ensembles.

Source of the scored/unscored code tables:
    https://github.com/physionetchallenges/evaluation-2020
    (dx_mapping_scored.csv, dx_mapping_unscored.csv)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 27 officially scored SNOMED CT codes -> abbreviation and long name.
# (code, abbreviation, long description)
# ---------------------------------------------------------------------------
_SCORED_ROWS = [
    ("270492004", "IAVB",   "first degree atrioventricular block"),
    ("164889003", "AF",     "atrial fibrillation"),
    ("164890007", "AFL",    "atrial flutter"),
    ("426627000", "Brady",  "bradycardia"),
    ("713427006", "CRBBB",  "complete right bundle branch block"),
    ("713426002", "IRBBB",  "incomplete right bundle branch block"),
    ("445118002", "LAnFB",  "left anterior fascicular block"),
    ("39732003",  "LAD",    "left axis deviation"),
    ("164909002", "LBBB",   "left bundle branch block"),
    ("251146004", "LQRSV",  "low QRS voltages"),
    ("698252002", "NSIVCB", "nonspecific intraventricular conduction block"),
    ("10370003",  "PR",     "pacing rhythm"),
    ("284470004", "PAC",    "premature atrial contraction"),
    ("427172004", "PVC",    "premature ventricular contraction"),
    ("164947007", "LPR",    "prolonged PR interval"),
    ("111975006", "LQT",    "prolonged QT interval"),
    ("164917005", "QAb",    "abnormal Q wave"),
    ("47665007",  "RAD",    "right axis deviation"),
    ("59118001",  "RBBB",   "right bundle branch block"),
    ("427393009", "SA",     "sinus arrhythmia"),
    ("426177001", "SB",     "sinus bradycardia"),
    ("426783006", "NSR",    "sinus rhythm"),
    ("427084000", "STach",  "sinus tachycardia"),
    ("63593006",  "SVPB",   "supraventricular premature beat"),
    ("164934002", "TAb",    "abnormal T wave"),
    ("59931005",  "TInv",   "T wave inversion"),
    ("17338001",  "VPB",    "ventricular premature beat"),
]

SCORED: dict[str, str] = {code: abbr for code, abbr, _ in _SCORED_ROWS}
SCORED_LONG: dict[str, str] = {code: name for code, _, name in _SCORED_ROWS}

# The Challenge scores these code pairs as the *same* diagnosis. We merge each
# pair onto a single canonical column (secondary code -> primary code).
EQUIVALENT: dict[str, str] = {
    "713427006": "59118001",   # CRBBB -> RBBB
    "63593006":  "284470004",  # SVPB  -> PAC
    "17338001":  "427172004",  # VPB   -> PVC
}


def canonical_code(code: str) -> str:
    """Collapse Challenge-equivalent SNOMED codes onto one representative code."""
    return EQUIVALENT.get(code, code)


# Ordered list of the 24 canonical scored classes (abbreviations), keeping the
# table order and dropping the merged-away secondary codes.
CLASSES_SCORED: list[str] = [
    SCORED[code]
    for code, _, _ in _SCORED_ROWS
    if code not in EQUIVALENT
]

# abbreviation -> representative SNOMED code (for the 24 canonical classes)
ABBR_TO_CODE: dict[str, str] = {
    SCORED[code]: code for code, _, _ in _SCORED_ROWS if code not in EQUIVALENT
}
# representative SNOMED code -> abbreviation
CODE_TO_ABBR: dict[str, str] = {code: abbr for abbr, code in ABBR_TO_CODE.items()}

# Short English descriptions used to build CLIP text prompts / captions.
DESCRIPTIONS: dict[str, str] = {
    "IAVB":   "first degree atrioventricular block",
    "AF":     "atrial fibrillation",
    "AFL":    "atrial flutter",
    "Brady":  "bradycardia",
    "IRBBB":  "incomplete right bundle branch block",
    "LAnFB":  "left anterior fascicular block",
    "LAD":    "left axis deviation",
    "LBBB":   "left bundle branch block",
    "LQRSV":  "low QRS voltages",
    "NSIVCB": "nonspecific intraventricular conduction block",
    "PR":     "pacing rhythm",
    "PAC":    "premature atrial contraction",
    "PVC":    "premature ventricular contraction",
    "LPR":    "prolonged PR interval",
    "LQT":    "prolonged QT interval",
    "QAb":    "an abnormal Q wave",
    "RAD":    "right axis deviation",
    "RBBB":   "right bundle branch block",
    "SA":     "sinus arrhythmia",
    "SB":     "sinus bradycardia",
    "NSR":    "normal sinus rhythm",
    "STach":  "sinus tachycardia",
    "TAb":    "an abnormal T wave",
    "TInv":   "T wave inversion",
}

# ---------------------------------------------------------------------------
# Approximate SNOMED CT -> PTB-XL diagnostic superclass mapping.
#
# Lets you evaluate Challenge 2020 with the *same* 5 classes the PTB-XL pipeline
# uses (NORM, MI, STTC, CD, HYP). This is deliberately partial and includes some
# UNSCORED codes (MI / hypertrophy are not in the scored set), so it draws on the
# full Dx list of a record, not only the scored codes. Treat it as a convenience
# for rough cross-dataset comparison, not a clinically exhaustive mapping.
# ---------------------------------------------------------------------------
SUPERCLASS_MAP: dict[str, str] = {
    # --- NORM ---
    "426783006": "NORM",   # sinus rhythm

    # --- CD: conduction disturbance ---
    "270492004": "CD",     # 1st degree AV block
    "195042002": "CD",     # 2nd degree AV block (unscored)
    "233917008": "CD",     # AV block (unscored)
    "27885002":  "CD",     # complete heart block (unscored)
    "54016002":  "CD",     # mobitz type I (unscored)
    "28189009":  "CD",     # mobitz type II (unscored)
    "713427006": "CD",     # complete RBBB
    "59118001":  "CD",     # RBBB
    "713426002": "CD",     # incomplete RBBB
    "164909002": "CD",     # LBBB
    "445118002": "CD",     # left anterior fascicular block
    "445211001": "CD",     # left posterior fascicular block (unscored)
    "698252002": "CD",     # nonspecific IVCB
    "10370003":  "CD",     # pacing rhythm
    "164947007": "CD",     # prolonged PR interval
    "6374002":   "CD",     # bundle branch block (unscored)

    # --- STTC: ST/T change & repolarization ---
    "164917005": "STTC",   # abnormal Q wave
    "164934002": "STTC",   # abnormal T wave
    "59931005":  "STTC",   # T wave inversion
    "111975006": "STTC",   # prolonged QT interval
    "429622005": "STTC",   # ST depression (unscored)
    "164931005": "STTC",   # ST elevation (unscored)
    "428750005": "STTC",   # nonspecific ST-T change (unscored)
    "426434006": "STTC",   # anterior ischemia (unscored)
    "425419005": "STTC",   # inferior ischaemia (unscored)
    "425623009": "STTC",   # lateral ischaemia (unscored)

    # --- MI (all unscored in Challenge 2020) ---
    "164865005": "MI",     # myocardial infarction
    "57054005":  "MI",     # acute myocardial infarction
    "54329005":  "MI",     # anterior myocardial infarction
    "704997005": "MI",     # inferior myocardial infarction (older/alt)
    "164861001": "MI",     # myocardial ischemia
    "413444003": "MI",     # acute myocardial ischemia

    # --- HYP: hypertrophy / chamber enlargement (all unscored) ---
    "164873001": "HYP",    # left ventricular hypertrophy
    "89792004":  "HYP",    # right ventricular hypertrophy
    "195126007": "HYP",    # atrial hypertrophy
    "446358003": "HYP",    # right atrial hypertrophy
    "67741000119109": "HYP",  # left atrial enlargement
    "446813000": "HYP",    # left atrial hypertrophy
}

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

SUPERCLASS_DESCRIPTIONS = {
    "NORM": "normal ECG",
    "MI":   "myocardial infarction",
    "STTC": "ST/T wave change",
    "CD":   "conduction disturbance",
    "HYP":  "cardiac hypertrophy",
}


# ---------------------------------------------------------------------------
# Label-vector helpers
# ---------------------------------------------------------------------------
def codes_to_scored_labels(codes) -> list[str]:
    """Map a set/iterable of raw SNOMED codes to the canonical 24-class list.

    Equivalent codes are merged; codes outside the scored set are ignored.
    Returns a sorted list of abbreviations.
    """
    out = set()
    for raw in codes:
        code = canonical_code(str(raw).strip())
        abbr = CODE_TO_ABBR.get(code)
        if abbr is not None:
            out.add(abbr)
    return sorted(out)


def codes_to_superclasses(codes) -> list[str]:
    """Map raw SNOMED codes to PTB-XL superclasses via ``SUPERCLASS_MAP``."""
    out = set()
    for raw in codes:
        sc = SUPERCLASS_MAP.get(str(raw).strip())
        if sc is not None:
            out.add(sc)
    return sorted(out)


# ---------------------------------------------------------------------------
# BiomedCoOp descriptive prompt ensembles (generated from frames).
#
# BiomedCoOp needs several descriptive sentences per class. Rather than hand
# writing 24x30 lines, we compose them from clinically-worded frames so every
# class gets a diverse, deterministic ensemble of the requested size.
# ---------------------------------------------------------------------------
_FRAMES = [
    "A 12-lead ECG showing {d}.",
    "This electrocardiogram demonstrates {d}.",
    "An ECG tracing consistent with {d}.",
    "The 12-lead electrocardiogram reveals {d}.",
    "A standard ECG with {d}.",
    "This 12-lead recording is consistent with {d}.",
    "Electrocardiographic findings of {d}.",
    "The tracing shows {d}.",
    "A 12-lead ECG image with features of {d}.",
    "This ECG is compatible with {d}.",
    "An electrocardiogram exhibiting {d}.",
    "A twelve-lead ECG demonstrating {d}.",
    "The recording indicates {d}.",
    "A clinical 12-lead ECG showing {d}.",
    "This tracing is diagnostic of {d}.",
    "An ECG with morphology of {d}.",
    "The 12-lead ECG is characteristic of {d}.",
    "A cardiac tracing showing {d}.",
    "This electrocardiogram is consistent with {d}.",
    "A 12-lead ECG revealing {d}.",
    "The ECG waveform shows {d}.",
    "A diagnostic 12-lead ECG with {d}.",
    "This ECG demonstrates the pattern of {d}.",
    "An electrocardiographic recording of {d}.",
    "The 12-lead tracing displays {d}.",
    "A resting 12-lead ECG showing {d}.",
    "This recording captures {d}.",
    "An ECG image demonstrating {d}.",
    "The electrocardiogram is typical of {d}.",
    "A 12-lead ECG whose morphology indicates {d}.",
]


def build_prompt_ensemble(description: str, n: int) -> list[str]:
    """Return ``n`` descriptive sentences for a class description.

    Cycles through the frames if ``n`` exceeds the number of frames so the
    caller always gets exactly ``n`` non-empty prompts.
    """
    if n <= 0:
        return []
    out = []
    i = 0
    while len(out) < n:
        frame = _FRAMES[i % len(_FRAMES)]
        suffix = "" if i < len(_FRAMES) else f" (variant {i // len(_FRAMES) + 1})"
        out.append(frame.format(d=description) + suffix)
        i += 1
    return out[:n]


def build_templates_dict(classes, descriptions, n: int) -> dict[str, list[str]]:
    """Build ``{class: [prompt, ...]}`` for BiomedCoOp for the given classes."""
    return {
        c: build_prompt_ensemble(descriptions.get(c, c), n)
        for c in classes
    }
