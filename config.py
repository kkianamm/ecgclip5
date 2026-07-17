"""
Central configuration for the BiomedCLIP x ECG project.

Two datasets are supported, selected with the ``DATASET`` environment variable
(or ``ECGCLIP_DATASET``):

    DATASET=ptbxl          (default) -> PTB-XL, 5 diagnostic superclasses
    DATASET=challenge2020            -> PhysioNet/CinC Challenge 2020, 12-lead

For PTB-XL, point ``DATA_DIR`` at the folder containing ``ptbxl_database.csv``.
For Challenge 2020, point ``DATA_DIR`` at the folder that contains the extracted
source directories (each with ``*.hea`` / ``*.mat`` records), e.g. the parent of
``WFDB_CPSC2018/``, ``WFDB_Ga/``, ``WFDB_PTBXL/`` ...

Everything else in the repo reads ``config as C`` and keys off ``C.CLASSES``,
``C.CLASS_DESCRIPTIONS``, ``C.PROMPT_TEMPLATES`` and the ``strat_fold`` split, so
switching datasets is just switching this file's active block.
"""
import os

# ----------------------------------------------------------------------------
# Which dataset?
# ----------------------------------------------------------------------------
DATASET = os.environ.get(
    "DATASET", os.environ.get("ECGCLIP_DATASET", "ptbxl")
).strip().lower()
if DATASET in ("challenge-2020", "cinc2020", "c2020"):
    DATASET = "challenge2020"
if DATASET not in ("ptbxl", "challenge2020"):
    raise ValueError(
        f"Unknown DATASET={DATASET!r}; expected 'ptbxl' or 'challenge2020'"
    )

# For Challenge 2020: 'scored' -> 24 canonical scored classes (multi-label,
# the native benchmark); 'superclass' -> the 5 PTB-XL superclasses via an
# approximate SNOMED mapping (for cross-dataset comparison).
C2020_CLASS_SET = os.environ.get("C2020_CLASS_SET", "scored").strip().lower()
if C2020_CLASS_SET not in ("scored", "superclass"):
    raise ValueError("C2020_CLASS_SET must be 'scored' or 'superclass'")

# ----------------------------------------------------------------------------
# Zero-shot prompt engineering (shared: BiomedCLIP's own template + ensemble)
# ----------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    "this is a photo of {}",
    "an electrocardiogram showing {}",
    "a 12-lead ECG with {}",
    "ECG tracing consistent with {}",
]

# ----------------------------------------------------------------------------
# Model (shared)
# ----------------------------------------------------------------------------
BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONTEXT_LENGTH = 256         # BiomedCLIP tokenizer default

# ----------------------------------------------------------------------------
# Training hyper-parameters (shared; used by linear_probe.py / finetune_clip.py)
# ----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4

LP_EPOCHS = 50               # linear probe
LP_LR = 1e-3
LP_WEIGHT_DECAY = 1e-4

FT_EPOCHS = 5                # contrastive fine-tune
FT_LR = 1e-5
FT_WEIGHT_DECAY = 0.1
FT_FREEZE_TEXT = True

# Split convention shared by both datasets: folds 1-8 train, 9 val, 10 test.
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

# BiomedCoOp descriptive-prompt ensemble (filled per dataset below; None -> use
# the curated ecg_prompts.py tables, as PTB-XL does).
BIOMEDCOOP_TEMPLATES = None

# ============================================================================
# Dataset-specific block
# ============================================================================
if DATASET == "ptbxl":
    DATA_DIR = os.environ.get(
        "DATA_DIR", "/lambda/nfs/Kiana2/ecgclip/data/ptbxl"
    )
    _default_work = "./work"

    SAMPLING_RATE = 100       # 100 -> records100 (fast), 500 -> records500
    FILENAME_COL = "filename_lr" if SAMPLING_RATE == 100 else "filename_hr"

    # 5 diagnostic superclasses (multi-label)
    CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
    CLASS_DESCRIPTIONS = {
        "NORM": "normal ECG",
        "MI":   "myocardial infarction",
        "STTC": "ST/T wave change",
        "CD":   "conduction disturbance",
        "HYP":  "cardiac hypertrophy",
    }
    # ecg_prompts.py holds the curated PTB-XL BiomedCoOp templates.

else:  # challenge2020
    import challenge2020_labels as CH

    DATA_DIR = os.environ.get(
        "DATA_DIR", "/lambda/nfs/Kiana2/ecgclip/data/challenge2020"
    )
    _default_work = "./work_challenge2020"

    # Records vary in sampling rate (257-1000 Hz) and length (6 s - 30 min).
    # We crop/pad each record to a fixed window at its native rate for rendering.
    C2020_WINDOW_SECONDS = float(os.environ.get("C2020_WINDOW_SECONDS", "10"))
    C2020_SPLIT_SEED = int(os.environ.get("C2020_SPLIT_SEED", "42"))
    SAMPLING_RATE = None      # not fixed; taken per-record from the header
    FILENAME_COL = "filename"

    if C2020_CLASS_SET == "scored":
        CLASSES = list(CH.CLASSES_SCORED)               # 24 classes
        CLASS_DESCRIPTIONS = dict(CH.DESCRIPTIONS)
    else:  # superclass
        CLASSES = list(CH.SUPERCLASSES)                 # 5 PTB-XL superclasses
        CLASS_DESCRIPTIONS = dict(CH.SUPERCLASS_DESCRIPTIONS)

    # Generate a BiomedCoOp prompt ensemble for the active classes (30 each).
    BIOMEDCOOP_TEMPLATES = CH.build_templates_dict(CLASSES, CLASS_DESCRIPTIONS, 30)

# ----------------------------------------------------------------------------
# Paths (derived; WORK_DIR env overrides the per-dataset default)
# ----------------------------------------------------------------------------
WORK_DIR = os.environ.get("WORK_DIR", _default_work)
IMG_DIR = os.path.join(WORK_DIR, "images")           # rendered ECG PNGs
FEAT_DIR = os.path.join(WORK_DIR, "features")         # cached embeddings
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")      # fine-tuned weights

for _d in (WORK_DIR, IMG_DIR, FEAT_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)
