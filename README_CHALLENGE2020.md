# Adding PhysioNet/CinC Challenge 2020 to this repo

This adds the **PhysioNet/Computing in Cardiology Challenge 2020** 12-lead ECG
dataset alongside PTB-XL, reusing the *same* BiomedCLIP pipeline (zero-shot,
linear probe, contrastive fine-tune, BiomedCoOp). You select the dataset with a
single environment variable; nothing else in the workflow changes.

```bash
export DATASET=challenge2020      # everything below then targets Challenge 2020
export DATASET=ptbxl              # (default) back to the original PTB-XL flow
```

## What was added / changed

| File | Change |
|------|--------|
| `challenge2020_labels.py` | **new** — SNOMED CT tables: the 24 scored classes (27 codes with 3 Challenge-equivalent pairs merged), English descriptions, a BiomedCoOp prompt generator, and an approximate SNOMED→PTB-XL-superclass map. |
| `challenge2020_data.py` | **new** — the data-prep script (the Challenge-2020 analogue of `prepare_data.py`): parses headers, builds labels, makes a stratified split, renders images. |
| `config.py` | dataset switch via `DATASET`; PTB-XL block is byte-for-byte the old behavior. |
| `prepare_data.py` | `image_path_for` now accepts string record IDs (PTB-XL integer names are unchanged). |
| `ecg_prompts.py` | BiomedCoOp falls back to the active dataset's prompts/descriptions when they aren't PTB-XL's. |

The downstream scripts (`zero_shot_eval.py`, `extract_features.py`,
`linear_probe.py`, `finetune_clip.py`, `train_biomedcoop.py`, `evaluation.py`)
are **untouched** — they already key off `config.CLASSES`, the `strat_fold`
split, and `work/labels.csv`, all of which the adapter reproduces.

## Why this dataset needs an adapter (and PTB-XL didn't)

Challenge 2020 differs from PTB-XL in three ways the adapter handles:

1. **Labels are SNOMED CT codes** on a `#Dx:` line in each WFDB `.hea` header,
   not PTB-XL's SCP superclasses. We map them to the 24 officially *scored*
   classes (multi-label), merging the three code pairs the Challenge scores as
   identical (`CRBBB≡RBBB`, `PAC≡SVPB`, `PVC≡VPB`).
2. **Records vary in sampling rate (257–1000 Hz) and length (6 s – 30 min)**
   across the six source databases. Each record is cropped/padded to a fixed
   window (default 10 s, centered) at its **native** rate before rendering, so
   the ECG images stay comparable. The 30-minute INCART Holter records are
   center-cropped.
3. **There is no official train/val/test split.** We build a reproducible
   multi-label **iterative-stratified** 10-fold split and write it as
   `strat_fold`, reusing the repo's convention (folds 1–8 train, 9 val, 10 test).

## Download the data (~7 GB)

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.1/
```

Extract every source tarball into one folder, e.g.:

```
data/challenge2020/
  WFDB_CPSC2018/    A*.hea A*.mat
  WFDB_CPSC2018_2/  Q*.hea Q*.mat
  WFDB_StPetersburg/ I*.hea I*.mat
  WFDB_PTB/         S*.hea S*.mat
  WFDB_PTBXL/       HR*.hea HR*.mat
  WFDB_Ga/          E*.hea E*.mat
```

Point `DATA_DIR` at that parent folder (headers are found recursively):

```bash
export DATA_DIR=/path/to/data/challenge2020
```

> Note: the PTB-XL records inside Challenge 2020 (`WFDB_PTBXL/`) are the *same*
> signals as the standalone PTB-XL dataset, just relabeled with SNOMED codes. If
> you evaluate on both, be aware of that overlap.

## Class set

Chosen with `C2020_CLASS_SET` (default `scored`):

- `scored` — the **24** canonical scored classes; multi-label; the native
  Challenge 2020 benchmark. Metric: macro AUROC/AUPRC over the 24 classes.
- `superclass` — the **5 PTB-XL superclasses** (`NORM/MI/STTC/CD/HYP`) via an
  approximate SNOMED map, for rough cross-dataset comparison with PTB-XL.
  This map draws on scored *and* unscored codes (MI and hypertrophy are not in
  the scored set), and is intentionally partial — use it for comparison, not as
  a clinical ground truth.

Other knobs (env vars): `C2020_WINDOW_SECONDS` (default 10),
`C2020_SPLIT_SEED` (default 42), `WORK_DIR` (default `./work_challenge2020`).

## Run it

```bash
export DATASET=challenge2020
export DATA_DIR=/path/to/data/challenge2020

# 1) Prepare labels + render images (this replaces prepare_data.py for C2020)
python challenge2020_data.py --limit 500     # quick smoke test first
python challenge2020_data.py                 # full run

# 2) Everything else is identical to the PTB-XL instructions:
python zero_shot_eval.py --task multi
python extract_features.py && python linear_probe.py
python finetune_clip.py
python train_biomedcoop.py --shots 16 --seed 1 --task multi
```

`challenge2020_data.py` writes `work_challenge2020/labels.csv` (index `ecg_id`;
columns `filename, source, fs, sig_len, report, superclasses`, one 0/1 column
per class, and `strat_fold`) plus `work_challenge2020/images/<record>.png` — the
exact contract the rest of the repo expects.

## Notes & caveats

- **Single-label mode** (`--task single`) filters to records with exactly one
  active class. With 24 multi-label classes those subsets are smaller and some
  rare classes may have too few examples for large `--shots`; multi-label is the
  primary protocol here.
- **Zero-shot expectations** are the same as the PTB-XL README: BiomedCLIP was
  never trained on ECG plots, so zero-shot on 24 classes will be modest. Linear
  probe and fine-tuning improve substantially.
- **Patient overlap**: most sources are one record per patient; INCART's 74
  records come from 32 patients, so a small amount of within-source patient
  leakage across folds is possible there. The split is record-level (standard
  for this dataset).
- The split and rendering are deterministic given `C2020_SPLIT_SEED` and
  `C2020_WINDOW_SECONDS`.
