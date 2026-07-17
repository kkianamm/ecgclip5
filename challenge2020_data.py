"""
Prepare the PhysioNet/CinC Challenge 2020 12-lead ECG dataset for this repo.

Challenge 2020 ships one WFDB record per ECG: a ``.hea`` header (with the SNOMED
CT diagnosis codes on a ``#Dx:`` line) and a ``.mat`` signal file. Records come
from several source databases and differ in sampling rate (257-1000 Hz) and
length (6 s - 30 min), and there is **no official train/val/test split**.

This script produces the *same* artifacts the PTB-XL pipeline consumes, so every
downstream script (zero_shot_eval / extract_features / linear_probe /
finetune_clip / train_biomedcoop) works unchanged:

    <WORK_DIR>/labels.csv          index=ecg_id, columns:
                                     filename, source, fs, strat_fold, report,
                                     superclasses (pipe-joined active labels),
                                     and one 0/1 column per class in C.CLASSES
    <WORK_DIR>/images/<id>.png     rendered ECG image per record

Run (after `export DATASET=challenge2020` and setting DATA_DIR):

    python challenge2020_data.py --limit 500     # quick check
    python challenge2020_data.py                 # full: labels + all images
    python challenge2020_data.py --no-render      # metadata only

Class set is chosen in config via C2020_CLASS_SET=scored (default, 24 classes)
or C2020_CLASS_SET=superclass (5 PTB-XL superclasses via an approximate map).
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

import config as C
import challenge2020_labels as CH
from ecg_to_image import render_to_file
from prepare_data import image_path_for

_DX_RE = re.compile(r"^#\s*Dx\s*:\s*(.*)$", re.IGNORECASE)
_AGE_RE = re.compile(r"^#\s*Age\s*:\s*(.*)$", re.IGNORECASE)
_SEX_RE = re.compile(r"^#\s*Sex\s*:\s*(.*)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Header parsing (text-only; does NOT read the signal, so it is fast)
# ---------------------------------------------------------------------------
def parse_header(hea_path: str) -> dict:
    """Parse a Challenge-2020 ``.hea`` file into a small metadata dict."""
    name = None
    fs = None
    sig_len = None
    dx_codes: list[str] = []
    age = ""
    sex = ""

    with open(hea_path, "r", errors="ignore") as fh:
        for line_no, line in enumerate(fh):
            line = line.rstrip("\n")
            if line_no == 0:
                # "<name> <n_sig> <fs> <n_samples> ..."
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    try:
                        fs = int(float(parts[2]))
                        sig_len = int(float(parts[3]))
                    except ValueError:
                        pass
                continue
            if not line.startswith("#"):
                continue
            m = _DX_RE.match(line)
            if m:
                dx_codes = [c.strip() for c in m.group(1).split(",") if c.strip()]
                continue
            m = _AGE_RE.match(line)
            if m:
                age = m.group(1).strip()
                continue
            m = _SEX_RE.match(line)
            if m:
                sex = m.group(1).strip()

    if name is None:
        name = os.path.splitext(os.path.basename(hea_path))[0]
    return {
        "name": name,
        "fs": fs,
        "sig_len": sig_len,
        "dx_codes": dx_codes,
        "age": age,
        "sex": sex,
    }


def find_headers(data_dir: str) -> list[str]:
    """Recursively list every ``.hea`` file under ``data_dir`` (sorted)."""
    paths = glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True)
    return sorted(paths)


def build_report(meta: dict, active_labels: list[str]) -> str:
    """A short English report string (used by finetune_clip --caption report)."""
    codes = meta["dx_codes"]
    long_names = [CH.SCORED_LONG.get(CH.canonical_code(c)) for c in codes]
    long_names = [n for n in long_names if n]
    demo = []
    if meta.get("age"):
        demo.append(f"age {meta['age']}")
    if meta.get("sex"):
        demo.append(str(meta["sex"]).lower())
    demo_str = f" ({', '.join(demo)})" if demo else ""
    if long_names:
        findings = ", ".join(sorted(set(long_names)))
    elif active_labels:
        findings = ", ".join(C.CLASS_DESCRIPTIONS.get(a, a) for a in active_labels)
    else:
        findings = "no scored diagnosis"
    return f"12-lead ECG{demo_str}. Findings: {findings}."


# ---------------------------------------------------------------------------
# Metadata table
# ---------------------------------------------------------------------------
def load_metadata(data_dir: str = C.DATA_DIR, limit: int | None = None) -> pd.DataFrame:
    """Scan all headers and return a DataFrame indexed by record name.

    Columns: filename (relative, no extension), source, fs, sig_len, report,
    superclasses (pipe-joined active labels), and one 0/1 column per class.
    """
    class_set = getattr(C, "C2020_CLASS_SET", "scored")
    to_labels = (
        CH.codes_to_scored_labels
        if class_set == "scored"
        else CH.codes_to_superclasses
    )

    hea_paths = find_headers(data_dir)
    if not hea_paths:
        raise FileNotFoundError(
            f"No .hea files found under {data_dir!r}. Point DATA_DIR at the "
            "folder containing the extracted Challenge 2020 record directories."
        )
    if limit:
        hea_paths = hea_paths[:limit]

    rows = []
    seen = set()
    for hea in tqdm(hea_paths, desc="Parsing headers"):
        meta = parse_header(hea)
        name = meta["name"]
        if name in seen:  # guard against duplicate record names across sources
            name = f"{os.path.basename(os.path.dirname(hea))}_{name}"
        seen.add(name)

        rel = os.path.relpath(os.path.splitext(hea)[0], data_dir)
        source = rel.split(os.sep)[0] if os.sep in rel else "root"
        active = to_labels(meta["dx_codes"])

        row = {
            "ecg_id": name,
            "filename": rel,
            "source": source,
            "fs": meta["fs"] if meta["fs"] else 0,
            "sig_len": meta["sig_len"] if meta["sig_len"] else 0,
            "report": build_report(meta, active),
            "superclasses": "|".join(active),
        }
        for cls in C.CLASSES:
            row[cls] = 1.0 if cls in active else 0.0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("ecg_id")
    return df


# ---------------------------------------------------------------------------
# Reproducible multi-label stratified 10-fold split (iterative stratification,
# Sechidis et al. 2011). Deterministic given the seed and row order.
# ---------------------------------------------------------------------------
def stratified_folds(
    y: np.ndarray,
    n_folds: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Assign each sample (row of multi-hot ``y``) to one of ``n_folds`` folds.

    Balances per-class positives across folds. Returns an int array of fold ids
    in ``1..n_folds`` (matching PTB-XL's 1-indexed ``strat_fold``).
    """
    n_samples, n_classes = y.shape
    rng = np.random.RandomState(seed)

    # Fold proportions are uniform; desired positives per (fold, class).
    proportion = np.full(n_folds, 1.0 / n_folds)
    class_totals = y.sum(axis=0)
    desired = np.outer(proportion, class_totals)          # (n_folds, n_classes)
    desired_size = proportion * n_samples                 # (n_folds,)
    fold_size = np.zeros(n_folds)

    # Process rarest-label samples first (harder to balance) with a stable,
    # seeded tie-break so the result is deterministic.
    order = np.lexsort((rng.rand(n_samples), y.sum(axis=1)))
    assignment = np.full(n_samples, -1, dtype=int)

    for idx in order:
        labels = np.nonzero(y[idx])[0]
        if labels.size == 0:
            # No active class: balance purely on fold size.
            fold = int(np.argmax(desired_size - fold_size))
        else:
            # Pick the label this sample has that is currently rarest overall,
            # then the fold that most needs that label.
            rarest = labels[np.argmin(class_totals[labels])]
            need = desired[:, rarest]
            best = np.max(need)
            candidates = np.nonzero(need >= best - 1e-9)[0]
            if candidates.size > 1:  # tie-break on remaining capacity
                cap = desired_size[candidates] - fold_size[candidates]
                candidates = candidates[np.nonzero(cap >= cap.max() - 1e-9)[0]]
            fold = int(candidates[0])
            desired[fold, labels] -= 1
        assignment[idx] = fold
        fold_size[fold] += 1
        desired_size_gap = desired_size[fold]  # noqa: F841 (readability)

    return assignment + 1  # 1-indexed to match strat_fold


def add_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    y = df[C.CLASSES].values.astype(np.int64)
    df = df.copy()
    df["strat_fold"] = stratified_folds(y, n_folds=10, seed=seed)
    return df


# ---------------------------------------------------------------------------
# Rendering: crop/pad each record to a fixed window at its native rate
# ---------------------------------------------------------------------------
def load_window(data_dir: str, filename: str, window_seconds: float):
    """Return (signal[window, 12] in mV, fs) cropped/padded to the window.

    Centered crop for long records (e.g. INCART Holter), zero-padded if short.
    """
    signal, fields = wfdb.rdsamp(os.path.join(data_dir, filename))
    signal = np.asarray(signal, dtype=np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    fs = int(fields["fs"])
    target = int(round(window_seconds * fs))

    n = signal.shape[0]
    if n > target:
        start = (n - target) // 2
        signal = signal[start:start + target]
    elif n < target:
        pad = np.zeros((target - n, signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, pad])
    return signal, fs


def render_all(df: pd.DataFrame, data_dir: str = C.DATA_DIR, limit: int | None = None):
    window = getattr(C, "C2020_WINDOW_SECONDS", 10.0)
    ids = df.index.tolist()
    if limit:
        ids = ids[:limit]
    failures = 0
    for ecg_id in tqdm(ids, desc="Rendering ECG images"):
        out = image_path_for(ecg_id)
        if os.path.exists(out):
            continue
        try:
            signal, fs = load_window(data_dir, df.loc[ecg_id, "filename"], window)
            render_to_file(signal, fs, out)
        except Exception as exc:  # keep going; report a count at the end
            failures += 1
            if failures <= 10:
                print(f"  ! failed to render {ecg_id}: {exc}")
    if failures:
        print(f"WARNING: {failures} records failed to render.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if C.DATASET != "challenge2020":
        raise SystemExit(
            "config.DATASET is not 'challenge2020'. Run with "
            "`DATASET=challenge2020 python challenge2020_data.py ...`"
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N records (quick test)")
    ap.add_argument("--no-render", action="store_true",
                    help="build metadata only, do not render images")
    ap.add_argument("--data-dir", default=C.DATA_DIR)
    args = ap.parse_args()

    print(f"Dataset: challenge2020 | class set: {C.C2020_CLASS_SET} "
          f"| {len(C.CLASSES)} classes")
    print(f"Scanning {args.data_dir} ...")
    df = load_metadata(args.data_dir, limit=args.limit)

    seed = getattr(C, "C2020_SPLIT_SEED", 42)
    df = add_split(df, seed=seed)

    train = df.strat_fold.isin(C.TRAIN_FOLDS)
    val = df.strat_fold == C.VAL_FOLD
    test = df.strat_fold == C.TEST_FOLD
    print(f"Records: {len(df)} | train {int(train.sum())} "
          f"val {int(val.sum())} test {int(test.sum())}")
    if "source" in df.columns:
        print("By source: " + ", ".join(
            f"{s}={n}" for s, n in df["source"].value_counts().items()
        ))

    print("Positives per class:")
    for cls in C.CLASSES:
        print(f"  {cls:7s}: {int(df[cls].sum())}")

    out_csv = os.path.join(C.WORK_DIR, "labels.csv")
    df.to_csv(out_csv)
    print(f"Saved labels -> {out_csv}")

    if not args.no_render:
        render_all(df, data_dir=args.data_dir, limit=args.limit)
        print(f"Images in {C.IMG_DIR}")


if __name__ == "__main__":
    main()
