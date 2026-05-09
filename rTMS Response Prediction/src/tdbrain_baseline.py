from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Band:
    name: str
    lo: float
    hi: float


BANDS: list[Band] = [
    Band("delta", 1.0, 4.0),
    Band("theta", 4.0, 8.0),
    Band("alpha", 8.0, 13.0),
    Band("beta", 13.0, 30.0),
    Band("gamma", 30.0, 45.0),
]


EEG_26 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC3",
    "FCz",
    "FC4",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP3",
    "CPz",
    "CP4",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
]


def _to_float(x: object) -> float | None:
    if x is None:
        return None
    if isinstance(x, (float, int)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"n/a", "na", "nan", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def compute_responder(row: pd.Series) -> int | None:
    bdi_pre = _to_float(row.get("BDI Pre"))
    bdi_post = _to_float(row.get("BDI Post"))
    if bdi_pre is not None and bdi_post is not None and bdi_pre > 0:
        return int(bdi_post <= 0.5 * bdi_pre)

    resp = row.get("Responder")
    if resp is None:
        return None
    s = str(resp).strip().lower()
    if s in {"1", "yes", "y", "true", "responder"}:
        return 1
    if s in {"0", "no", "n", "false", "non-responder", "nonresponder"}:
        return 0
    return None


def extract_bandpower_features(eeg_csv: Path, *, fs_hz: float = 250.0) -> dict[str, float]:
    df = pd.read_csv(eeg_csv)
    missing = [c for c in EEG_26 if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected eeg columns: {missing}")

    x = df[EEG_26].to_numpy(dtype=np.float64, copy=False)
    x = x - np.mean(x, axis=0, keepdims=True)

    nperseg = int(fs_hz * 2)
    nperseg = min(nperseg, x.shape[0])
    freqs, psd = welch(
        x,
        fs=fs_hz,
        nperseg=nperseg,
        axis=0,
        detrend="constant",
        scaling="density",
    )

    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = np.trapz(psd[total_mask, :], freqs[total_mask], axis=0)
    total_power = np.maximum(total_power, 1e-12)

    feats: dict[str, float] = {}
    for band in BANDS:
        m = (freqs >= band.lo) & (freqs < band.hi)
        bp = np.trapz(psd[m, :], freqs[m], axis=0)
        rel = bp / total_power
        for i, ch in enumerate(EEG_26):
            feats[f"{ch}_rel_{band.name}"] = float(rel[i])

    return feats


def build_dataset(
    participants_path: Path,
    derivatives_dir: Path,
    *,
    max_participants: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if not participants_path.is_file():
        raise FileNotFoundError(
            f"participants.tsv not found at {participants_path}. "
            "if you only extracted derivatives, pass --participants path to the full "
            "TDBRAIN participants.tsv from the main dataset package."
        )

    participants = pd.read_csv(participants_path, sep="\t")
    participants["y"] = participants.apply(compute_responder, axis=1)
    participants = participants[participants["y"].notna()].copy()
    participants["y"] = participants["y"].astype(int)

    rows: list[dict[str, float]] = []
    ys: list[int] = []

    for _, row in participants.iterrows():
        pid = str(row["participant_id"])
        sid = row.get("sessID", 1)
        sid_f = _to_float(sid)
        sess = str(int(sid_f)) if sid_f is not None else (str(sid).strip() if sid is not None else "1") or "1"
        eeg_path = (
            derivatives_dir / pid / f"ses-{sess}" / "eeg" / f"{pid}_ses-{sess}_task-restEC_eeg.csv"
        )
        if not eeg_path.is_file():
            continue

        feats = extract_bandpower_features(eeg_path)
        feats["age"] = float(row["age"]) if _to_float(row.get("age")) is not None else np.nan
        feats["gender"] = float(row["gender"]) if _to_float(row.get("gender")) is not None else np.nan
        rows.append(feats)
        ys.append(int(row["y"]))

        if max_participants is not None and len(rows) >= max_participants:
            break

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.Series(ys, name="y")
    return X, y


def train_and_eval(X: pd.DataFrame, y: pd.Series) -> None:
    if len(y) < 10 or y.nunique() < 2:
        print(f"not enough labeled samples to train (n={len(y)}, classes={sorted(y.unique().tolist())})")
        return

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    bal_acc = balanced_accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, prob)
    except ValueError:
        auc = float("nan")

    print(f"n_train={len(y_train)} n_test={len(y_test)}")
    print(f"balanced_accuracy={bal_acc:.3f}")
    if not math.isnan(auc):
        print(f"roc_auc={auc:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="BIDS root folder containing participants.tsv and derivatives/ (optional if both paths set)",
    )
    ap.add_argument(
        "--participants",
        type=Path,
        default=None,
        help="path to participants.tsv (defaults to <root>/participants.tsv)",
    )
    ap.add_argument(
        "--derivatives-dir",
        type=Path,
        default=None,
        help="folder containing sub-*/ses-*/eeg (defaults to <root>/derivatives)",
    )
    ap.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help="stop after this many successfully loaded EEG rows (smoke test)",
    )
    args = ap.parse_args()

    if args.root is None and (args.participants is None or args.derivatives_dir is None):
        ap.error("pass either --root or both --participants and --derivatives-dir")

    participants_path = args.participants or (args.root / "participants.tsv")
    derivatives_dir = args.derivatives_dir or (args.root / "derivatives")

    X, y = build_dataset(
        participants_path,
        derivatives_dir,
        max_participants=args.max_participants,
    )
    print(f"loaded X shape={X.shape} labeled y={len(y)}")
    train_and_eval(X, y)


if __name__ == "__main__":
    main()

