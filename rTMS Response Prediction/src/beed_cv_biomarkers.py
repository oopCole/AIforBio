"""
cv-fold permutation importance for BEED (objective 4).

trains a fresh model per fold on train split only, then scores permutation
importance on the validation fold (no leakage). aggregates which features
appear in the top-k by importance in each fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


def load_beed(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    x_cols = [f"X{i}" for i in range(1, 17)]
    missing = [c for c in x_cols + ["y"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in BEED_Data.csv: {missing}")
    X = df[x_cols].astype(float)
    y = df["y"].astype(int)
    return X, y


def make_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def run_cv_permutation_biomarkers(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    seed: int,
    top_k: int,
    n_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    feature_names = list(X.columns)
    n_features = len(feature_names)

    fold_rows: list[dict[str, object]] = []
    pi_matrix = np.zeros((n_splits, n_features), dtype=np.float64)
    in_top_k = np.zeros(n_features, dtype=np.int32)
    rank_matrix = np.zeros((n_splits, n_features), dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        model = make_model(seed + fold_idx)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        perm = permutation_importance(
            model,
            X.iloc[val_idx],
            y.iloc[val_idx],
            n_repeats=n_repeats,
            random_state=seed,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        pi = np.asarray(perm.importances_mean, dtype=np.float64)
        pi_matrix[fold_idx] = pi

        order = np.argsort(-pi)
        in_top_k[order[:top_k]] += 1

        # rank 1 = highest importance (largest mean drop in score when shuffled)
        rank_matrix[fold_idx] = rankdata(-pi, method="average")

        for j, name in enumerate(feature_names):
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "feature": name,
                    "permutation_importance_mean": float(pi[j]),
                }
            )

    mean_pi = pi_matrix.mean(axis=0)
    std_pi = pi_matrix.std(axis=0)
    mean_rank = rank_matrix.mean(axis=0)

    summary_rows: list[dict[str, object]] = []
    for j, name in enumerate(feature_names):
        summary_rows.append(
            {
                "feature": name,
                "mean_permutation_importance": float(mean_pi[j]),
                "std_permutation_importance": float(std_pi[j]),
                f"folds_in_top_{top_k}": int(in_top_k[j]),
                "mean_rank_across_folds": float(mean_rank[j]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=[f"folds_in_top_{top_k}", "mean_permutation_importance"],
        ascending=[False, False],
    )

    consistent = summary_df[summary_df[f"folds_in_top_{top_k}"] == n_splits]["feature"].tolist()

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, summary_df, consistent


def main() -> None:
    ap = argparse.ArgumentParser(description="BEED: cv-fold permutation importance biomarkers")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(r"C:\Users\coleb\Downloads\beed_+bangalore+eeg+epilepsy+dataset\BEED_Data.csv"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    X, y = load_beed(args.csv)
    print(f"loaded X shape={X.shape} classes={sorted(y.unique().tolist())}")
    print(f"cv={args.n_splits}-fold, top_k={args.top_k}, perm repeats={args.n_repeats}")

    fold_df, summary_df, consistent = run_cv_permutation_biomarkers(
        X,
        y,
        n_splits=args.n_splits,
        seed=args.seed,
        top_k=args.top_k,
        n_repeats=args.n_repeats,
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.artifacts_dir / "beed_permutation_importance_by_fold.csv"
    summary_path = args.artifacts_dir / "beed_biomarker_summary.csv"
    json_path = args.artifacts_dir / "beed_biomarkers_consistent_topk.json"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "definition": (
            f"top {args.top_k} features by mean permutation importance "
            f"(balanced_accuracy) on each validation fold; "
            f"'consistent' = in top-{args.top_k} in all {args.n_splits} folds"
        ),
        "n_splits": args.n_splits,
        "top_k": args.top_k,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "consistent_features_all_folds": consistent,
        "top5_by_mean_importance": summary_df.head(5)["feature"].tolist(),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"wrote {fold_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {json_path}")
    print()
    print(f"features in top-{args.top_k} in every fold ({len(consistent)}):")
    if consistent:
        for name in consistent:
            row = summary_df.loc[summary_df["feature"] == name].iloc[0]
            print(
                f"  - {name}: mean_pi={row['mean_permutation_importance']:.6f}, "
                f"mean_rank={row['mean_rank_across_folds']:.2f}"
            )
    else:
        print("  (none — see summary csv for partial consistency)")

    print()
    print("summary table (sorted by folds-in-top-k, then mean importance):")
    print(summary_df.to_string(index=False))

    # final model on all data for deployment template (same as future tdbrain flow)
    final = make_model(args.seed)
    final.fit(X, y)
    model_path = args.artifacts_dir / f"beed_random_forest_full_cv_seed{args.seed}.joblib"
    joblib.dump(
        {
            "model": final,
            "feature_names": X.columns.tolist(),
            "classes": sorted(y.unique().tolist()),
            "biomarker_summary_path": str(summary_path),
            "biomarker_json_path": str(json_path),
        },
        model_path,
    )
    print()
    print(f"saved full-data model to {model_path}")


if __name__ == "__main__":
    main()

