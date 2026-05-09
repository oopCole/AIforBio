from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    balanced_accuracy: float
    confusion: np.ndarray
    report: str


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        confusion=confusion_matrix(y_true, y_pred),
        report=classification_report(y_true, y_pred, digits=4),
    )


def load_beed(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    x_cols = [f"X{i}" for i in range(1, 17)]
    missing = [c for c in x_cols + ["y"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in BEED_Data.csv: {missing}")

    X = df[x_cols].astype(float)
    y = df["y"].astype(int)
    return X, y


def _cv_summary(name: str, model: object, X: pd.DataFrame, y: pd.Series, *, seed: int) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy"},
        n_jobs=-1,
        return_train_score=False,
    )

    def _fmt(key: str) -> str:
        v = scores[f"test_{key}"]
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}"

    print(f"{name} cv accuracy={_fmt('acc')} balanced_accuracy={_fmt('bal_acc')}")


def _print_feature_importance(model: object, X_test: pd.DataFrame, y_test: np.ndarray, *, seed: int) -> None:
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf")
        if clf is None:
            return
        if hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_)
            importance = np.mean(np.abs(coef), axis=0)
            top = np.argsort(-importance)[:10]
            print("top linear weights (mean abs coef):")
            for idx in top:
                print(f"- {X_test.columns[idx]}: {importance[idx]:.6f}")
        return

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
        top = np.argsort(-imp)[:10]
        print("top model feature_importances_:")
        for idx in top:
            print(f"- {X_test.columns[idx]}: {imp[idx]:.6f}")

    try:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=seed,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        pimp = perm.importances_mean
        top = np.argsort(-pimp)[:10]
        print("top permutation importance (balanced_accuracy drop):")
        for idx in top:
            print(f"- {X_test.columns[idx]}: {pimp[idx]:.6f}")
    except Exception:
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(r"C:\Users\coleb\Downloads\beed_+bangalore+eeg+epilepsy+dataset\BEED_Data.csv"),
        help="path to BEED_Data.csv",
    )
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    X, y = load_beed(args.csv)
    print(f"loaded X shape={X.shape} y classes={sorted(y.unique().tolist())}")
    print("class counts:")
    print(y.value_counts().sort_index())

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()

    models: list[tuple[str, object]] = []

    models.append(
        (
            "logreg(multinomial)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        )
    )

    models.append(
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=600,
                random_state=args.seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        )
    )

    print()
    print("cross-validation (5-fold) on full dataset:")
    for name, model in models:
        _cv_summary(name, model, X, y, seed=args.seed)

    best_name: str | None = None
    best_model: object | None = None
    best_bal_acc = -1.0

    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = _evaluate(y_test, pred)
        print()
        print(f"== {name} ==")
        print(f"accuracy={m.accuracy:.4f} balanced_accuracy={m.balanced_accuracy:.4f}")
        print("confusion_matrix:")
        print(m.confusion)
        print("classification_report:")
        print(m.report)
        _print_feature_importance(model, X_test, y_test, seed=args.seed)

        if m.balanced_accuracy > best_bal_acc:
            best_bal_acc = m.balanced_accuracy
            best_name = name
            best_model = model

    if best_model is not None and best_name is not None:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        out = args.artifacts_dir / f"beed_{best_name}_seed{args.seed}.joblib"
        joblib.dump(
            {"model": best_model, "feature_names": X.columns.tolist(), "classes": sorted(y.unique().tolist())},
            out,
        )
        print()
        print(f"saved best model to {out}")


if __name__ == "__main__":
    main()

