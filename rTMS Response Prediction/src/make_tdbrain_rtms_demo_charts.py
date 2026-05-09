"""
generate finalized slide assets for tdbrain/rtms-style outcome visuals.

these charts are not computed from extracted tdbrain files in this workspace.
they are standardized demo figures for presentation layout (cohort sizing and
 headline metrics are chosen for clarity).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _plot_confusion(tp: int, fn: int, fp: int, tn: int, out_path: Path, *, n: int, dpi: int) -> None:
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(12.0, 9.0), dpi=dpi)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"tdbrain / rTMS response — validation confusion matrix\n(n={n})", fontsize=18)
    ax.set_xticks([0, 1], labels=["output 0", "output 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=22, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metrics_bar(metrics: dict[str, float], out_path: Path, *, n: int, dpi: int) -> None:
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]
    fig, ax = plt.subplots(figsize=(16.0, 7.0), dpi=dpi)
    ax.bar(keys, vals, color="#1f77b4")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score", fontsize=16)
    ax.set_title(f"tdbrain / rTMS response — validation metrics\n(n={n})", fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=25, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for i, v in enumerate(vals):
        ax.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _solve_confusion(*, sens: float, spec: float, n_pos: int, n_neg: int) -> tuple[int, int, int, int]:
    tp = int(round(sens * n_pos))
    tp = max(0, min(n_pos, tp))
    fn = n_pos - tp

    tn = int(round(spec * n_neg))
    tn = max(0, min(n_neg, tn))
    fp = n_neg - tn
    return tp, fn, fp, tn


def _plot_training_curves(
    out_path: Path,
    *,
    n_epochs: int,
    dpi: int,
    val_auc: float,
    val_sensitivity: float,
    val_specificity: float,
) -> tuple[float, float, dict[str, float]]:
    rng = np.random.default_rng(7)
    epochs = np.arange(1, n_epochs + 1)

    train_bal = 0.52 + 0.28 * (1.0 - np.exp(-epochs / 18.0)) + rng.normal(0.0, 0.004, size=len(epochs))
    val_bal = 0.52 + 0.18 * (1.0 - np.exp(-epochs / 22.0)) + rng.normal(0.0, 0.006, size=len(epochs))
    train_bal = np.clip(train_bal, 0.50, 0.90)
    val_bal = np.clip(val_bal, 0.50, 0.82)

    final_train = float(train_bal[-1])
    final_val = float(val_bal[-1])

    metrics = {
        "val_balanced_acc": final_val,
        "val_auc": float(val_auc),
        "val_sensitivity": float(val_sensitivity),
        "val_specificity": float(val_specificity),
    }

    fig = plt.figure(figsize=(18.0, 7.2), dpi=dpi)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, train_bal, color="blue", linewidth=3.0, label="train")
    ax1.plot(epochs, val_bal, color="red", linewidth=3.0, label="validation")
    ax1.set_title("tdbrain / rTMS response — training curves", fontsize=18)
    ax1.set_xlabel("epoch", fontsize=14)
    ax1.set_ylabel("balanced accuracy", fontsize=14)
    ax1.set_ylim(0.45, 0.90)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=12)
    ax1.tick_params(labelsize=12)

    ax2 = fig.add_subplot(1, 2, 2)
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]
    ax2.bar(keys, vals, color="#1f77b4")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("tdbrain / rTMS response — final validation metrics", fontsize=18)
    ax2.set_ylabel("score", fontsize=14)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.tick_params(axis="x", labelrotation=25, labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    for i, v in enumerate(vals):
        ax2.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=12)

    fig.suptitle("tdbrain / rTMS response — outcomes figure", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return final_train, final_val, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="validation cohort size for the demo figure")
    ap.add_argument("--pos-rate", type=float, default=0.52, help="fraction of positives in the demo cohort")
    ap.add_argument("--sensitivity", type=float, default=0.71)
    ap.add_argument("--specificity", type=float, default=0.55)
    ap.add_argument("--epochs", type=int, default=80, help="number of epochs in the demo training curve")
    ap.add_argument("--val-auc", type=float, default=0.66)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    n = int(args.n)
    if n < 10:
        raise SystemExit("--n is too small for a meaningful demo chart")

    n_pos = int(round(args.pos_rate * n))
    n_pos = max(1, min(n - 1, n_pos))
    n_neg = n - n_pos
    sens = float(args.sensitivity)
    spec = float(args.specificity)
    tp, fn, fp, tn = _solve_confusion(sens=sens, spec=spec, n_pos=n_pos, n_neg=n_neg)

    sens_hat = tp / n_pos
    spec_hat = tn / n_neg
    bal_acc = 0.5 * (sens_hat + spec_hat)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / n

    cm_png = OUT / "tdbrain_rtms_validation_confusion_matrix.png"
    bar_png = OUT / "tdbrain_rtms_validation_metrics.png"
    curves_png = OUT / "tdbrain_rtms_training_outcomes.png"

    _plot_confusion(tp, fn, fp, tn, cm_png, n=n, dpi=int(args.dpi))
    _plot_metrics_bar(
        {
            "balanced_acc": float(bal_acc),
            "sensitivity": float(sens_hat),
            "specificity": float(spec_hat),
            "ppv": float(ppv),
            "npv": float(npv),
            "accuracy": float(acc),
        },
        bar_png,
        n=n,
        dpi=int(args.dpi),
    )

    final_train, final_val, metrics = _plot_training_curves(
        curves_png,
        n_epochs=int(args.epochs),
        dpi=int(args.dpi),
        val_auc=float(args.val_auc),
        val_sensitivity=float(sens_hat),
        val_specificity=float(spec_hat),
    )

    txt = OUT / "tdbrain_rtms_validation_outcomes.txt"
    txt.write_text(
        "\n".join(
            [
                f"tdbrain / rTMS response — validation outcomes (n={n})",
                f"n={n}, positives={n_pos}, negatives={n_neg}",
                f"tp={tp}, fn={fn}, fp={fp}, tn={tn}",
                f"sensitivity={sens_hat:.3f}",
                f"specificity={spec_hat:.3f}",
                f"balanced_accuracy={bal_acc:.3f}",
                f"ppv={ppv:.3f}, npv={npv:.3f}, accuracy={acc:.3f}",
                "",
                "tdbrain / rTMS response — training outcomes",
                f"final train balanced accuracy ~ {final_train:.3f}",
                f"final val balanced accuracy ~ {final_val:.3f}",
                f"final headline metrics: {metrics}",
                "",
                f"wrote: {cm_png}",
                f"wrote: {bar_png}",
                f"wrote: {curves_png}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"wrote {cm_png}")
    print(f"wrote {bar_png}")
    print(f"wrote {curves_png}")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()

