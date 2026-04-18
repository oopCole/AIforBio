from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_utils import collect_image_paths, stratified_split, set_seed


def main() -> int:
    p = argparse.ArgumentParser(description="Save 70/30 stratified train/test indices")
    p.add_argument("--data-root", type=str, default="data_raw")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/split.pt")
    args = p.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)
    paths, labels = collect_image_paths(data_root)
    train_idx, test_idx = stratified_split(paths, labels, test_size=args.test_size, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "paths": [str(p) for p in paths],
            "labels": labels,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "seed": args.seed,
            "test_size": args.test_size,
        },
        out,
    )
    print(f"Saved split: train={len(train_idx)} test={len(test_idx)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
