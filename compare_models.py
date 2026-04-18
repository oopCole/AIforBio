from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils import BINARY_NAMES, ImagePathDataset, eval_transform_classifier, set_seed
from models_classifier import build_resnet18_binary
from train_classifier import evaluate_metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Compare model A vs B on the same real test set")
    p.add_argument("--split-file", type=str, default="outputs/split.pt")
    p.add_argument("--ckpt-a", type=str, default="outputs/model_a/best.pt")
    p.add_argument("--ckpt-b", type=str, default="outputs/model_b/best.pt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = torch.load(args.split_file, map_location="cpu")
    paths = [Path(s) for s in split["paths"]]
    labels = split["labels"]
    test_idx = split["test_idx"]
    te_tf = eval_transform_classifier()
    test_ds = ImagePathDataset(paths, labels, test_idx, transform=te_tf)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    for name, ckpt in [("A (real HR train)", args.ckpt_a), ("B (SRGAN train)", args.ckpt_b)]:
        if not Path(ckpt).is_file():
            print(f"Missing checkpoint {ckpt}, skip {name}")
            continue
        model = build_resnet18_binary(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        acc, f1, auc, _, _ = evaluate_metrics(model, loader, device)
        rows.append((name, acc, f1, auc))
        print(f"{name}: accuracy={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")

    print(f"\nClasses: {BINARY_NAMES[0]}=0, {BINARY_NAMES[1]}=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
