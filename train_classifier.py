from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from data_utils import (
    BINARY_NAMES,
    collect_image_paths,
    eval_transform_classifier,
    set_seed,
    train_transform_classifier,
    ImagePathDataset,
)
from models_classifier import build_resnet18_binary


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, list[float], list[int]]:
    """Returns accuracy, f1, auc, probs positive, labels."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    ys: list[int] = []
    probs: list[float] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pr = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        probs.extend(pr)
        ys.extend(yb.cpu().tolist())
    import numpy as np

    y_arr = np.array(ys)
    p_arr = np.array(probs)
    acc = float(accuracy_score(y_arr, (p_arr >= 0.5).astype(int)))
    f1 = float(f1_score(y_arr, (p_arr >= 0.5).astype(int), zero_division=0))
    try:
        auc = float(roc_auc_score(y_arr, p_arr))
    except ValueError:
        auc = float("nan")
    return acc, f1, auc, probs, ys


def main() -> int:
    p = argparse.ArgumentParser(description="Binary classifier (transfer learning ResNet18)")
    p.add_argument("--split-file", type=str, default="outputs/split.pt")
    p.add_argument("--train-root", type=str, default="", help="If set, torchvision ImageFolder train root (for model B on SR images)")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Default: 40 for real training (A), 150 for --train-root (B)",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/model_a")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = 150 if args.train_root else 40

    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_root:
        # Model B: folder dataset (SRGAN outputs): subfolders 0 and 1
        tr_tf = train_transform_classifier()
        te_tf = eval_transform_classifier()
        train_ds = datasets.ImageFolder(args.train_root, transform=tr_tf)
        # test still from real images via split file
        split = torch.load(args.split_file, map_location="cpu")
        paths = [Path(s) for s in split["paths"]]
        labels = split["labels"]
        test_idx = split["test_idx"]
        test_ds = ImagePathDataset(paths, labels, test_idx, transform=te_tf)
        print(f"Train (SR folder): {len(train_ds)}  Test (real HR): {len(test_ds)}")
    else:
        split = torch.load(args.split_file, map_location="cpu")
        paths = [Path(s) for s in split["paths"]]
        labels = split["labels"]
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        train_ds = ImagePathDataset(paths, labels, train_idx, transform=train_transform_classifier())
        test_ds = ImagePathDataset(paths, labels, test_idx, transform=eval_transform_classifier())
        print(f"Train (real): {len(train_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet18_binary(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    hist: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)
        acc, f1, auc, _, _ = evaluate_metrics(model, test_loader, device)
        hist.append({"epoch": epoch, "train_loss": train_loss, "test_acc": acc, "test_f1": f1, "test_auc": auc})
        print(
            f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
            f"acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_dir / "best.pt")
        torch.save({"history": hist}, out_dir / "history.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    acc, f1, auc, _, _ = evaluate_metrics(model, test_loader, device)
    print(f"Final test: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc}\nf1={f1}\nauc={auc}\nclasses={BINARY_NAMES}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
