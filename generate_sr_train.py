from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data_utils import (
    collect_image_paths,
    gan_tensor_to_minus1_1,
    minus1_1_to_gan_tensor,
    set_seed,
    SRGANPairDataset,
)
from models_srgan import SRGANGenerator


def main() -> int:
    p = argparse.ArgumentParser(description="Generate 128x128 SR images for training model B")
    p.add_argument("--data-root", type=str, default="data_raw")
    p.add_argument("--split-file", type=str, default="outputs/split.pt")
    p.add_argument("--checkpoint", type=str, default="outputs/srgan/srgan.pt")
    p.add_argument("--output-dir", type=str, default="outputs/sr_train_images")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = torch.load(args.split_file, map_location="cpu")
    paths = [Path(s) for s in split["paths"]]
    labels = split["labels"]
    train_idx = split["train_idx"]

    data_root = Path(args.data_root)
    ds = SRGANPairDataset(paths, train_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = torch.load(args.checkpoint, map_location=device)
    G = SRGANGenerator(n_residual_blocks=8).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    out_root = Path(args.output_dir)
    for c in (0, 1):
        (out_root / str(c)).mkdir(parents=True, exist_ok=True)

    idx_global = 0
    with torch.no_grad():
        for batch_i, (lr, _hr) in enumerate(tqdm(loader)):
            lr = lr.to(device, non_blocking=True)
            sr = G(gan_tensor_to_minus1_1(lr))
            sr01 = minus1_1_to_gan_tensor(sr)
            for j in range(sr01.size(0)):
                ti = batch_i * args.batch_size + j
                if ti >= len(train_idx):
                    break
                k = train_idx[ti]
                y = labels[k]
                path = paths[k]
                name = f"{idx_global:06d}_{path.stem}.png"
                save_image(sr01[j], out_root / str(y) / name)
                idx_global += 1

    print(f"Wrote {idx_global} images to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
