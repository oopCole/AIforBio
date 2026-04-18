from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import SRGANPairDataset, gan_tensor_to_minus1_1, minus1_1_to_gan_tensor, set_seed
from models_srgan import SRGANDiscriminator, SRGANGenerator, VGGPerceptualLoss, weights_init


def spatial_bce_loss(logits: torch.Tensor, target: float) -> torch.Tensor:
    """Average BCE over batch for mean of spatial logits."""
    v = logits.mean(dim=(1, 2, 3))
    t = torch.full_like(v, target)
    return nn.functional.binary_cross_entropy_with_logits(v, t)


def train_epoch(
    G: nn.Module,
    D: nn.Module,
    vgg_loss: VGGPerceptualLoss,
    loader: DataLoader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
    lambda_vgg: float,
    lambda_l1: float,
) -> tuple[float, float]:
    G.train()
    D.train()
    tot_g = 0.0
    tot_d = 0.0
    n = 0
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        lr_m11 = gan_tensor_to_minus1_1(lr)
        hr_m11 = gan_tensor_to_minus1_1(hr)

        # --- D ---
        with torch.no_grad():
            sr = G(lr_m11)
        sr_det = sr.detach()
        logits_real = D(hr_m11)
        logits_fake = D(sr_det)
        loss_d = spatial_bce_loss(logits_real, 1.0) + spatial_bce_loss(logits_fake, 0.0)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # --- G ---
        sr = G(lr_m11)
        logits_fake_g = D(sr)
        loss_adv = spatial_bce_loss(logits_fake_g, 1.0)
        sr01 = minus1_1_to_gan_tensor(sr)
        loss_l1 = nn.functional.l1_loss(sr01, hr)
        loss_v = vgg_loss(sr01, hr)
        loss_g = loss_adv + lambda_l1 * loss_l1 + lambda_vgg * loss_v
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        bs = lr.size(0)
        tot_g += loss_g.item() * bs
        tot_d += loss_d.item() * bs
        n += bs
    return tot_g / max(n, 1), tot_d / max(n, 1)


@torch.no_grad()
def save_samples(
    G: nn.Module,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    max_save: int = 4,
) -> None:
    G.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    import torchvision.utils as vutils

    lr, hr = next(iter(loader))
    lr = lr[:max_save].to(device)
    hr = hr[:max_save].to(device)
    lr_m11 = gan_tensor_to_minus1_1(lr)
    sr = G(lr_m11)
    sr01 = minus1_1_to_gan_tensor(sr)
    grid = torch.cat([lr, sr01, hr], dim=0)
    vutils.save_image(grid, out_dir / "lr_sr_hr_strip.png", nrow=max_save, normalize=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Train SRGAN (32 -> 128)")
    p.add_argument("--data-root", type=str, default="data_raw")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-vgg", type=float, default=1e-3)
    p.add_argument("--lambda-l1", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/srgan")
    p.add_argument("--split-file", type=str, default="outputs/split.pt", help="From prepare_split.py")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from data_utils import collect_image_paths

    split_path = Path(args.split_file)
    if split_path.is_file():
        split = torch.load(split_path, map_location="cpu")
        paths = [Path(s) for s in split["paths"]]
        train_idx = split["train_idx"]
    else:
        paths, labels = collect_image_paths(data_root)
        from data_utils import stratified_split

        train_idx, _ = stratified_split(paths, labels, test_size=0.3, seed=args.seed)

    ds = SRGANPairDataset(paths, train_idx)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    G = SRGANGenerator(n_residual_blocks=8).to(device)
    D = SRGANDiscriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    vgg_loss = VGGPerceptualLoss().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))

    hist_g: list[float] = []
    hist_d: list[float] = []
    for epoch in range(1, args.epochs + 1):
        g_loss, d_loss = train_epoch(
            G,
            D,
            vgg_loss,
            loader,
            opt_g,
            opt_d,
            device,
            lambda_vgg=args.lambda_vgg,
            lambda_l1=args.lambda_l1,
        )
        hist_g.append(g_loss)
        hist_d.append(d_loss)
        print(f"epoch {epoch}/{args.epochs}  G={g_loss:.4f}  D={d_loss:.4f}")
        if epoch % 10 == 0 or epoch == 1:
            save_samples(G, loader, out_dir / f"epoch_{epoch:04d}", device)

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, out_dir / "srgan.pt")
    torch.save({"train_g": hist_g, "train_d": hist_d}, out_dir / "history.pt")
    save_samples(G, loader, out_dir / "final", device)
    print(f"Saved checkpoint to {out_dir / 'srgan.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
