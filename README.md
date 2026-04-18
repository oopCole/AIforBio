# SRGAN assignment (binary OCT classification)

This repo trains a **binary classifier A** (ResNet18 transfer learning) on real **128×128** images, trains an **SRGAN** (32×128), generates synthetic HR training images for **classifier B**, and compares **accuracy, F1, and ROC-AUC** on the same real test split.

## Task and labels

- **Class 0 — NORMAL** (`NORMAL` folder).
- **Class 1 — PATHOLOGY** (`DME` and `DRUSEN` folders merged).

The train/test **index split** is stratified 70/30 with **seed `42`** (`prepare_split.py`). Using the same data paths and seed reproduces the same `outputs/split.pt`; classifiers set `set_seed(42)` before training.

**Note:** PyTorch can still show small run-to-run differences on GPU (non-deterministic ops). For closest replication, use the same PyTorch/CUDA versions and optionally set `CUBLAS_WORKSPACE_CONFIG=:4098:8` and `torch.use_deterministic_algorithms(True)` (not enabled in these scripts by default).

## Requirements

- Python 3.10+ recommended.
- **PyTorch** with CUDA optional but recommended for SRGAN and training speed.

```bash
pip install -r requirements.txt
```

Install a **GPU** build of PyTorch if available ([pytorch.org](https://pytorch.org/get-started/locally/)); the code uses `cuda` when `torch.cuda.is_available()`.

## Dataset layout

Place the **Srinivasan** OCT dataset under `data_raw/` so that both original `train` and `test` folders are included:

```text
data_raw/
  Srinivasan/
    train/
      DME/
      DRUSEN/
      NORMAL/
    test/
      DME/
      DRUSEN/
      NORMAL/
```

Images are read as RGB (`.tif` supported). `data_raw/` is gitignored; obtain the dataset separately.

## End-to-end reproduction (run from this directory)

Create `outputs/` automatically:

### 1. Stratified split

Writes `outputs/split.pt` (paths, labels, `train_idx`, `test_idx`).

```bash
python prepare_split.py --data-root data_raw --test-size 0.3 --seed 42 --output outputs/split.pt
```

### 2. Model A (real HR, 128×128)

Default: **40 epochs**, batch **16**, Adam **lr=1e-4**, ResNet18 ImageNet weights. Saves `outputs/model_a/best.pt`, `last.pt`, `history.pt`, `metrics.txt`.

```bash
python train_classifier.py --split-file outputs/split.pt --output-dir outputs/model_a --seed 42
```

### 3. SRGAN (LR 32×32 → HR 128×128, train split only)

Default: **50 epochs**, batch **8**, **lr=1e-4**, λ_l1=**1e-2**, λ_vgg=**1e-3**, 8 residual blocks. Saves `outputs/srgan/srgan.pt` and sample strips under `outputs/srgan/`.

```bash
python train_srgan.py --data-root data_raw --split-file outputs/split.pt --output-dir outputs/srgan --epochs 50 --seed 42
```

### 4. Generate SR training images for model B

Writes PNGs under `outputs/sr_train_images/0/` and `outputs/sr_train_images/1/` (one file per training index).

```bash
python generate_sr_train.py --split-file outputs/split.pt --checkpoint outputs/srgan/srgan.pt --output-dir outputs/sr_train_images --seed 42
```

### 5. Model B (train on SRGAN outputs, test on real HR)

Default when `--train-root` is set: **150 epochs** (same optimizer/lr/batch defaults as A unless overridden). Saves under `outputs/model_b/`.

```bash
python train_classifier.py --split-file outputs/split.pt --train-root outputs/sr_train_images --output-dir outputs/model_b --seed 42
```

### 6. Compare A vs B on the shared real test set

```bash
python compare_models.py --split-file outputs/split.pt --ckpt-a outputs/model_a/best.pt --ckpt-b outputs/model_b/best.pt
```

### Notebook

Open `srgan_assignment.ipynb` with the working directory set to this repo. It demonstrates augmentations, LR / bicubic / SRGAN / HR visuals (needs a trained `outputs/srgan/srgan.pt` for GAN outputs), and loads the same metrics if checkpoints exist.

## Outputs (local, not in git)

| Path | Contents |
|------|----------|
| `outputs/split.pt` | Stratified indices and label list |
| `outputs/model_a/` | Classifier A checkpoints and `metrics.txt` |
| `outputs/srgan/` | SRGAN checkpoint and sample images |
| `outputs/sr_train_images/` | PNGs for training B |
| `outputs/model_b/` | Classifier B checkpoints |

## Repository

Public remote: [https://github.com/oopCole/AIforBio](https://github.com/oopCole/AIforBio)
