# SRGAN assignment (binary OCT classification)

Train **model A** (ResNet18 transfer learning) on real **128×128** images, train an **SRGAN** (32×128), generate SR training images for **model B**, and compare metrics (accuracy, F1, ROC-AUC) on a shared test split.

## Layout

- `prepare_split.py` — stratified 70/30 split → `outputs/split.pt` (created locally; not committed)
- `train_classifier.py` — model A (real HR) or B (`--train-root` for SRGAN PNG folders)
- `train_srgan.py` — SRGAN on the training split
- `generate_sr_train.py` — write SR images for model B
- `compare_models.py` — evaluate A vs B on the same real test set
- `srgan_assignment.ipynb` — augmentation demos, LR/SR/HR examples, metric comparison

Place the **Srinivasan** dataset under `data_raw/Srinivasan/` (see `data_utils.collect_image_paths`).

## Setup

```bash
pip install -r requirements.txt
python prepare_split.py --data-root data_raw
```

## GitHub

Remote: [https://github.com/oopCole/AIforBio](https://github.com/oopCole/AIforBio)

After installing [Git for Windows](https://git-scm.com/download/win), from this folder run:

```bash
git init
git add .
git commit -m "Add SRGAN assignment: classifiers, SRGAN, notebook"
git branch -M main
git remote add origin https://github.com/oopCole/AIforBio.git
git push -u origin main
```

If the remote already has commits, use `git pull origin main --rebase` before pushing. For authentication, use GitHub HTTPS with a [personal access token](https://github.com/settings/tokens) or `gh auth login`.
