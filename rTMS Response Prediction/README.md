# rTMS Response Prediction (EEG + ML)

This folder contains a complete end-to-end workflow for EEG-derived modeling:

- BEED epilepsy benchmark: multiclass classification, cross-validation, holdout evaluation, and stable biomarkers via permutation importance.
- TDBRAIN rTMS workflow: encrypted archive extraction helper, responder labeling logic from clinical tables, band-power feature extraction from resting-state EEG, and a baseline classifier.
- Reporting assets: LaTeX report source under `report/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## BEED

Data: `BEED_Data.csv` with `X1`–`X16` and multiclass `y` (0–3).

Run baseline training/evaluation:

```bash
python -m src.beed_baseline --csv "C:\path\to\BEED_Data.csv"
```

Run stable biomarker analysis (fold-wise permutation importance on validation only):

```bash
python -m src.beed_cv_biomarkers --csv "C:\path\to\BEED_Data.csv"
```

Artifacts are written to `artifacts/`.

## TDBRAIN

Run baseline (requires `participants.tsv` and extracted `derivatives/`):

```powershell
python -m src.tdbrain_baseline `
  --participants "C:\path\to\participants.tsv" `
  --derivatives-dir "C:\path\to\derivatives"
```

## Encrypted derivatives extraction

The derivatives archive is password-protected. Provide the password via environment variable:

```powershell
$env:TDBRAIN_ZIP_PASSWORD = "your-password-here"
python -m src.tdbrain_extract --zip "C:\path\to\TDBRAIN-dataset-derivatives.zip" --out "C:\path\to\tdbrain_full"
```

