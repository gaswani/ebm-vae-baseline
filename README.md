# EBM VAT Anomaly Detection (Baseline VAE)

Baseline **Variational Autoencoder (VAE)** for anomaly detection in Rwanda Revenue Authority (RRA) **Electronic Billing Machine (EBM)** VAT ecosystem data.

The VAE is trained **unsupervised** to learn “normal” patterns. Records are flagged as anomalies using **reconstruction error** (high error → unusual pattern → candidate for review).

> This repository is intended as a **baseline prototype**: start simple, validate top anomalies with domain experts, then expand features/segmentation.

---

## Project Runtime

- **Python**: 3.11.9
- Recommended: VS Code + Python + Jupyter extensions

Verify:

```bash
python --version
# or
python3 --version
# Windows alternative
py --version
```

---

## Repository Structure

```text
ebm-vae-baseline/
├── data/
│   ├── ebm_columns.csv                         # Column header reference (safe to commit if approved)
│   └── ebm-anomaly-detection-dataset.parquet   # RAW DATA (DO NOT COMMIT)
├── model/
│   ├── encoder/                                # SavedModel
│   ├── decoder/                                # SavedModel
│   └── vae_weights*                            # VAE weights (optional)
├── scaler/
│   └── standard_scaler.joblib                  # Saved StandardScaler
├── outputs/
│   ├── ebm_vae_scored_records.csv              # Record-level scores
│   ├── ebm_vae_taxpayer_summary.csv            # Taxpayer-level rollup
│   └── ebm_vae_top_anomalies.csv               # Top-N anomalies
├── ebm_vae_baseline_notebook.ipynb             # Main notebook
├── requirements.txt
├── .gitignore
├── LICENSE
└── COPYRIGHT_NOTICE.md
```

---

## Data Handling & Governance

**Do not commit** the raw EBM dataset to GitHub.

- Keep `data/ebm-anomaly-detection-dataset.parquet` locally or in an approved secure storage location.
- The repo can safely contain only the **schema reference** (`ebm_columns.csv`) if approved.

---

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/gaswani/ebm-vae-baseline.git
cd ebm-vae-baseline
```

### 2) Create and activate a virtual environment

#### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

> If script execution is blocked on Windows, run PowerShell as Admin and execute:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Running the Notebook

1. Open the project in **VS Code**.
2. Select the interpreter: `.venv` (Python 3.11.9).
3. Open and run:

```text
ebm_vae_baseline_notebook.ipynb
```

The notebook performs:
- Feature selection (baseline set) + hashed `taxpayer_tin` for traceability
- Preprocessing (type coercion, missing value handling, clipping, scaling)
- VAE training (TensorFlow)
- Scoring (reconstruction error) + thresholding (e.g., 99.5th percentile)
- Exports to `outputs/`
- Saves artifacts to `scaler/` and `model/`

---

## Outputs

After a successful run, you should see:

- `outputs/ebm_vae_scored_records.csv`  
  Record-level reconstruction error + anomaly flag (and hashed taxpayer id)

- `outputs/ebm_vae_taxpayer_summary.csv`  
  Taxpayer-level rollup (counts, max/mean error, anomaly record counts)

- `outputs/ebm_vae_top_anomalies.csv`  
  Highest-scoring records for domain validation

---

## Model Artifacts

The notebook saves:

- **Scaler**: `scaler/standard_scaler.joblib`
- **Encoder**: `model/encoder/` (SavedModel)
- **Decoder**: `model/decoder/` (SavedModel)
- **VAE weights**: `model/vae_weights*` (weights-only checkpoint)

These allow inference without retraining (subject to environment compatibility and governance controls).

---

## Reproducibility Notes

- For consistent `tin_hashed` values across environments, set a salt via environment variable:

```bash
export TIN_HASH_SALT="your_internal_salt"
```

On Windows PowerShell:

```powershell
$env:TIN_HASH_SALT="your_internal_salt"
```

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgements

Developed as part of Pathways Technologies’ work on anomaly detection for the RRA EBM ecosystem.
