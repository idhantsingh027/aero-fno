# Airfoil Flow Prediction with Fourier Neural Operator (FNO)

Predict airflow around airfoil shapes using a deep learning model (FNO), replacing slow CFD simulations. Ultimately used for inverse design — automatically reshaping airfoils to reduce drag.

---

## Project Status

| Stage | Description | Status |
|-------|-------------|--------|
| Stage 1 | Data exploration, normalization, pipeline validation | ✅ Done |
| Stage 2 | FNO model training | 🔜 Next |
| Stage 3 | Evaluation vs baseline | 🔜 Upcoming |
| Stage 4 | Inverse design (drag reduction) | 🔜 Upcoming |

---

## Repo Structure

```
aero-fno/
├── stage1_explore.ipynb       # Stage 1: data exploration & pipeline check
├── requirements.txt           # Python dependencies (pinned versions)
├── norm_stats.npy             # Per-channel normalization stats (auto-generated)
└── data/                      # ⚠ NOT in git — download separately (see below)
    ├── train/                 # 6,400 CFD simulation .npz files
    └── test/                  # 90 CFD simulation .npz files
```
---

## Dataset

This project uses the **TUM Deep Flow Prediction** dataset — 6,400 RANS CFD simulations of 2D airfoils at varying Reynolds numbers and angles of attack.

Each sample is a `.npz` file with key `'a'`, shape `(6, 128, 128)`:

| Channels | Name | Description |
|----------|------|-------------|
| 0 | `inlet_x` | Freestream x-velocity (uniform) |
| 1 | `inlet_y` | Freestream y-velocity (uniform) |
| 2 | `geometry` | Binary airfoil mask |
| 3 | `pressure` | CFD pressure field |
| 4 | `u` | CFD x-velocity field |
| 5 | `v` | CFD y-velocity field |

---

## Prerequisites & Setup

### 1. Clone the repo

```bash
git clone https://github.com/idhantsingh027/aero-fno.git
cd aero-fno
```

### 2. Create a virtual environment

**Windows (Command Prompt):**
```cmd
python -m venv aerofno
aerofno\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv aerofno
.\aerofno\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

> **Why isn't the data in the repo?**  
> The dataset is ~1.3GB (6,400 files). GitHub rejects files over 100MB.  
> This is standard practice for ML projects — code lives in git, data is downloaded separately.  
> It's a one-time download per machine.

**Option A — FTP via curl (Mac/Linux/WSL, recommended):**
```bash
curl -u "m1470791:m1470791" \
  "ftp://dataserv.ub.tum.de/data_6k.tar.gz" \
  -o data_6k.tar.gz \
  --progress-bar

tar -xzf data_6k.tar.gz
rm data_6k.tar.gz
```

> **Note:** The HTTP download endpoint on dataserv.ub.tum.de returns 0 bytes — FTP is the only working method.

**Option B — Windows (no WSL):** Use [WinSCP](https://winscp.net) (free FTP GUI):
- Host: `dataserv.ub.tum.de`, Protocol: FTP, Username: `m1470791`, Password: `m1470791`
- Download `data_6k.tar.gz`, then extract with 7-Zip

After extracting, ensure files are at:
```
aero-fno/data/train/   ← 6,400 .npz files
aero-fno/data/test/    ← 90 .npz files
```

**Option C — Google Drive (if someone has shared a link):**
```bash
pip install gdown
gdown "YOUR_DRIVE_SHARE_LINK" -O data_6k.tar.gz && tar -xzf data_6k.tar.gz
```

### 5. Open `stage1_explore.ipynb` and run all cells.

---

## Stage 1 Results

- ✅ All 6,400 training samples loaded and inspected (no NaN/Inf)
- ✅ Per-channel normalization stats computed and saved to `norm_stats.npy`
- ✅ `AirfoilDataset` class: 80/20 train/val split, normalized `(input, output)` pairs
- ✅ Overfit sanity check: MiniCNN loss → **0.0047** on 1 batch (500 epochs) — pipeline healthy

---

## Tech Stack

- Python 3.10
- PyTorch 2.10 (MPS / CUDA / CPU)
- neuralop 2.0
- numpy, matplotlib, scipy, tqdm
