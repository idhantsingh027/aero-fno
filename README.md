# Geometry-Aware Fourier Neural Operators (FNO) for Real-Time Inverse Aerodynamic Design

> **Deep learning surrogate for CFD** — Predict airflow around airfoils ~3,000× faster than traditional simulations, enabling real-time inverse design and optimization.

---

## 🎯 Project Overview

This project trains a **Fourier Neural Operator (FNO)** to predict aerodynamic flow fields (pressure, velocity) around 2D airfoils. Once trained, the model replaces expensive CFD simulations with millisecond-scale inference, enabling:

- **Real-time flow prediction** (~ms vs hours for CFD)
- **Inverse aerodynamic design** (optimize airfoil shape to reduce drag)
- **What-if analysis** (explore thousands of designs rapidly)

---

## 📚 Background & Terminology

### What is CFD?
**Computational Fluid Dynamics (CFD)** solves the Navier-Stokes equations numerically to simulate fluid flow. For airfoils, a single RANS simulation takes **minutes to hours** depending on mesh resolution.

### What is an FNO?
The **Fourier Neural Operator** is a neural network architecture that learns mappings between function spaces. Unlike CNNs that learn local patterns, FNOs learn in the **frequency domain** via Fast Fourier Transform, making them:
- Resolution-invariant (train at 128×128, test at 64×64 or 256×256)
- Efficient for smooth physics (pressure/velocity fields)
- Much faster than mesh-based solvers

### Key Physical Quantities

| Symbol | Name | Description |
|--------|------|-------------|
| **Cp** | Pressure coefficient | Non-dimensional pressure |
| **u*** | Normalized x-velocity | `u / U∞` where U∞ is freestream velocity |
| **v*** | Normalized y-velocity | `v / U∞` |
| **AoA** | Angle of Attack | Angle between airfoil chord and incoming flow |
| **Wake deficit** | Drag proxy | Velocity reduction behind airfoil (lower = less drag) |

### Why Non-Dimensional Normalization?
The dataset spans Reynolds numbers from 10,000 to 900,000 (90× range), causing raw pressure to vary by 100×. Physics-based normalization (Cp, u*, v*) produces values ~O(1) regardless of Re, making learning tractable.

---

## 🗂️ Project Structure

```
aero-fno/
├── 📓 Notebooks
│   ├── stage1_explore.ipynb              # Data exploration & pipeline validation
│   ├── stage2_v3_fno_train.ipynb         # FNO v3 training
│   ├── stage2_v5_surface_weighted_1.ipynb # FNO v5 training (part 1)
│   ├── stage2_v5_surface_weighted_2.ipynb # FNO v5 training (part 2)
│   ├── stage3_evaluation.ipynb           # v3 evaluation on test set
│   ├── stage3_evaluation_v5.ipynb        # v3 vs v5 comparison, ablation study
│   └── stage4_inverse_design.ipynb       # Inverse design demonstration
│
├── 📁 evaluation_v5/                     # Generated figures
│   ├── paper_summary_figure.png
│   ├── predictions_cp.png
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## 🔬 Pipeline Stages

### Stage 1: Data Exploration
**Notebook:** `stage1_explore.ipynb`

- Load and inspect 6,400 CFD simulations
- Visualize input/output channels
- Implement `AirfoilDataset` class with physics-based normalization
- Validate pipeline with CNN overfitting test (loss → 0.0047)

**Dataset format:** Each `.npz` file contains array `'a'` with shape `(6, 128, 128)`:

| Channel | Name | Description |
|---------|------|-------------|
| 0 | inlet_x | Freestream x-velocity (uniform) |
| 1 | inlet_y | Freestream y-velocity (uniform) |
| 2 | geometry | Binary airfoil mask |
| 3 | pressure | CFD pressure field |
| 4 | u | CFD x-velocity field |
| 5 | v | CFD y-velocity field |

---

### Stage 2: FNO Training

#### Version 3 (prev attempt)
**Notebook:** `stage2_v3_fno_train.ipynb`

| Component | Specification |
|-----------|--------------|
| Architecture | FNO2d, 48×48 Fourier modes |
| Parameters | ~30M |
| Loss | Weighted relative L2 (pressure 2×) |
| Optimizer | Adam (lr=1e-3, cosine annealing) |
| Training | 50 epochs on T4 GPU |
| Best val loss | 0.1291 |

**Problem identified:** While full-field errors were good (Cp ~21%), **surface Cp error was 143.5%** — the model learned the easy far-field but struggled at the airfoil boundary.

#### Version 5 (Surface-Weighted - new approach) ⭐
**Notebooks:** `stage2_v5_surface_weighted_1.ipynb`, `stage2_v5_surface_weighted_2.ipynb`

**Key Innovation: Surface-Weighted Loss**

The standard L2 loss treats all pixels equally, but:
- ~95% of pixels are far-field (Cp ≈ 0) — easy to predict
- ~5% of pixels are on the airfoil surface — hard but critical for engineering

**Solution:** Add a surface penalty term:
```
L_total = L_base + λ × L_surface
```
where `λ = 5.0` and surface is extracted via morphological operations.

| Component | v3 → v5 Change |
|-----------|----------------|
| Fourier modes | 48×48 → **64×64** (+33%) |
| Surface weight λ | 0 → **5.0** |
| Optimizer | Adam → **AdamW** |
| Training | 50 → **120 epochs** |
| Best val loss | 0.1291 → **0.1266** |

**Result:** Surface Cp error dropped from **143.5% → 34.7%** (76% improvement!)

---

### Stage 3: Evaluation
**Notebooks:** `stage3_evaluation.ipynb`, `stage3_evaluation_v5.ipynb`

Comprehensive evaluation on 90 held-out test samples:

#### Ablation Study: Effect of Surface Weight

| λ | Surface Cp Error | Full-field Cp |
|---|------------------|---------------|
| 0 (v3) | 143.5% | 24.1% |
| 5 (v5) | **34.7%** | 21.8% |

#### Resolution Invariance

| Resolution | Cp Error | u* Error | v* Error | Status |
|------------|----------|----------|----------|--------|
| 64×64 | 20.3% | 5.8% | 10.0% | ✅ Good |
| 128×128 | 21.8% | 4.6% | 8.0% | ✅ Trained |
| 256×256 | 22.3% | 5.2% | 8.8% | ✅ Good |
| 512×512 | 83.3% | 21.3% | 42.4% | ⚠️ Degrades |

The model generalizes well to lower resolutions but struggles at 4× the training resolution.

#### Computational Cost Comparison

| Method | Time | Speedup vs FNO |
|--------|------|----------------|
| **FNO v5** | 34 ms | 1× |
| RANS 2D (coarse) | 1 min | ~2,000× slower |
| RANS 2D (fine) | 5 min | ~9,000× slower |
| RANS 3D | 1 hour | ~100,000× slower |

---

### Stage 4: Inverse Design
**Notebook:** `stage4_inverse_design.ipynb`

**Goal:** Automatically optimize airfoil shape to minimize drag.

**Approach:** Use FNO as a fast surrogate, then search over geometric transformations:
- Scale (sx, sy): Stretch/compress airfoil
- Translate (tx, ty): Shift position
- Bias: Adjust thickness

**Drag Proxy:** Wake deficit — velocity reduction behind the airfoil. Lower wake deficit = lower drag.

**Results on 5 test airfoils:**
| Metric | Value |
|--------|-------|
| Mean improvement | +2-20% drag reduction |
| Time per optimization | ~40 seconds |
| CFD equivalent | ~33 hours |
| **Speedup** | **~3,000×** |

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/idhantsingh027/aero-fno.git
cd aero-fno

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision
pip install git+https://github.com/neuraloperator/neuraloperator.git
pip install numpy scipy matplotlib tqdm jupyter
```

### 2. Download Dataset

The dataset is ~1.3GB (not included in git). Download via FTP:

```bash
curl -u "m1470791:m1470791" \
  "ftp://dataserv.ub.tum.de/data_6k.tar.gz" \
  -o data_6k.tar.gz

tar -xzf data_6k.tar.gz
rm data_6k.tar.gz
```

Verify structure:
```
data/train/  ← 6,400 .npz files
data/test/   ← 90 .npz files
```

### 3. Download Pretrained Weights

- **v3 checkpoints:** [Google Drive](https://drive.google.com/drive/folders/1G40-BzbLAZysnt68e_gO9zo15L-ZBEGM?usp=sharing)
- **v5 checkpoints:** [Google Drive](https://drive.google.com/drive/folders/1WdiwGvw_SBMU_ZWt6dsRGmX3U97QwEkb?usp=sharing)

### 4. Run Notebooks

Open Jupyter and run notebooks in order:
1. `stage1_explore.ipynb` — Verify data loading
2. `stage3_evaluation_v5.ipynb` — See model performance
3. `stage4_inverse_design.ipynb` — Try inverse design

---

## 📊 Model Architecture

```
Input: [u_x/U∞, u_y/U∞, geometry]  →  FNO2d  →  Output: [Cp, u*, v*]
       (3 × 128 × 128)                         (3 × 128 × 128)
```

**FNO v5 Configuration:**
- Fourier modes: 64 × 64
- Hidden channels: 64
- Layers: 6
- Parameters: ~30M
- Loss: Surface-weighted relative L2 (λ=5)

---

## 📈 Results Summary

### Prediction Quality

| Channel | v3 (prev) | v5 (new) | Improvement |
|---------|---------------|-----------|-------------|
| Cp (full-field) | 24.1% | 21.8% | +10% |
| Cp (surface) | 143.5% | 34.7% | **+76%** ⭐ |
| u* | 4.9% | 4.4% | +10% |
| v* | 9.7% | 8.3% | +17% |

### Key Achievements

✅ **76% surface error reduction** via surface-weighted loss  
✅ **~3,000× speedup** over CFD  
✅ **Resolution invariant** (64×64 to 256×256)  
✅ **Inverse design** demonstrated with success rate  

---

## 🔧 Technical Notes

### Surface-Weighted Loss Implementation

```python
def surface_weighted_loss(pred, target, geometry, lambda_surf=5.0):
    # Extract surface via morphological operations
    dilated = F.max_pool2d(geometry, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-geometry, 3, stride=1, padding=1)
    surface_mask = dilated - eroded
    
    # Base loss (all pixels)
    base_loss = relative_l2(pred, target)
    
    # Surface loss (boundary pixels only)
    surface_loss = relative_l2(pred * surface_mask, target * surface_mask)
    
    return base_loss + lambda_surf * surface_loss
```

### Why FNO Gradients Fail for Inverse Design

Early attempts used gradient descent through the FNO for shape optimization. This failed because:
1. FNO learns a smooth approximation, not exact physics
2. Gradients point toward "adversarial" shapes that fool the network
3. Resulting geometries are non-physical (fragmented, disconnected)

**Solution:** Gradient-free parametric search over valid transformations.

---

## Tech Stack

- Python 3.10
- PyTorch 2.10 (MPS / CUDA / CPU)
- neuraloperator (from GitHub)
- numpy, matplotlib, scipy, tqdm
