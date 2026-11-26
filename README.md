# Uncertainty-Aware Predictive Maintenance

> Probabilistic Temporal Convolutional Networks that predict the Remaining Useful Life (RUL) of turbofan engines as a distribution, then simulate maintenance policies that minimise expected cost.

**Team:** Team 6 • **Track:** 2 — Applied research • **Course:** DLE-AI-202 Deep Learning, AI Academy (Cohort II 2026)

---

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/Nurana100/uncertainty-aware-predictive-maintenance
cd uncertainty-aware-predictive-maintenance
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Install PyTorch (pick one)
pip install torch --index-url https://download.pytorch.org/whl/cpu     # laptop (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # server (CUDA 12.8)

# 3. Get the data (see "Data" below), then check the pipeline
python src/data.py
```

---

## Data

NASA C-MAPSS turbofan engine degradation simulation (FD001–FD004).

```bash
# Download and unzip into data/CMAPSS/
https://data.nasa.gov/docs/legacy/CMAPSSData.zip
```

The `data/` folder is not tracked by Git.

---

## Approach

1. **Data:** engine-level train/validation split, per-sensor scaling, 30-cycle windows, RUL capped at 125.
2. **Models:** point-prediction TCN and baselines.
3. **Uncertainty:** Gaussian and quantile heads, deep ensembles, conformal calibration.
4. **Decision simulation:** maintenance policies compared under breakdown and service costs.
5. **Evaluation:** RUL RMSE, interval coverage, expected maintenance cost.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Source code |
| `runs/` | Tracked experiment runs and metric logs |
| `report/` | IEEE paper |
| `presentation/` | Slides |

---

## Team

| Module | Owner |
|---|---|
| Data | |
| Models | |
| Uncertainty | |
| Decision simulation | |
| Evaluation, writing, repo | |

---

## Reproducing the headline result

> TODO: one command goes here before the code freeze.