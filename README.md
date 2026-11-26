\# Uncertainty-Aware Predictive Maintenance



Deep Learning Final Project (AI Academy, DLE-AI-202, Cohort II 2026), Track 2: applied research.



Probabilistic Temporal Convolutional Networks predict the Remaining Useful Life (RUL)

of turbofan engines as a distribution (NASA C-MAPSS), and maintenance policies are

simulated to minimise expected cost under breakdown and service costs.



\## Setup



```bash

python -m venv .venv

\# Windows: .venv\\Scripts\\Activate.ps1      Linux: source .venv/bin/activate

pip install -r requirements.txt

```



PyTorch (install separately):

\- Laptop (CPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu`

\- Server (CUDA 12.8): `pip install torch --index-url https://download.pytorch.org/whl/cu128`



\## Data



Download CMAPSSData.zip from https://data.nasa.gov/docs/legacy/CMAPSSData.zip

and unzip it into `data/CMAPSS/`. The `data/` folder is not tracked by Git.



\## Quick check



```bash

python src/data.py

```



\## Team and modules



| Module | Owner |

|---|---|

| Data | |

| Models | |

| Uncertainty | |

| Decision simulation | |

| Evaluation, writing, repo | |



\## Reproducing the headline result



TODO: one command goes here before the code freeze.

