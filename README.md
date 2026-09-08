# FedPDA-IDS

**Privacy-Preserving Drift-Adaptive Federated Learning Framework for IoT Intrusion Detection**

Final-year B.Tech (AI & Data Science) project. Extends the federated
transfer-learning approach of C. Sri Abhijit, Jerusha, Syed Ibrahim &
Varadharajan, *"Federated transfer learning for rare attack class
detection in network intrusion detection systems,"* Scientific
Reports (2025). DOI: [10.1038/s41598-025-02068-x](https://doi.org/10.1038/s41598-025-02068-x)

## What this project adds over the base paper

| Base paper gap | Our extension |
|---|---|
| No formal privacy mechanism | Client-level differential privacy on shared model updates |
| Server can inspect individual client updates | Flower SecAgg+ secure aggregation |
| Raw sample/exemplar channel for rare-class transfer | DP-protected latent class prototypes (no raw samples ever leave the client) |
| Label-dependent drift detection | Unsupervised ADWIN drift detection on reconstruction error |
| Shallow DNN encoder | LSTM sequence autoencoder over temporally-ordered traffic windows |

## Architecture (frozen)

Each client runs a shared LSTM autoencoder (encoder produces a
32-dim latent representation) plus a **local, personalized**
classification head that never leaves the client. Only the shared
encoder/decoder updates are communicated, and only after adaptive
clipping, Gaussian DP noise, and secure aggregation.

```
Raw Traffic -> Feature Extraction -> Preprocessing -> Sequence Construction (W=10, stride=5)
    -> Shared LSTM Encoder (F->64->32, latent=32)
        -> Decoder (reconstruction / anomaly & drift signal)
        -> Local Classification Head (personalized, stays on client)
        -> Class Prototypes (DP-protected, for rare/zero-day detection)
```

Full architectural, privacy, and experimental specification is
treated as frozen for this project; see project memory / design
discussion for the complete spec. Deviations are only made after
explicit approval and are labeled `PRE-CODING CORRECTION` when
proposed.

## Two independent federations

CICIDS2017 and N-BaIoT are **never merged** — they are evaluated as
two separate federated learning experiments with their own client
partitioning, drift phases, and zero-day holdout classes.

## Repository layout

```
configs/            # config.yaml - single source of truth for all hyperparameters
src/fedpda_ids/      # installable Python package (src layout)
  utils/             # config loading, seeding, logging
data/                # raw/processed/external datasets (gitignored, not committed)
experiments/         # results, metrics, checkpoints (gitignored)
logs/                # run logs (gitignored)
notebooks/           # exploratory analysis
scripts/             # CLI entry points for each phase
tests/               # pytest test suite
```

## Setup

Requires **Python 3.10** (see rationale below).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Why Python 3.10

PyTorch, Flower (`flwr[simulation]`), and Opacus all have mature,
well-tested wheels for 3.10 as of this writing; 3.10 is also the
version already installed on this machine, so no version conflicts.
3.12+ support across this exact combination of libraries is not yet
reliably tested, so we pin to 3.10 for the life of the project to
avoid dependency surprises mid-experiment.

## Reproducibility

- All hyperparameters live in `configs/config.yaml`, not hardcoded in source.
- `fedpda_ids.utils.seed.set_seed()` seeds Python `random`, NumPy, and PyTorch/CUDA.
- Final reported experiment numbers use 3 seeds (42, 123, 2024) and are reported as mean ± std.
- Logs are written to `logs/` per run; results/metrics/checkpoints are written to `experiments/`. Neither is committed to git.

## Project status

Implementation proceeds in 14 phases, each built, tested, validated,
and committed before the next begins.

- [x] Phase 1 — Project setup
- [ ] Phase 2 — Data pipeline
- [ ] Phase 3 — Sequence construction
- [ ] Phase 4 — Centralized/local baseline
- [ ] Phase 5 — Federated baseline (FedAvg)
- [ ] Phase 6 — Personalization
- [ ] Phase 7 — Latent class prototypes
- [ ] Phase 8 — Differential privacy
- [ ] Phase 9 — Secure aggregation
- [ ] Phase 10 — Concept drift detection
- [ ] Phase 11 — Privacy attacks (gradient inversion, membership inference)
- [ ] Phase 12 — Experiments (E1–E6, T7)
- [ ] Phase 13 — Monitoring (Prometheus/Grafana)
- [ ] Phase 14 — Final evaluation, paper, viva prep
