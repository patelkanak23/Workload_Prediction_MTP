# AGMS-Net: Adaptive Gated Multi-Scale Network for Dynamic Cloud Workload Prediction

**Major Technical Project (MTP)**

## Overview

AGMS-Net extends MSCNet (IEEE TSC 2025) with a learned, input-conditioned scale-weighting mechanism for long-term cloud workload forecasting. The core idea: instead of using fixed patch scales uniformly across all input windows, AGMS-Net learns to dynamically weight different temporal scales based on the properties of each individual window.

## Architecture

```
Input Workload (L steps, D metrics)
        │
   Instance Norm (RevIN)
        │
   ┌────┴──────────────────────────┐
   │                               │
Trend Block               Multi-Scale Encoder
(AvgPool + Linear)        (5 patch scales + Transformer)
   │                               │
   │                    ┌──────────┤
   │                    │   Adaptive Scale Gating ◄── Regime Detector
   │                    │   (Phase 1: MLP → softmax)   (Phase 2: stats + k-means)
   │                    │          │
   │                    └──────────┤
   │                         X_fused
   └────────┬──────────────────┘
            │
     Y = RevIN⁻¹(Y_trend + Y_multiscale)
```

## Phases

| Phase | What | Status |
|-------|------|--------|
| **Phase 0** | Reproduce MSCNet baseline — fixed scale weights | 🔄 In Progress |
| **Phase 1** | Adaptive Scale Gating — learned per-window weights | ⬜ Not Started |
| **Phase 2** | Regime-Conditioned Gating — stats + k-means embedding | ⬜ Not Started |
| **Phase 3** | SLA-aware evaluation — provisioning simulator | ⬜ Not Started |

## Datasets

- **Alibaba Cluster Trace 2018** — 100 sampled machines, CPU utilization
- **Google Cluster Trace** — 100 sampled machines, CPU utilization
- **Azure Public Dataset V1** — 100 sampled machines, CPU utilization

Protocol: L=96, H∈{24, 48, 72, 96}, 7:1:2 train/val/test split.

## Project Structure

```
├── MSCNet_baseline/          # Official MSCNet code (reference)
│   ├── models/MSCNet.py      # Main model (5 fixed scales)
│   ├── layers/               # RevIN, Transformer, Attention, etc.
│   ├── exp/                  # Training/evaluation pipeline
│   ├── data_provider/        # Dataset loading
│   ├── utils/                # Metrics, tools
│   ├── scripts/              # Training shell scripts
│   └── run.py                # CLI entry point
├── data_preprocessing/       # Raw trace → preprocessed CSV
├── notebooks/                # Colab training notebooks
├── dataset/                  # Preprocessed CSVs (git-ignored)
├── checkpoints/              # Trained models (git-ignored)
├── results/                  # Predictions & metrics (git-ignored)
├── requirements.txt
└── README.md
```

## Base Papers

1. **MSCNet** — Zhao et al., "MSCNet: Multi-Scale Network with Convolutions for Long-term Cloud Workload Prediction", IEEE TSC 2025. [Paper](https://ieeexplore.ieee.org/abstract/document/10858411) | [Code](https://github.com/ACAT-SCUT/MSCNet)
2. **SMPM** — Seshadri et al., "Design and Evaluation of a Hierarchical Characterization and Adaptive Prediction Model for Cloud Workloads", IEEE TCC 2024.

## Setup

```bash
# Clone
git clone https://github.com/patelkanak23/Workload_Prediction_MTP.git
cd Workload_Prediction_MTP

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Training is done on Google Colab (free T4 GPU)
# See notebooks/ for Colab notebooks
```

## Training (on Colab)

```python
# Mount Drive, clone repo, install deps, then:
!python run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset --data_path alibaba.csv \
    --data custom --model MSCNet --features M \
    --seq_len 96 --label_len 96 --pred_len 96 \
    --enc_in 100 --dec_in 100 --c_out 100 \
    --e_layers 3 --n_heads 4 --d_model 128 \
    --train_epochs 100 --patience 3
```
