# arHCA — Autoregressive Higher Order Coupling Analysis

This repository provides code to allow reproducibility of the submitted paper of the same name to ISMB 2026.

Autoregressive Higher Order Coupling Analysis (arHCA) is a Python-based framework for modeling, analyzing, and visualizing higher‑order statistical couplings using autoregressive generative models.
This repository contains implementations of several autoregressive methods (including arHCA, arDCA, and neural-network–based approaches) along with utilities for data handling and visualization.

The main file is arHCA.py which implements the training process with preset parameters saved at the top of the file.

## Key Features
- Higher‑Order Coupling Analysis using autoregressive probability models.
- Implementations of ablation models: arHCA, arDCA, and arNN neural approaches.
- Visualization Utilities for plotting and analysis.
- PyTorch implementation of arDCA.

## Installation
Prerequisits:
 - python version >=3.11,<3.15
 - [poetry](https://python-poetry.org/docs/)
 
```bash
git clone https://github.com/SandroHauri/arHCA.git
cd arHCA
poetry install
```

<img width="433" height="352" alt="image" src="https://github.com/user-attachments/assets/d5fd4b27-751d-4690-9e89-45355953a941" />
<img width="305" height="375" alt="image" src="https://github.com/user-attachments/assets/1fb7b28d-31b7-4e39-aafb-54103b4fdb70" />
