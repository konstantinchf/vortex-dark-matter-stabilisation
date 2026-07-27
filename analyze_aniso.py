#!/usr/bin/env python3
"""
Analyze c/a for anisotropic field.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Путь к файлу (если он в ./expanding_3d_aniso/psi_final.npz)
field_path = './expanding_3d_aniso/psi_final.npz'
if not os.path.exists(field_path):
    # попробуем в текущей папке
    field_path = './psi_final.npz'
    if not os.path.exists(field_path):
        raise FileNotFoundError("psi_final.npz not found")

data = np.load(field_path)
psi = torch.tensor(data['psi'], dtype=torch.complex64, device=device)
rho = torch.abs(psi)**2
L = 40.0
N = rho.shape[0]
dx = L / N
print(f"Loaded field, N={N}, L={L}, rho.max={rho.max().item():.3f}")

thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
for th in thresholds:
    threshold = th * rho.max().item()
    mask = rho > threshold
    idx = torch.nonzero(mask)
    if idx.numel() > 0:
        coords = idx.float() * dx - L/2
        coords = coords.cpu().numpy()
        pca = PCA(n_components=3)
        pca.fit(coords)
        axes = np.sqrt(pca.explained_variance_) * 2
        a, b, c = axes
        c_over_a = c / a
        print(f"threshold={th:.1f}: c/a={c_over_a:.4f} (a={a:.2f}, b={b:.2f}, c={c:.2f})")
    else:
        print(f"threshold={th:.1f}: no points")