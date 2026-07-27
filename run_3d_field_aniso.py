#!/usr/bin/env python3
"""
3D simulation with anisotropic initial conditions (spin or flattening).
"""

import torch
import torch.fft
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- ПАРАМЕТРЫ ----------
m = 1.0
g_nl = 0.1
scale_rho = 1.0
G = 1.0
steps = 2000
dt = 0.01
N = 128
L0 = 40.0
dx = L0 / N
t0 = 10.0

# Анизотропия
flattening = 0.5          # 0 = сжато по z, 1 = изотропно
spin = 1.0                # амплитуда фазы вращения

output_dir = './expanding_3d_aniso'
os.makedirs(output_dir, exist_ok=True)

# Сетка
kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
kz = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
K2[0,0,0] = 1.0

# Создание начального поля с анизотропией
def create_anisotropic_state(N, L, seed, flattening=1.0, spin=0.0):
    np.random.seed(seed)
    kx = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + (flattening * kz[None, None, :])**2
    K2[0,0,0] = 1.0
    amp = torch.sqrt(K2 ** (-2))
    real = torch.randn(N, N, N, device=device) * amp
    imag = torch.randn(N, N, N, device=device) * amp
    psi = real + 1j * imag
    # Добавляем вращение (спин) в фазу
    if spin != 0.0:
        X, Y, Z = torch.meshgrid(
            torch.linspace(-L/2, L/2, N, device=device),
            torch.linspace(-L/2, L/2, N, device=device),
            torch.linspace(-L/2, L/2, N, device=device),
            indexing='ij'
        )
        phase = spin * torch.atan2(Y, X)
        psi = psi * torch.exp(1j * phase)
    psi = psi / torch.sqrt(torch.mean(torch.abs(psi)**2))
    return psi

def compute_potential_3d(rho, G, scale_rho, K2, a):
    rho_scaled = rho * scale_rho / a
    rho_k = torch.fft.fftn(rho_scaled)
    phi_k = -4 * torch.pi * G * rho_k / K2
    phi_k[0,0,0] = 0.0
    phi = torch.real(torch.fft.ifftn(phi_k))
    return phi

psi = create_anisotropic_state(N, L0, 42, flattening=flattening, spin=spin)
kinetic_base = torch.exp(-1j * dt * K2 / (2*m))

print(f"Starting anisotropic simulation: flattening={flattening}, spin={spin}")
for step in range(steps):
    t = t0 + step * dt
    a = (t / t0) ** (2.0/3.0)
    H = (2.0/3.0) / t
    rho = torch.abs(psi)**2
    phi = compute_potential_3d(rho, G, scale_rho, K2, a)
    V = m * phi + g_nl * rho
    damp = torch.exp(torch.tensor(-0.75 * H * dt, device=device))
    psi = psi * torch.exp(-0.5j * dt * V) * damp
    kinetic = torch.exp(-1j * dt * K2 / (2 * m * a**2))
    psi_k = torch.fft.fftn(psi)
    psi_k = psi_k * kinetic
    psi = torch.fft.ifftn(psi_k)
    rho = torch.abs(psi)**2
    phi = compute_potential_3d(rho, G, scale_rho, K2, a)
    V = m * phi + g_nl * rho
    psi = psi * torch.exp(-0.5j * dt * V) * damp
    if step % 100 == 0:
        print(f"Step {step}, rho_max={rho.max().item():.3f}")

psi_cpu = psi.cpu().numpy()
np.savez(os.path.join(output_dir, 'psi_final.npz'), psi=psi_cpu)
print(f"Field saved to {output_dir}/psi_final.npz")