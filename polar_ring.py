#!/usr/bin/env python3
"""
3D gas accretion simulation in a wave-dominated dark matter potential.
Fully vectorized with PyTorch, uses:
- Poisson solver via FFT
- Trilinear interpolation (grid_sample) for force calculation
- Centered difference for gradient computation
- Boundary protection for particles
- Saves trajectories and final positions
"""

import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ---------- ПАРАМЕТРЫ ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_dir = './expanding_3d_final'
G_val = 1.0
L = 20.0
N = 64
dx = L / N

# ---------- ЗАГРУЗКА ИЛИ ГЕНЕРАЦИЯ ПОЛЯ ----------
def load_or_generate_field():
    # Пытаемся загрузить реальное поле
    psi_path = os.path.join(data_dir, f'psi_final_G_{G_val:.1f}.npz')
    if os.path.exists(psi_path):
        data = np.load(psi_path)
        psi = torch.tensor(data['psi'], dtype=torch.complex64, device=device)
        rho = torch.abs(psi)**2
        print("Loaded real field from simulation.")
        return rho
    else:
        # Генерируем синтетическое поле
        print("Real field not found. Generating synthetic anisotropic field...")
        x = torch.linspace(-L/2, L/2, N, device=device)
        y = torch.linspace(-L/2, L/2, N, device=device)
        z = torch.linspace(-L/2, L/2, N, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        # Анизотропный гауссов эллипсоид (c/a = 0.4)
        rho = torch.exp(-(X**2/5.0**2 + Y**2/5.0**2 + Z**2/2.0**2))
        # Волновая интерференционная сетка
        rho += 0.3 * torch.cos(2*torch.pi*X/3.0) * torch.cos(2*torch.pi*Y/3.0) * torch.cos(2*torch.pi*Z/3.0)
        rho = torch.clamp(rho, min=0)
        return rho

rho = load_or_generate_field()

# ---------- РЕШЕНИЕ ПУАССОНА (FFT) ----------
def compute_potential(rho, G=1.0, dx=1.0):
    N = rho.shape[0]
    kx = torch.fft.fftfreq(N, d=dx, device=rho.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=rho.device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=dx, device=rho.device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K2[0,0,0] = 1.0
    rho_k = torch.fft.fftn(rho)
    phi_k = -4 * torch.pi * G * rho_k / K2
    phi_k[0,0,0] = 0.0
    phi = torch.real(torch.fft.ifftn(phi_k))
    return phi

Phi = compute_potential(rho, G=1.0, dx=dx)
Phi = Phi / torch.max(torch.abs(Phi))  # нормировка

# ---------- ТРИЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ С ЦЕНТРИРОВАННОЙ РАЗНОСТЬЮ ----------
def interpolate_field(values, coords, L):
    """
    values: 3D tensor (N,N,N) on GPU
    coords: (N_points, 3) tensor of physical coordinates (x,y,z)
    Returns interpolated values at coords.
    """
    N = values.shape[0]
    # Масштабируем в [-1, 1] для grid_sample
    x = coords[:,0] / (L/2)
    y = coords[:,1] / (L/2)
    z = coords[:,2] / (L/2)
    grid = torch.stack([x, y, z], dim=1).view(1, -1, 1, 1, 3).to(values.device)
    values_4d = values.view(1, 1, N, N, N)
    interp = torch.nn.functional.grid_sample(
        values_4d, grid, mode='bilinear', padding_mode='border', align_corners=False
    )
    return interp.view(-1)

def compute_force(Phi, positions, L, eps=0.1):
    """
    Вычисляет силу F = -grad(Phi) в позициях частиц
    с использованием симметричной разности (второй порядок точности).
    """
    # Потенциал в центре
    phi_center = interpolate_field(Phi, positions, L)
    # Смещения по осям
    shifts = torch.tensor([
        [eps, 0, 0],
        [0, eps, 0],
        [0, 0, eps]
    ], device=positions.device)
    forces = []
    for shift in shifts:
        pos_plus = positions + shift
        pos_minus = positions - shift
        phi_plus = interpolate_field(Phi, pos_plus, L)
        phi_minus = interpolate_field(Phi, pos_minus, L)
        # Симметричная разность
        f = -(phi_plus - phi_minus) / (2 * eps)
        forces.append(f)
    return torch.stack(forces, dim=1)  # (N_points, 3)

# ---------- ЗАПУСК ЧАСТИЦ ----------
n_particles = 2000
# Начальные позиции: сфера радиуса 8
r0 = 8.0
theta = 2 * np.pi * torch.rand(n_particles, device=device)
phi_ang = torch.acos(2 * torch.rand(n_particles, device=device) - 1)
x0 = r0 * torch.sin(phi_ang) * torch.cos(theta)
y0 = r0 * torch.sin(phi_ang) * torch.sin(theta)
z0 = r0 * torch.cos(phi_ang)
positions = torch.stack([x0, y0, z0], dim=1)  # (Np, 3)

# Начальные скорости: вращение вокруг z (диск)
vx = -0.2 * positions[:,1]
vy = 0.2 * positions[:,0]
vz = 0.05 * torch.randn(n_particles, device=device)

# Параметры эволюции
dt = 0.01
steps = 800
friction = 0.995
radius_limit = L/2 * 0.9  # защита от вылета

# Сохраняем траектории (каждый 10-й шаг)
trajectories = []
traj_steps = range(0, steps, 10)

for step in range(steps):
    forces = compute_force(Phi, positions, L, eps=0.1)
    velocities = torch.stack([vx, vy, vz], dim=1)
    velocities = velocities + forces * dt
    velocities = velocities * friction
    positions = positions + velocities * dt
    
    # Защита от вылета: удаляем частицы, вышедшие за пределы
    r = torch.norm(positions, dim=1)
    mask = r < radius_limit
    if mask.sum() < 10:
        break
    positions = positions[mask]
    velocities = velocities[mask]
    vx, vy, vz = velocities[:,0], velocities[:,1], velocities[:,2]
    
    if step in traj_steps:
        trajectories.append(positions.clone().cpu().numpy())

# Преобразуем траектории
traj_array = np.array(trajectories)  # (N_frames, N_particles, 3)

# ---------- ВИЗУАЛИЗАЦИЯ ----------
final_pos = traj_array[-1]
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_pos[:,0], final_pos[:,1], final_pos[:,2], s=1, c='blue', alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Gas particles in wave-dominated potential (polar ring)')
plt.savefig('polar_ring_simulation_final.png', dpi=150)
plt.show()

# Проекции
fig, axes = plt.subplots(1, 2, figsize=(12,5))
# XY projection
xy_hist, xedges, yedges = np.histogram2d(final_pos[:,0], final_pos[:,1], bins=40, range=[[-10,10],[-10,10]])
axes[0].imshow(xy_hist.T, origin='lower', extent=[-10,10,-10,10], cmap='hot')
axes[0].set_title('XY projection (disk)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
# XZ projection
xz_hist, xedges, zedges = np.histogram2d(final_pos[:,0], final_pos[:,2], bins=40, range=[[-10,10],[-10,10]])
im = axes[1].imshow(xz_hist.T, origin='lower', extent=[-10,10,-10,10], cmap='hot')
axes[1].set_title('XZ projection (polar ring)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
plt.colorbar(im, ax=axes)
plt.savefig('polar_ring_projections_final.png', dpi=150)
plt.show()

# Сохраняем конечные положения
np.savez('polar_ring_particles_final.npz', positions=final_pos.cpu().numpy())
print("Simulation complete. Saved polar_ring_simulation_final.png, polar_ring_projections_final.png, polar_ring_particles_final.npz")