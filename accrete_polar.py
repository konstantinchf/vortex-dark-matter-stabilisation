#!/usr/bin/env python3
"""
Accretion of gas from a polar direction (XZ plane) into the anisotropic field.
"""

import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- ПОИСК ПОЛЯ ----------
possible_paths = [
    './psi_final.npz',
    './expanding_3d_aniso/psi_final.npz',
    './expanding_3d_L40/psi_final.npz',
]
field_path = None
for p in possible_paths:
    if os.path.exists(p):
        field_path = p
        break
if field_path is None:
    raise FileNotFoundError("psi_final.npz not found")

data = np.load(field_path)
psi = torch.tensor(data['psi'], dtype=torch.complex64, device=device)
rho = torch.abs(psi)**2
L = 40.0
N = rho.shape[0]
dx = L / N
print(f"Loaded field from {field_path}, N={N}, L={L}, rho.max={rho.max().item():.3f}")

# ---------- ПОТЕНЦИАЛ ----------
def compute_potential(rho, G=1.0):
    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K2[0,0,0] = 1.0
    rho_k = torch.fft.fftn(rho)
    phi_k = -4 * torch.pi * G * rho_k / K2
    phi_k[0,0,0] = 0.0
    phi = torch.real(torch.fft.ifftn(phi_k))
    return phi

Phi = compute_potential(rho, G=1.0)
Phi = Phi / torch.max(torch.abs(Phi))

# ---------- ИНТЕРПОЛЯЦИЯ ----------
def interpolate_field(values, coords, L):
    N = values.shape[0]
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
    phi_center = interpolate_field(Phi, positions, L)
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
        f = -(phi_plus - phi_minus) / (2 * eps)
        forces.append(f)
    return torch.stack(forces, dim=1)

# ---------- ПОЛЯРНЫЙ СТАРТ (в плоскости XZ) ----------
n_particles = 3000
r0 = 8.0

# Угол в плоскости XZ
theta = 2 * np.pi * torch.rand(n_particles, device=device)
# Разброс по толщине (ось Y) – небольшой
y0 = 0.5 * torch.randn(n_particles, device=device)
x0 = r0 * torch.cos(theta)
z0 = r0 * torch.sin(theta)

positions = torch.stack([x0, y0, z0], dim=1)

# Начальные скорости: вращение вокруг оси Y в плоскости XZ
v_rot = 0.3
vx = -v_rot * torch.sin(theta)
vz = v_rot * torch.cos(theta)
vy = 0.02 * torch.randn(n_particles, device=device)

velocities = torch.stack([vx, vy, vz], dim=1)

dt = 0.01
steps = 2000
friction = 0.99
radius_limit = L/2 * 0.85

trajectories = []
traj_steps = range(0, steps, 20)

for step in range(steps):
    forces = compute_force(Phi, positions, L, eps=0.1)
    velocities = velocities + forces * dt
    velocities = velocities * friction
    positions = positions + velocities * dt

    r = torch.norm(positions, dim=1)
    mask = r < radius_limit
    if mask.sum() < 20:
        print(f"Step {step}: only {mask.sum()} particles left, stopping.")
        break
    positions = positions[mask]
    velocities = velocities[mask]

    if step in traj_steps:
        trajectories.append(positions.clone().cpu().numpy())

if not trajectories:
    print("ERROR: No particles survived.")
    exit()

final_pos = trajectories[-1]
print(f"Final particles: {final_pos.shape[0]}")

# ---------- ВИЗУАЛИЗАЦИЯ ----------
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_pos[:,0], final_pos[:,1], final_pos[:,2], s=2, c='blue', alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Gas accretion from polar direction')
ax.set_xlim(-12,12); ax.set_ylim(-12,12); ax.set_zlim(-12,12)
plt.savefig('accretion_polar.png', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,5))
xy_hist, _, _ = np.histogram2d(final_pos[:,0], final_pos[:,1], bins=40, range=[[-12,12],[-12,12]])
axes[0].imshow(xy_hist.T, origin='lower', extent=[-12,12,-12,12], cmap='hot')
axes[0].set_title('XY projection')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
xz_hist, _, _ = np.histogram2d(final_pos[:,0], final_pos[:,2], bins=40, range=[[-12,12],[-12,12]])
im = axes[1].imshow(xz_hist.T, origin='lower', extent=[-12,12,-12,12], cmap='hot')
axes[1].set_title('XZ projection (polar ring)')
axes[1].set_xlabel('x'); axes[1].set_ylabel('z')
plt.colorbar(im, ax=axes)
plt.savefig('accretion_polar_projections.png', dpi=150)
plt.show()

print("Done. Check accretion_polar.png and accretion_polar_projections.png")