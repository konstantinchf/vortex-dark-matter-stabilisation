#!/usr/bin/env python3
"""
3D simulations with cosmic expansion (FULLY VECTORIZED).
- Полностью GPU-векторизованный подсчёт вихревых нитей через циркуляцию на гранях.
- Сглаживание фазы в начальных условиях через 3D-свёртку.
- N=64, steps=1000, t0=10, dt=0.01, n_realisations=3.
"""

import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- ПАРАМЕТРЫ ----------
m = 1.0
g_nl = 0.1
scale_rho = 1.0
G_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_realisations = 3
steps = 1000
dt = 0.01
record_every = 20
N = 64
L0 = 20.0
dx = L0 / N
t0 = 10.0

output_dir = './expanding_3d_final'
os.makedirs(output_dir, exist_ok=True)

# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------
def create_initial_state_3d(N, L, seed):
    np.random.seed(seed)
    kx = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K2[0,0,0] = 1.0
    amp = torch.sqrt(K2 ** (-2))
    real = torch.randn(N, N, N, device=device) * amp
    imag = torch.randn(N, N, N, device=device) * amp
    psi = real + 1j * imag
    psi = psi / torch.sqrt(torch.mean(torch.abs(psi)**2))
    # Сглаживание фазы через 3D-свёртку
    phase = torch.angle(psi)
    kernel = torch.ones((1, 1, 3, 3, 3), device=device) / 27.0
    phase_padded = phase[None, None, :, :, :]
    phase_smooth = F.conv3d(phase_padded, kernel, padding=1)[0, 0]
    psi = torch.abs(psi) * torch.exp(1j * phase_smooth)
    psi = psi / torch.sqrt(torch.mean(torch.abs(psi)**2))
    return psi

def compute_potential_3d(rho, G, scale_rho, K2, a):
    rho_scaled = rho * scale_rho / a
    rho_k = torch.fft.fftn(rho_scaled)
    phi_k = -4 * torch.pi * G * rho_k / K2
    phi_k[0,0,0] = 0.0
    phi = torch.real(torch.fft.ifftn(phi_k))
    return phi

def count_vortex_lines_3d(psi, threshold_factor=0.02, dx=1.0, L=20.0):
    """
    Полностью векторизованный (GPU) подсчёт длины 3D вихревых нитей.
    """
    phase = torch.angle(psi)
    amp = torch.abs(psi)
    mean_amp = torch.mean(amp).item()
    threshold = max(threshold_factor * mean_amp, 0.01)

    # 1. Циркуляция по плакеткам XY (вокруг оси Z)
    d1_z = (torch.roll(phase, -1, 0) - phase + np.pi) % (2*np.pi) - np.pi
    d2_z = (torch.roll(phase, (-1, -1), (0, 1)) - torch.roll(phase, -1, 0) + np.pi) % (2*np.pi) - np.pi
    d3_z = (torch.roll(phase, -1, 1) - torch.roll(phase, (-1, -1), (0, 1)) + np.pi) % (2*np.pi) - np.pi
    d4_z = (phase - torch.roll(phase, -1, 1) + np.pi) % (2*np.pi) - np.pi
    charge_z = torch.round((d1_z + d2_z + d3_z + d4_z) / (2 * np.pi)).int()

    # 2. Циркуляция по плакеткам YZ (вокруг оси X)
    d1_x = (torch.roll(phase, -1, 1) - phase + np.pi) % (2*np.pi) - np.pi
    d2_x = (torch.roll(phase, (-1, -1), (1, 2)) - torch.roll(phase, -1, 1) + np.pi) % (2*np.pi) - np.pi
    d3_x = (torch.roll(phase, -1, 2) - torch.roll(phase, (-1, -1), (1, 2)) + np.pi) % (2*np.pi) - np.pi
    d4_x = (phase - torch.roll(phase, -1, 2) + np.pi) % (2*np.pi) - np.pi
    charge_x = torch.round((d1_x + d2_x + d3_x + d4_x) / (2 * np.pi)).int()

    # 3. Циркуляция по плакеткам ZX (вокруг оси Y)
    d1_y = (torch.roll(phase, -1, 2) - phase + np.pi) % (2*np.pi) - np.pi
    d2_y = (torch.roll(phase, (-1, -1), (2, 0)) - torch.roll(phase, -1, 2) + np.pi) % (2*np.pi) - np.pi
    d3_y = (torch.roll(phase, -1, 0) - torch.roll(phase, (-1, -1), (2, 0)) + np.pi) % (2*np.pi) - np.pi
    d4_y = (phase - torch.roll(phase, -1, 0) + np.pi) % (2*np.pi) - np.pi
    charge_y = torch.round((d1_y + d2_y + d3_y + d4_y) / (2 * np.pi)).int()

    # Объединенная маска наличия нити в узле
    has_vortex = (charge_z != 0) | (charge_x != 0) | (charge_y != 0)

    # Фильтр по малой локальной амплитуде
    amp_avg = (amp + 
               torch.roll(amp, -1, 0) + 
               torch.roll(amp, -1, 1) + 
               torch.roll(amp, -1, 2)) / 4.0
    
    mask = has_vortex & (amp_avg < threshold)
    
    # Суммарная длина нитей ~ dx * количество узлов с проходящей нитью
    return mask.sum().item() * dx

def run_simulation_3d(G, seed):
    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K2[0,0,0] = 1.0

    psi = create_initial_state_3d(N, L0, seed)
    history = {'steps': [], 'rho_max': [], 'E_kin': [], 'E_grav': [], 'vortex_length': []}

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

        if step % record_every == 0:
            rho = torch.abs(psi)**2
            phi = compute_potential_3d(rho, G, scale_rho, K2, a)
            psi_k_current = torch.fft.fftn(psi)
            E_kin = torch.sum(torch.abs(psi_k_current)**2 * K2) / (2 * m * a**2) * (dx**3) / (N**3)
            E_grav = 0.5 * torch.sum(rho * phi) * dx**3
            rho_max = torch.max(rho)
            vlength = count_vortex_lines_3d(psi, threshold_factor=0.02, dx=dx, L=L0)
            history['steps'].append(step)
            history['rho_max'].append(rho_max.item())
            history['E_kin'].append(E_kin.item())
            history['E_grav'].append(E_grav.item())
            history['vortex_length'].append(vlength)
    return history

# ---------- ЗАПУСК ----------
all_results = {}
for G in G_vals:
    print(f"\n=== 3D Expanding G = {G} ===")
    hist_list = []
    for seed in range(n_realisations):
        print(f"  realisation {seed+1}/{n_realisations}")
        hist = run_simulation_3d(G, seed+1000)
        hist_list.append(hist)
    # Агрегация
    steps_arr = hist_list[0]['steps']
    vlength_mean = np.mean([h['vortex_length'] for h in hist_list], axis=0)
    vlength_std = np.std([h['vortex_length'] for h in hist_list], axis=0)
    rho_max_mean = np.mean([h['rho_max'] for h in hist_list], axis=0)
    rho_max_std = np.std([h['rho_max'] for h in hist_list], axis=0)
    E_kin_mean = np.mean([h['E_kin'] for h in hist_list], axis=0)
    E_kin_std = np.std([h['E_kin'] for h in hist_list], axis=0)
    E_grav_mean = np.mean([h['E_grav'] for h in hist_list], axis=0)
    E_grav_std = np.std([h['E_grav'] for h in hist_list], axis=0)
    all_results[G] = {
        'steps': steps_arr,
        'vortex_length_mean': vlength_mean,
        'vortex_length_std': vlength_std,
        'rho_max_mean': rho_max_mean,
        'rho_max_std': rho_max_std,
        'E_kin_mean': E_kin_mean,
        'E_kin_std': E_kin_std,
        'E_grav_mean': E_grav_mean,
        'E_grav_std': E_grav_std,
    }
    np.savez(os.path.join(output_dir, f'3D_G_{G:.1f}_expanding_final.npz'), **all_results[G])

# ---------- ГРАФИКИ ----------
print("\nGenerating 3D plots...")
G_list = list(all_results.keys())
L_avg = []
L_err = []
for G in G_list:
    data = all_results[G]
    steps_arr = np.array(data['steps'])
    l_mean = data['vortex_length_mean']
    l_std = data['vortex_length_std']
    idx = np.where(steps_arr >= 700)[0]
    if len(idx) > 0:
        L_avg.append(np.mean(l_mean[idx]))
        L_err.append(np.mean(l_std[idx]))
    else:
        L_avg.append(0)
        L_err.append(0)

plt.figure(figsize=(8,6))
plt.errorbar(G_list, L_avg, yerr=L_err, fmt='bo-', capsize=5)
plt.xlabel('G')
plt.ylabel('Total vortex line length (3D)')
plt.title('3D phase diagram with expansion (N=64, fully vectorized)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, '3D_phase_diagram_expanding.png'), dpi=150)
plt.show()

if 1.0 in all_results:
    data = all_results[1.0]
    steps_arr = np.array(data['steps'])
    ratio = np.abs(data['E_grav_mean']) / (data['E_kin_mean'] + 1e-12)
    plt.figure()
    plt.plot(steps_arr, ratio)
    plt.xlabel('Step')
    plt.ylabel('|E_grav|/E_kin')
    plt.title('3D energy ratio (expanding, G=1, fully vectorized)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '3D_energy_ratio_expanding.png'), dpi=150)
    plt.show()

print(f"\nAll 3D results saved in {output_dir}")