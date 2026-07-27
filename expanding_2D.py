#!/usr/bin/env python3
"""
2D simulations with cosmic expansion (PHYSICALLY CORRECTED).
- Исправлено дублирование scale_rho в E_grav.
- Строгий порог для вихрей (threshold_factor=0.02).
- Деление на a(t) в уравнении Пуассона (сопутствующие координаты).
- scale_rho=1.0, N=128, steps=2000, dt=0.01, t0=10.
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
scale_rho = 1.0                  # физическая нормировка
G_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_realisations = 5
steps = 2000
dt = 0.01
record_every = 20
N = 128
L0 = 20.0
dx = L0 / N
t0 = 10.0

output_dir = './expanding_phys'
os.makedirs(output_dir, exist_ok=True)

# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------
def create_initial_state(N, L, seed):
    np.random.seed(seed)
    kx = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    K2 = kx[:, None]**2 + ky[None, :]**2
    K2[0,0] = 1.0
    amp = torch.sqrt(K2 ** (-2))
    real = torch.randn(N, N, device=device) * amp
    imag = torch.randn(N, N, device=device) * amp
    psi = real + 1j * imag
    psi = psi / torch.sqrt(torch.mean(torch.abs(psi)**2))
    phase = torch.angle(psi)
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32, device=device)
    kernel = kernel / kernel.sum()
    kernel = kernel[None, None, :, :]
    phase_padded = phase[None, None, :, :]
    phase_smooth = F.conv2d(phase_padded, kernel, padding=1)[0,0]
    psi = torch.abs(psi) * torch.exp(1j * phase_smooth)
    psi = psi / torch.sqrt(torch.mean(torch.abs(psi)**2))
    return psi

def compute_potential(rho, G, scale_rho, K2, a):
    """
    Пуассон в сопутствующих координатах: div^2 phi = 4πG * rho / a
    """
    rho_scaled = rho * scale_rho / a   # <-- деление на a(t)
    rho_k = torch.fft.fftn(rho_scaled)
    phi_k = -4 * torch.pi * G * rho_k / K2
    phi_k[0,0] = 0.0
    phi = torch.real(torch.fft.ifftn(phi_k))
    return phi

def count_vortices_adaptive(psi, threshold_factor=0.02, dx=1.0, L=20.0):
    """
    Строгий порог: 0.02 от средней амплитуды (чтобы не считать фоновый шум).
    """
    phase = torch.angle(psi)
    amp = torch.abs(psi)
    mean_amp = torch.mean(amp).item()
    threshold = max(threshold_factor * mean_amp, 0.01)
    theta_00 = phase
    theta_10 = torch.roll(phase, shifts=(-1, 0), dims=(0, 1))
    theta_11 = torch.roll(phase, shifts=(-1, -1), dims=(0, 1))
    theta_01 = torch.roll(phase, shifts=(0, -1), dims=(0, 1))
    d1 = (theta_10 - theta_00 + np.pi) % (2*np.pi) - np.pi
    d2 = (theta_11 - theta_10 + np.pi) % (2*np.pi) - np.pi
    d3 = (theta_01 - theta_11 + np.pi) % (2*np.pi) - np.pi
    d4 = (theta_00 - theta_01 + np.pi) % (2*np.pi) - np.pi
    winding = (d1 + d2 + d3 + d4) / (2*np.pi)
    charge = torch.round(winding).int()
    amp_avg = (amp + torch.roll(amp, shifts=(-1,0), dims=(0,1)) +
               torch.roll(amp, shifts=(-1,-1), dims=(0,1)) +
               torch.roll(amp, shifts=(0,-1), dims=(0,1))) / 4.0
    mask = (charge != 0) & (amp_avg < threshold)
    return mask.sum().item()

def run_simulation(G, seed):
    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    K2 = kx[:, None]**2 + ky[None, :]**2
    K2[0,0] = 1.0

    psi = create_initial_state(N, L0, seed)
    history = {'steps': [], 'rho_max': [], 'E_kin': [], 'E_grav': [], 'N_vort': []}

    for step in range(steps):
        t = t0 + step * dt
        a = (t / t0) ** (2.0/3.0)
        H = (2.0/3.0) / t

        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2, a)   # <-- передаём a
        V = m * phi + g_nl * rho

        damp = torch.exp(torch.tensor(-0.75 * H * dt, device=device))
        psi = psi * torch.exp(-0.5j * dt * V) * damp

        kinetic = torch.exp(-1j * dt * K2 / (2 * m * a**2))
        psi_k = torch.fft.fftn(psi)
        psi_k = psi_k * kinetic
        psi = torch.fft.ifftn(psi_k)

        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2, a)
        V = m * phi + g_nl * rho
        psi = psi * torch.exp(-0.5j * dt * V) * damp

        if step % record_every == 0:
            rho = torch.abs(psi)**2
            phi = compute_potential(rho, G, scale_rho, K2, a)
            psi_k_current = torch.fft.fftn(psi)
            E_kin = torch.sum(torch.abs(psi_k_current)**2 * K2) / (2 * m * a**2) * (dx**2) / (N**2)
            # ИСПРАВЛЕНО: убран лишний scale_rho
            E_grav = 0.5 * torch.sum(rho * phi) * dx**2
            rho_max = torch.max(rho)
            n_vort = count_vortices_adaptive(psi, threshold_factor=0.02, dx=dx, L=L0)
            history['steps'].append(step)
            history['rho_max'].append(rho_max.item())
            history['E_kin'].append(E_kin.item())
            history['E_grav'].append(E_grav.item())
            history['N_vort'].append(n_vort)
    return history

# ---------- ЗАПУСК ----------
all_results = {}
for G in G_vals:
    print(f"\n=== Expanding G = {G}, phys corrected ===")
    hist_list = []
    for seed in range(n_realisations):
        print(f"  realisation {seed+1}/{n_realisations}")
        hist = run_simulation(G, seed+1000)
        hist_list.append(hist)
    # Агрегация
    steps_arr = hist_list[0]['steps']
    N_vort_mean = np.mean([h['N_vort'] for h in hist_list], axis=0)
    N_vort_std = np.std([h['N_vort'] for h in hist_list], axis=0)
    rho_max_mean = np.mean([h['rho_max'] for h in hist_list], axis=0)
    rho_max_std = np.std([h['rho_max'] for h in hist_list], axis=0)
    E_kin_mean = np.mean([h['E_kin'] for h in hist_list], axis=0)
    E_kin_std = np.std([h['E_kin'] for h in hist_list], axis=0)
    E_grav_mean = np.mean([h['E_grav'] for h in hist_list], axis=0)
    E_grav_std = np.std([h['E_grav'] for h in hist_list], axis=0)
    all_results[G] = {
        'steps': steps_arr,
        'N_vort_mean': N_vort_mean,
        'N_vort_std': N_vort_std,
        'rho_max_mean': rho_max_mean,
        'rho_max_std': rho_max_std,
        'E_kin_mean': E_kin_mean,
        'E_kin_std': E_kin_std,
        'E_grav_mean': E_grav_mean,
        'E_grav_std': E_grav_std,
    }
    np.savez(os.path.join(output_dir, f'expanding_G_{G:.1f}_phys.npz'), **all_results[G])

# ---------- ГРАФИКИ ----------
print("\nGenerating plots...")
G_list = list(all_results.keys())
N_avg = []
N_err = []
for G in G_list:
    data = all_results[G]
    steps_arr = np.array(data['steps'])
    n_mean = data['N_vort_mean']
    n_std = data['N_vort_std']
    idx = np.where(steps_arr >= 1500)[0]
    if len(idx) > 0:
        N_avg.append(np.mean(n_mean[idx]))
        N_err.append(np.mean(n_std[idx]))
    else:
        N_avg.append(0)
        N_err.append(0)

plt.figure(figsize=(8,6))
plt.errorbar(G_list, N_avg, yerr=N_err, fmt='ro-', capsize=5)
plt.xlabel('G')
plt.ylabel('Average vortices (t > 1500)')
plt.title(f'Expanding box (phys corrected), N={N}, t0={t0}')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'expanding_phase_diagram_phys.png'), dpi=150)
plt.show()

if 1.0 in all_results:
    data = all_results[1.0]
    steps_arr = np.array(data['steps'])
    ratio = np.abs(data['E_grav_mean']) / (data['E_kin_mean'] + 1e-12)
    plt.figure()
    plt.plot(steps_arr, ratio)
    plt.xlabel('Step')
    plt.ylabel('|E_grav|/E_kin')
    plt.title('Energy ratio (expanding, G=1, phys corrected)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'expanding_energy_ratio_phys.png'), dpi=150)
    plt.show()
    
    
# ---------- ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: rho_max для G=0 и G=1 ----------
if 0.0 in all_results and 1.0 in all_results:
    data0 = all_results[0.0]
    data1 = all_results[1.0]
    steps0 = np.array(data0['steps'])
    steps1 = np.array(data1['steps'])
    plt.figure(figsize=(8,6))
    plt.plot(steps0, data0['rho_max_mean'], label='G=0', color='blue')
    plt.fill_between(steps0, 
                     data0['rho_max_mean'] - data0['rho_max_std'], 
                     data0['rho_max_mean'] + data0['rho_max_std'], 
                     alpha=0.2, color='blue')
    plt.plot(steps1, data1['rho_max_mean'], label='G=1', color='orange')
    plt.fill_between(steps1, 
                     data1['rho_max_mean'] - data1['rho_max_std'], 
                     data1['rho_max_mean'] + data1['rho_max_std'], 
                     alpha=0.2, color='orange')
    plt.xlabel('Step')
    plt.ylabel(r'$\rho_{\max}$')
    plt.title('Maximum density evolution (expanding Universe)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rho_max_expanding_phys.png'), dpi=150)
    plt.show()
print(f"\nAll results saved in {output_dir}")