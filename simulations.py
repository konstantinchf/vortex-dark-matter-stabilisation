"""
Final simulation script for vortex stabilisation in self-gravitating scalar field dark matter.

Features:
- Multiple independent realisations (seeds) for each G.
- Tracking of individual vortices (lifetimes, trajectories).
- Convergence tests over grid size N and time step dt.
- Automatic saving of results and figures.
- Command-line arguments for quick testing or full production.

Usage:
    python final_script.py --mode full          # full production run (20 realisations, N=128, dt=0.0002)
    python final_script.py --mode test          # quick test (2 realisations, N=64, dt=0.001)
    python final_script.py --mode convergence   # convergence tests only
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
import csv
import os
import argparse
import time

# ============================================================
# Global variables (will be set by set_grid)
# ============================================================
K2 = None
dx = None
L_box = None

# ============================================================
# Default parameters
# ============================================================
DEFAULT_N = 128
DEFAULT_L = 20.0
DEFAULT_DT = 0.0002
DEFAULT_STEPS = 2000
DEFAULT_M = 1.0
DEFAULT_G_NL = 0.1
DEFAULT_SCALE_RHO = 100.0
DEFAULT_THRESHOLD = 0.05
DEFAULT_RECORD_EVERY = 20
DEFAULT_G_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_N_REALIZATIONS = 20   # for full run
DEFAULT_SEED_BASE = 12345

# ============================================================
# Grid initialisation
# ============================================================
def set_grid(N, L):
    global K2, dx, L_box
    L_box = L
    dx = L / N
    kx = fftfreq(N, d=dx) * 2 * np.pi
    ky = fftfreq(N, d=dx) * 2 * np.pi
    K2 = kx[:, None]**2 + ky[None, :]**2
    K2[0,0] = 1.0

# ============================================================
# Helper functions (use global K2, dx, L_box)
# ============================================================
def random_gaussian_field(N, power_law_index=-2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kx = fftfreq(N, d=dx) * 2 * np.pi
    ky = fftfreq(N, d=dx) * 2 * np.pi
    K2_loc = kx[:, None]**2 + ky[None, :]**2
    K2_loc[0,0] = 1.0
    amp = np.sqrt(K2_loc ** power_law_index)
    phase = np.exp(2j * np.pi * np.random.rand(N, N))
    field = np.real(ifft2(amp * phase))
    field = (field - field.mean()) / field.std()
    return field

def create_initial_state(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kx = fftfreq(N, d=dx) * 2 * np.pi
    ky = fftfreq(N, d=dx) * 2 * np.pi
    K2_loc = kx[:, None]**2 + ky[None, :]**2
    K2_loc[0,0] = 1.0
    amp_spectrum = np.sqrt(K2_loc ** (-2))
    real_part = np.random.randn(N, N) * amp_spectrum
    imag_part = np.random.randn(N, N) * amp_spectrum
    field = real_part + 1j * imag_part
    field = field / np.sqrt(np.mean(np.abs(field)**2))
    phase = np.angle(field)
    phase = gaussian_filter(phase, sigma=1.0)
    amp = np.abs(field)
    psi = amp * np.exp(1j * phase)
    psi = psi / np.sqrt(np.mean(np.abs(psi)**2))
    return psi

def compute_potential(rho, G, scale_rho):
    rho_scaled = rho * scale_rho
    rho_k = fft2(rho_scaled)
    phi_k = -4 * np.pi * G * rho_k / K2
    phi_k[0,0] = 0.0
    phi = np.real(ifft2(phi_k))
    return phi

def find_vortices(psi, threshold=0.05):
    phase = np.angle(psi)
    Nx, Ny = psi.shape
    vortices = []
    for i in range(Nx-1):
        for j in range(Ny-1):
            theta_00 = phase[i, j]
            theta_10 = phase[i+1, j]
            theta_11 = phase[i+1, j+1]
            theta_01 = phase[i, j+1]
            d1 = (theta_10 - theta_00 + np.pi) % (2*np.pi) - np.pi
            d2 = (theta_11 - theta_10 + np.pi) % (2*np.pi) - np.pi
            d3 = (theta_01 - theta_11 + np.pi) % (2*np.pi) - np.pi
            d4 = (theta_00 - theta_01 + np.pi) % (2*np.pi) - np.pi
            winding = (d1 + d2 + d3 + d4) / (2*np.pi)
            charge = int(round(winding))
            if charge != 0:
                amp_avg = (np.abs(psi[i,j]) + np.abs(psi[i+1,j]) +
                           np.abs(psi[i+1,j+1]) + np.abs(psi[i,j+1])) / 4.0
                if amp_avg < threshold:
                    xc = (i + 0.5) * dx - L_box/2
                    yc = (j + 0.5) * dx - L_box/2
                    vortices.append((xc, yc, charge))
    return vortices

# ============================================================
# Vortex tracking
# ============================================================
def track_vortices(prev_vortices, curr_vortices, max_dist=0.5):
    """Match vortices between two frames. Returns matches, appeared, disappeared."""
    if not prev_vortices or not curr_vortices:
        return [], list(range(len(curr_vortices))), list(range(len(prev_vortices)))
    prev_pos = np.array([[v[0], v[1]] for v in prev_vortices])
    curr_pos = np.array([[v[0], v[1]] for v in curr_vortices])
    tree = KDTree(prev_pos)
    matches = []
    used_prev = set()
    used_curr = set()
    for i_curr, (x,y,c) in enumerate(curr_vortices):
        dist, idx = tree.query([x,y])
        if dist < max_dist and idx not in used_prev:
            matches.append((idx, i_curr, dist))
            used_prev.add(idx)
            used_curr.add(i_curr)
    appeared = [i for i in range(len(curr_vortices)) if i not in used_curr]
    disappeared = [i for i in range(len(prev_vortices)) if i not in used_prev]
    return matches, appeared, disappeared

# ============================================================
# Single simulation with tracking
# ============================================================
def run_simulation_with_tracking(psi0, dt, steps, m, G, g, scale_rho, record_every=20):
    kinetic = np.exp(-1j * dt * K2 / (2*m))
    psi = psi0.copy()
    history = {'steps': [], 'rho_max': [], 'rho_std': [], 'E_kin': [], 'E_grav': [], 'N_vort': []}
    # Tracking structures
    vortex_tracks = {}   # track_id -> {'birth': step, 'positions': [], 'charge': 0}
    next_id = 0
    prev_vortices = []
    prev_ids = []        # list of track ids corresponding to prev_vortices
    track_history = {}   # step -> {track_id: (x, y, charge)}

    for step in range(steps):
        rho = np.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho)
        V = m * phi + g * rho
        # Strang split
        psi = psi * np.exp(-0.5j * dt * V)
        psi_k = fft2(psi)
        psi_k = psi_k * kinetic
        psi = ifft2(psi_k)
        rho = np.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho)
        V = m * phi + g * rho
        psi = psi * np.exp(-0.5j * dt * V)

        if step % record_every == 0:
            rho = np.abs(psi)**2
            phi = compute_potential(rho, G, scale_rho)
            psi_k = fft2(psi)
            E_kin = 0.5 * np.sum(np.abs(psi_k)**2 * K2) / (2*m) * dx**2
            E_grav = 0.5 * np.sum(rho * scale_rho * phi) * dx**2
            curr_vortices = find_vortices(psi, threshold=DEFAULT_THRESHOLD)
            history['steps'].append(step)
            history['rho_max'].append(np.max(rho))
            history['rho_std'].append(np.std(rho))
            history['E_kin'].append(E_kin)
            history['E_grav'].append(E_grav)
            history['N_vort'].append(len(curr_vortices))

            # ---------- Tracking ----------
            if prev_vortices:
                matches, appeared, disappeared = track_vortices(prev_vortices, curr_vortices)
                curr_ids = [None] * len(curr_vortices)
                # matched vortices
                for idx_prev, idx_curr, dist in matches:
                    track_id = prev_ids[idx_prev]
                    vortex_tracks[track_id]['positions'].append((curr_vortices[idx_curr][0], curr_vortices[idx_curr][1]))
                    curr_ids[idx_curr] = track_id
                # newly appeared
                for idx_curr in appeared:
                    xc, yc, charge = curr_vortices[idx_curr]
                    new_id = next_id
                    next_id += 1
                    vortex_tracks[new_id] = {'birth': step, 'positions': [(xc, yc)], 'charge': charge, 'alive': True}
                    curr_ids[idx_curr] = new_id
                # disappeared: mark as dead
                for idx_prev in disappeared:
                    track_id = prev_ids[idx_prev]
                    vortex_tracks[track_id]['alive'] = False
                prev_ids = curr_ids
            else:
                # first frame: all new
                curr_ids = []
                for idx_curr, (xc, yc, charge) in enumerate(curr_vortices):
                    new_id = next_id
                    next_id += 1
                    vortex_tracks[new_id] = {'birth': step, 'positions': [(xc, yc)], 'charge': charge, 'alive': True}
                    curr_ids.append(new_id)
                prev_ids = curr_ids

            prev_vortices = curr_vortices
            # store track history for this step (optional)
            track_history[step] = {tid: (vortex_tracks[tid]['positions'][-1] if vortex_tracks[tid]['positions'] else (0,0)) for tid in curr_ids if tid is not None}

    # Compute lifetimes from tracks
    lifetimes = []
    for tid, data in vortex_tracks.items():
        if data['positions']:
            # lifetime = (last recorded step - birth) * record_every? Actually steps are already recorded.
            # We stored birth as actual step. Last recorded step is birth + (len(positions)-1)*record_every
            last_step = data['birth'] + (len(data['positions'])-1) * record_every
            lifetime = last_step - data['birth']
            lifetimes.append(lifetime)
    history['lifetimes'] = lifetimes
    history['vortex_tracks'] = vortex_tracks
    return history

# ============================================================
# Multiple realisations
# ============================================================
def run_realizations(G, n_realizations, base_seed, dt, steps, m, g, scale_rho, record_every, N):
    set_grid(N, DEFAULT_L)   # will update global K2, dx, L_box
    histories = []
    for i in range(n_realizations):
        seed = base_seed + i
        psi0 = create_initial_state(N, seed=seed)
        print(f"G={G}, realization {i+1}/{n_realizations} (seed={seed})")
        hist = run_simulation_with_tracking(psi0, dt, steps, m, G, g, scale_rho, record_every)
        if hist is not None:
            histories.append(hist)
        else:
            print(f"  Warning: realisation {i} failed, skipped.")
    return histories

# ============================================================
# Aggregation
# ============================================================
def aggregate_histories(histories):
    if not histories:
        return None
    steps = histories[0]['steps']
    # Arrays
    rho_max = np.array([h['rho_max'] for h in histories])
    rho_std = np.array([h['rho_std'] for h in histories])
    E_kin   = np.array([h['E_kin']   for h in histories])
    E_grav  = np.array([h['E_grav']  for h in histories])
    N_vort  = np.array([h['N_vort']  for h in histories])
    # Lifetimes: combine all
    all_lifetimes = []
    for h in histories:
        all_lifetimes.extend(h['lifetimes'])
    agg = {
        'steps': steps,
        'rho_max_mean': np.mean(rho_max, axis=0),
        'rho_max_std':  np.std(rho_max, axis=0),
        'rho_std_mean': np.mean(rho_std, axis=0),
        'rho_std_std':  np.std(rho_std, axis=0),
        'E_kin_mean':   np.mean(E_kin, axis=0),
        'E_kin_std':    np.std(E_kin, axis=0),
        'E_grav_mean':  np.mean(E_grav, axis=0),
        'E_grav_std':   np.std(E_grav, axis=0),
        'N_vort_mean':  np.mean(N_vort, axis=0),
        'N_vort_std':   np.std(N_vort, axis=0),
        'lifetimes':    all_lifetimes,
    }
    return agg

# ============================================================
# Convergence tests
# ============================================================
def convergence_test(G_values, n_realizations=5, base_seed=12345):
    """
    Test convergence over N and dt. Runs a minimal set of parameters.
    """
    N_list = [64, 128, 256]
    dt_list = [0.0004, 0.0002, 0.0001]  # factor 2 each
    results = {}
    for N in N_list:
        dt = 0.0002  # fixed
        print(f"\nConvergence: N={N}, dt={dt}")
        set_grid(N, DEFAULT_L)
        histories = []
        for G in G_values:
            for i in range(n_realizations):
                seed = base_seed + i
                psi0 = create_initial_state(N, seed=seed)
                hist = run_simulation_with_tracking(psi0, dt, DEFAULT_STEPS, DEFAULT_M, G, DEFAULT_G_NL, DEFAULT_SCALE_RHO, DEFAULT_RECORD_EVERY)
                if hist:
                    histories.append(hist)
        # aggregate
        agg = aggregate_histories(histories)
        results[f'N={N}'] = agg
    for dt in dt_list:
        N = 128
        print(f"\nConvergence: N={N}, dt={dt}")
        set_grid(N, DEFAULT_L)
        histories = []
        for G in G_values:
            for i in range(n_realizations):
                seed = base_seed + i
                psi0 = create_initial_state(N, seed=seed)
                hist = run_simulation_with_tracking(psi0, dt, DEFAULT_STEPS, DEFAULT_M, G, DEFAULT_G_NL, DEFAULT_SCALE_RHO, DEFAULT_RECORD_EVERY)
                if hist:
                    histories.append(hist)
        agg = aggregate_histories(histories)
        results[f'dt={dt}'] = agg
    return results

# ============================================================
# Plotting functions
# ============================================================
def plot_vortex_evolution(all_results, G_values, title_suffix=''):
    plt.figure(figsize=(10,6))
    for G in G_values:
        agg = all_results[G]
        steps = agg['steps']
        mean = agg['N_vort_mean']
        std = agg['N_vort_std']
        plt.plot(steps, mean, label=f'G={G}')
        plt.fill_between(steps, mean-std, mean+std, alpha=0.2)
    plt.xlabel('Time step')
    plt.ylabel('Number of vortices')
    plt.title(f'Vortex number evolution {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'vortex_evolution{title_suffix.replace(" ","_")}.png', dpi=150)
    plt.show()

def plot_phase_diagram(G_values, all_results, title_suffix=''):
    late_avg = []
    for G in G_values:
        agg = all_results[G]
        steps_arr = np.array(agg['steps'])
        idx = np.where(steps_arr >= 1500)[0]
        if len(idx) > 0:
            n_mean = np.mean(agg['N_vort_mean'][idx])
            n_std = np.mean(agg['N_vort_std'][idx])
            late_avg.append((G, n_mean, n_std))
        else:
            late_avg.append((G, 0, 0))
    late_avg = np.array(late_avg)
    G_vals = late_avg[:,0]; N_avg = late_avg[:,1]; N_err = late_avg[:,2]
    plt.figure(figsize=(8,6))
    plt.errorbar(G_vals, N_avg, yerr=N_err, fmt='bo', capsize=5, label='Simulation')
    # fit
    mask = G_vals > 0.3
    if np.sum(mask) >= 3:
        def power_law(G, A, Gc, beta):
            return A * np.maximum(G - Gc, 0)**beta
        try:
            popt, _ = curve_fit(power_law, G_vals[mask], N_avg[mask], p0=[30, 0.3, 0.8])
            A_fit, Gc_fit, beta_fit = popt
            G_plot = np.linspace(0, 1.2, 100)
            N_plot = power_law(G_plot, *popt)
            plt.plot(G_plot, N_plot, 'r--', label=f'Fit: N={A_fit:.1f} (G-{Gc_fit:.2f})$^{{{beta_fit:.2f}}}$')
        except:
            pass
    plt.xlabel('G (gravitational constant)')
    plt.ylabel('Average vortex number (t > 1500 steps)')
    plt.title(f'Stabilisation threshold {title_suffix}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'phase_diagram{title_suffix.replace(" ","_")}.png', dpi=150)
    plt.show()

def plot_energy_ratio(all_results, G=1.0, title_suffix=''):
    if G not in all_results: return
    agg = all_results[G]
    steps = agg['steps']
    ratio_mean = np.abs(agg['E_grav_mean']) / (agg['E_kin_mean'] + 1e-12)
    ratio_std = ratio_mean * np.sqrt( (agg['E_grav_std']/np.abs(agg['E_grav_mean']+1e-12))**2 + (agg['E_kin_std']/(agg['E_kin_mean']+1e-12))**2 )
    plt.figure(figsize=(8,6))
    plt.plot(steps, ratio_mean, 'r-', label='mean')
    plt.fill_between(steps, ratio_mean-ratio_std, ratio_mean+ratio_std, alpha=0.2)
    plt.xlabel('Time step')
    plt.ylabel('|E_grav| / E_kin')
    plt.title(f'Energy ratio for G={G} {title_suffix}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'energy_ratio_G{G}{title_suffix.replace(" ","_")}.png', dpi=150)
    plt.show()

def plot_lifetime_distribution(all_results, G_values, title_suffix=''):
    plt.figure(figsize=(10,6))
    for G in G_values:
        agg = all_results[G]
        lifetimes = agg['lifetimes']
        if lifetimes:
            plt.hist(lifetimes, bins=20, alpha=0.5, label=f'G={G}')
    plt.xlabel('Lifetime (steps)')
    plt.ylabel('Count')
    plt.title(f'Vortex lifetime distribution {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'lifetime_distribution{title_suffix.replace(" ","_")}.png', dpi=150)
    plt.show()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vortex simulation')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'test', 'convergence'],
                        help='Run mode: full (20 realisations, N=128), test (2 realisations, N=64), convergence')
    args = parser.parse_args()

    if args.mode == 'test':
        n_realizations = 2
        N = 64
        dt = 0.001
        steps = 500
        G_values = [0.0, 1.0]   # only two for quick test
        title_suffix = ' (test)'
    elif args.mode == 'convergence':
        # run convergence tests only
        print("Running convergence tests...")
        conv_results = convergence_test([0.0, 0.5, 1.0], n_realizations=3, base_seed=12345)
        # Plot convergence results (simplified)
        for key, agg in conv_results.items():
            plt.figure()
            plt.plot(agg['steps'], agg['N_vort_mean'], label=key)
            plt.title('Convergence test: N_vort')
            plt.legend()
            plt.savefig(f'convergence_{key}.png')
        print("Convergence tests done. Exiting.")
        exit()
    else:  # full
        n_realizations = 20
        N = 128
        dt = DEFAULT_DT
        steps = DEFAULT_STEPS
        G_values = DEFAULT_G_VALUES
        title_suffix = ' (full)'

    # L is fixed to DEFAULT_L
    set_grid(N, DEFAULT_L)   # sets global K2, dx, L_box

    all_results = {}
    for G in G_values:
        print(f"\n===== Running for G = {G} with {n_realizations} realizations =====")
        histories = run_realizations(G, n_realizations, DEFAULT_SEED_BASE, dt, steps, DEFAULT_M, DEFAULT_G_NL, DEFAULT_SCALE_RHO, DEFAULT_RECORD_EVERY, N)
        if histories:
            agg = aggregate_histories(histories)
            all_results[G] = agg
        else:
            print(f"  No successful realisations for G={G}")

    # Save aggregated data
    with open('simulation_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['G', 'step', 'N_vort_mean', 'N_vort_std', 'rho_max_mean', 'rho_max_std',
                         'E_kin_mean', 'E_kin_std', 'E_grav_mean', 'E_grav_std'])
        for G, agg in all_results.items():
            for i, step in enumerate(agg['steps']):
                writer.writerow([G, step,
                                 agg['N_vort_mean'][i], agg['N_vort_std'][i],
                                 agg['rho_max_mean'][i], agg['rho_max_std'][i],
                                 agg['E_kin_mean'][i], agg['E_kin_std'][i],
                                 agg['E_grav_mean'][i], agg['E_grav_std'][i]])

    # Generate plots
    plot_vortex_evolution(all_results, G_values, title_suffix)
    plot_phase_diagram(G_values, all_results, title_suffix)
    plot_energy_ratio(all_results, G=1.0, title_suffix=title_suffix)
    # Max density for G=0 and G=1
    plt.figure()
    for G in [0.0, 1.0]:
        if G in all_results:
            agg = all_results[G]
            steps = agg['steps']
            mean = agg['rho_max_mean']
            std = agg['rho_max_std']
            plt.plot(steps, mean, label=f'G={G}')
            plt.fill_between(steps, mean-std, mean+std, alpha=0.2)
    plt.xlabel('Time step')
    plt.ylabel(r'$\rho_{\max}$')
    plt.title(f'Maximum density evolution {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'rho_max{title_suffix.replace(" ","_")}.png', dpi=150)
    plt.show()

    # Lifetime distribution
    plot_lifetime_distribution(all_results, G_values, title_suffix)

    print("All plots saved. Data written to simulation_results.csv")