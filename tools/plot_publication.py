#!/usr/bin/env python3
# =============================================================================
# plot_publication.py — PUBLICATION-GRADE HIGH-TECH SCIENTIFIC JOURNAL FIGURES
# Sleek cohesive color system: Navy Blue (Earth), Rust Red (Moon Cohesive Jammed),
# Emerald Teal (Moon Fluidized Active Flow).
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
import pandas as pd
import os
from PIL import Image

# Output directory for the ASR latex figures
output_dir = '/home/neo/liggghts/project/paper/Figures'
results_dir = '/home/neo/liggghts/project/data'

os.makedirs(output_dir, exist_ok=True)

# --- Sleek High-Tech Color Tokens ---
C_TEXT     = '#1E293B'  # Sleek Charcoal/Slate-900 (crisp high-contrast text)
C_EARTH    = '#1A365D'  # Deep Slate Navy Blue (Terrestrial baseline)
C_MOON     = '#C53030'  # Muted Rust/Brick Red (Cohesive jammed lunar condition)
C_FLUID    = '#0D9488'  # Energetic Teal/Emerald (Active fluidized flow)
C_TARGET   = '#EF4444'  # Crimson Red (crisp indicator/benchmark)
C_GRID     = '#F1F5F9'  # Very soft, clean slate-100 grid lines
C_SPINE    = '#CBD5E1'  # Clean border line color
C_BG_ZONE  = '#F8FAFC'  # Very soft grey fill for threshold zones

# --- Premium Global Matplotlib Styling ---
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'axes.linewidth': 0.6,
    'axes.edgecolor': C_SPINE,
    'axes.labelcolor': C_TEXT,
    'text.color': C_TEXT,
    'xtick.color': C_TEXT,
    'ytick.color': C_TEXT,
    'grid.alpha': 1.0,
    'grid.color': C_GRID,
    'grid.linestyle': '-',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def crop_and_load_image(img_path):
    """Load an image and crop transparent or white borders to maximize display size."""
    if not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Simple bounding box crop for white/transparent borders
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        return np.array(img)
    except Exception as e:
        print(f"[-] Warning: Failed to process image {img_path}: {e}")
        return None

def finalize_plot(ax, title, xlabel, ylabel):
    """Apply modern clean spines and thin gridlines."""
    ax.set_title(title, fontweight='bold', pad=12, color=C_TEXT)
    ax.set_xlabel(xlabel, labelpad=6, color=C_TEXT)
    ax.set_ylabel(ylabel, labelpad=6, color=C_TEXT)
    
    # Clean modern spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_color(C_SPINE)
    ax.spines['bottom'].set_color(C_SPINE)
    
    # Crisp thin grid
    ax.grid(True, linestyle='-', color=C_GRID, linewidth=0.5, zorder=0)
    ax.tick_params(colors=C_TEXT, width=0.6, length=4)

# =============================================================================
# Figure 1: Cohesion Calibration & Repose Pile Convergence (4 Panels, 2x2 Grid)
# =============================================================================
def generate_composite_calibration():
    print("Generating Composite Figure 1 (Calibration & Repose)...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # --------------------------------------------------------
    # Panel (a): Scale Convergence from raw LIGGGHTS dumps
    # --------------------------------------------------------
    ax_a = axes[0, 0]
    paths = {
        '5k Particles': '/home/neo/liggghts/project/research_attic/Angle of Repose/AoR_Calibration_Study/run_D1.0_S1/dump/final_positions.txt',
        '20k Particles': '/home/neo/liggghts/project/research_attic/Angle of Repose/Calibration_20k/dump/seed_1/final_positions.txt'
    }
    colors = ['#38BDF8', '#1A365D']  # Modern bright blue vs premium dark navy
    
    has_a = False
    for i, (label, path) in enumerate(paths.items()):
        if os.path.exists(path):
            data = []
            reading = False
            with open(path, 'r') as f:
                for line in f:
                    if "ITEM: ATOMS" in line:
                        reading = True
                        continue
                    if reading:
                        parts = line.split()
                        if len(parts) >= 5:
                            data.append([float(parts[2]), float(parts[3]), float(parts[4])])
            data = np.array(data)
            if len(data) > 0:
                indices = np.random.choice(len(data), min(len(data), 1500), replace=False)
                x, y, z = data[indices, 0], data[indices, 1], data[indices, 2]
                r = np.sqrt(x**2 + y**2) * 1000  # to mm
                z_mm = z * 1000
                ax_a.scatter(r, z_mm, s=1.5, alpha=0.3, color=colors[i], label=label, rasterized=True, zorder=3)
                has_a = True
                
    # Add target slope line (38.3 deg)
    target_slope = np.tan(np.radians(38.3))
    x_ref = np.array([15, 55])
    y_ref = 32 - target_slope * (x_ref - 15)
    ax_a.plot(x_ref, y_ref, color=C_TARGET, linestyle='--', linewidth=1.5, label=r'38.3$^\circ$ Target Slope', zorder=4)
    
    finalize_plot(ax_a, '(a) Scale Convergence Profile', 'Radial Distance (mm)', 'Height (mm)')
    ax_a.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9, edgecolor='none')
    ax_a.set_ylim(0, 50)
    ax_a.set_xlim(0, 100)

    # --------------------------------------------------------
    # Panel (b): Fitted cross-sectional profile (aor_final_20k)
    # --------------------------------------------------------
    ax_b = axes[0, 1]
    profile_img_path = os.path.join(output_dir, 'aor_final_20k.png')
    img_b = crop_and_load_image(profile_img_path)
    if img_b is not None:
        ax_b.imshow(img_b)
        ax_b.axis('off')
        ax_b.set_title('(b) Fitted Slope Profile (20k)', fontweight='bold', pad=12, color=C_TEXT)
    else:
        ax_b.text(0.5, 0.5, 'aor_final_20k.png\nnot found', ha='center', va='center', color=C_TEXT)
        ax_b.axis('off')

    # --------------------------------------------------------
    # Panel (c): 3D settled rest pile for 5k particles
    # --------------------------------------------------------
    ax_c = axes[1, 0]
    pv_5k_path = os.path.join(output_dir, 'paraview_5k.png')
    img_c = crop_and_load_image(pv_5k_path)
    if img_c is not None:
        ax_c.imshow(img_c)
        ax_c.axis('off')
        ax_c.set_title('(c) 5k Conical Heap (ParaView)', fontweight='bold', pad=12, color=C_TEXT)
    else:
        ax_c.text(0.5, 0.5, 'paraview_5k.png\nnot found', ha='center', va='center', color=C_TEXT)
        ax_c.axis('off')

    # --------------------------------------------------------
    # Panel (d): 3D settled rest pile for 20k particles
    # --------------------------------------------------------
    ax_d = axes[1, 1]
    pv_20k_path = os.path.join(output_dir, 'paraview_20k.png')
    img_d = crop_and_load_image(pv_20k_path)
    if img_d is not None:
        ax_d.imshow(img_d)
        ax_d.axis('off')
        ax_d.set_title('(d) 20k Conical Heap (ParaView)', fontweight='bold', pad=12, color=C_TEXT)
    else:
        ax_d.text(0.5, 0.5, 'paraview_20k.png\nnot found', ha='center', va='center', color=C_TEXT)
        ax_d.axis('off')

    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig_composite_calibration.png')
    pdf_path = os.path.join(output_dir, 'fig_composite_calibration.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved composite calibration figure to: {png_path} and {pdf_path}")

# =============================================================================
# Figure 2: Granular Flow Dynamics & Fluidization Diagnostics (2 Panels)
# =============================================================================
def generate_composite_flow_dynamics():
    print("Generating Composite Figure 2 (Flow Dynamics)...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    
    # --------------------------------------------------------
    # Panel (a): Coordination Number transient timeseries
    # --------------------------------------------------------
    ax_a = axes[0]
    cases = {
        "E_A05": (C_EARTH, ':', 'Earth 0.5mm (Jammed / Compact)'), 
        "M_A05": (C_MOON, '-', 'Moon 0.5mm (Jammed Clump)'), 
        "M_A20": (C_FLUID, '--', 'Moon 2.0mm (Fluidized Flow)')
    }
    
    has_data = False
    for case, (color, ls, label) in cases.items():
        data_path = os.path.join(results_dir, f"{case}_data.npz")
        if os.path.exists(data_path):
            data = np.load(data_path)
            ax_a.plot(data['times'], data['cn_history'], label=label, color=color, linestyle=ls, lw=1.6, zorder=3)
            has_data = True
            
    if has_data:
        # Simple, ultra-clean horizontal dashed lines for CN regimes
        ax_a.axhline(y=1.2, color='#64748B', linestyle=':', lw=1.0, zorder=2)
        ax_a.text(0.05, 1.3, 'Fluidization Threshold (CN = 1.2)', fontsize=8, color='#64748B', style='italic')
        
        ax_a.axhline(y=2.0, color='#64748B', linestyle=':', lw=1.0, zorder=2)
        ax_a.text(0.05, 2.1, 'Jammed Threshold (CN = 2.0)', fontsize=8, color='#64748B', style='italic')
        
        finalize_plot(ax_a, '(a) Bed State (Coordination Number)', 'Time (s)', 'Mean Coordination Number (CN)')
        ax_a.set_ylim(0, 4.2)
        ax_a.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none')
    else:
        ax_a.text(0.5, 0.5, 'CN simulation data\nnot found', ha='center', va='center', color=C_TEXT)

    # --------------------------------------------------------
    # Panel (b): Granular Bond Number analytical analysis
    # --------------------------------------------------------
    ax_b = axes[1]
    
    rho = 2920.0        # kg/m^3
    g_earth = 9.81
    g_moon = 1.62
    
    f_vdw_mean = 15e-9  # 15 nN
    f_vdw_low = 10e-9   # 10 nN
    f_vdw_high = 20e-9  # 20 nN
    
    d_range = np.logspace(-5, -3.3, 100) # 10um to 500um
    m_range = rho * (np.pi / 6.0) * d_range**3
    
    bo_earth = f_vdw_mean / (m_range * g_earth)
    bo_moon = f_vdw_mean / (m_range * g_moon)
    
    bo_earth_low = f_vdw_low / (m_range * g_earth)
    bo_earth_high = f_vdw_high / (m_range * g_earth)
    bo_moon_low = f_vdw_low / (m_range * g_moon)
    bo_moon_high = f_vdw_high / (m_range * g_moon)
    
    # Ultra-soft transparent fills
    ax_b.fill_between(d_range * 1e6, bo_earth_low, bo_earth_high, alpha=0.06, color=C_EARTH, zorder=1)
    ax_b.fill_between(d_range * 1e6, bo_moon_low, bo_moon_high, alpha=0.06, color=C_MOON, zorder=1)
    
    ax_b.loglog(d_range * 1e6, bo_earth, label='Earth (9.81 m/s$^2$)', color=C_EARTH, lw=1.8, zorder=3)
    ax_b.loglog(d_range * 1e6, bo_moon, label='Moon (1.62 m/s$^2$)', color=C_MOON, lw=1.8, zorder=3)
    
    # Threshold line (Bo_g = 1)
    ax_b.axhline(y=1.0, color=C_TEXT, linestyle=':', alpha=0.5, lw=1.0, zorder=2)
    ax_b.text(12, 1.2, r'$Bo_g = 1$ crossover', fontsize=8.5, color=C_TEXT, style='italic')
    
    # Mark LMS-1 mean grain size (91 um)
    d_lms1 = 91.0
    ax_b.axvline(x=d_lms1, color=C_TARGET, linestyle='--', lw=1.2, label='LMS-1 Mean Size (91 $\mu$m)', zorder=2)
    
    finalize_plot(ax_b, '(b) Analytical Granular Bond Number', 'Particle Diameter ($d$, $\mu$m)', 'Granular Bond Number ($Bo_g$)')
    ax_b.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9, edgecolor='none')
    ax_b.set_xlim(10, 500)
    ax_b.set_ylim(0.05, 100)
    ax_b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig_composite_flow_dynamics.png')
    pdf_path = os.path.join(output_dir, 'fig_composite_flow_dynamics.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved composite flow dynamics figure to: {png_path} and {pdf_path}")

# =============================================================================
# Figure 3: Triboelectric Charging & Collision Mechanics (3 Panels)
# =============================================================================
def generate_composite_charging():
    print("Generating Composite Figure 3 (Triboelectric Charging)...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    
    # --------------------------------------------------------
    # Panel (a): Transient mean charge history
    # --------------------------------------------------------
    ax_a = axes[0]
    has_a = False
    
    for case in ["E_A10", "M_A10"]:
        data_path = os.path.join(results_dir, f"{case}_data.npz")
        if os.path.exists(data_path):
            data = np.load(data_path)
            q_mean_pc = data['q_history'] * 1e12
            q_std_pc = data['q_std_history'] * 1e12
            
            color = C_EARTH if 'E' in case else C_MOON
            label = 'Earth 1.0mm (Terrestrial)' if 'E' in case else 'Moon 1.0mm (Lunar)'
            
            ax_a.plot(data['times'], q_mean_pc, label=label, color=color, lw=1.8, zorder=3)
            ax_a.fill_between(data['times'], q_mean_pc - q_std_pc, q_mean_pc + q_std_pc, 
                            color=color, alpha=0.06, lw=0, zorder=2)
            has_a = True
            
    if has_a:
        finalize_plot(ax_a, '(a) Charge Accumulation History', 'Time (s)', r'Avg. Particle Charge $q$ (pC)')
        ax_a.legend(frameon=True, loc='upper left', facecolor='white', framealpha=0.9, edgecolor='none')
        ax_a.set_ylim(0, 1000)
    else:
        ax_a.text(0.5, 0.5, 'Charge history data\nnot found', ha='center', va='center', color=C_TEXT)

    # --------------------------------------------------------
    # Panel (b): Scipy Gaussian Kernel Density Estimation (KDE)
    # --------------------------------------------------------
    ax_b = axes[1]
    has_b = False
    
    for case in ["E_A10", "M_A10"]:
        data_path = os.path.join(results_dir, f"{case}_data.npz")
        if os.path.exists(data_path):
            data = np.load(data_path)
            energies = data['energies']
            if len(energies) > 10:
                energies = energies[energies > 1e-11]
                if len(energies) > 50:
                    color = C_EARTH if 'E' in case else C_MOON
                    label = 'Earth 1.0mm' if 'E' in case else 'Moon 1.0mm'
                    
                    log_energies = np.log10(energies)
                    if len(log_energies) > 10000:
                        np.random.seed(42)  # Deterministic downsampling
                        log_energies = np.random.choice(log_energies, 10000, replace=False)
                    kde = gaussian_kde(log_energies)
                    
                    x_eval = np.linspace(-10, -3, 200)
                    kde_vals = kde.evaluate(x_eval)
                    
                    ax_b.plot(10**x_eval, kde_vals, color=color, lw=1.8, label=label, zorder=3)
                    ax_b.fill_between(10**x_eval, 0, kde_vals, color=color, alpha=0.08, zorder=2)
                    has_b = True
                    
    if has_b:
        finalize_plot(ax_b, '(b) Collision Energy Distribution', 'Collision Energy $\Delta E$ (J)', 'Probability Density')
        ax_b.set_xscale('log')
        ax_b.set_xlim(1e-10, 1e-4)
        ax_b.set_ylim(0, 0.8)
        ax_b.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9, edgecolor='none')
    else:
        ax_b.text(0.5, 0.5, 'Collision energy data\nnot found', ha='center', va='center', color=C_TEXT)

    # --------------------------------------------------------
    # Panel (c): Parametric Q/M ratio vs. Amplitude (Bar Chart)
    # --------------------------------------------------------
    ax_c = axes[2]
    summary_path = os.path.join(results_dir, "charging_summary.csv")
    
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        earth = df[df['gravity'] == 'Earth']
        moon = df[df['gravity'] == 'Moon']
        
        if len(earth) > 0 or len(moon) > 0:
            x = np.arange(max(len(earth), len(moon)))
            width = 0.30
            
            # Clean flat bars without edge outlines
            if len(earth) > 0:
                ax_c.bar(x[:len(earth)] - width/2, earth['qm_ratio_ucg_final'], width, 
                       yerr=earth['qm_std_ucg_final'], capsize=3, 
                       error_kw={'alpha':0.5, 'lw':0.7, 'ecolor': C_TEXT},
                       label='Earth (9.81 m/s$^2$)', color=C_EARTH, edgecolor='none', zorder=3)
                       
            if len(moon) > 0:
                # Use C_FLUID for Moon 2.0mm to show fluidization rescue, C_MOON for lower amplitudes
                moon_colors = [C_MOON, C_MOON, C_FLUID]
                for idx, row in enumerate(moon.itertuples()):
                    # We plot each bar individually to support our premium multi-color theme
                    bar_x = x[idx] + width/2
                    ax_c.bar(bar_x, row.qm_ratio_ucg_final, width,
                           yerr=row.qm_std_ucg_final, capsize=3,
                           error_kw={'alpha':0.5, 'lw':0.7, 'ecolor': C_TEXT},
                           color=moon_colors[idx], edgecolor='none', zorder=3,
                           label='Moon (1.62 m/s$^2$)' if idx == 0 else "")
            
            # Ultra-clean grey band for Trigwell experimental benchmark range
            ax_c.axhspan(0.45, 0.55, color='#F1F5F9', alpha=0.7, label='Trigwell Target Range', zorder=1)
            ax_c.axhline(y=0.5, color=C_TARGET, linestyle='--', lw=1.2, alpha=0.8, zorder=2)
            
            finalize_plot(ax_c, '(c) Final Charge-to-Mass Ratio ($Q/M$)', 'Vibration Amplitude (mm)', r'Average $Q/M$ ($\mu$C/g)')
            
            labels = earth['amplitude'].tolist() if len(earth) > 0 else moon['amplitude'].tolist()
            ax_c.set_xticks(x[:len(labels)])
            ax_c.set_xticklabels(labels)
            ax_c.set_ylim(0, 1.4)
            ax_c.legend(frameon=True, loc='upper left', facecolor='white', framealpha=0.9, edgecolor='none')
    else:
        ax_c.text(0.5, 0.5, 'charging_summary.csv\nnot found', ha='center', va='center', color=C_TEXT)

    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig_composite_charging.png')
    pdf_path = os.path.join(output_dir, 'fig_composite_charging.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved composite charging figure to: {png_path} and {pdf_path}")

if __name__ == "__main__":
    print("[+] Starting high-tech premium figure rebuild...")
    generate_composite_calibration()
    generate_composite_flow_dynamics()
    generate_composite_charging()
    print("[+] All publication figures rebuilt successfully to vector PDF and premium PNG standards.")
