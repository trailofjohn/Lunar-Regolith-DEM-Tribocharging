#!/usr/bin/env python3
# =============================================================================
# plot_publication.py — PREMIUM MULTI-PANEL ASR JOURNAL FIGURES
# Optimized for publication in Advances in Space Research (Elsevier)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
import pandas as pd
import os
from PIL import Image

# Output directory for the ASR latex figures
output_dir = '/home/neo/liggghts/project/latex_files/paper_asr/Figures'
results_dir = '/home/neo/liggghts/project/final_polished_production_code/results'

os.makedirs(output_dir, exist_ok=True)

# --- Premium Elsevier Journal Styling ---
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10.5,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Premium color-blind safe HSL-curated color palette
C_EARTH   = '#2E5B88'  # Premium Muted Slate Blue (Earth baseline)
C_MOON    = '#606C71'  # Clean Steel/Charcoal Gray (Lunar regolith)
C_TARGET  = '#C0504D'  # Dusty Terracotta Rose (Trigwell benchmark range)
C_FLOW    = '#4CAF50'  # Sage Green (Fluidized state)
C_JAM     = '#E57373'  # Soft Coral (Jammed state)
C_GRID    = '#E8E8E8'  # Clean light gray for grid lines

def crop_and_load_image(img_path):
    """Load an image and crop transparent or white borders to maximize display size."""
    if not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path)
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get bounding box of non-white pixels
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        diff = Image.new('RGBA', img.size)
        # Simple bounding box crop for white/transparent borders
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        return np.array(img)
    except Exception as e:
        print(f"[-] Warning: Failed to process image {img_path}: {e}")
        return None

def finalize_plot(ax, title, xlabel, ylabel):
    """Apply unified styling and borders to a plot."""
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Use standard clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(True, linestyle='--', color=C_GRID, alpha=0.7)

# =============================================================================
# Figure 1: Cohesion Calibration & Repose Pile Convergence (3 Panels)
# =============================================================================
def generate_composite_calibration():
    print("Generating Composite Figure 1 (Calibration & Repose)...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    
    # --------------------------------------------------------
    # Panel (a): Scale Convergence from raw LIGGGHTS dumps
    # --------------------------------------------------------
    ax_a = axes[0]
    paths = {
        '5k Particles': '/home/neo/liggghts/project/Angle of Repose/AoR_Calibration_Study/run_D1.0_S1/dump/final_positions.txt',
        '20k Particles': '/home/neo/liggghts/project/Angle of Repose/Calibration_20k/dump/seed_1/final_positions.txt'
    }
    colors = ['#5499C7', '#1F4E79']
    markers = ['o', 's']
    
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
                # We scatter a subset to make it clean and fast
                indices = np.random.choice(len(data), min(len(data), 1500), replace=False)
                x, y, z = data[indices, 0], data[indices, 1], data[indices, 2]
                r = np.sqrt(x**2 + y**2) * 1000  # to mm
                z_mm = z * 1000
                ax_a.scatter(r, z_mm, s=1.5, alpha=0.3, color=colors[i], label=label, rasterized=True)
                has_a = True
                
    # Add target slope line (38.3 deg)
    target_slope = np.tan(np.radians(38.3))
    x_ref = np.array([15, 55])
    y_ref = 32 - target_slope * (x_ref - 15)
    ax_a.plot(x_ref, y_ref, color=C_TARGET, linestyle='--', linewidth=2.0, label=r'38.3$^\circ$ Target Slope')
    
    finalize_plot(ax_a, '(a) Scale Convergence Profile', 'Radial Distance (mm)', 'Height (mm)')
    ax_a.legend(frameon=True, loc='upper right')
    ax_a.set_ylim(0, 50)
    ax_a.set_xlim(0, 100)

    # --------------------------------------------------------
    # Panel (b): Fitted cross-sectional profile (aor_final_20k)
    # --------------------------------------------------------
    ax_b = axes[1]
    profile_img_path = os.path.join(output_dir, 'aor_final_20k.png')
    img_b = crop_and_load_image(profile_img_path)
    if img_b is not None:
        ax_b.imshow(img_b)
        ax_b.axis('off')
        ax_b.set_title('(b) Fitted Slope Profile (20k)', fontweight='bold', pad=10)
    else:
        ax_b.text(0.5, 0.5, 'aor_final_20k.png\nnot found', ha='center', va='center')
        ax_b.axis('off')

    # --------------------------------------------------------
    # Panel (c): 3D settled rest piles (paraview_20k)
    # --------------------------------------------------------
    ax_c = axes[2]
    pv_img_path = os.path.join(output_dir, 'paraview_20k.png')
    img_c = crop_and_load_image(pv_img_path)
    if img_c is not None:
        ax_c.imshow(img_c)
        ax_c.axis('off')
        ax_c.set_title('(c) settled 3D Heap (ParaView)', fontweight='bold', pad=10)
    else:
        ax_c.text(0.5, 0.5, 'paraview_20k.png\nnot found', ha='center', va='center')
        ax_c.axis('off')

    plt.tight_layout()
    
    # Save composite figure in both PNG and vector PDF format
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
        "E_A05": (C_EARTH, '--', 'Earth 0.5mm (Flow)'), 
        "M_A05": (C_MOON, '-', 'Moon 0.5mm (Jammed)'), 
        "M_A20": ('#E67E22', '-', 'Moon 2.0mm (Flow)')
    }
    
    has_data = False
    for case, (color, ls, label) in cases.items():
        data_path = os.path.join(results_dir, f"{case}_data.npz")
        if os.path.exists(data_path):
            data = np.load(data_path)
            ax_a.plot(data['times'], data['cn_history'], label=label, color=color, linestyle=ls, lw=1.8)
            has_data = True
            
    if has_data:
        # Shading for the fluidized regime (CN < 1.0)
        ax_a.axhspan(0, 1.0, color='#4CAF50', alpha=0.08, label='Fluidized Regime')
        # Shading for the jammed regime (CN > 2.0)
        ax_a.axhspan(2.0, 5.0, color='#E57373', alpha=0.08, label='Jammed Regime')
        
        finalize_plot(ax_a, '(a) Bed State (Coordination Number)', 'Time (s)', 'Mean Coordination Number (CN)')
        ax_a.set_ylim(0, 4.5)
        ax_a.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white')
    else:
        ax_a.text(0.5, 0.5, 'CN simulation data\nnot found', ha='center', va='center')

    # --------------------------------------------------------
    # Panel (b): Granular Bond Number analytical analysis
    # --------------------------------------------------------
    ax_b = axes[1]
    
    # LMS-1 parameters
    rho = 2920.0        # kg/m^3
    g_earth = 9.81
    g_moon = 1.62
    
    # F_vdW central and limits
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
    
    # Plot bounds
    ax_b.fill_between(d_range * 1e6, bo_earth_low, bo_earth_high, alpha=0.15, color='#4A90E2')
    ax_b.fill_between(d_range * 1e6, bo_moon_low, bo_moon_high, alpha=0.15, color='#C0392B')
    
    ax_b.loglog(d_range * 1e6, bo_earth, label='Earth (9.81 m/s$^2$)', color=C_EARTH, lw=2.0)
    ax_b.loglog(d_range * 1e6, bo_moon, label='Moon (1.62 m/s$^2$)', color='#C62828', lw=2.0)
    
    # Threshold line (Bo_g = 1)
    ax_b.axhline(y=1.0, color='black', linestyle='--', alpha=0.6, lw=1.2)
    ax_b.text(12, 1.2, r'$Bo_g = 1$ threshold', fontsize=8, color='black', style='italic')
    
    # Mark LMS-1 mean grain size (91 um)
    d_lms1 = 91.0
    ax_b.axvline(x=d_lms1, color='#E67E22', linestyle=':', lw=1.5, label='LMS-1 Mean Size (91 $\mu$m)')
    
    finalize_plot(ax_b, '(b) Analytical Granular Bond Number', 'Particle Diameter ($d$, $\mu$m)', 'Granular Bond Number ($Bo_g$)')
    ax_b.legend(frameon=True, loc='upper right')
    ax_b.set_xlim(10, 500)
    ax_b.set_ylim(0.05, 100)
    ax_b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    
    # Save composite figure in both PNG and vector PDF format
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
            q_mean_uc = data['q_history'] * 1e6
            q_std_uc = data['q_std_history'] * 1e6
            
            color = C_EARTH if 'E' in case else C_MOON
            label = 'Earth (9.81 m/s$^2$)' if 'E' in case else 'Moon (1.62 m/s$^2$)'
            
            ax_a.plot(data['times'], q_mean_uc, label=label, color=color, lw=2.0)
            ax_a.fill_between(data['times'], q_mean_uc - q_std_uc, q_mean_uc + q_std_uc, 
                            color=color, alpha=0.15, lw=0)
            has_a = True
            
    if has_a:
        finalize_plot(ax_a, '(a) Charge Accumulation History', 'Time (s)', r'Avg. Particle Charge $q$ ($\mu$C)')
        ax_a.legend(frameon=True, loc='upper left')
        ax_a.set_ylim(0, 0.6)
    else:
        ax_a.text(0.5, 0.5, 'Charge history data\nnot found', ha='center', va='center')

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
                # Remove zero collision energies
                energies = energies[energies > 1e-11]
                if len(energies) > 50:
                    color = C_EARTH if 'E' in case else C_MOON
                    label = 'Earth' if 'E' in case else 'Moon'
                    
                    # Compute Scipy Kernel Density Estimation (KDE)
                    log_energies = np.log10(energies)
                    kde = gaussian_kde(log_energies)
                    
                    # Compute smooth plot line
                    x_eval = np.linspace(-10, -3, 200)
                    kde_vals = kde.evaluate(x_eval)
                    
                    # Display x-axis as true energies (10^x)
                    ax_b.plot(10**x_eval, kde_vals, color=color, lw=2.0, label=label)
                    ax_b.fill_between(10**x_eval, 0, kde_vals, color=color, alpha=0.12)
                    has_b = True
                    
    if has_b:
        finalize_plot(ax_b, '(b) Collision Energy Distribution', 'Collision Energy $\Delta E$ (J)', 'Probability Density')
        ax_b.set_xscale('log')
        ax_b.set_xlim(1e-10, 1e-4)
        ax_b.set_ylim(0, 0.8)
        ax_b.legend(frameon=True, loc='upper right')
    else:
        ax_b.text(0.5, 0.5, 'Collision energy data\nnot found', ha='center', va='center')

    # --------------------------------------------------------
    # Panel (c): Parametric Q/M ratio vs. Amplitude
    # --------------------------------------------------------
    ax_c = axes[2]
    summary_path = os.path.join(results_dir, "charging_summary.csv")
    
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        earth = df[df['gravity'] == 'Earth']
        moon = df[df['gravity'] == 'Moon']
        
        if len(earth) > 0 or len(moon) > 0:
            x = np.arange(max(len(earth), len(moon)))
            width = 0.32
            
            if len(earth) > 0:
                ax_c.bar(x[:len(earth)] - width/2, earth['qm_ratio_ucg_final'], width, 
                       yerr=earth['qm_std_ucg_final'], capsize=4, error_kw={'alpha':0.6, 'lw':0.8},
                       label='Earth (9.81 m/s$^2$)', color=C_EARTH, edgecolor='black', lw=0.5)
                       
            if len(moon) > 0:
                ax_c.bar(x[:len(moon)] + width/2, moon['qm_ratio_ucg_final'], width, 
                       yerr=moon['qm_std_ucg_final'], capsize=4, error_kw={'alpha':0.6, 'lw':0.8},
                       label='Moon (1.62 m/s$^2$)', color=C_MOON, edgecolor='black', lw=0.5)
            
            # Target range shading from Trigwell
            ax_c.axhspan(0.45, 0.55, color=C_TARGET, alpha=0.15, label='Trigwell Target Range')
            ax_c.axhline(y=0.5, color=C_TARGET, linestyle='--', lw=1.5, alpha=0.7)
            
            finalize_plot(ax_c, '(c) Final Charge-to-Mass Ratio ($Q/M$)', 'Vibration Amplitude (mm)', r'Average $Q/M$ ($\mu$C/g)')
            
            labels = earth['amplitude'].tolist() if len(earth) > 0 else moon['amplitude'].tolist()
            ax_c.set_xticks(x[:len(labels)])
            ax_c.set_xticklabels(labels)
            ax_c.set_ylim(0, 1.1)
            ax_c.legend(frameon=True, loc='upper left')
    else:
        ax_c.text(0.5, 0.5, 'charging_summary.csv\nnot found', ha='center', va='center')

    plt.tight_layout()
    
    # Save composite figure in both PNG and vector PDF format
    png_path = os.path.join(output_dir, 'fig_composite_charging.png')
    pdf_path = os.path.join(output_dir, 'fig_composite_charging.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved composite charging figure to: {png_path} and {pdf_path}")

if __name__ == "__main__":
    print("[+] Starting high-standard ASR figures rebuild...")
    generate_composite_calibration()
    generate_composite_flow_dynamics()
    generate_composite_charging()
    print("[+] All publication figures rebuilt successfully to vector PDF and high-res PNG standards.")
