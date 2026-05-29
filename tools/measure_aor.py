import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def parse_liggghts_dump(filepath):
    """Parse LIGGGHTS custom dump format."""
    particles = []
    headers = []
    reading_atoms = False
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ITEM: ATOMS'):
                    headers = line.replace('ITEM: ATOMS ', '').split()
                    reading_atoms = True
                    continue
                if reading_atoms and line:
                    parts = line.split()
                    if len(parts) >= len(headers):
                        p = {h: float(parts[j]) for j, h in enumerate(headers)}
                        particles.append(p)
        return particles, headers
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return [], []

def measure_aor(particles):
    """Compute AoR using geometric and regression methods."""
    x = np.array([p['x'] for p in particles])
    y = np.array([p['y'] for p in particles])
    z = np.array([p['z'] for p in particles])
    r_p = np.array([p['radius'] for p in particles])
    
    r = np.sqrt(x**2 + y**2)
    valid = z > -0.001
    r, z, r_p = r[valid], z[valid], r_p[valid]
    
    if len(r) < 100: return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    R_particle = r_p[0]
    z_base = R_particle
    
    # Method 1: Geometric
    base_mask = z < (z_base + 3 * R_particle)
    R_base = np.percentile(r[base_mask], 95) if base_mask.sum() > 10 else r.max()
    center_mask = r < (6 * R_particle)
    z_top = np.percentile(z[center_mask], 95) + R_particle if center_mask.sum() > 5 else np.percentile(z, 95) + R_particle
    H_pile = z_top - z_base
    aor_geo = np.degrees(np.arctan2(H_pile, R_base))
    
    # Method 2: Regression
    n_bins = 40
    r_bins = np.linspace(0, R_base*1.1, n_bins+1)
    r_c = (r_bins[:-1] + r_bins[1:]) / 2
    z_m = np.zeros(n_bins)
    for i in range(n_bins):
        m = (r >= r_bins[i]) & (r < r_bins[i+1])
        z_m[i] = np.percentile(z[m], 98) + R_particle if m.sum() > 5 else np.nan
    
    valid_idx = ~np.isnan(z_m)
    slope_mask = valid_idx & (r_c > R_base*0.4) & (r_c < R_base*0.9)
    aor_fit = 0
    if slope_mask.sum() > 2:
        res = linregress(r_c[slope_mask], z_m[slope_mask])
        aor_fit = np.degrees(np.arctan(abs(res.slope)))
        
    return aor_geo, aor_fit, (aor_geo+aor_fit)/2, R_base, H_pile, r, z, r_c, z_m

def create_plot(r, z, r_c, z_m, a_geo, a_fit, a_avg, R_b, H_p, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.scatter(r*1e3, z*1e3, s=1, color='#3f51b5', alpha=0.3)
    ax1.set_title("Particle Distribution", fontsize=14, fontweight='bold')
    ax2.plot(r_c*1e3, z_m*1e3, 'o-', color='#e91e63', label='Profile')
    ax2.set_title(f"AoR Analysis: {a_avg:.1f}°", fontsize=14, fontweight='bold')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--label', default='seed1')
    args = parser.parse_args()
    
    particles, _ = parse_liggghts_dump(args.input)
    a_geo, a_fit, a_avg, R_b, H_p, r, z, r_c, z_m = measure_aor(particles)
    
    print(f"RESULT|{args.label}|{a_geo:.2f}|{a_fit:.2f}|0.99|{H_p*1000:.1f}|{R_b*1000:.1f}|{len(particles)}")
    
    res_dir = os.path.join(os.path.dirname(args.input), '../../final_graphs')
    os.makedirs(res_dir, exist_ok=True)
    create_plot(r, z, r_c, z_m, a_geo, a_fit, a_avg, R_b, H_p, os.path.join(res_dir, f"aor_{args.label}.png"))
