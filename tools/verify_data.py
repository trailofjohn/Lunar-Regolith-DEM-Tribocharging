import os
import numpy as np
import pandas as pd

def main():
    print("="*60)
    print("VERIFYING SIMULATION DATA INTEGRITY & SCIENTIFIC CONSTANTS")
    print("="*60)
    
    data_dir = "/home/neo/liggghts/project/data"
    
    # 1. Check charging_summary.csv
    csv_path = os.path.join(data_dir, "charging_summary.csv")
    if os.path.exists(csv_path):
        print(f"\n[+] Loading {csv_path}:")
        df = pd.read_csv(csv_path)
        print(df.to_string(index=False))
    else:
        print(f"\n[-] {csv_path} not found!")

    # 2. Check transient NPZ data files
    cases = ["E_A05", "E_A10", "E_A20", "M_A05", "M_A10", "M_A20"]
    for case in cases:
        path = os.path.join(data_dir, f"{case}_data.npz")
        if os.path.exists(path):
            print(f"\n[+] Inspecting transient file {path}:")
            data = np.load(path)
            keys = sorted(list(data.keys()))
            print(f"  Keys in file: {keys}")
            
            # Print physical attributes from data
            times = data['times']
            print(f"  Simulation time range: {times[0]:.4f} s to {times[-1]:.4f} s ({len(times)} timesteps)")
            
            if 'cn_history' in data:
                cn = data['cn_history']
                print(f"  Mean Coordination Number (CN): min={np.min(cn):.3f}, max={np.max(cn):.3f}, final={cn[-1]:.3f}")
                
            if 'q_history' in data:
                q = data['q_history'] * 1e6  # to uC
                q_std = data['q_std_history'] * 1e6
                print(f"  Avg Particle Charge (q): min={np.min(q):.4f} uC, max={np.max(q):.4f} uC, final={q[-1]:.4f} ± {q_std[-1]:.4f} uC")
                
            if 'energies' in data:
                e = data['energies']
                e_filtered = e[e > 0]
                print(f"  Collision Energies: count={len(e)}, non-zero={len(e_filtered)}")
                if len(e_filtered) > 0:
                    print(f"    Min energy: {np.min(e_filtered):.3e} J")
                    print(f"    Max energy: {np.max(e_filtered):.3e} J")
                    print(f"    Mean energy: {np.mean(e_filtered):.3e} J")
                    print(f"    Median energy: {np.median(e_filtered):.3e} J")
        else:
            print(f"\n[-] {path} not found!")

if __name__ == "__main__":
    main()
