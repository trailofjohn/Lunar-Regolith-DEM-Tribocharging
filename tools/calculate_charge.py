#!/usr/bin/env python3
# =============================================================================
# calculate_charge.py — Energy-based Tribocharging Model (Q = kd * E^n)
# Post-processes LIGGGHTS velocity data to predict charge acquisition.
# =============================================================================

import numpy as np
import os
import glob
import pandas as pd

def write_vtk(filename, ids, positions, radii, charges, cns, vels):
    """Writes particle data including charge to an ASCII VTK file for ParaView."""
    n = len(positions)
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Particle Data with Tribocharge\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n} float\n")
        for p in positions:
            f.write(f"{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}\n")
            
        f.write(f"CELLS {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
            
        f.write(f"CELL_TYPES {n}\n")
        for i in range(n):
            f.write("1\n") # 1 = VTK_VERTEX
            
        f.write(f"POINT_DATA {n}\n")
        
        f.write("SCALARS id float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in ids: f.write(f"{i}\n")
        
        f.write("SCALARS charge float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for c in charges: f.write(f"{c:.6g}\n")

        f.write("SCALARS radius float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for r in radii: f.write(f"{r:.6g}\n")
        
        f.write("SCALARS cn float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for c in cns: f.write(f"{c:.6g}\n")
        
        f.write("VECTORS velocity float\n")
        for v in vels: f.write(f"{v[0]:.6g} {v[1]:.6g} {v[2]:.6g}\n")


# Constants
PARTICLE_DENSITY = 2920 # kg/m3
PARTICLE_RADIUS = 0.5e-3 # m
PARTICLE_MASS = PARTICLE_DENSITY * (4/3) * np.pi * (PARTICLE_RADIUS**3)
TRIGWELL_TARGET_QM = 0.5e-3 # 0.5 uC/g = 0.5e-3 C/kg
DUMP_INTERVAL = 0.0005 # 0.5 ms (1000 steps * 5e-7)

def analyze_case(case_dir, kd=1.0, n=1.0):
    """Computes charge accumulation and coordination number for a single case."""
    files = sorted(glob.glob(os.path.join(case_dir, "charge_*.txt")))
    if not files:
        return None

    prev_velocities = {}
    charges = {}
    cn_history = []
    energy_history = []
    total_q_history = []
    
    # Track time
    times = []
    q_std_history = []
    
    for i, filename in enumerate(files):
        time = i * DUMP_INTERVAL
        times.append(time)
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            header_idx = 0
            for j, line in enumerate(lines):
                if "ITEM: ATOMS" in line:
                    header_idx = j
                    break
            
            step_charges = []
            step_cns = []
            
            # Data for VTK export
            step_ids = []
            step_pos = []
            step_radii = []
            step_vels = []
            
            for line in lines[header_idx+1:]:
                parts = line.split()
                if len(parts) < 9: continue
                
                p_id = parts[0]
                vx, vy, vz = float(parts[4]), float(parts[5]), float(parts[6])
                cn = float(parts[8])
                v_curr = np.array([vx, vy, vz])
                
                step_cns.append(cn)
                
                # Store for VTK
                step_ids.append(float(p_id))
                step_pos.append((float(parts[1]), float(parts[2]), float(parts[3])))
                step_radii.append(float(parts[7]))
                step_vels.append((vx, vy, vz))

                
                if p_id not in charges: charges[p_id] = 0.0
                
                if p_id in prev_velocities:
                    v_prev = prev_velocities[p_id]
                    dv = np.linalg.norm(v_curr - v_prev)
                    # Threshold to detect collision (0.001 m/s)
                    if dv > 0.001:
                        de = 0.5 * PARTICLE_MASS * (dv**2)
                        energy_history.append(de)
                        # Apply Q = kd * E^n
                        dq = kd * (de**n)
                        charges[p_id] += dq
                
                prev_velocities[p_id] = v_curr
                step_charges.append(charges[p_id])
            
            # Generate VTK for Paraview with full resolution (0.5ms)
            vtk_filename = filename.replace('charge_', 'viz_charge_').replace('.txt', '.vtk')
            write_vtk(vtk_filename, step_ids, step_pos, step_radii, step_charges, step_cns, step_vels)
            
            cn_history.append(np.mean(step_cns) if step_cns else 0)
            total_q_history.append(np.mean(step_charges) if step_charges else 0)
            q_std_history.append(np.std(step_charges) if step_charges else 0)

    # Calculate final Q/M and statistics
    final_charges = np.array(list(charges.values()))
    qm_ratios = final_charges / PARTICLE_MASS
    
    return {
        'times': times,
        'q_history': total_q_history,
        'q_std_history': q_std_history,
        'cn_history': cn_history,
        'energies': energy_history,
        'qm_ratio': np.mean(qm_ratios) if len(qm_ratios) > 0 else 0,
        'qm_std': np.std(qm_ratios) if len(qm_ratios) > 0 else 0,
        'particles': len(charges)
    }

def main():
    cases = ["E_A05", "E_A10", "E_A20", "M_A05", "M_A10", "M_A20"]
    base_dir = "dump"
    
    # 1. CALIBRATE KD on Case E_A10 (Earth, 1.0mm)
    print(">>> CALIBRATING KD ON BASELINE CASE (E_A10)...")
    calibration_data = analyze_case(os.path.join(base_dir, "E_A10"), kd=1.0, n=1.0)
    
    if calibration_data is None:
        print("ERROR: Could not find data for E_A10. Run the simulation first!")
        return

    # Q/M_measured = 1.0 * (Sum(E^n) / TotalMass)
    # kd = TargetQM / (Sum(E^n) / TotalMass)
    kd_calibrated = TRIGWELL_TARGET_QM / calibration_data['qm_ratio']
    print(f">>> CALIBRATED KD = {kd_calibrated:.4e} C/J")

    # 2. ANALYZE ALL CASES with calibrated KD
    results = []
    print(">>> ANALYZING ALL CASES...")
    for case in cases:
        print(f"Processing {case}...")
        data = analyze_case(os.path.join(base_dir, case), kd=kd_calibrated, n=1.0)
        if data:
            results.append({
                'case': case,
                'gravity': 'Earth' if case.startswith('E') else 'Moon',
                'amplitude': float(case.split('A')[1])/10.0,
                'qm_ratio_ucg_final': (data['qm_ratio'] * 1e6) / 1000.0,
                'qm_std_ucg_final': (data['qm_std'] * 1e6) / 1000.0,
                'data': data
            })
            # Save raw data for plotting
            np.savez(f"results/{case}_data.npz", 
                     times=data['times'], 
                     q_history=data['q_history'], 
                     q_std_history=data['q_std_history'],
                     cn_history=data['cn_history'],
                     energies=data['energies'])

    # Save summary CSV
    df = pd.DataFrame([{k:v for k,v in r.items() if k != 'data'} for r in results])
    df.to_csv("results/charging_summary.csv", index=False)
    print(">>> ANALYSIS COMPLETE. Results saved to results/charging_summary.csv")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
