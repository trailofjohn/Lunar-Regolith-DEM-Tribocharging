import numpy as np
import matplotlib.pyplot as plt
import os

def plot_side_view(dump_file, output_file):
    # Load data
    data = []
    reading = False
    with open(dump_file, 'r') as f:
        for line in f:
            if "ITEM: ATOMS" in line:
                reading = True
                continue
            if reading:
                parts = line.split()
                if len(parts) >= 5:
                    # x, y, z
                    data.append([float(parts[2]), float(parts[3]), float(parts[4])])
    
    data = np.array(data)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    
    # Calculate radial distance
    r = np.sqrt(x**2 + y**2)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(r * 1000, z * 1000, s=1, alpha=0.5, color='brown')
    plt.xlabel('Radial Distance (mm)')
    plt.ylabel('Height (mm)')
    plt.title('Particle Pile Side-View Profile')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(output_file, dpi=300)
    print(f"Profile saved to {output_file}")

if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else ""
    
    if version:
        dump_path = f"/home/neo/liggghts/project/Angle_of_repose_v2/dump/final_positions_v{version}.txt"
        output_path = f"/home/neo/liggghts/project/Angle_of_repose_v2/results/raw_pile_profile_v{version}.png"
    else:
        dump_path = "/home/neo/liggghts/project/Angle_of_repose_v2/dump/final_positions.txt"
        output_path = "/home/neo/liggghts/project/Angle_of_repose_v2/results/raw_pile_profile.png"
        
    if os.path.exists(dump_path):
        plot_side_view(dump_path, output_path)
    else:
        print(f"Dump file not found: {dump_path}")
