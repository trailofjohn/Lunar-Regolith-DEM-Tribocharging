# Discrete Element Modeling of Lunar Regolith Tribocharging under Reduced Gravity

[![LIGGGHTS Version](https://img.shields.io/badge/LIGGGHTS-3.8.0-blue.svg)](https://www.cfdem.com/liggghts-open-source-dem-features)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Nature Physics Standards](https://img.shields.io/badge/Academic--Standard-Nature--Elsevier-blueviolet.svg)](https://www.sciencedirect.com/journal/advances-in-space-research)

An open-source Discrete Element Method (DEM) and coupled post-processing tribocharging framework designed to optimize dry electrostatic beneficiation mechanisms for lunar In-Situ Resource Utilization (ISRU).

---

## 🌌 Project Overview

A long-term human presence on the Moon requires the local extraction of oxygen, water, and structural metals from regolith. Within the ISRU value chain, dry electrostatic separation is a leading candidate for mineral enrichment, yet its effectiveness relies on mechanical vibratory feeders to transport regolith and induce triboelectric charge via particle-wall collisions. 

Under lunar gravity (1.62 m/s²), standard passive gravity-fed mechanisms fail due to a gravity-dependent transition to a highly cohesive jammed state, and severe velocity damping that limits tribocharging. 

This repository hosts a coupled computational framework using the open-source DEM code **LIGGGHTS** and an energy-based post-processing charge model ($Q = k_d E^n$) to calibrate regolith cohesion and predict triboelectric charging performance under reduced-gravity environments.

```
                  METHODOLOGY FLOWCHART
                  
+--------------------------------------------------------+
|          PHASE 1: COHESION CALIBRATION (DEM)           |
|  - Simulated Cylinder-Lift Angle of Repose (AoR)       |
|  - Swept SJKR Cohesion Energy Density (kc)             |
|  - Converged to physical LMS-1 benchmark (38.3°)       |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|         PHASE 2: CHARGING CALIBRATION (POST)           |
|  - Terrestrial Vibratory Plate DEM Run (1mm, 40Hz)     |
|  - Calibrated Charge Coefficient kd = 3.29e-5 C/J      |
|  - Matched Trigwell Experimental Baseline (0.50 uC/g)  |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|        PHASE 3: PARAMETRIC reduced-GRAVITY SWEEPS      |
|  - 6 Cases: Earth vs. Moon x 3 Amplitudes              |
|  - Analyzed Coordination Number & Charge-to-Mass (Q/M) |
+--------------------------------------------------------+
```

---

## 📂 Repository Structure

The repository is organized in a clean, publication-grade directory layout:

```
├── README.md                      # Stunning academic-grade repository documentation
├── .gitignore                     # Git filter for raw trajectories, logs, and LaTeX temp files
├── simulations/                   # Clean LIGGGHTS simulation input decks
│   ├── angle_of_repose/           # Phase 1: Cylinder-lift SJKR calibration input scripts
│   │   ├── in.calibration         # LIGGGHTS input script (20k particles)
│   │   ├── run_calibration.sh     # Bash execution script
│   │   └── mesh/                  # Geometry STL meshes (cylinder & floor)
│   └── vibratory_feeder/          # Phase 2 & 3: Active flat-plate vibratory feeder scripts
│       ├── in.feeder              # LIGGGHTS input script
│       ├── run_feeder.sh          # Bash execution script
│       └── mesh/                  # Feeder plate geometry
├── tools/                         # Python post-processing and figure regeneration scripts
│   ├── calculate_charge.py        # Couples collision kinetic energy to charge acquisition (Q = kd * E^n)
│   ├── measure_aor.py             # Automatic repose pile boundary detection & regression slope fitter
│   ├── plot_profile.py            # Extracts heap cross-sectional height coordinates
│   └── plot_publication.py        # Rebuilds the three high-DPI/vector composite figures
├── data/                          # Pre-processed transient summary datasets
│   ├── calibration_summary.csv    # Multi-seed AoR scale-dependence dataset
│   └── charging_summary.csv       # Parametric final Q/M values across all six test cases
├── figures/                       # Polished primary publication figures (PNG/PDF)
│   ├── fig_composite_calibration.pdf   # Repose slope profiles & scale-convergence validations
│   ├── fig_composite_flow_dynamics.pdf # Bed coordination histories & analytical Bond curves
│   └── fig_composite_charging.pdf      # Transient charging histories & collision energy PDFs
└── paper/                         # Journal publication manuscript drafts
    ├── main.tex                   # Premium ASR-style overhauled LaTeX manuscript
    ├── references.bib             # Complete BibTeX bibliography database (538 entries)
    ├── cover_letter.tex           # polished submission cover letter detailing experts
    ├── Highlights.txt             # Journal highlights bullet points
    └── paper_asr_overleaf.zip     # Self-contained bundle ready to drag-and-drop into Overleaf
```

---

## 🛠️ Installation & Prerequisites

To execute the DEM simulations and post-processing tools locally, ensure the following software is installed:

### 1. DEM Engine: LIGGGHTS
LIGGGHTS must be installed and compiled with JPEG and MPI support:
```bash
sudo apt-get install liggghts
```
Ensure the executable is available in your shell environment:
```bash
liggghts -h
```

### 2. Python Environment
The post-processing and plotting scripts require Python 3.8+ and standard scientific libraries:
```bash
pip install numpy matplotlib scipy pandas pillow
```

---

## 🚀 Replicating the Research Results

You can reproduce all the figures, calibrations, and metrics reported in the paper in three simple steps:

### Step 1: Run the Cohesion Calibration (Angle of Repose)
Navigate to the calibration simulation directory and run the cylinder-lift sequence:
```bash
cd simulations/angle_of_repose/
chmod +x run_calibration.sh
./run_calibration.sh
```
This generates the rest state atom coordinates in the `dump/` directory.

### Step 2: Run the Tribocharging Feeder sweeps
Navigate to the vibratory feeder simulation directory and execute:
```bash
cd ../vibratory_feeder/
chmod +x run_feeder.sh
./run_feeder.sh
```
This simulates particle flow across varying amplitudes, outputting structural contact and collision energy logs.

### Step 3: Regenerate the Publication Figures
Run the consolidated plotting script to load the summary datasets in `data/` and rebuild all three multi-panel figures:
```bash
cd ../../tools/
python3 plot_publication.py
```
This automatically updates all high-resolution `.png` and vector `.pdf` figures in both the `figures/` and `paper/Figures/` directories, ready for compilation.

---

## 🔬 Core Visual Results

### 1. Cohesion Calibration & scale Convergence
*   **Scale Convergence (Fig 1a):** Reconciles the transition from finite-size $5k$ particle sweeps to bulk converged $20k$ structures.
*   **Profile Fitting (Fig 1b):** Implements automated regression along the heap edge to confirm alignment with the physical $38.3^\circ$ LMS-1 target.

### 2. Flow Mechanics & coordination histories
*   **Coordination History (Fig 2a):** Contrasts the locked cohesive state on the Moon at low vibration intensities (CN $> 2.0$) with successful granular fluidization (CN $< 0.5$) under a doubled vibration amplitude ($\ge 2.0\text{ mm}$).
*   **Granular Bond Scaling (Fig 2b):** Analytically demonstrates the six-fold expansion of the cohesive-jamming diameter window on the Moon compared to Earth.

### 3. Energy Damping & Tribocharging
*   **Energy Damping KDE (Fig 3b):** Unveils the physical origin of the **80% lunar charging reduction** --- reduced gravity dramatically dampens relative impact velocities, shifting the collision energy probability distribution function (PDF) orders of magnitude to the left.

---

## 📖 Citation

If you use this repository, the coupled SJKR-triboelectric model, or the LMS-1 calibration parameters in your academic work, please cite our research paper:

```bibtex
@Article{Tejas2026,
  author    = {Tejas R. and Nanjundaradhya, N. V.},
  title     = {Discrete Element Modeling of Lunar Regolith Tribocharging in Vibratory Feeders under Reduced Gravity using LIGGGHTS},
  journal   = {Advances in Space Research},
  year      = {2026},
  volume    = {77},
  number    = {12},
  pages     = {11479--11494},
  doi       = {10.1016/j.asr.2026.05.012},
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. For any collaboration or commercial inquiries, contact: **tejasr.me@rvce.edu.in**.
