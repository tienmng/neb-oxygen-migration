# NEB Oxygen Migration Toolkit

Automated DFT workflow for computing **oxygen vacancy migration barriers in perovskite oxides** (LaNiO₃, LaVO₃, LaNi₀.₅V₀.₅O₃) using **VASP**, **pymatgen**, **ASE**, and the **climbing-image Nudged Elastic Band (CI-NEB)** method on SLURM-based HPC clusters.

The toolkit takes a relaxed bulk structure and, end to end, predicts favorable vacancy sites, enumerates symmetry-distinct O→vacancy hop pathways, generates and interpolates NEB images, submits and monitors multi-stage VASP calculations with automatic restarts, and post-processes the results into energy barriers, transition-state analyses, and publication-quality figures.

## Workflow overview

```
 Bulk structure (POSCAR)
        │
        ▼
 1. Vacancy prediction ──────────  Oxygen_pathways/O_vacancy_predictor.py
    (optional ShakeNBreak            oxygen_workflow_improved.py
     distortion sampling)            oxygen_workflow_shakenbreak.py
        │
        ▼
 2. Pathway enumeration ─────────  oxygen_hopping_vacancy_pathway_generation.py
    initial/final structures         pathway_analyzer.py, report_generator.py
     for each distinct hop
        │
        ▼
 3. Endpoint relaxation ─────────  relax_NEB/multistage_relaxation.py
    multi-stage (coarse→fine),       parallel initial+final relaxations
        │
        ▼
 4. NEB setup + run ─────────────  relax_NEB/neb_workflow_from_migration_analysis.py
    ASE interpolation with           neb_path_generation_with_constraint_control.py
     constraint control,             neb_multistage.py (2/3/5-stage CI-NEB)
     SLURM auto-restart              slurm_manager.py, vasp_inputs.py
        │
        ▼
 5. Analysis + visualization ────  energy_analyzer.py, neb_path_visualizer.py
    barriers, TS detection,          improved_transition_state_detection.py
     TS frequencies,                 ts_frequency_calc.py
     local-environment analysis      oxygen_migration_analyzer_complete.py
```

## Key features

- **Multi-stage NEB** (2, 3, or 5 stages): rough optimization → climbing image → tight convergence, with automatic CONTCAR→POSCAR handoff between stages and per-stage SLURM resource sizing.
- **Multi-stage endpoint relaxation** with progressive convergence criteria (EDIFFG −0.1 → −0.01), run in parallel for initial and final structures.
- **Constraint control** during ASE path interpolation — selective freezing/unfreezing of the host lattice around the migrating oxygen.
- **HPC automation**: SLURM script generation, job submission, live monitoring, failure detection, and automatic restart from the last checkpoint.
- **Vacancy-site prediction** with optional [ShakeNBreak](https://shakenbreak.readthedocs.io/) symmetry-breaking distortions to escape metastable defect configurations.
- **Transition-state characterization**: TS detection from the converged path, vibrational frequency calculations, and imaginary-mode verification.
- **Analysis pipeline**: energy profiles from OUTCAR/NEBEF.dat, migration-path geometry tracking, oxygen coordination/local-environment evolution (CrystalNN), and automated plots and CSV/JSON reports.

## Repository layout

| Path | Contents |
|---|---|
| `relax_NEB/` | Core NEB workflow: relaxation, path generation, multi-stage NEB, SLURM + VASP I/O, analysis, visualization |
| `Oxygen_pathways/` | Vacancy prediction and O-hop pathway enumeration (pre-NEB stage) |
| `examples/data/` | Sample outputs: converged pathways for LaNiO₃, LaVO₃, LaNi₀.₅V₀.₅O₃ (energy profiles, migrating-O trajectories, bond analysis, per-image POSCARs) |
| `docs/` | Component reference and workflow notes |

## Installation

```bash
git clone https://github.com/<your-username>/neb-oxygen-migration.git
cd neb-oxygen-migration
pip install -r requirements.txt
```

Runtime requirements on the cluster side: VASP (with [VTST tools](https://theory.cm.utexas.edu/vtsttools/) for CI-NEB) and a SLURM scheduler. Set your `POTCAR` library path and SLURM account/partition in the workflow configuration block of the driver scripts.

## Quick start

Generate migration pathways from a relaxed bulk structure:

```python
from Oxygen_pathways.oxygen_workflow_improved import ImprovedEnergyDrivenOxygenWorkflow
# see Oxygen_pathways/start_oxygen_workflow.py for a complete driver script
```

Run a full NEB calculation from a generated pathway (initial/final structures + moving O index):

```python
# relax_NEB/neb_workflow_from_migration_analysis.py
# - multi-stage relaxation of both endpoints (parallel)
# - ASE image interpolation with constraint control
# - multi-stage CI-NEB with SLURM auto-restart
# - energy profile + barrier extraction
```

Visualize a finished calculation:

```bash
python relax_NEB/neb_visualizing.py   # energy profile, path geometry, TS analysis
```

Example output (LaNiO₃, O→V<sub>O</sub> hop, 7 images): see
[`examples/data/LaNiO3_pathway_001_7images/`](examples/data/LaNiO3_pathway_001_7images) — per-image energies, forces, band gap, and magnetization in `*_energy_profile.csv`, the migrating-oxygen trajectory in `*_moving_O_path.csv`, and the full image structures in `structures/`.

## Physics background

Oxygen-ion transport governs the performance of solid-oxide fuel cell cathodes, oxygen-permeation membranes, and resistive-switching memristors. The rate-limiting step is the hop of a lattice O²⁻ into a neighboring vacancy; its activation energy is the NEB barrier computed here. Comparing barriers across LaNiO₃, LaVO₃, and the mixed B-site LaNi₀.₅V₀.₅O₃ quantifies how B-site chemistry and octahedral distortion tune ionic mobility.

## Requirements

Python ≥ 3.9 with pymatgen, ASE, NumPy, SciPy, pandas, matplotlib, seaborn, networkx (see `requirements.txt`). ShakeNBreak is optional (vacancy distortion sampling).

## License

MIT — see [LICENSE](LICENSE).
