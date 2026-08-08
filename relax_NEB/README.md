# relax_NEB — core NEB workflow

Modular components for running multi-stage relaxation + CI-NEB on SLURM/VASP.

**Core modules:** `file_manager.py` (POTCAR/CONTCAR handling), `structure_analyzer.py` (vacancy ID, PBC-aware analysis), `vasp_inputs.py` (INCAR/KPOINTS with LDAU + magmoms), `slurm_manager.py` (job scripts, submission, monitoring, auto-restart).

**Workflow drivers:** `neb_workflow_from_migration_analysis.py` (full pipeline from pathway endpoints), `multistage_relaxation.py` (2–3 stage endpoint relaxation), `neb_multistage.py` (2/3/5-stage CI-NEB), `neb_path_generation_with_constraint_control.py` (ASE interpolation).

**Analysis:** `energy_analyzer.py`, `improved_transition_state_detection.py`, `ts_frequency_calc.py` (+ `ts_freq_example.py`), `oxygen_migration_analyzer_complete.py` (local-environment evolution), `neb_path_visualizer.py`, `neb_visualizing.py`.

See the repository root README for the end-to-end workflow.
