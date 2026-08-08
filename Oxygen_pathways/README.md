# Oxygen_pathways — vacancy prediction & pathway enumeration

Pre-NEB stage: predicts favorable O-vacancy sites and enumerates symmetry-distinct O→vacancy hop pathways, producing the initial/final structures consumed by `relax_NEB`.

**Entry points:** `start_oxygen_workflow.py` (driver), `oxygen_workflow_improved.py` (energy-driven workflow), `oxygen_workflow_shakenbreak.py` + `example_shakenbreak_usage.py` (with ShakeNBreak distortion sampling).

**Components:** `O_vacancy_predictor.py` (vacancy formation energetics and coordination), `oxygen_hopping_vacancy_pathway_generation.py` (hop enumeration), `pathway_analyzer.py`, `atom_movement_tracker.py`, `structure_manager.py`, `vasp_manager.py`, `shakenbreak_manager.py`, `report_generator.py`, `oxygen_pathway_visualization.py`, `enhanced_oxygen_migration_analysis.py`.

Notes: if not using ShakeNBreak, disable it in the driver script; see `docs/oxygen_pathways_notes.txt`.
