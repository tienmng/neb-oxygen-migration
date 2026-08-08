# Example data

Converged NEB results for one representative O→vacancy hop in LaNiO₃ (9 images, 7 intermediate): `LaNiO3_pathway_001_7images/`

Contents:

- `*_energy_profile.csv` — per-image energy, relative energy, max force, band gap, total magnetization vs. migration distance
- `*_moving_O_path.csv` — Cartesian trajectory of the migrating oxygen
- `*_structure_bonding.csv` — bond lengths / coordination of the migrating O at each image
- `*_path.extxyz` — full path as a multi-frame extended-XYZ file (viewable in OVITO/ASE)
- `*_summary.json` — barrier, reaction energy, and pathway metadata
- `structures/NN/POSCAR` — per-image VASP structures

The migration barrier is `max(rel_energy_eV)` in the energy profile.
