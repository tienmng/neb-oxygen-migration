# Example data

Converged NEB results for one representative O→vacancy hop in each material:

| Directory | Material | Images |
|---|---|---|
| `LaNiO3_pathway_001_7images/` | LaNiO₃ (rhombohedral) | 9 (7 intermediate) |
| `LaVO3_pathway_001/` | LaVO₃ (orthorhombic) | 7 |
| `LaNiVO3-ortho_pathway_000/` | LaNi₀.₅V₀.₅O₃ (orthorhombic) | 7 |

Each directory contains:

- `*_energy_profile.csv` — per-image energy, relative energy, max force, band gap, total magnetization vs. migration distance
- `*_moving_O_path.csv` — Cartesian trajectory of the migrating oxygen
- `*_structure_bonding.csv` — bond lengths / coordination of the migrating O at each image
- `*_path.extxyz` — full path as a multi-frame extended-XYZ file (viewable in OVITO/ASE)
- `*_summary.json` — barrier, reaction energy, and pathway metadata
- `structures/NN/POSCAR` — per-image VASP structures

The migration barrier is `max(rel_energy_eV)` in the energy profile.
