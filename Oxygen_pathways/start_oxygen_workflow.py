
import sys
# Import VASP calculation modules
sys.path.append('/path/to/Oxygen_pathways')  # UPDATE THIS PATH
#sys.path.append('/path/to/pathways')

from oxygen_workflow_improved import run_improved_workflow

# Configuration
config = {
    'potcar_path': "/path/to/potcar/potpaw_PBE_54",
    'potcar_mapping': {
        "O": "O_s", "La": "La", "Ni": "Ni_pv", "V": "V_sv",
        "Fe": "Fe_pv", "Co": "Co_pv", "Mn": "Mn_pv"
    },
    'ldau_settings': {
        'LDAU': True,
        "La": {"L": 0, "U": 0}, "Ni": {"L": 2, "U": 7}, "V": {"L": 2, "U": 3},
        "Ti": {"L": 2,"U": 14.5}, "Fe": {"L": 2, "U": 5}, "Co": {"L": 2, "U": 3.5},
        "Mn": {"L": 2, "U": 4}, "Nb": {"L": 2,"U": 5}, "O": {"L": 0, "U": 0}
    },
    'nodes': 1,
    'ntasks_per_node': 128,
    'partition': 'bigmem'
}

print("Starting IMPROVED energy-driven oxygen migration analysis...")
print("Features: Better atom tracking, robust structure matching, persistent metadata")

viable_pathways = run_improved_workflow(
    structure_file="POSCAR_defect_La_O",
    config=config,
    reference_file="POSCAR_perfect",
    calc_type="quick_relax",
    max_pathways=5,
    energy_tolerance=0.1,
    test_mode=False
)

print("\nWorkflow complete!")
if viable_pathways:
    best = viable_pathways[0]
    print(f"Best pathway: {best['type']} {best['index']}")
    print(f"  Class: {best['pathway_class']}")
    print(f"  Migration score: {best['migration_score']:.3f}")
    print(f"  Forward barrier: {best['forward_barrier']:.3f} eV")
    print(f"  Energy change: {best['delta_energy']:+.3f} eV")
