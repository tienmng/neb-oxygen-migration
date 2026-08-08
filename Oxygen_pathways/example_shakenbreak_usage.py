#!/usr/bin/env python3
"""
Example Usage of ShakeNBreak-Enhanced Oxygen Migration Workflow
==============================================================
"""

import sys
#sys.path.append('/nfs/home/6/nguyenm/pymatgen-packages')  # UPDATE THIS PATH
sys.path.append('/Users/tiennguyen/Downloads/relax_NEB_examples')

from Oxygen_pathways.oxygen_workflow_shakenbreak import run_shakenbreak_workflow

# Configuration for VASP calculations
config = {
    # POTCAR settings
    'potcar_path': "/nfs/home/6/nguyenm/sensor/POTCAR-files/potpaw_PBE_54",
    'potcar_mapping': {
        "O": "O_s", 
        "La": "La", 
        "Ni": "Ni_pv", 
        "V": "V_sv",
        "Fe": "Fe_pv", 
        "Co": "Co_pv", 
        "Mn": "Mn_pv",
        "Ti": "Ti_pv",
        "Nb": "Nb_pv"
    },
    
    # DFT+U settings
    'ldau_settings': {
        'LDAU': True,
        "La": {"L": 0, "U": 0}, 
        "Ni": {"L": 2, "U": 7}, 
        "V": {"L": 2, "U": 3},
        "Ti": {"L": 2, "U": 14.5}, 
        "Fe": {"L": 2, "U": 5}, 
        "Co": {"L": 2, "U": 3.5},
        "Mn": {"L": 2, "U": 4}, 
        "Nb": {"L": 2, "U": 5}, 
        "O": {"L": 0, "U": 0}
    },
    
    # SLURM settings
    'nodes': 1,
    'ntasks_per_node': 128,
    'partition': 'bigmem'
}

# Run workflow with different options

# Option 1: Quick test run (no calculations)
print("="*70)
print("OPTION 1: Test run (no calculations)")
print("="*70)
"""
test_pathways = run_shakenbreak_workflow(
    structure_file="POSCAR_defect_La_O",
    config=config,
    reference_file="POSCAR_perfect",
    output_dir="test_shakenbreak_workflow",
    test_mode=True,  # This skips VASP calculations
    max_pathways=2
)
"""

# Option 2: Full run with ShakeNBreak (all distortion types)
print("\n" + "="*70)
print("OPTION 2: Full run with ShakeNBreak - All Distortion Types")
print("="*70)

viable_pathways = run_shakenbreak_workflow(
    structure_file="POSCAR_defect_La_O",
    config=config,
    reference_file="POSCAR_perfect",
    output_dir=None,  # Auto-generated timestamp directory
    
    # Migration barrier calculation settings
    calc_type="quick_relax",  # or "single_point" for faster calculations
    max_pathways=5,
    energy_tolerance=0.1,  # eV
    
    # ShakeNBreak settings - ALL DISTORTION TYPES
    apply_shakenbreak=True,
    shakenbreak_calc_type="relax",  # Full relaxation for ShakeNBreak
    bond_distortions=[-0.4, -0.2, 0.2, 0.4],  # Bond distortion factors
    apply_dimer=True,  # Apply dimer distortions
    apply_local_rattle=True,  # Use global rattling (set True for local)
    stdev=0.2,  # Rattling standard deviation (Å)
    d_min=1.8,  # Minimum interatomic distance (Å)
    num_nearest_neighbours=2,  # Number of neighbours for bond distortions
    min_e_diff=0.05,  # Minimum energy difference to consider significant (eV)
    
    test_mode=False  # Run actual calculations
)
"""
# Option 2b: Full run with local rattling
print("\n" + "="*70)
print("OPTION 2b: Full run with ShakeNBreak - Local Rattling")
print("="*70)

local_rattle_pathways = run_shakenbreak_workflow(
    structure_file="POSCAR-defect",
    config=config,
    reference_file="POSCAR-perfect",
    output_dir="shakenbreak_local_rattle",
    
    # Use local rattling - displacements tail off with distance from defect
    apply_local_rattle=True,
    stdev=0.3,  # Can use larger stdev with local rattling
    
    # Other settings
    calc_type="quick_relax",
    max_pathways=3,
    apply_shakenbreak=True,
    bond_distortions=[-0.3, 0.3],  # Fewer distortions for testing
    apply_dimer=True,
    test_mode=False
)

# Option 3: Run without ShakeNBreak (original workflow)
print("\n" + "="*70)
print("OPTION 3: Run without ShakeNBreak")
print("="*70)

standard_pathways = run_shakenbreak_workflow(
    structure_file="POSCAR-defect",
    config=config,
    reference_file="POSCAR-perfect",
    output_dir="standard_oxygen_workflow",
    calc_type="single_point",
    max_pathways=3,
    apply_shakenbreak=False,  # Skip ShakeNBreak
    test_mode=False
)

# Option 4: Recovery mode (continue from existing directory)
print("\n" + "="*70)
print("OPTION 4: Recovery mode")
print("="*70)

# If a previous run was interrupted, you can continue from where it left off
recovery_pathways = run_shakenbreak_workflow(
    structure_file="POSCAR-defect",
    config=config,
    reference_file="POSCAR-perfect",
    output_dir="shakenbreak_oxygen_workflow_20240115_120000",  # Existing directory
    calc_type="quick_relax",
    max_pathways=5,
    apply_shakenbreak=True,
    test_mode=False
)

# Analyze results
if viable_pathways:
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nFound {len(viable_pathways)} viable migration pathways")
    
    # Best pathway
    if viable_pathways:
        best = viable_pathways[0]
        print(f"\nBest pathway:")
        print(f"  Type: {best['type']}")
        print(f"  Index: {best['index']}")
        print(f"  Classification: {best['pathway_class']}")
        print(f"  Migration score: {best['migration_score']:.3f}")
        print(f"  Forward barrier: {best['forward_barrier']:.3f} eV")
        print(f"  Reverse barrier: {best['reverse_barrier']:.3f} eV")
        print(f"  Overall energy change: {best['delta_energy']:+.3f} eV")
        
        # Check if intermediate minimum exists
        if best.get('has_intermediate_minimum'):
            print(f"  ⚠️  Has intermediate minimum - consider segmented NEB")
        
        # Thermodynamic favorability
        if best['is_downhill']:
            print(f"  ✓ Thermodynamically favorable (downhill)")
        elif best['is_uphill']:
            print(f"  ✗ Thermodynamically unfavorable (uphill)")
        else:
            print(f"  ≈ Thermodynamically neutral")
    
    # Summary by pathway class
    print("\nPathway distribution by class:")
    class_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for pathway in viable_pathways:
        class_counts[pathway['pathway_class']] += 1
    
    for pclass, count in class_counts.items():
        if count > 0:
            print(f"  Class {pclass}: {count} pathways")
    
    # Files generated
    print("\nOutput files generated:")
    print("  - energy_report.txt: Comprehensive analysis report")
    print("  - shakenbreak_report.txt: ShakeNBreak distortion analysis")
    print("  - relaxed_structures/: Optimized structures for NEB")
    print("  - shakenbreak_distortions/: All distorted structures tested")
    print("  - enhanced_analysis/: Pathway visualization and analysis")
"""
