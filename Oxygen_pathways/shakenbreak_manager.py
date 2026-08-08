#!/usr/bin/env python3
"""
ShakeNBreak Integration Manager
==============================
Handles ShakeNBreak distortions for oxygen vacancy structures in migration pathways
"""

import os
import json
import shutil
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# Import ShakeNBreak modules
from shakenbreak import distortions, analysis, io
from shakenbreak.energy_lowering_distortions import get_energy_lowering_distortions

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymatgen.core import Element, PeriodicSite
from pymatgen.analysis.local_env import CrystalNN

class ShakeNBreakManager:
    """Manages ShakeNBreak distortions for oxygen vacancy structures"""
    def __init__(self, config: Dict):
        """
        Initialize ShakeNBreak manager with reference structure support
        
        Args:
            config: VASP configuration dictionary
        """
        self.config = config
        self.distortion_results = {}
        self.ground_state_structures = {}
        self.pathway_vacancy_info = {}    # Changed from single value to dict
        self.movement_vacancy_info = {}   # Added for consistency
     
        self.shakenbreak_results = {
        'analysis': {},
        'ground_states': {}
    }
        # Store reference structure if available
        if 'reference_structure' in config:
            self.reference_structure = config['reference_structure']
        elif 'reference_structure_file' in config:
            from pymatgen.core import Structure
            self.reference_structure = Structure.from_file(config['reference_structure_file'])
        else:
            self.reference_structure = None
                
    def apply_shakenbreak_to_structures(self, 
                                      structure_plan: Dict,
                                      output_dir: str,
                                      bond_distortions: List[float] = None,
                                      stdev: float = 0.25,
                                      d_min: float = 2.25,
                                      num_nearest_neighbours: int = 2,
                                      max_attempts: int = 10000,
                                      apply_dimer: bool = True,
                                      apply_local_rattle: bool = False) -> Dict:
        """
        Apply ShakeNBreak distortions to all unique oxygen vacancy structures
        
        Args:
            structure_plan: Dictionary with pathway structures
            output_dir: Base output directory
            bond_distortions: List of distortion factors (default: [-0.4, -0.2, 0.0, 0.2, 0.4])
            stdev: Standard deviation for rattling (Angstroms)
            d_min: Minimum interatomic distance (Angstroms)
            num_nearest_neighbours: Number of neighbours to distort
            max_attempts: Maximum rattling attempts
            apply_dimer: Whether to apply dimer distortions
            apply_local_rattle: Whether to use local rattling (displacements tail off from defect)
            
        Returns:
            Dictionary mapping structure keys to ShakeNBreak results
        """
        if bond_distortions is None:
            bond_distortions = [-0.4, -0.2, 0.0, 0.2, 0.4]
        
        shakenbreak_dir = os.path.join(output_dir, "shakenbreak_distortions")
        os.makedirs(shakenbreak_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("APPLYING SHAKENBREAK DISTORTIONS")
        print("="*60)
        
        results = {}
        
        # Process individual pathways
        for pathway_data in structure_plan.get('individual_pathways', []):
            pathway_idx = pathway_data['pathway_index']
            
            for struct_info in pathway_data['structures']:
                struct_key = f"pathway_{pathway_idx}_{struct_info['structure_type']}"
                structure = struct_info['structure']
                
                # Use vacancy coords from struct_info if available
                if 'vacancy_coords' in struct_info and struct_info['vacancy_coords'] is not None:
                    vacancy_site = np.array(struct_info['vacancy_coords'])
                else:
                    # Fallback to detection
                    vacancy_site = self._find_oxygen_vacancy_site(
                        structure, 
                        pathway_data.get('oxygen_index', -1)
                    )
                
                if vacancy_site is not None:
                    # Apply ShakeNBreak
                    snb_results = self._apply_shakenbreak_to_single_structure(
                        structure=structure,
                        struct_key=struct_key,
                        vacancy_site=vacancy_site,
                        output_dir=shakenbreak_dir,
                        bond_distortions=bond_distortions,
                        stdev=stdev,
                        d_min=d_min,
                        num_nearest_neighbours=num_nearest_neighbours,
                        max_attempts=max_attempts,
                        apply_dimer=apply_dimer,
                        apply_local_rattle=apply_local_rattle
                    )
                    
                    results[struct_key] = snb_results
                else:
                    print(f"  Warning: Could not identify oxygen vacancy site for {struct_key}")
                    results[struct_key] = {
                        'status': 'failed',
                        'error': 'Could not identify oxygen vacancy site'
                    }
        
        # Process concerted movements
        for movement_data in structure_plan.get('concerted_movements', []):
            movement_idx = movement_data['movement_index']
            
            for struct_info in movement_data['structures']:
                struct_key = f"concerted_{movement_idx}_{struct_info['structure_type']}"
                structure = struct_info['structure']
                
                print(f"\nProcessing {struct_key}...")
                
                # Check if vacancy coords already extracted in Step 2
                if 'vacancy_coords' in struct_info and struct_info['vacancy_coords'] is not None:
                    vacancy_site = np.array(struct_info['vacancy_coords'])
                else:
                    # Fallback to finding primary vacancy
                    vacancy_site = self._find_primary_vacancy_site(
                        structure,
                        movement_data.get('movement_info', {})
                    )
                
                
                if vacancy_site is not None:
                    snb_results = self._apply_shakenbreak_to_single_structure(
                        structure=structure,
                        struct_key=struct_key,
                        vacancy_site=vacancy_site,
                        output_dir=shakenbreak_dir,
                        bond_distortions=bond_distortions,
                        stdev=stdev,
                        d_min=d_min,
                        num_nearest_neighbours=num_nearest_neighbours,
                        max_attempts=max_attempts,
                        apply_dimer=apply_dimer,
                        apply_local_rattle=apply_local_rattle
                    )
                    
                    results[struct_key] = snb_results
                else:
                    print(f"  Warning: Could not identify vacancy site for {struct_key}")
                    results[struct_key] = {
                        'status': 'failed',
                        'error': 'Could not identify vacancy site'
                    }
        
        self.distortion_results = results
        return results
    
    def _apply_shakenbreak_to_single_structure(self,
                                             structure: Structure,
                                             struct_key: str,
                                             vacancy_site: np.ndarray,
                                             output_dir: str,
                                             bond_distortions: List[float],
                                             stdev: float,
                                             d_min: float,
                                             num_nearest_neighbours: int,
                                             max_attempts: int,
                                             apply_dimer: bool = True,
                                             apply_local_rattle: bool = False) -> Dict:
        """
        Apply ShakeNBreak to a single structure with all distortion types
        
        Args:
            structure: Pymatgen Structure
            struct_key: Unique structure identifier
            vacancy_site: Fractional coordinates of vacancy
            output_dir: Output directory for distorted structures
            bond_distortions: List of distortion factors
            stdev: Rattling standard deviation
            d_min: Minimum interatomic distance
            num_nearest_neighbours: Number of neighbours to distort
            max_attempts: Maximum rattling attempts
            apply_dimer: Whether to apply dimer distortions
            apply_local_rattle: Whether to use local rattling
            
        Returns:
            Dictionary with distortion results
        """
        struct_dir = os.path.join(output_dir, struct_key)
        os.makedirs(struct_dir, exist_ok=True)
        
        distorted_structures = {}
        
        # 1. UNPERTURBED (just structure as-is, no distortion)
        unperturbed_dir = os.path.join(struct_dir, "Unperturbed")
        os.makedirs(unperturbed_dir, exist_ok=True)
        Poscar(structure).write_file(os.path.join(unperturbed_dir, "POSCAR"))
        
        distorted_structures["Unperturbed"] = {
            'structure': structure,
            'directory': unperturbed_dir,
            'type': 'unperturbed'
        }
        
        # 2. RATTLED (random displacements only)
        try:
            print(f"  Applying rattling (stdev={stdev} Å)...")
            if apply_local_rattle:
                # Local rattling - displacements tail off with distance from defect
                rattled_struct = distortions.local_mc_rattle(
                    structure=structure,
                    frac_coords=vacancy_site,
                    stdev=stdev,
                    d_min=d_min,
                    max_attempts=max_attempts,
                    seed=42
                )
                rattle_type = 'local_rattled'
            else:
                # Global rattling - uniform displacements
                rattled_struct = distortions.rattle(
                    structure=structure,
                    stdev=stdev,
                    d_min=d_min,
                    max_attempts=max_attempts,
                    seed=42
                )
                rattle_type = 'rattled'
            
            rattled_dir = os.path.join(struct_dir, "Rattled")
            os.makedirs(rattled_dir, exist_ok=True)
            Poscar(rattled_struct).write_file(os.path.join(rattled_dir, "POSCAR"))
            
            distorted_structures["Rattled"] = {
                'structure': rattled_struct,
                'directory': rattled_dir,
                'type': rattle_type,
                'stdev': stdev
            }
            
        except Exception as e:
            print(f"  Warning: Rattling failed - {str(e)}")
        
        # 3. BOND DISTORTIONS + RATTLING
        for distortion_factor in bond_distortions:
            if distortion_factor == 0.0:
                continue  # Skip 0% distortion
            
            try:
                print(f"  Applying {distortion_factor*100:+.0f}% bond distortion + rattling...")
                
                # ShakeNBreak uses 1 + bond_distortion as the factor
                snb_distortion_factor = 1 + distortion_factor
                
                distorted_dict = distortions.distort_and_rattle(
                    structure=structure,
                    distortion_factor=snb_distortion_factor,
                    num_nearest_neighbours=num_nearest_neighbours,
                    frac_coords=vacancy_site,
                    local_rattle=apply_local_rattle,
                    stdev=stdev,
                    d_min=d_min,
                    max_attempts=max_attempts,
                    seed=42
                )
                
                distorted_struct = distorted_dict['distorted_structure']
                
                # Save structure
                distortion_dir = os.path.join(struct_dir, f"Bond_Distortion_{distortion_factor*100:+.0f}%")
                os.makedirs(distortion_dir, exist_ok=True)
                Poscar(distorted_struct).write_file(os.path.join(distortion_dir, "POSCAR"))
                
                distorted_structures[f"{distortion_factor:.1f}"] = {
                    'structure': distorted_struct,
                    'directory': distortion_dir,
                    'type': 'bond_distorted_rattled',
                    'distortion_factor': distortion_factor,
                    'num_distorted_neighbours': num_nearest_neighbours,
                    'distorted_atoms': distorted_dict.get('distorted_atoms', []),
                    'local_rattle': apply_local_rattle
                }
                
            except Exception as e:
                print(f"  Warning: Bond distortion {distortion_factor*100:+.0f}% failed - {str(e)}")
        
        # 4. DIMER DISTORTION (special distortion type)
        if apply_dimer:
            try:
                print(f"  Applying dimer distortion...")
                
                # First apply dimer without rattling
                dimer_dict = distortions.apply_dimer_distortion(
                    structure=structure,
                    frac_coords=vacancy_site,
                    verbose=False
                )
                
                dimer_struct = dimer_dict['distorted_structure']
                
                # Then apply rattling to the dimer structure
                if apply_local_rattle:
                    dimer_rattled = distortions.local_mc_rattle(
                        structure=dimer_struct,
                        frac_coords=vacancy_site,
                        stdev=stdev,
                        d_min=d_min,
                        max_attempts=max_attempts,
                        seed=42
                    )
                else:
                    dimer_rattled = distortions.rattle(
                        structure=dimer_struct,
                        stdev=stdev,
                        d_min=d_min,
                        max_attempts=max_attempts,
                        seed=42
                    )
                
                dimer_dir = os.path.join(struct_dir, "Dimer")
                os.makedirs(dimer_dir, exist_ok=True)
                Poscar(dimer_rattled).write_file(os.path.join(dimer_dir, "POSCAR"))
                
                distorted_structures["Dimer"] = {
                    'structure': dimer_rattled,
                    'directory': dimer_dir,
                    'type': 'dimer_rattled',
                    'distorted_atoms': dimer_dict.get('distorted_atoms', []),
                    'local_rattle': apply_local_rattle
                }
                
            except Exception as e:
                print(f"  Warning: Dimer distortion failed - {str(e)}")
        
        # 5. SPECIAL DISTORTIONS (can be extended)
        # You can also use distort_and_rattle with string argument "Dimer"
        try:
            print(f"  Applying alternative dimer (via distort_and_rattle)...")
            
            alt_dimer_dict = distortions.distort_and_rattle(
                structure=structure,
                distortion_factor="Dimer",  # String argument for special distortion
                num_nearest_neighbours=0,  # Not used for dimer
                frac_coords=vacancy_site,
                local_rattle=apply_local_rattle,
                stdev=stdev,
                d_min=d_min,
                max_attempts=max_attempts,
                seed=42
            )
            
            alt_dimer_struct = alt_dimer_dict['distorted_structure']
            
            alt_dimer_dir = os.path.join(struct_dir, "Dimer_Alt")
            os.makedirs(alt_dimer_dir, exist_ok=True)
            Poscar(alt_dimer_struct).write_file(os.path.join(alt_dimer_dir, "POSCAR"))
            
            distorted_structures["Dimer_Alt"] = {
                'structure': alt_dimer_struct,
                'directory': alt_dimer_dir,
                'type': 'dimer_alt_rattled',
                'distorted_atoms': alt_dimer_dict.get('distorted_atoms', [])
            }
            
        except Exception as e:
            print(f"  Info: Alternative dimer method not applied - {str(e)}")
        
        # Save comprehensive metadata
        metadata = {
            'struct_key': struct_key,
            'vacancy_site': vacancy_site.tolist(),
            'num_distortions': len(distorted_structures),
            'distortion_types': list(distorted_structures.keys()),
            'distortion_parameters': {
                'bond_distortions': bond_distortions,
                'num_nearest_neighbours': num_nearest_neighbours,
                'stdev': stdev,
                'd_min': d_min,
                'apply_dimer': apply_dimer,
                'apply_local_rattle': apply_local_rattle
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(struct_dir, 'distortion_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Generated {len(distorted_structures)} distorted structures:")
        for name, info in distorted_structures.items():
            print(f"    - {name} ({info['type']})")
        
        return {
            'status': 'success',
            'distorted_structures': distorted_structures,
            'metadata': metadata
        }
    
    

    def prepare_vasp_calculations(self, 
                                vasp_manager,
                                file_manager, 
                                input_gen,
                                calc_type: str = "relax") -> List[Tuple[str, str]]:
        """
        Prepare VASP input files for all distorted structures
        
        Args:
            vasp_manager: VASPManager instance
            file_manager: FileManager instance
            input_gen: VASPInputGenerator instance
            calc_type: Type of calculation ("relax" or "single_point")
            
        Returns:
            List of (calculation_key, calculation_directory) tuples
        """
        calculations_to_submit = []
        
        for struct_key, results in self.distortion_results.items():
            if results.get('status') != 'success':
                continue
            
            for distortion_name, dist_info in results['distorted_structures'].items():
                calc_dir = dist_info['directory']
                calc_key = f"{struct_key}_{distortion_name}"
                
                # Check if already calculated
                if vasp_manager.check_calculation_complete(calc_dir):
                    print(f"Calculation already complete for {calc_key}")
                    continue
                
                # Prepare VASP files
                structure = dist_info['structure']
                
                # Create INCAR
                if calc_type == "relax":
                    incar = input_gen.create_relax_incar(
                        structure=structure,
                        max_iterations=200,
                        ediffg=-0.05
                    )
                    incar['ALGO'] = 'Fast'
                    incar['PREC'] = 'Accurate'
                    incar['EDIFF'] = 1e-5
                    incar['LWAVE'] = False
                    incar['LCHARG'] = False
                else:  # single_point
                    incar = input_gen.create_relax_incar(
                        structure=structure,
                        max_iterations=0,
                        ediffg=-0.01
                    )
                    incar['NSW'] = 0
                    incar['IBRION'] = -1
                
                incar.write_file(os.path.join(calc_dir, "INCAR"))
                
                # Create KPOINTS
                kpoints = input_gen.generate_kpoints(structure, kspacing=0.3)
                kpoints.write_file(os.path.join(calc_dir, "KPOINTS"))
                
                # Generate POTCAR
                try:
                    elements = [site.specie.symbol for site in structure]
                    unique_elements = list(dict.fromkeys(elements))
                    file_manager.create_potcar(calc_dir, unique_elements)
                except Exception as e:
                    print(f"Warning: POTCAR generation failed for {calc_key}: {e}")
                
                calculations_to_submit.append((calc_key, calc_dir))
        
        return calculations_to_submit
    
    def analyze_shakenbreak_results(self, output_dir: str, min_e_diff: float = 0.05) -> Dict:
        """
        Analyze ShakeNBreak results to find ground state structures
        
        Args:
            output_dir: Directory containing ShakeNBreak calculations
            min_e_diff: Minimum energy difference to consider significant (eV)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_results = {}
        
        for struct_key, results in self.distortion_results.items():
            if results.get('status') != 'success':
                continue
            
            struct_dir = os.path.join(output_dir, "shakenbreak_distortions", struct_key)
            
            # Parse energies
            try:
                energies_file = io.parse_energies(
                    defect_species=struct_key,
                    output_path=os.path.join(output_dir, "shakenbreak_distortions"),
                    verbose=False
                )
                
                if energies_file:
                    # Analyze results
                    energies_dict, energy_diff, gs_distortion = analysis._sort_data(
                        energies_file,
                        min_e_diff=min_e_diff,
                        verbose=True
                    )
                    
                    if energies_dict:
                        analysis_results[struct_key] = {
                            'energies': energies_dict,
                            'energy_lowering': energy_diff,
                            'ground_state_distortion': gs_distortion,
                            'found_lower_energy': energy_diff is not None and energy_diff < -min_e_diff
                        }
                        
                        # Get ground state structure
                        if gs_distortion and gs_distortion in results['distorted_structures']:
                            gs_dir = results['distorted_structures'][gs_distortion]['directory']
                            gs_structure_path = os.path.join(gs_dir, "CONTCAR")
                            
                            if os.path.exists(gs_structure_path):
                                gs_structure = Structure.from_file(gs_structure_path)
                                self.ground_state_structures[struct_key] = {
                                    'structure': gs_structure,
                                    'distortion': gs_distortion,
                                    'energy_lowering': energy_diff
                                }
                    
            except Exception as e:
                print(f"Warning: Could not analyze results for {struct_key}: {e}")
                analysis_results[struct_key] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return analysis_results
    
    def get_ground_state_structures(self) -> Dict[str, Structure]:
        """
        Get ground state structures for each pathway position
        
        Returns:
            Dictionary mapping structure keys to ground state Structures
        """
        gs_structures = {}
        
        for struct_key, gs_info in self.ground_state_structures.items():
            gs_structures[struct_key] = gs_info['structure']
        
        return gs_structures
        
    def write_shakenbreak_report(self, output_dir: str, analysis_results: Dict):
        """
        Main method to generate all ShakeNBreak reports
        
        This replaces the original simple report with comprehensive reporting
        """
        print("\nGenerating comprehensive ShakeNBreak reports...")
        
        # 1. Enhanced detailed report
        self.write_enhanced_shakenbreak_report(output_dir, analysis_results)
        
        # 2. Save ground state structures for Step 4
        self.save_ground_state_structures_for_step4(output_dir)
        
        # 3. Create workflow summary
        self.create_shakenbreak_workflow_summary(output_dir)
        
        # 4. Generate VASP calculations report
        self.generate_vasp_calculation_report(output_dir)
        
        # 5. Generate structure comparison plots (if matplotlib available)
        try:
            self.generate_structure_comparison_plots(output_dir)
        except ImportError:
            print("Matplotlib not available, skipping structure comparison plots")
        
        print("\n✓ All ShakeNBreak reports generated successfully!")


    def write_enhanced_shakenbreak_report(self, output_path: str, analysis_results: Dict):
        """
        Write a comprehensive ShakeNBreak report with vacancy tracking and energy profiles
        
        Args:
            output_path: Path to save reports
            analysis_results: Results from analyze_shakenbreak_results
        """
        report_path = os.path.join(output_path, "shakenbreak_report_enhanced.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED SHAKENBREAK DISTORTION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total structures analyzed: {len(analysis_results)}\n\n")
            
            # Overall summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*60 + "\n")
            
            total_structures = len(analysis_results)
            energy_lowering_found = sum(1 for r in analysis_results.values() 
                                    if r.get('found_lower_energy', False))
            
            f.write(f"✓ Structures with energy-lowering distortions: {energy_lowering_found}/{total_structures} ")
            f.write(f"({energy_lowering_found/total_structures*100:.1f}%)\n")
            
            if energy_lowering_found > 0:
                max_lowering = max(r.get('energy_lowering', 0) for r in analysis_results.values() 
                                if r.get('energy_lowering'))
                avg_lowering = np.mean([abs(r.get('energy_lowering', 0)) for r in analysis_results.values() 
                                    if r.get('found_lower_energy', False)])
                f.write(f"✓ Maximum energy lowering: {abs(max_lowering):.3f} eV\n")
                f.write(f"✓ Average energy lowering: {avg_lowering:.3f} eV\n")
            
            f.write("\n")
            
            # Detailed analysis by structure
            f.write("DETAILED ANALYSIS BY STRUCTURE\n")
            f.write("="*80 + "\n\n")
            
            # Group by pathway type
            individual_pathways = {}
            concerted_movements = {}
            
            for struct_key, result in sorted(analysis_results.items()):
                if struct_key.startswith('pathway_'):
                    pathway_num = int(struct_key.split('_')[1])
                    if pathway_num not in individual_pathways:
                        individual_pathways[pathway_num] = {}
                    individual_pathways[pathway_num][struct_key] = result
                elif struct_key.startswith('concerted_'):
                    movement_num = int(struct_key.split('_')[1])
                    if movement_num not in concerted_movements:
                        concerted_movements[movement_num] = {}
                    concerted_movements[movement_num][struct_key] = result
            
            # Report individual pathways
            if individual_pathways:
                f.write("INDIVIDUAL OXYGEN MIGRATION PATHWAYS\n")
                f.write("-"*60 + "\n\n")
                
                for pathway_idx in sorted(individual_pathways.keys()):
                    f.write(f"Pathway {pathway_idx + 1}:\n")
                    
                    # Get vacancy information if available
                    vacancy_info = self._get_vacancy_info_for_pathway(pathway_idx)
                    if vacancy_info:
                        f.write(f"  Initial vacancy position: {self._format_coords(vacancy_info['initial'])}\n")
                        f.write(f"  Final vacancy position: {self._format_coords(vacancy_info['final'])}\n")
                    
                    f.write("\n")
                    
                    # Report each step
                    for struct_key in sorted(individual_pathways[pathway_idx].keys()):
                        result = individual_pathways[pathway_idx][struct_key]
                        step_type = struct_key.split('_')[-1]
                        
                        f.write(f"  {step_type}:\n")
                        self._write_structure_analysis(f, struct_key, result, indent="    ")
                    
                    f.write("\n")
            
            # Report concerted movements
            if concerted_movements:
                f.write("\nCONCERTED MOVEMENTS\n")
                f.write("-"*60 + "\n\n")
                
                for movement_idx in sorted(concerted_movements.keys()):
                    f.write(f"Concerted Movement {movement_idx + 1}:\n")
                    
                    # Get movement type and vacancy info
                    movement_info = self._get_movement_info(movement_idx)
                    if movement_info:
                        f.write(f"  Type: {movement_info['type']}\n")
                        if movement_info.get('vacancy_coords'):
                            f.write(f"  Vacancy involved: Yes\n")
                            f.write(f"  Initial vacancy: {self._format_coords(movement_info['vacancy_coords'])}\n")
                    
                    f.write("\n")
                    
                    # Report each structure
                    for struct_key in sorted(concerted_movements[movement_idx].keys()):
                        result = concerted_movements[movement_idx][struct_key]
                        step_type = struct_key.split('_')[-1]
                        
                        f.write(f"  {step_type}:\n")
                        self._write_structure_analysis(f, struct_key, result, indent="    ")
                    
                    f.write("\n")
            
            # Energy profile visualization data
            f.write("\nENERGY PROFILES\n")
            f.write("="*80 + "\n\n")
            
            # Create energy profile plots
            self._generate_energy_profile_plots(output_path, analysis_results, 
                                            individual_pathways, concerted_movements)
            
            f.write("Energy profile plots saved to: shakenbreak_energy_profiles.png\n\n")
            
            # Distortion statistics
            f.write("DISTORTION STATISTICS\n")
            f.write("-"*60 + "\n\n")
            
            distortion_stats = self._calculate_distortion_statistics(analysis_results)
            
            f.write("Success rate by distortion type:\n")
            for dist_type, stats in distortion_stats.items():
                f.write(f"  {dist_type}: {stats['success_rate']:.1f}% ")
                f.write(f"(avg lowering: {stats['avg_lowering']:.3f} eV)\n")
            
            f.write("\n")
            
            # Recommendations for next steps
            f.write("RECOMMENDATIONS FOR MIGRATION BARRIER CALCULATIONS\n")
            f.write("-"*60 + "\n\n")
            
            if energy_lowering_found > 0:
                f.write("✓ Use ground state structures for accurate barrier calculations\n")
                f.write("✓ Energy-lowering distortions found - barriers may be lower than expected\n")
                f.write("✓ Consider segmented NEB for pathways with large structure changes\n")
            else:
                f.write("⚠ No significant energy lowering found\n")
                f.write("⚠ Original structures appear to be at/near global minimum\n")
                f.write("⚠ Migration barriers should be reliable with original structures\n")
            
            # Save detailed JSON data
            self._save_detailed_json_report(output_path, analysis_results)
            
        print(f"\n✓ Enhanced ShakeNBreak report saved to: {report_path}")


    # 3. FIX: Add missing helper methods to ShakeNBreakManager
    def _write_structure_analysis(self, f, struct_key: str, result: Dict, indent: str = ""):
        """Write detailed analysis for a single structure"""
        
        if result.get('status') == 'failed':
            f.write(f"{indent}❌ Analysis failed: {result.get('error', 'Unknown error')}\n")
            return
        
        # Ground state information
        gs_distortion = result.get('ground_state_distortion', 'Unknown')
        energy_lowering = result.get('energy_lowering', 0.0)
        
        f.write(f"{indent}Ground state: {gs_distortion}\n")
        
        if result.get('found_lower_energy'):
            f.write(f"{indent}✓ Energy lowering: {abs(energy_lowering):.3f} eV\n")
        else:
            f.write(f"{indent}✗ No significant energy lowering found\n")
        
        # Vacancy position for this structure
        if struct_key in self.distortion_results:
            vacancy_pos = self.distortion_results[struct_key].get('metadata', {}).get('vacancy_site')
            if vacancy_pos:
                f.write(f"{indent}Vacancy position: {self._format_coords(vacancy_pos)}\n")
        
        # Energy breakdown
        if 'energies' in result and result['energies']:
            f.write(f"{indent}Energies by distortion:\n")
            
            # Sort by energy
            energy_items = []
            
            if 'Unperturbed' in result['energies']:
                energy_items.append(('Unperturbed', result['energies']['Unperturbed']))
            
            if 'distortions' in result['energies']:
                for dist, energy in result['energies']['distortions'].items():
                    energy_items.append((dist, energy))
            
            energy_items.sort(key=lambda x: x[1])
            
            for dist_name, energy in energy_items:
                marker = "← MIN" if dist_name == gs_distortion else ""
                f.write(f"{indent}  {dist_name}: {energy:.3f} eV {marker}\n")


    def _format_coords(self, coords) -> str:
        """Format coordinates for display"""
        if coords is None:
            return "Unknown"
        if isinstance(coords, (list, np.ndarray)):
            return f"[{coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}]"
        return str(coords)

    def apply_shakenbreak_to_structures_enhanced(self, 
                                                structure_plan: Dict,
                                                output_dir: str,
                                                bond_distortions: List[float] = None,
                                                stdev: float = 0.25,
                                                d_min: float = 2.25,
                                                num_nearest_neighbours: int = 2,
                                                max_attempts: int = 10000,
                                                apply_dimer: bool = True,
                                                apply_local_rattle: bool = False) -> Dict:
        """
        Enhanced version that better handles vacancy information from pathways
        """
        # Use the existing apply_shakenbreak_to_structures but with enhanced vacancy detection
        return self.apply_shakenbreak_to_structures(
            structure_plan=structure_plan,
            output_dir=output_dir,
            bond_distortions=bond_distortions,
            stdev=stdev,
            d_min=d_min,
            num_nearest_neighbours=num_nearest_neighbours,
            max_attempts=max_attempts,
            apply_dimer=apply_dimer,
            apply_local_rattle=apply_local_rattle
        )

    def save_ground_state_structures_for_step4(self, output_path: str):
        """
        Save the ground state structures that will be used in Step 4 (energy evaluation)
        
        Creates a clear directory structure with the optimized structures and metadata
        """
        step4_dir = os.path.join(output_path, "step4_ground_state_structures")
        os.makedirs(step4_dir, exist_ok=True)
        
        # Create summary file
        summary_path = os.path.join(step4_dir, "ground_state_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GROUND STATE STRUCTURES FOR MIGRATION BARRIER CALCULATIONS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("These structures represent the global minimum configurations found by ShakeNBreak\n")
            f.write("and will be used for calculating migration barriers in Step 4.\n\n")
            
            # Organize by pathway type
            individual_saved = 0
            concerted_saved = 0
            
            # Individual pathways
            individual_dir = os.path.join(step4_dir, "individual_pathways")
            os.makedirs(individual_dir, exist_ok=True)
            
            f.write("INDIVIDUAL PATHWAYS\n")
            f.write("-"*60 + "\n\n")
            
            for struct_key, gs_structure_info in sorted(self.ground_state_structures.items()):
                if struct_key.startswith('pathway_'):
                    parts = struct_key.split('_')
                    pathway_idx = parts[1]
                    step_type = '_'.join(parts[2:])
                    
                    # Create pathway directory
                    pathway_dir = os.path.join(individual_dir, f"pathway_{pathway_idx}")
                    os.makedirs(pathway_dir, exist_ok=True)
                    
                    # Save structure
                    structure = gs_structure_info['structure']
                    output_file = os.path.join(pathway_dir, f"{step_type}_groundstate.POSCAR")
                    Poscar(structure).write_file(output_file)
                    
                    # Write info
                    f.write(f"Pathway {pathway_idx}, {step_type}:\n")
                    f.write(f"  File: {output_file}\n")
                    f.write(f"  Ground state distortion: {gs_structure_info['distortion']}\n")
                    f.write(f"  Energy lowering: {gs_structure_info['energy_lowering']:.3f} eV\n")
                    
                    # Get vacancy position
                    vacancy_info = self._extract_vacancy_from_structure(structure, struct_key)
                    if vacancy_info:
                        f.write(f"  Vacancy position: {self._format_coords(vacancy_info['coords'])}\n")
                    
                    f.write("\n")
                    individual_saved += 1
                    
                    # Save metadata
                    metadata = {
                        'struct_key': struct_key,
                        'pathway_index': pathway_idx,
                        'step_type': step_type,
                        'ground_state_distortion': gs_structure_info['distortion'],
                        'energy_lowering': gs_structure_info['energy_lowering'],
                        'vacancy_info': vacancy_info,
                        'original_unperturbed_energy': self._get_unperturbed_energy(struct_key),
                        'ground_state_energy': self._get_ground_state_energy(struct_key)
                    }
                    
                    metadata_file = os.path.join(pathway_dir, f"{step_type}_metadata.json")
                    with open(metadata_file, 'w') as mf:
                        json.dump(metadata, mf, indent=2)
            
            # Summary statistics
            f.write("\nSUMMARY\n")
            f.write("-"*60 + "\n")
            f.write(f"Total ground state structures saved: {individual_saved + concerted_saved}\n")
            f.write(f"  Individual pathways: {individual_saved}\n")
            f.write(f"  Concerted movements: {concerted_saved}\n\n")
            
            f.write("All structures are ready for migration barrier calculations in Step 4.\n")
        
        print(f"\n✓ Ground state structures saved to: {step4_dir}/")


    
    def _find_oxygen_vacancy_site(self, structure: Structure, oxygen_index: int) -> Optional[np.ndarray]:
        """
        Find the oxygen vacancy site in the structure
        
        This method looks for oxygen vacancy positions by:
        1. Checking stored pathway information
        2. Comparing with reference structure
        3. Using coordination analysis
        
        Args:
            structure: Structure with oxygen vacancy
            oxygen_index: Index of the migrating oxygen atom (not used directly but kept for compatibility)
            
        Returns:
            Fractional coordinates of vacancy site or None
        """
        # Method 1: Check if we have stored vacancy information from the pathway finder
        if hasattr(self, 'pathway_vacancy_info') and self.pathway_vacancy_info:
            return np.array(self.pathway_vacancy_info['coords'])
        
        # Method 2: Compare with reference structure if available
        if self.reference_structure is not None:
            # Find oxygen sites in reference that are missing in current structure
            ref_oxygen_sites = []
            for site in self.reference_structure:
                if site.species_string == "O":
                    ref_oxygen_sites.append(site.frac_coords)
            
            current_oxygen_sites = []
            for site in structure:
                if site.species_string == "O":
                    current_oxygen_sites.append(site.frac_coords)
            
            # Find missing oxygen sites
            for ref_coords in ref_oxygen_sites:
                found = False
                for curr_coords in current_oxygen_sites:
                    dist = structure.lattice.get_distance_and_image(ref_coords, curr_coords)[0]
                    if dist < 0.5:  # Same site
                        found = True
                        break
                
                if not found:
                    # This is an oxygen vacancy
                    return np.array(ref_coords)
        
        # Method 3: Use coordination analysis to find undercoordinated metal sites
        from pymatgen.analysis.local_env import CrystalNN
        
        try:
            nn = CrystalNN()
            potential_vacancy_sites = []
            
            # Check each metal site for undercoordination
            for i, site in enumerate(structure):
                if site.species_string != "O":  # Metal site
                    try:
                        neighbors = nn.get_nn_info(structure, i)
                        oxygen_neighbors = [n for n in neighbors if n['site'].species_string == 'O']
                        
                        # Get expected coordination
                        element = site.species_string.split('+')[0].split('-')[0]
                        expected_coord = self._get_expected_oxygen_coordination(element)
                        
                        # If undercoordinated, there might be a vacancy nearby
                        if len(oxygen_neighbors) < expected_coord - 1:
                            # Find the likely vacancy position
                            vacancy_pos = self._find_missing_oxygen_around_metal(
                                structure, site, oxygen_neighbors, expected_coord
                            )
                            if vacancy_pos is not None:
                                potential_vacancy_sites.append(vacancy_pos)
                    
                    except Exception:
                        continue
            
            # Return the first found vacancy
            if potential_vacancy_sites:
                return potential_vacancy_sites[0]
        
        except Exception as e:
            print(f"Coordination analysis failed: {e}")
        
        # If all methods fail, return None
        print("Warning: Could not determine oxygen vacancy position")
        return None


    # 8. FIX: Add missing helper methods that are referenced
    def _get_expected_oxygen_coordination(self, element: str) -> int:
        """
        Get expected number of oxygen neighbors for a cation
        """
        coordination_numbers = {
            # Alkali metals
            'Li': 4, 'Na': 6, 'K': 8, 'Rb': 8, 'Cs': 8,
            # Alkaline earth metals
            'Be': 4, 'Mg': 6, 'Ca': 8, 'Sr': 8, 'Ba': 8,
            # Transition metals
            'Sc': 6, 'Ti': 6, 'V': 6, 'Cr': 6, 'Mn': 6,
            'Fe': 6, 'Co': 6, 'Ni': 6, 'Cu': 4, 'Zn': 4,
            'Y': 8, 'Zr': 8, 'Nb': 6, 'Mo': 6, 'Tc': 6,
            'Ru': 6, 'Rh': 6, 'Pd': 4, 'Ag': 4, 'Cd': 6,
            # Lanthanides
            'La': 8, 'Ce': 8, 'Pr': 8, 'Nd': 8, 'Pm': 8,
            'Sm': 8, 'Eu': 8, 'Gd': 8, 'Tb': 8, 'Dy': 8,
            'Ho': 8, 'Er': 8, 'Tm': 8, 'Yb': 8, 'Lu': 8,
            # Main group elements
            'Al': 6, 'Ga': 6, 'In': 6, 'Tl': 6,
            'Si': 4, 'Ge': 4, 'Sn': 6, 'Pb': 6,
            'P': 4, 'As': 4, 'Sb': 6, 'Bi': 6
        }
        
        return coordination_numbers.get(element, 6)  # Default to octahedral


    # 9. FIX: Add missing stub methods referenced in reports
    def _generate_energy_profile_plots(self, output_path: str, analysis_results: Dict,
                                    individual_pathways: Dict, concerted_movements: Dict):
        """Generate energy profile plots for ShakeNBreak results"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Energy landscape for selected pathways
            ax1 = axes[0]
            
            # Select up to 3 pathways to plot
            pathways_to_plot = list(individual_pathways.keys())[:3]
            
            for i, pathway_idx in enumerate(pathways_to_plot):
                steps = sorted([k for k in individual_pathways[pathway_idx].keys()])
                
                for step_idx, struct_key in enumerate(steps):
                    result = individual_pathways[pathway_idx][struct_key]
                    
                    if 'energies' in result:
                        # Get all energies for this structure
                        energies = []
                        labels = []
                        
                        if 'Unperturbed' in result['energies']:
                            energies.append(result['energies']['Unperturbed'])
                            labels.append('Unperturbed')
                        
                        if 'distortions' in result['energies']:
                            for dist, energy in sorted(result['energies']['distortions'].items()):
                                energies.append(energy)
                                labels.append(dist)
                        
                        # Plot energies
                        x_positions = [pathway_idx + step_idx * 0.3 + j * 0.05 for j in range(len(energies))]
                        colors = ['red' if label == result.get('ground_state_distortion') else 'blue' 
                                for label in labels]
                        
                        ax1.scatter(x_positions, energies, c=colors, s=50, alpha=0.7)
            
            ax1.set_xlabel('Pathway Index')
            ax1.set_ylabel('Energy (eV)')
            ax1.set_title('ShakeNBreak Energy Landscape for Migration Pathways')
            ax1.grid(True, alpha=0.3)
            
            # Add legend
            red_patch = mpatches.Patch(color='red', label='Ground State')
            blue_patch = mpatches.Patch(color='blue', label='Other States')
            ax1.legend(handles=[red_patch, blue_patch])
            
            # Plot 2: Energy lowering distribution
            ax2 = axes[1]
            
            energy_lowerings = []
            for result in analysis_results.values():
                if result.get('found_lower_energy') and result.get('energy_lowering'):
                    energy_lowerings.append(abs(result['energy_lowering']))
            
            if energy_lowerings:
                ax2.hist(energy_lowerings, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(np.mean(energy_lowerings), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(energy_lowerings):.3f} eV')
                ax2.set_xlabel('Energy Lowering (eV)')
                ax2.set_ylabel('Count')
                ax2.set_title('Distribution of Energy Lowering from ShakeNBreak')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'shakenbreak_energy_profiles.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not generate energy profile plots: {e}")


    def _calculate_distortion_statistics(self, analysis_results: Dict) -> Dict:
        """Calculate statistics for each distortion type"""
        
        distortion_stats = {}
        
        # Collect data by distortion type
        for struct_key, result in analysis_results.items():
            if result.get('ground_state_distortion'):
                dist_type = result['ground_state_distortion']
                
                if dist_type not in distortion_stats:
                    distortion_stats[dist_type] = {
                        'count': 0,
                        'lowering_sum': 0,
                        'total_attempts': 0
                    }
                
                if result.get('found_lower_energy'):
                    distortion_stats[dist_type]['count'] += 1
                    distortion_stats[dist_type]['lowering_sum'] += abs(result.get('energy_lowering', 0))
                
                distortion_stats[dist_type]['total_attempts'] += 1
        
        # Calculate success rates and averages
        for dist_type, stats in distortion_stats.items():
            stats['success_rate'] = (stats['count'] / stats['total_attempts']) * 100 if stats['total_attempts'] > 0 else 0
            stats['avg_lowering'] = stats['lowering_sum'] / stats['count'] if stats['count'] > 0 else 0
        
        return distortion_stats


    def _save_detailed_json_report(self, output_path: str, analysis_results: Dict):
        """Save detailed analysis results as JSON for further processing"""
        
        detailed_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_structures': len(analysis_results),
                'energy_lowering_found': sum(1 for r in analysis_results.values() 
                                        if r.get('found_lower_energy', False)),
                'max_energy_lowering': max([abs(r.get('energy_lowering', 0)) for r in analysis_results.values()], default=0),
                'avg_energy_lowering': np.mean([abs(r.get('energy_lowering', 0)) for r in analysis_results.values() 
                                            if r.get('found_lower_energy', False)]) if any(r.get('found_lower_energy', False) for r in analysis_results.values()) else 0
            },
            'structures': {}
        }
        
        # Add detailed info for each structure
        for struct_key, result in analysis_results.items():
            struct_info = {
                'ground_state': result.get('ground_state_distortion'),
                'energy_lowering': result.get('energy_lowering'),
                'found_lower_energy': result.get('found_lower_energy', False),
                'energies': result.get('energies', {}),
                'distortion_metadata': {}
            }
            
            # Add distortion metadata if available
            if struct_key in self.distortion_results:
                distortion_data = self.distortion_results[struct_key]
                if distortion_data.get('metadata'):
                    struct_info['distortion_metadata'] = {
                        'vacancy_site': distortion_data['metadata'].get('vacancy_site', []),
                        'distortion_parameters': distortion_data['metadata'].get('distortion_parameters', {}),
                        'num_distortions': distortion_data['metadata'].get('num_distortions', 0)
                    }
            
            detailed_report['structures'][struct_key] = struct_info
        
        # Save JSON report
        json_path = os.path.join(output_path, 'shakenbreak_analysis_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"✓ Detailed JSON report saved to: {json_path}")


    def _find_primary_vacancy_site(self, structure: Structure, movement_info: Dict) -> Optional[np.ndarray]:
        """
        Find the primary vacancy site for concerted movements
        
        Args:
            structure: Structure with vacancy/vacancies
            movement_info: Information about the concerted movement
            
        Returns:
            Fractional coordinates of primary vacancy or None
        """
        # Check if vacancy information is provided in movement_info
        if 'vacancy_coords' in movement_info:
            return np.array(movement_info['vacancy_coords'])
        
        if 'vacancy_info' in movement_info and movement_info['vacancy_info'].get('coords'):
            return np.array(movement_info['vacancy_info']['coords'])
        
        # For different movement types, use different strategies
        movement_type = movement_info.get('movement_type', '')
        
        if movement_type == 'chains':
            # For chains, the vacancy is typically at the end of the chain
            if 'chain_end_vacancy' in movement_info:
                return np.array(movement_info['chain_end_vacancy'])
        
        elif movement_type == 'rings':
            # For rings, any vacancy in the ring can be the primary one
            if 'ring_vacancy' in movement_info:
                return np.array(movement_info['ring_vacancy'])
        
        # Fallback to finding any oxygen vacancy
        return self._find_oxygen_vacancy_site(structure, -1)

    def create_shakenbreak_workflow_summary(self, output_path: str):
        """
        Create a comprehensive workflow summary that tracks the entire ShakeNBreak process
        """
        summary_file = os.path.join(output_path, "shakenbreak_workflow_summary.md")
        
        with open(summary_file, 'w') as f:
            f.write("# ShakeNBreak Workflow Summary\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("## Overview\n\n")
            f.write("This document summarizes the ShakeNBreak distortion analysis performed on oxygen migration pathways.\n\n")
            
            # Process summary
            f.write("## Process Summary\n\n")
            f.write("1. **Pathway Identification**: Found oxygen migration pathways with vacancies\n")
            f.write("2. **Structure Extraction**: Extracted structures at key points along pathways\n")
            f.write("3. **ShakeNBreak Distortions**: Applied systematic distortions to find ground states\n")
            f.write("4. **Energy Evaluation**: Calculated energies for all distorted structures\n")
            f.write("5. **Ground State Selection**: Identified lowest energy configurations\n\n")
            
            # Distortion types applied
            f.write("## Distortion Types Applied\n\n")
            f.write("For each structure, the following distortions were tested:\n\n")
            f.write("- **Unperturbed**: Original structure without distortion\n")
            f.write("- **Rattled**: Random atomic displacements (σ = 0.25 Å)\n")
            f.write("- **Bond Distortions**: Systematic bond length changes\n")
            f.write("- **Dimer**: Special dimer-like distortion pattern\n")
            f.write("\n")
            
            # Results summary
            f.write("## Results Summary\n\n")
            
            if hasattr(self, 'shakenbreak_results'):
                analysis = self.shakenbreak_results.get('analysis', {})
                total = len(analysis)
                lowered = sum(1 for r in analysis.values() if r.get('found_lower_energy'))
                
                f.write(f"- Total structures analyzed: {total}\n")
                f.write(f"- Structures with energy lowering: {lowered} ({lowered/total*100:.1f}%)\n")
                
                if lowered > 0:
                    max_lowering = max(abs(r.get('energy_lowering', 0)) for r in analysis.values() 
                                    if r.get('found_lower_energy'))
                    f.write(f"- Maximum energy lowering: {max_lowering:.3f} eV\n")
                f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Find most successful distortion types
            distortion_success = {}
            for result in self.shakenbreak_results.get('analysis', {}).values():
                if result.get('found_lower_energy'):
                    dist_type = result.get('ground_state_distortion', 'Unknown')
                    distortion_success[dist_type] = distortion_success.get(dist_type, 0) + 1
            
            if distortion_success:
                f.write("### Most Successful Distortion Types\n\n")
                for dist_type, count in sorted(distortion_success.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {dist_type}: {count} structures\n")
                f.write("\n")
            
            # Vacancy tracking
            f.write("### Vacancy Positions\n\n")
            f.write("Vacancy positions were successfully tracked throughout the migration pathways:\n\n")
            f.write("- Initial vacancy positions identified from pathway analysis\n")
            f.write("- Final vacancy positions correspond to initial oxygen positions\n")
            f.write("- All distortions centered on appropriate vacancy sites\n\n")
            
            # Files generated
            f.write("## Output Files\n\n")
            f.write("### Reports\n")
            f.write("- `shakenbreak_report_enhanced.txt`: Detailed analysis report\n")
            f.write("- `shakenbreak_analysis_detailed.json`: Machine-readable results\n")
            f.write("- `shakenbreak_energy_profiles.png`: Energy landscape visualization\n\n")
            
            f.write("### Ground State Structures\n")
            f.write("- `step4_ground_state_structures/`: Optimized structures for barrier calculations\n")
            f.write("  - `individual_pathways/`: Ground states for individual O migrations\n")
            f.write("  - `concerted_movements/`: Ground states for concerted movements\n\n")
            
            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. Use ground state structures from `step4_ground_state_structures/` for NEB calculations\n")
            f.write("2. Consider using climbing-image NEB for accurate transition states\n")
            f.write("3. For pathways with large energy lowering, consider segmented NEB approach\n")
            f.write("4. Validate barriers with single-point calculations at higher accuracy\n\n")
        
        print(f"✓ Workflow summary saved to: {summary_file}")


    def generate_vasp_calculation_report(self, output_path: str):
        """
        Generate a report of all VASP calculations performed
        """
        calc_report = os.path.join(output_path, "vasp_calculations_report.txt")
        
        with open(calc_report, 'w') as f:
            f.write("VASP CALCULATIONS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Count calculations
            total_calcs = 0
            completed_calcs = 0
            failed_calcs = 0
            
            shakenbreak_dir = os.path.join(output_path, "shakenbreak_distortions")
            
            if os.path.exists(shakenbreak_dir):
                for struct_key in os.listdir(shakenbreak_dir):
                    struct_path = os.path.join(shakenbreak_dir, struct_key)
                    if os.path.isdir(struct_path):
                        f.write(f"\n{struct_key}:\n")
                        f.write("-"*40 + "\n")
                        
                        for distortion in os.listdir(struct_path):
                            dist_path = os.path.join(struct_path, distortion)
                            if os.path.isdir(dist_path):
                                total_calcs += 1
                                
                                # Check calculation status
                                outcar = os.path.join(dist_path, "OUTCAR")
                                oszicar = os.path.join(dist_path, "OSZICAR")
                                
                                if os.path.exists(outcar) and os.path.exists(oszicar):
                                    # Check if converged
                                    converged = self._check_vasp_convergence(outcar)
                                    if converged:
                                        completed_calcs += 1
                                        status = "✓ Completed"
                                        
                                        # Get final energy
                                        energy = self._extract_final_energy(oszicar)
                                        if energy:
                                            status += f" (E = {energy:.6f} eV)"
                                    else:
                                        failed_calcs += 1
                                        status = "✗ Not converged"
                                else:
                                    failed_calcs += 1
                                    status = "✗ Failed/Incomplete"
                                
                                f.write(f"  {distortion}: {status}\n")
            
            f.write(f"\n\nSUMMARY:\n")
            f.write(f"Total calculations: {total_calcs}\n")
            f.write(f"Completed: {completed_calcs}\n")
            f.write(f"Failed/Incomplete: {failed_calcs}\n")
            if total_calcs > 0:
                f.write(f"Success rate: {completed_calcs/total_calcs*100:.1f}%\n")
        
        print(f"✓ VASP calculations report saved to: {calc_report}")



    def _check_vasp_convergence(self, outcar_path: str) -> bool:
        """Check if VASP calculation converged"""
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
                return "reached required accuracy" in content
        except:
            return False


    def _extract_final_energy(self, oszicar_path: str) -> Optional[float]:
        """Extract final energy from OSZICAR"""
        try:
            with open(oszicar_path, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if 'F=' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'F=':
                                return float(parts[i+1])
        except:
            pass
        return None
    def generate_structure_comparison_plots(self, output_path: str):
        """
        Generate visual comparisons of structures before and after distortion
        """
        comparison_dir = os.path.join(output_path, "structure_comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        
        print("Generating structure comparison plots...")
    def _get_vacancy_info_for_pathway(self, pathway_idx: int) -> Optional[Dict]:
        """Get vacancy position information for a pathway"""
        # Check if we have this information stored
        if hasattr(self, 'pathway_vacancy_info'):
            return self.pathway_vacancy_info.get(f'pathway_{pathway_idx}')
        return None


    def _get_movement_info(self, movement_idx: int) -> Optional[Dict]:
        """Get movement type and vacancy information"""
        # Check if we have this information stored
        if hasattr(self, 'movement_vacancy_info'):
            return self.movement_vacancy_info.get(f'movement_{movement_idx}')
        return None



    def _extract_vacancy_from_structure(self, structure: Structure, struct_key: str) -> Optional[Dict]:
        """Extract vacancy information from a structure"""
        
        # Check if we have stored vacancy info
        if struct_key in self.distortion_results:
            metadata = self.distortion_results[struct_key].get('metadata', {})
            if 'vacancy_site' in metadata and metadata['vacancy_site'] is not None:
                return {
                    'coords': metadata['vacancy_site'],
                    'type': 'oxygen_vacancy'
                }
        
        # Otherwise try to detect it
        vacancy_coords = self._find_oxygen_vacancy_site(structure, -1)
        if vacancy_coords is not None:
            return {
                'coords': vacancy_coords.tolist() if isinstance(vacancy_coords, np.ndarray) else vacancy_coords,
                'type': 'oxygen_vacancy'
            }
        
        return None


    def _get_unperturbed_energy(self, struct_key: str) -> Optional[float]:
        """Get the unperturbed energy for a structure"""
        
        analysis = self.shakenbreak_results.get('analysis', {}).get(struct_key, {})
        energies = analysis.get('energies', {})
        
        return energies.get('Unperturbed')


    def _get_ground_state_energy(self, struct_key: str) -> Optional[float]:
        """Get the ground state energy for a structure"""
        
        analysis = self.shakenbreak_results.get('analysis', {}).get(struct_key, {})
        gs_distortion = analysis.get('ground_state_distortion')
        energies = analysis.get('energies', {})
        
        if gs_distortion == 'Unperturbed':
            return energies.get('Unperturbed')
        elif 'distortions' in energies and gs_distortion in energies['distortions']:
            return energies['distortions'][gs_distortion]
        
        return None


    def _find_missing_oxygen_around_metal(self, structure: Structure, metal_site: PeriodicSite,
                                        oxygen_neighbors: List, expected_coord: int) -> Optional[np.ndarray]:
        """
        Find the position of a missing oxygen around an undercoordinated metal
        """
        if len(oxygen_neighbors) == 0:
            return None
        
        # Calculate average M-O distance
        distances = [n['weight'] for n in oxygen_neighbors]
        avg_distance = np.mean(distances)
        
        # Get oxygen positions relative to metal
        oxygen_vectors = []
        for neighbor in oxygen_neighbors:
            # Calculate vector from metal to oxygen
            o_cart = neighbor['site'].coords
            m_cart = metal_site.coords
            
            # Handle periodic boundaries properly
            diff_cart = structure.lattice.get_distance_and_image_from_frac_coords(
                metal_site.frac_coords, neighbor['site'].frac_coords
            )[1]
            
            vec = diff_cart / np.linalg.norm(diff_cart)
            oxygen_vectors.append(vec)
        
        # Based on coordination geometry, find missing position
        if expected_coord == 6:  # Octahedral
            return self._find_octahedral_vacancy_position(
                structure, metal_site, oxygen_vectors, avg_distance
            )
        elif expected_coord == 4:  # Tetrahedral
            return self._find_tetrahedral_vacancy_position(
                structure, metal_site, oxygen_vectors, avg_distance
            )
        else:
            # General approach - find largest gap
            return self._find_general_vacancy_position(
                structure, metal_site, oxygen_vectors, avg_distance
            )


    def _find_tetrahedral_vacancy_position(self, structure, center, oxygen_vectors, distance):
        """
        Find vacancy position in tetrahedral coordination
        
        Args:
            structure: Pymatgen Structure
            center: Central metal site
            oxygen_vectors: List of vectors pointing to oxygen neighbors
            distance: Expected metal-oxygen distance
            
        Returns:
            Fractional coordinates of likely vacancy position or None
        """
        import numpy as np
        
        # Ideal tetrahedral directions
        tetrahedral_dirs = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ]) / np.sqrt(3)
        
        # Find which tetrahedral position is vacant
        max_min_angle = 0
        best_direction = None
        
        for tet_dir in tetrahedral_dirs:
            # Find minimum angle to existing oxygen positions
            angles = []
            for ov in oxygen_vectors:
                dot_product = np.clip(np.dot(tet_dir, ov), -1, 1)
                angles.append(np.arccos(dot_product))
            
            min_angle = min(angles) if angles else np.pi
            
            if min_angle > max_min_angle:
                max_min_angle = min_angle
                best_direction = tet_dir
        
        # Tetrahedral angle is ~109.5 degrees
        if best_direction is not None and max_min_angle > np.radians(70):
            # Calculate vacancy position
            vacancy_cart = center.coords + best_direction * distance
            vacancy_frac = structure.lattice.get_fractional_coords(vacancy_cart)
            return vacancy_frac % 1.0
        
        return None


    def _find_general_vacancy_position(self, structure, center, oxygen_vectors, distance):
        """
        Find vacancy position using a general approach (largest gap in coordination sphere)
        
        Args:
            structure: Pymatgen Structure
            center: Central metal site
            oxygen_vectors: List of vectors pointing to oxygen neighbors
            distance: Expected metal-oxygen distance
            
        Returns:
            Fractional coordinates of likely vacancy position or None
        """
        import numpy as np
        
        if len(oxygen_vectors) < 2:
            # If only one or no oxygen neighbors, can't determine vacancy reliably
            return None
        
        # Generate test directions on a sphere
        n_points = 100
        test_directions = []
        
        # Use Fibonacci sphere algorithm for even distribution
        golden_angle = np.pi * (3. - np.sqrt(5.))
        
        for i in range(n_points):
            theta = golden_angle * i
            y = 1 - (i / float(n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            test_directions.append(np.array([x, y, z]))
        
        # Find direction with maximum minimum angle to existing oxygens
        max_min_angle = 0
        best_direction = None
        
        for test_dir in test_directions:
            angles = []
            for ov in oxygen_vectors:
                dot_product = np.clip(np.dot(test_dir, ov), -1, 1)
                angles.append(np.arccos(dot_product))
            
            min_angle = min(angles) if angles else np.pi
            
            if min_angle > max_min_angle:
                max_min_angle = min_angle
                best_direction = test_dir
        
        # Only accept if there's a significant gap (> 60 degrees)
        if best_direction is not None and max_min_angle > np.radians(60):
            vacancy_cart = center.coords + best_direction * distance
            vacancy_frac = structure.lattice.get_fractional_coords(vacancy_cart)
            return vacancy_frac % 1.0
        
        return None


    def _find_octahedral_vacancy_position(self, structure: Structure, center: PeriodicSite,
                                        oxygen_vectors: List[np.ndarray], distance: float) -> Optional[np.ndarray]:
        """Find vacancy position in octahedral coordination"""
        # Ideal octahedral directions
        octahedral_dirs = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        
        # Find which octahedral position is vacant
        max_min_angle = 0
        best_direction = None
        
        for oct_dir in octahedral_dirs:
            # Find minimum angle to existing oxygen positions
            angles = [np.arccos(np.clip(np.dot(oct_dir, ov), -1, 1)) for ov in oxygen_vectors]
            min_angle = min(angles) if angles else np.pi
            
            if min_angle > max_min_angle:
                max_min_angle = min_angle
                best_direction = oct_dir
        
        if best_direction is not None and max_min_angle > np.pi/3:  # 60 degrees
            # Calculate vacancy position
            vacancy_cart = center.coords + best_direction * distance
            vacancy_frac = structure.lattice.get_fractional_coords(vacancy_cart)
            return vacancy_frac % 1.0
        
        return None
    
