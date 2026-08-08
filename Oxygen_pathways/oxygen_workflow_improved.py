#!/usr/bin/env python3
"""
Improved Energy-Driven Oxygen Migration Workflow
===============================================
Main workflow using modular components for better organization and robustness
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# Import VASP calculation modules
sys.path.append('/path/to/packages')  # UPDATE THIS PATH
#sys.path.append('/path/to/packages')

# Import the enhanced analysis module
from Oxygen_pathways.enhanced_oxygen_migration_analysis import analyze_oxygen_migration, find_vacancy_involved_pathways


from relax_NEB.file_manager import FileManager
from relax_NEB.vasp_inputs import VASPInputGenerator
from relax_NEB.slurm_manager import SLURMManager

# Import our new modular components
from pathway_analyzer import PathwayAnalyzer
from structure_manager import StructureManager
from atom_movement_tracker import AtomMovementTracker
from vasp_manager import VASPManager


class ImprovedEnergyDrivenOxygenWorkflow:
    """
    Improved workflow with modular components and better robustness
    """
    
    def __init__(self, config: Dict):
        """Initialize with VASP calculation configuration"""
        self.config = config
        self.file_manager = FileManager(config['potcar_path'], config['potcar_mapping'])
        self.input_gen = VASPInputGenerator()
        self.slurm_manager = SLURMManager()
        
        # Initialize modular components
        self.structure_manager = StructureManager()
        self.pathway_analyzer = PathwayAnalyzer()
        self.movement_tracker = AtomMovementTracker()
        self.vasp_manager = VASPManager()
        
        # Results storage
        self.pathway_structures = {}
        self.energy_results = {}
        self.viable_pathways = []
        self.finder = None
    def run_energy_driven_analysis(self, 
                             structure_file: str,
                             reference_file: str = None,
                             output_dir: str = None,
                             calc_type: str = "single_point",
                             max_pathways: int = 10,
                             energy_tolerance: float = 0.1,
                             test_mode: bool = False):
    
        # Check if output_dir exists (recovery mode)
        if output_dir and os.path.exists(output_dir):
            print(f"Detected existing workflow directory: {output_dir}")
            print("Running in recovery/continuation mode...")
            self.output_dir = output_dir
            # Continue with normal flow - it will skip completed calculations
        elif output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"energy_oxygen_workflow_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.test_mode = test_mode
        
        print("="*70)
        print("IMPROVED ENERGY-DRIVEN OXYGEN MIGRATION WORKFLOW")
        print("="*70)
        print(f"Output directory: {output_dir}")
        if test_mode:
            print("⚠️  RUNNING IN TEST MODE - No VASP calculations will be submitted")
        
        # Save finder for potential recovery
        self._save_workflow_state()
        
        # STEP 1: Generate pathways
        print("\n" + "="*60)
        print("STEP 1: Enhanced Pathway Generation")
        print("="*60)
        
        finder, viz = self.step1_generate_enhanced_pathways(structure_file, reference_file)
        
        # STEP 2: Extract all structures
        print("\n" + "="*60)
        print("STEP 2: Extract All Structures")
        print("="*60)
        
        structure_plan = self.step2_extract_all_structures(finder, max_pathways)
        
        if not test_mode:
            # STEP 3: Energy evaluation
            print("\n" + "="*60)
            print("STEP 3: Combined Energy Evaluation")
            print("="*60)
            
            energy_results = self.step3_evaluate_all_energies(structure_plan, calc_type)
            
            # STEP 4: Analyze energy profiles
            print("\n" + "="*60)
            print("STEP 4: Energy Profile Analysis")
            print("="*60)
            
            viable_pathways = self.step4_analyze_complete_energy_profiles(energy_results, energy_tolerance)
            
            # STEP 5: Save results
            print("\n" + "="*60)
            print("STEP 5: Saving Results and Structures")
            print("="*60)
            
            self._save_relaxed_structures_with_pathway_info()
            self.generate_enhanced_report()
            
        else:
            print("\n⚠️  Skipping energy calculations in test mode")
            viable_pathways = []
        
        return viable_pathways
    
    def step1_generate_enhanced_pathways(self, structure_file: str, reference_file: str = None):
        """Use enhanced analysis to generate pathways"""
        
        print("Running enhanced oxygen migration analysis...")
        finder, viz = analyze_oxygen_migration(
            structure_file=structure_file,
            reference_file=reference_file,
            output_dir=os.path.join(self.output_dir, "enhanced_analysis"),
            max_hop_distance=4.0,
            max_pathway_length=6,
            create_animations=False,
            verbose=False
        )
        
        self.finder = finder
        self.viz = viz
        
        # Save finder for recovery
        self._save_finder()
        
        print(f"✓ Generated {len(finder.migration_pathways)} individual pathways")
        if hasattr(finder, 'concerted_pathways'):
            print(f"✓ Generated {len(finder.concerted_pathways.get('all', []))} concerted movements")
        
        return finder, viz
    
    def step2_extract_all_structures(self, finder, max_pathways: int) -> Dict:
        """Extract all structures with improved metadata tracking"""
        
        all_structures = {
            'individual_pathways': [],
            'concerted_movements': []
        }
        
        # Process individual pathways
        print(f"Extracting structures from top {max_pathways} individual pathways...")
        
        for i in range(min(max_pathways, len(finder.migration_pathways))):
            pathway = finder.migration_pathways[i]
            
            # Get oxygen index from pathway
            oxygen_idx = self._extract_oxygen_index(pathway, i)
            
            # Generate pathway structures
            pathway_dir = os.path.join(self.output_dir, "temp_pathways", f"pathway_{i}")
            os.makedirs(pathway_dir, exist_ok=True)
            
            structures = finder.execute_migration_pathway(
                pathway_index=i,
                output_prefix=os.path.join(pathway_dir, "step")
            )
            
            if structures and len(structures) >= 2:
                structure_files = []
                
                for j, structure in enumerate(structures):
                    if j == 0:
                        structure_type = 'start'
                    elif j == len(structures) - 1:
                        structure_type = 'end'
                    else:
                        structure_type = f'hop_{j}'
                    
                    file_path = os.path.join(self.output_dir, "all_structures", 
                                           f"pathway_{i}_{structure_type}.POSCAR")
                    
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    Poscar(structure).write_file(file_path)
                    
                    structure_files.append({
                        'structure': structure,
                        'file_path': file_path,
                        'structure_type': structure_type,
                        'hop_index': j
                    })
                
                all_structures['individual_pathways'].append({
                    'pathway_index': i,
                    'oxygen_index': oxygen_idx,
                    'pathway_info': pathway,
                    'structures': structure_files,
                    'num_hops': len(structures) - 1
                })
                
                print(f"  Pathway {i}: Extracted {len(structures)} structures")
        
        # Process concerted movements with better atom tracking
        if hasattr(finder, 'concerted_pathways') and finder.concerted_pathways.get('all'):
            print(f"Extracting structures from top {max_pathways} concerted movements...")
            
            for i in range(min(max_pathways, len(finder.concerted_pathways['all']))):
                movement = finder.concerted_pathways['all'][i]
                
                # Extract movement details
                movement_info = self._extract_movement_info(movement, i)
                
                movement_dir = os.path.join(self.output_dir, "temp_concerted", f"movement_{i}")
                os.makedirs(movement_dir, exist_ok=True)
                
                structures = finder.execute_concerted_movement(
                    movement_index=i,
                    output_prefix=os.path.join(movement_dir, "step")
                )
                
                if structures and len(structures) >= 2:
                    structure_files = []
                    
                    for j, structure in enumerate(structures):
                        if j == 0:
                            structure_type = 'start'
                        elif j == len(structures) - 1:
                            structure_type = 'end'
                        else:
                            structure_type = f'step_{j}'
                        
                        file_path = os.path.join(self.output_dir, "all_structures", 
                                               f"concerted_{i}_{structure_type}.POSCAR")
                        
                        Poscar(structure).write_file(file_path)
                        
                        structure_files.append({
                            'structure': structure,
                            'file_path': file_path,
                            'structure_type': structure_type,
                            'step_index': j
                        })
                    
                    all_structures['concerted_movements'].append({
                        'movement_index': i,
                        'movement_info': movement_info,
                        'structures': structure_files,
                        'num_steps': len(structures) - 1
                    })
                    
                    print(f"  Movement {i}: Extracted {len(structures)} structures")
        
        # Clean up temporary directories
        import shutil
        for temp_dir in [os.path.join(self.output_dir, "temp_pathways"), 
                        os.path.join(self.output_dir, "temp_concerted")]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        # Count total structures
        total_structures = sum(len(p['structures']) for p in all_structures['individual_pathways'])
        total_structures += sum(len(m['structures']) for m in all_structures['concerted_movements'])
        
        print(f"\n✓ Extracted {total_structures} total structures")
        
        self.pathway_structures = all_structures
        return all_structures
    
    def step3_evaluate_all_energies(self, structure_plan: Dict, calc_type: str) -> Dict:
        """Evaluate energies with improved structure matching"""
        
        calc_dir = os.path.join(self.output_dir, "energy_calculations")
        os.makedirs(calc_dir, exist_ok=True)
        
        # Use structure manager to find unique structures
        unique_structures, structure_to_calc = self.structure_manager.find_unique_structures(structure_plan)
        
        # Prepare and submit calculations
        energies = {}
        calculations_to_submit = []
        
        print(f"\nPreparing {len(unique_structures)} calculations...")
        
        for i, (unique_key, structure) in enumerate(unique_structures.items()):
            struct_calc_dir = os.path.join(calc_dir, unique_key)
            
            # Check if already calculated
            if self.vasp_manager.check_calculation_complete(struct_calc_dir):
                energy = self.vasp_manager.read_energy(struct_calc_dir)
                if energy is not None:
                    energies[unique_key] = energy
                    print(f"  [{i+1:3d}/{len(unique_structures)}] {unique_key}: Using existing energy {energy:.3f} eV")
                    continue
            
            # Prepare calculation
            os.makedirs(struct_calc_dir, exist_ok=True)
            self._prepare_vasp_calculation(structure, struct_calc_dir, calc_type)
            calculations_to_submit.append((unique_key, struct_calc_dir))
            print(f"  [{i+1:3d}/{len(unique_structures)}] {unique_key}: Prepared for calculation")
        
        # Submit and monitor jobs
        if calculations_to_submit:
            job_ids = self._submit_all_jobs(calculations_to_submit)
            
            if job_ids:
                # Wait for completion
                self.vasp_manager.wait_for_slurm_jobs([jid for jid, _, _ in job_ids])
                
                # Collect results
                print("\nCollecting results...")
                for i, (job_id, unique_key, calc_dir) in enumerate(job_ids):
                    energy = self.vasp_manager.read_energy(calc_dir)
                    if energy is not None:
                        energies[unique_key] = energy
                        print(f"  [{i+1:3d}/{len(job_ids)}] {unique_key}: Energy = {energy:.3f} eV ✓")
        
        # Map energies back to pathways
        pathway_energies = self._map_energies_to_pathways(energies, structure_to_calc)
        
        self.energy_results = {
            'unique_energies': energies,
            'pathway_energies': pathway_energies,
            'structure_mapping': structure_to_calc
        }
        
        return self.energy_results
    
    def step4_analyze_complete_energy_profiles(self, energy_results: Dict, energy_tolerance: float) -> List[Dict]:
        """Analyze energy profiles with improved classification"""
        
        viable_pathways = []
        
        print("Analyzing complete energy profiles...")
        
        # Process individual pathways
        for pathway_data in self.pathway_structures.get('individual_pathways', []):
            pathway_idx = pathway_data['pathway_index']
            
            # Collect energy profile
            energy_profile = []
            all_energies_found = True
            
            for struct_info in pathway_data['structures']:
                struct_key = f"pathway_{pathway_idx}_{struct_info['structure_type']}"
                
                if struct_key in energy_results['pathway_energies']['individual_pathways']:
                    energy = energy_results['pathway_energies']['individual_pathways'][struct_key]
                    energy_profile.append({
                        'position': struct_info['structure_type'],
                        'hop_index': struct_info['hop_index'],
                        'energy': energy
                    })
                else:
                    all_energies_found = False
                    break
            
            if all_energies_found and len(energy_profile) >= 2:
                # Use pathway analyzer for analysis
                analysis = self.pathway_analyzer.analyze_energy_profile(energy_profile, energy_tolerance)
                
                # Add pathway-specific info
                pathway_analysis = {
                    'type': 'individual',
                    'index': pathway_idx,
                    'oxygen_index': pathway_data.get('oxygen_index', -1),
                    'pathway_info': pathway_data['pathway_info'],
                    'energy_profile': energy_profile,
                    **analysis
                }
                
                # Calculate migration score and classification
                pathway_analysis['migration_score'] = self.pathway_analyzer.calculate_migration_score(
                    analysis['forward_barrier'], analysis['delta_energy'])
                pathway_analysis['pathway_class'] = self.pathway_analyzer.classify_pathway(
                    analysis['forward_barrier'], analysis['delta_energy'])
                
                # Check for intermediate minima
                if analysis['has_intermediate_minimum']:
                    pathway_analysis['suggested_segments'] = self.pathway_analyzer.suggest_pathway_segments(energy_profile)
                
                viable_pathways.append(pathway_analysis)
                
                print(f"  Pathway {pathway_idx}: Class {pathway_analysis['pathway_class']}, "
                      f"Score = {pathway_analysis['migration_score']:.3f}")
        
        # Process concerted movements
        for movement_data in self.pathway_structures.get('concerted_movements', []):
            movement_idx = movement_data['movement_index']
            
            # Collect energy profile
            energy_profile = []
            all_energies_found = True
            
            for struct_info in movement_data['structures']:
                struct_key = f"concerted_{movement_idx}_{struct_info['structure_type']}"
                
                if struct_key in energy_results['pathway_energies']['concerted_movements']:
                    energy = energy_results['pathway_energies']['concerted_movements'][struct_key]
                    energy_profile.append({
                        'position': struct_info['structure_type'],
                        'step_index': struct_info['step_index'],
                        'energy': energy
                    })
                else:
                    all_energies_found = False
                    break
            
            if all_energies_found and len(energy_profile) >= 2:
                analysis = self.pathway_analyzer.analyze_energy_profile(energy_profile, energy_tolerance)
                
                movement_analysis = {
                    'type': 'concerted',
                    'index': movement_idx,
                    'movement_info': movement_data['movement_info'],
                    'energy_profile': energy_profile,
                    'no_intermediates': len(energy_profile) == 2,
                    **analysis
                }
                
                movement_analysis['migration_score'] = self.pathway_analyzer.calculate_migration_score(
                    analysis['forward_barrier'], analysis['delta_energy'])
                movement_analysis['pathway_class'] = self.pathway_analyzer.classify_pathway(
                    analysis['forward_barrier'], analysis['delta_energy'])
                
                viable_pathways.append(movement_analysis)
                
                print(f"  Concerted {movement_idx}: Class {movement_analysis['pathway_class']}, "
                      f"Score = {movement_analysis['migration_score']:.3f}")
        
        # Sort by migration score
        viable_pathways.sort(key=lambda x: x['migration_score'])
        
        print(f"\n✓ Analyzed {len(viable_pathways)} pathways")
        
        self.viable_pathways = viable_pathways
        return viable_pathways
    
    def _save_relaxed_structures_with_pathway_info(self):
        """Save relaxed structures with comprehensive metadata"""
        
        print("\nSaving relaxed structures with pathway information...")
        
        relaxed_dir = os.path.join(self.output_dir, "relaxed_structures")
        os.makedirs(relaxed_dir, exist_ok=True)
        
        calc_dir = os.path.join(self.output_dir, "energy_calculations")
        
        # Save individual pathways
        individual_dir = os.path.join(relaxed_dir, "individual_pathways")
        os.makedirs(individual_dir, exist_ok=True)
        
        for pathway_data in self.pathway_structures.get('individual_pathways', []):
            pathway_idx = pathway_data['pathway_index']
            pathway_relaxed_dir = os.path.join(individual_dir, f"pathway_{pathway_idx:03d}")
            os.makedirs(pathway_relaxed_dir, exist_ok=True)
            
            # Find analyzed pathway
            analyzed_pathway = None
            for vp in self.viable_pathways:
                if vp['type'] == 'individual' and vp['index'] == pathway_idx:
                    analyzed_pathway = vp
                    break
            
            # Save movement info
            if self.finder:
                movement_data = self.movement_tracker.extract_pathway_movements(
                    self.finder, pathway_idx, pathway_data.get('oxygen_index', -1))
                self.movement_tracker.save_movement_info(
                    movement_data, os.path.join(pathway_relaxed_dir, "atom_movements.json"))
                self.movement_tracker.write_movement_report(
                    movement_data, analyzed_pathway, 
                    os.path.join(pathway_relaxed_dir, "atom_movements.txt"))
            
            # Save structures and metadata
            structure_sequence = []
            manifest_file = os.path.join(pathway_relaxed_dir, "structure_manifest.txt")
            
            with open(manifest_file, 'w') as f:
                f.write("Structure Manifest for NEB Setup\n")
                f.write("="*50 + "\n")
                f.write(f"Pathway index: {pathway_idx}\n")
                f.write(f"Total structures: {len(pathway_data['structures'])}\n\n")
            
            for i, struct_info in enumerate(pathway_data['structures']):
                struct_key = f"pathway_{pathway_idx}_{struct_info['structure_type']}"
                
                if struct_key in self.energy_results.get('structure_mapping', {}):
                    calc_key = self.energy_results['structure_mapping'][struct_key]
                    contcar_src = os.path.join(calc_dir, calc_key, "CONTCAR")
                    
                    seq_name = f"structure_{i:02d}_{struct_info['structure_type']}"
                    
                    # Copy relaxed structure
                    if self.structure_manager.copy_relaxed_structure(
                        contcar_src, pathway_relaxed_dir, f"{seq_name}_relaxed.POSCAR"):
                        
                        # Save original
                        original_dst = os.path.join(pathway_relaxed_dir, f"{seq_name}_original.POSCAR")
                        Poscar(struct_info['structure']).write_file(original_dst)
                        
                        # Get energy
                        if calc_key in self.energy_results.get('unique_energies', {}):
                            energy = self.energy_results['unique_energies'][calc_key]
                            structure_sequence.append({
                                'index': i,
                                'type': struct_info['structure_type'],
                                'energy': energy,
                                'relaxed_file': f"{seq_name}_relaxed.POSCAR",
                                'original_file': f"{seq_name}_original.POSCAR"
                            })
                            
                            with open(manifest_file, 'a') as f:
                                f.write(f"\n{i}: {struct_info['structure_type']}\n")
                                f.write(f"   Energy: {energy:.6f} eV\n")
                                f.write(f"   Files: {seq_name}_*.POSCAR\n")
            
            # Save structure sequence
            self.structure_manager.save_structure_sequence(
                structure_sequence, pathway_relaxed_dir)
            
            # Save pathway metadata
            self.structure_manager.save_pathway_metadata(
                pathway_data, analyzed_pathway, pathway_relaxed_dir, "individual")
        
        # Save concerted movements
        concerted_dir = os.path.join(relaxed_dir, "concerted_movements")
        os.makedirs(concerted_dir, exist_ok=True)
        
        for movement_data in self.pathway_structures.get('concerted_movements', []):
            movement_idx = movement_data['movement_index']
            movement_relaxed_dir = os.path.join(concerted_dir, f"movement_{movement_idx:03d}")
            os.makedirs(movement_relaxed_dir, exist_ok=True)
            
            # Find analyzed movement
            analyzed_movement = None
            for vp in self.viable_pathways:
                if vp['type'] == 'concerted' and vp['index'] == movement_idx:
                    analyzed_movement = vp
                    break
            
            # Save movement info
            if self.finder:
                movement_tracking = self.movement_tracker.extract_concerted_movements(
                    self.finder, movement_idx, 
                    movement_data['movement_info'].get('movement_type', 'unknown'))
                self.movement_tracker.save_movement_info(
                    movement_tracking, os.path.join(movement_relaxed_dir, "atom_movements.json"))
                self.movement_tracker.write_movement_report(
                    movement_tracking, analyzed_movement,
                    os.path.join(movement_relaxed_dir, "atom_movements.txt"), "concerted")
            
            # Save structures similar to individual pathways
            structure_sequence = []
            manifest_file = os.path.join(movement_relaxed_dir, "structure_manifest.txt")
            
            with open(manifest_file, 'w') as f:
                f.write("Structure Manifest for Concerted Movement\n")
                f.write("="*50 + "\n")
                f.write(f"Movement index: {movement_idx}\n")
                f.write(f"Movement type: {movement_data.get('movement_info', {}).get('movement_type', 'Unknown')}\n")
                f.write(f"Total structures: {len(movement_data['structures'])}\n\n")
            
            for i, struct_info in enumerate(movement_data['structures']):
                struct_key = f"concerted_{movement_idx}_{struct_info['structure_type']}"
                
                if struct_key in self.energy_results.get('structure_mapping', {}):
                    calc_key = self.energy_results['structure_mapping'][struct_key]
                    contcar_src = os.path.join(calc_dir, calc_key, "CONTCAR")
                    
                    seq_name = f"structure_{i:02d}_{struct_info['structure_type']}"
                    
                    if self.structure_manager.copy_relaxed_structure(
                        contcar_src, movement_relaxed_dir, f"{seq_name}_relaxed.POSCAR"):
                        
                        original_dst = os.path.join(movement_relaxed_dir, f"{seq_name}_original.POSCAR")
                        Poscar(struct_info['structure']).write_file(original_dst)
                        
                        if calc_key in self.energy_results.get('unique_energies', {}):
                            energy = self.energy_results['unique_energies'][calc_key]
                            structure_sequence.append({
                                'index': i,
                                'type': struct_info['structure_type'],
                                'energy': energy,
                                'relaxed_file': f"{seq_name}_relaxed.POSCAR",
                                'original_file': f"{seq_name}_original.POSCAR"
                            })
                            
                            with open(manifest_file, 'a') as f:
                                f.write(f"\n{i}: {struct_info['structure_type']}\n")
                                f.write(f"   Energy: {energy:.6f} eV\n")
                                f.write(f"   Files: {seq_name}_*.POSCAR\n")
            
            # Save structure sequence
            self.structure_manager.save_structure_sequence(
                structure_sequence, movement_relaxed_dir)
            
            # Save movement metadata
            self.structure_manager.save_pathway_metadata(
                movement_data, analyzed_movement, movement_relaxed_dir, "concerted")
        
        print(f"✓ Relaxed structures saved to: {relaxed_dir}")
        return relaxed_dir
    
    def generate_enhanced_report(self):
        """Generate comprehensive report with pathway classification"""
        from report_generator import ReportGenerator
        
        report_gen = ReportGenerator()
        report_path = os.path.join(self.output_dir, "energy_report.txt")
        
        report_gen.generate_energy_report(
            self.viable_pathways,
            report_path,
            self.finder
        )
        
        print(f"✓ Enhanced report saved to: {report_path}")
    
    # Helper methods
    def _extract_oxygen_index(self, pathway: Dict, pathway_idx: int) -> int:
        """Extract oxygen index from pathway data"""
        # Try multiple methods to get oxygen index
        if 'oxygen_index' in pathway:
            return pathway['oxygen_index']
        
        # Try from node sequence
        if 'node_sequence' in pathway and self.finder:
            first_node = pathway['node_sequence'][0]
            node_data = self.finder.migration_graph.nodes[first_node]
            
            # Find closest oxygen
            for o_idx in self.finder.oxygen_indices:
                if np.allclose(self.finder.structure[o_idx].frac_coords,
                             node_data['coords'], atol=0.01):
                    return o_idx
        
        return pathway_idx  # Fallback
    
    def _extract_movement_info(self, movement: Dict, movement_idx: int) -> Dict:
        """Extract comprehensive movement information"""
        movement_info = {
            'movement_index': movement_idx,
            'movement_type': movement.get('type', 'unknown')
        }
        
        # Extract participating atoms based on movement type
        if movement.get('type') == 'swaps':
            movement_info['oxygen1'] = movement.get('oxygen1', -1)
            movement_info['oxygen2'] = movement.get('oxygen2', -1)
            movement_info['participating_oxygens'] = [
                movement.get('oxygen1', -1), 
                movement.get('oxygen2', -1)
            ]
        
        elif movement.get('type') in ['rings', 'chains']:
            # Extract from node data if available
            participating = []
            
            if movement.get('type') == 'rings' and 'cycle_nodes' in movement:
                nodes = movement['cycle_nodes']
            elif movement.get('type') == 'chains' and 'chain_nodes' in movement:
                nodes = movement['chain_nodes']
            else:
                nodes = []
            
            if nodes and self.finder:
                for node in nodes:
                    node_data = self.finder.migration_graph.nodes[node]
                    for o_idx in self.finder.oxygen_indices:
                        if np.allclose(self.finder.structure[o_idx].frac_coords,
                                     node_data['coords'], atol=0.01):
                            participating.append(o_idx)
                            break
            
            movement_info['participating_oxygens'] = participating
        
        return movement_info
    
    def _prepare_vasp_calculation(self, structure: Structure, calc_dir: str, calc_type: str):
        """Prepare VASP input files"""
        # Write POSCAR
        poscar_path = os.path.join(calc_dir, "POSCAR")
        Poscar(structure).write_file(poscar_path)
        
        # Clean POSCAR if needed
        if hasattr(self.file_manager, 'clean_contcar_elements'):
            self.file_manager.clean_contcar_elements(poscar_path)
        
        # Create INCAR
        if calc_type == "single_point":
            incar = self.input_gen.create_relax_incar(
                structure=structure,
                max_iterations=0,
                ediffg=-0.01
            )
            incar['NSW'] = 0
            incar['IBRION'] = -1
            incar['LWAVE'] = False
            incar['LCHARG'] = False
        else:  # quick_relax
            incar = self.input_gen.create_relax_incar(
                structure=structure,
                max_iterations=100,
                ediffg=-0.2
            )
            incar['ALGO'] = 'Fast'
            incar['PREC'] = 'Normal'
            incar['EDIFF'] = 1e-4
            incar['LWAVE'] = False
            incar['LCHARG'] = False
            incar['ENCUT'] = 400
            incar['LREAL'] = 'Auto'
            incar['IBRION'] = 2
            incar['SIGMA'] = 0.1
        
        incar.write_file(os.path.join(calc_dir, "INCAR"))
        
        # Create KPOINTS
        kpoints = self.input_gen.generate_kpoints(structure, kspacing=0.3)
        kpoints.write_file(os.path.join(calc_dir, "KPOINTS"))
        
        # Generate POTCAR
        try:
            elements = [site.specie.symbol for site in structure]
            unique_elements = list(dict.fromkeys(elements))
            self.file_manager.create_potcar(calc_dir, unique_elements)
        except Exception as e:
            print(f"Warning: POTCAR generation failed: {e}")
    
    def _submit_all_jobs(self, calculations_to_submit: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """Submit all VASP jobs"""
        print(f"\nSubmitting {len(calculations_to_submit)} VASP jobs...")
        job_ids = []
        
        for i, (unique_key, calc_dir) in enumerate(calculations_to_submit):
            print(f"  [{i+1:3d}/{len(calculations_to_submit)}] Submitting {unique_key}...", end=' ')
            
            job_id = self._submit_vasp_job(calc_dir, unique_key)
            if job_id:
                job_ids.append((job_id, unique_key, calc_dir))
                print(f"✓ Job ID: {job_id}")
            else:
                print("✗ Failed")
        
        print(f"\n✓ Submitted {len(job_ids)} jobs successfully")
        return job_ids
    
    def _submit_vasp_job(self, calc_dir: str, job_name: str) -> Optional[str]:
        """Submit single VASP job"""
        nodes = self.config.get('nodes', 1)
        ntasks_per_node = self.config.get('ntasks_per_node', 64)
        partition = self.config.get('partition', None)
        
        job_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --qos=normal
#SBATCH --output=slurm.out"""
        
        if partition:
            job_script += f"\n#SBATCH --partition={partition}"
        
        job_script += f"""

cd {os.path.abspath(calc_dir)}
module load vasp/6.4.2/standard_vtst199
srun vasp > vasp.out
"""
        
        script_path = os.path.join(calc_dir, "job.sh")
        with open(script_path, 'w') as f:
            f.write(job_script)
        
        os.chmod(script_path, 0o755)
        
        try:
            return self.slurm_manager.submit_job(script_path)
        except Exception as e:
            print(f"Warning: Failed to submit job: {e}")
            return None
    
    def _map_energies_to_pathways(self, unique_energies: Dict, structure_mapping: Dict) -> Dict:
        """Map unique energies back to pathway structures"""
        pathway_energies = {
            'individual_pathways': {},
            'concerted_movements': {}
        }
        
        for structure_key, calc_key in structure_mapping.items():
            if calc_key in unique_energies:
                if structure_key.startswith('pathway_'):
                    pathway_energies['individual_pathways'][structure_key] = unique_energies[calc_key]
                elif structure_key.startswith('concerted_'):
                    pathway_energies['concerted_movements'][structure_key] = unique_energies[calc_key]
        
        return pathway_energies
    
    def _save_workflow_state(self):
        """Save workflow state for potential recovery"""
        state_file = os.path.join(self.output_dir, "workflow_state.json")
        state = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'output_dir': self.output_dir
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save workflow state: {e}")
    
    def _save_finder(self):
        """Save finder object for recovery"""
        if self.finder:
            finder_pkl = os.path.join(self.output_dir, "finder.pkl")
            try:
                with open(finder_pkl, 'wb') as f:
                    pickle.dump(self.finder, f)
            except Exception as e:
                print(f"Warning: Could not save finder: {e}")


def run_improved_workflow(structure_file: str, 
                         config: Dict,
                         reference_file: str = None,
                         output_dir: str = None,
                         calc_type: str = "single_point",
                         max_pathways: int = 10,
                         energy_tolerance: float = 0.1,
                         test_mode: bool = False):
    """
    Run improved energy-driven oxygen migration analysis
    
    Args:
        structure_file: POSCAR or CIF file with vacancy
        config: Configuration dictionary for VASP calculations
        reference_file: Optional reference structure
        output_dir: Output directory
        calc_type: "single_point" or "quick_relax"
        max_pathways: Maximum pathways to evaluate
        energy_tolerance: Energy tolerance for viable pathways (eV)
        test_mode: If True, skip VASP calculations for testing
    
    Returns:
        List of viable pathways ranked by migration score
    """
    workflow = ImprovedEnergyDrivenOxygenWorkflow(config)
    viable_pathways = workflow.run_energy_driven_analysis(
        structure_file=structure_file,
        reference_file=reference_file,
        output_dir=output_dir,
        calc_type=calc_type,
        max_pathways=max_pathways,
        energy_tolerance=energy_tolerance,
        test_mode=test_mode
    )
    
    return viable_pathways


# Example usage
if __name__ == "__main__":
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
        structure_file="POSCAR-defect",
        config=config,
        reference_file="POSCAR-perfect",
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
