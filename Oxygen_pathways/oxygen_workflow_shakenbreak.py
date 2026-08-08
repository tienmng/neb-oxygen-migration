
"""
Oxygen Migration Workflow with ShakeNBreak Integration
====================================================
Enhanced workflow that applies ShakeNBreak distortions to find global minima
before analyzing migration barriers
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pymatgen.core import Structure, Element
from pymatgen.io.vasp import Poscar

# Import VASP calculation modules
sys.path.append('/path/to/Oxygen_pathways')

#sys.path.append('/path/to/Oxygen_pathways')  # UPDATE THIS PATH

# Import the enhanced analysis module
from enhanced_oxygen_migration_analysis import analyze_oxygen_migration

# Import workflow modules
from relax_NEB.file_manager import FileManager
from relax_NEB.vasp_inputs import VASPInputGenerator
from relax_NEB.slurm_manager import SLURMManager

# Import modular components
from pathway_analyzer import PathwayAnalyzer
from structure_manager import StructureManager
from atom_movement_tracker import AtomMovementTracker
from vasp_manager import VASPManager
from shakenbreak_manager import ShakeNBreakManager
from report_generator import ReportGenerator

# Import base workflow
from oxygen_workflow_improved import ImprovedEnergyDrivenOxygenWorkflow


class ShakeNBreakOxygenWorkflow(ImprovedEnergyDrivenOxygenWorkflow):
    """
    Enhanced oxygen migration workflow with ShakeNBreak integration
    """
    
    def __init__(self, config: Dict):
        """Initialize with VASP calculation configuration"""
        super().__init__(config)
        self.shakenbreak_manager = ShakeNBreakManager(config)
        self.shakenbreak_results = {}
    

    def run_shakenbreak_enhanced_analysis(self,
                                        structure_file: str,
                                        reference_file: str = None,
                                        output_dir: str = None,
                                        calc_type: str = "single_point",
                                        max_pathways: int = 10,
                                        energy_tolerance: float = 0.1,
                                        test_mode: bool = False,
                                        apply_shakenbreak: bool = True,
                                        shakenbreak_calc_type: str = "relax",
                                        bond_distortions: List[float] = None,
                                        min_e_diff: float = 0.05,
                                        apply_dimer: bool = True,
                                        apply_local_rattle: bool = False,
                                        stdev: float = 0.25,
                                        d_min: float = 2.25,
                                        num_nearest_neighbours: int = 2):
        """
        Run energy-driven analysis with ShakeNBreak distortion testing
        
        Args:
            structure_file: POSCAR or CIF file with vacancy
            reference_file: Optional reference structure
            output_dir: Output directory
            calc_type: Calculation type for migration barriers
            max_pathways: Maximum pathways to evaluate
            energy_tolerance: Energy tolerance for viable pathways (eV)
            test_mode: If True, skip VASP calculations
            apply_shakenbreak: Whether to apply ShakeNBreak distortions
            shakenbreak_calc_type: Calculation type for ShakeNBreak ("relax" or "single_point")
            bond_distortions: List of distortion factors for ShakeNBreak
            min_e_diff: Minimum energy difference for ShakeNBreak analysis (eV)
            apply_dimer: Whether to apply dimer distortions
            apply_local_rattle: Whether to use local rattling (displacements tail off from defect)
            stdev: Standard deviation for rattling (Angstroms)
            d_min: Minimum interatomic distance (Angstroms)
            num_nearest_neighbours: Number of neighbours to distort in bond distortions
            
        Returns:
            List of viable pathways with ground state structures
        """
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"shakenbreak_oxygen_workflow_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.test_mode = test_mode
        
        print("="*70)
        print("OXYGEN MIGRATION WORKFLOW WITH SHAKENBREAK")
        print("="*70)
        print(f"Output directory: {output_dir}")
        if test_mode:
            print("⚠️  RUNNING IN TEST MODE - No VASP calculations will be submitted")
        
        # Save workflow state
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
        
        structure_plan = self.step2_extract_all_structures_enhanced(finder, max_pathways)
        
        if apply_shakenbreak and not test_mode:
            # STEP 3: Apply ShakeNBreak distortions
            print("\n" + "="*60)
            print("STEP 3: Apply ShakeNBreak Distortions")
            print("="*60)
            
            shakenbreak_structures = self.step3_apply_shakenbreak(
                structure_plan, 
                shakenbreak_calc_type,
                bond_distortions,
                min_e_diff,
                apply_dimer,
                apply_local_rattle,
                stdev,
                d_min,
                num_nearest_neighbours
            )
            
            # Update structure plan with ground state structures
            structure_plan = self._update_structure_plan_with_ground_states(
                structure_plan, shakenbreak_structures
            )
        
        if not test_mode:
            # STEP 4: Energy evaluation with ground state structures
            print("\n" + "="*60)
            print("STEP 4: Energy Evaluation of Migration Barriers")
            print("="*60)
            
            energy_results = self.step3_evaluate_all_energies(structure_plan, calc_type)
            
            # STEP 5: Analyze energy profiles
            print("\n" + "="*60)
            print("STEP 5: Energy Profile Analysis")
            print("="*60)
            
            viable_pathways = self.step4_analyze_complete_energy_profiles(energy_results, energy_tolerance)
            
            # STEP 6: Save results
            print("\n" + "="*60)
            print("STEP 6: Saving Results and Structures")
            print("="*60)
            
            self._save_relaxed_structures_with_shakenbreak_info()
            self.generate_enhanced_shakenbreak_report()
            
        else:
            print("\n⚠️  Skipping calculations in test mode")
            viable_pathways = []
        
        return viable_pathways
    
    def step3_apply_shakenbreak(self, 
                            structure_plan: Dict,
                            calc_type: str = "relax",
                            bond_distortions: List[float] = None,
                            min_e_diff: float = 0.05,
                            apply_dimer: bool = True,
                            apply_local_rattle: bool = False,
                            stdev: float = 0.25,
                            d_min: float = 2.25,
                            num_nearest_neighbours: int = 2) -> Dict:
        """
        Apply ShakeNBreak distortions and find ground state structures
        Enhanced to pass vacancy information to ShakeNBreak manager
        """
        # Pass reference structure to ShakeNBreak manager if available
        if hasattr(self, 'reference_structure'):
            self.shakenbreak_manager.reference_structure = self.reference_structure
        elif hasattr(self, 'finder') and hasattr(self.finder, 'reference_structure'):
            self.shakenbreak_manager.reference_structure = self.finder.reference_structure
        
        # Pass vacancy information from structure plan
        for pathway_data in structure_plan.get('individual_pathways', []):
            
            for struct_info in pathway_data['structures']:
                if 'vacancy_coords' in struct_info:
                    struct_key = f"pathway_{pathway_data['pathway_index']}_{struct_info['structure_type']}"
                    self.shakenbreak_manager.pathway_vacancy_info[struct_key] = {
                        'coords': struct_info['vacancy_coords'],
                        'type': 'oxygen_vacancy',
                        'pathway_index': pathway_data['pathway_index']
                    }
                    break
        
        # Apply ShakeNBreak distortions
        snb_results = self.shakenbreak_manager.apply_shakenbreak_to_structures_enhanced(
            structure_plan=structure_plan,
            output_dir=self.output_dir,
            bond_distortions=bond_distortions,
            stdev=stdev,
            d_min=d_min,
            num_nearest_neighbours=num_nearest_neighbours,
            apply_dimer=apply_dimer,
            apply_local_rattle=apply_local_rattle
        )
        
        
        # Prepare VASP calculations
        calculations = self.shakenbreak_manager.prepare_vasp_calculations(
            vasp_manager=self.vasp_manager,
            file_manager=self.file_manager,
            input_gen=self.input_gen,
            calc_type=calc_type
        )
        
        if calculations:
            print(f"\nSubmitting {len(calculations)} ShakeNBreak calculations...")
            
            # Submit jobs
            job_ids = []
            for calc_key, calc_dir in calculations:
                job_id = self._submit_vasp_job(calc_dir, f"snb_{calc_key}")
                if job_id:
                    job_ids.append((job_id, calc_key, calc_dir))
            
            if job_ids:
                # Wait for completion
                self.vasp_manager.wait_for_slurm_jobs([jid for jid, _, _ in job_ids])
                
                print("\nShakeNBreak calculations complete. Analyzing results...")
        
        # Analyze results
        analysis_results = self.shakenbreak_manager.analyze_shakenbreak_results(
            output_dir=self.output_dir,
            min_e_diff=min_e_diff
        )
        
        # Write ShakeNBreak report
        self.shakenbreak_manager.write_shakenbreak_report(
            output_dir=self.output_dir,
            analysis_results=analysis_results
        )
        
        # Get ground state structures
        ground_state_structures = self.shakenbreak_manager.get_ground_state_structures()
        
        self.shakenbreak_results = {
            'analysis': analysis_results,
            'ground_states': ground_state_structures
        }
        
        return ground_state_structures
    
    def _update_structure_plan_with_ground_states(self, 
                                                 structure_plan: Dict,
                                                 ground_state_structures: Dict) -> Dict:
        """
        Update structure plan to use ground state structures from ShakeNBreak
        
        Args:
            structure_plan: Original structure plan
            ground_state_structures: Ground state structures from ShakeNBreak
            
        Returns:
            Updated structure plan
        """
        updated_plan = {
            'individual_pathways': [],
            'concerted_movements': []
        }
        
        # Update individual pathways
        for pathway_data in structure_plan.get('individual_pathways', []):
            updated_pathway = pathway_data.copy()
            updated_structures = []
            
            for struct_info in pathway_data['structures']:
                struct_key = f"pathway_{pathway_data['pathway_index']}_{struct_info['structure_type']}"
                
                if struct_key in ground_state_structures:
                    # Use ground state structure
                    updated_struct_info = struct_info.copy()
                    updated_struct_info['structure'] = ground_state_structures[struct_key]
                    updated_struct_info['shakenbreak_applied'] = True
                    updated_struct_info['ground_state_info'] = self.shakenbreak_results['analysis'].get(struct_key, {})
                    updated_structures.append(updated_struct_info)
                else:
                    # Keep original structure
                    updated_structures.append(struct_info)
            
            updated_pathway['structures'] = updated_structures
            updated_plan['individual_pathways'].append(updated_pathway)
        
        # Update concerted movements
        for movement_data in structure_plan.get('concerted_movements', []):
            updated_movement = movement_data.copy()
            updated_structures = []
            
            for struct_info in movement_data['structures']:
                struct_key = f"concerted_{movement_data['movement_index']}_{struct_info['structure_type']}"
                
                if struct_key in ground_state_structures:
                    # Use ground state structure
                    updated_struct_info = struct_info.copy()
                    updated_struct_info['structure'] = ground_state_structures[struct_key]
                    updated_struct_info['shakenbreak_applied'] = True
                    updated_struct_info['ground_state_info'] = self.shakenbreak_results['analysis'].get(struct_key, {})
                    updated_structures.append(updated_struct_info)
                else:
                    # Keep original structure
                    updated_structures.append(struct_info)
            
            updated_movement['structures'] = updated_structures
            updated_plan['concerted_movements'].append(updated_movement)
        
        return updated_plan
    
    def _save_relaxed_structures_with_shakenbreak_info(self):
        """Save relaxed structures with ShakeNBreak metadata"""
        
        print("\nSaving relaxed structures with ShakeNBreak information...")
        
        relaxed_dir = os.path.join(self.output_dir, "relaxed_structures")
        os.makedirs(relaxed_dir, exist_ok=True)
        
        # Call parent method first
        super()._save_relaxed_structures_with_pathway_info()
        
        # Add ShakeNBreak metadata
        for pathway_data in self.pathway_structures.get('individual_pathways', []):
            pathway_idx = pathway_data['pathway_index']
            pathway_dir = os.path.join(relaxed_dir, "individual_pathways", f"pathway_{pathway_idx:03d}")
            
            if os.path.exists(pathway_dir):
                # Save ShakeNBreak info
                snb_info = {}
                for struct_info in pathway_data['structures']:
                    struct_key = f"pathway_{pathway_idx}_{struct_info['structure_type']}"
                    if struct_key in self.shakenbreak_results.get('analysis', {}):
                        snb_info[struct_info['structure_type']] = self.shakenbreak_results['analysis'][struct_key]
                
                if snb_info:
                    with open(os.path.join(pathway_dir, "shakenbreak_info.json"), 'w') as f:
                        json.dump(snb_info, f, indent=2)
        
        # Same for concerted movements
        for movement_data in self.pathway_structures.get('concerted_movements', []):
            movement_idx = movement_data['movement_index']
            movement_dir = os.path.join(relaxed_dir, "concerted_movements", f"movement_{movement_idx:03d}")
            
            if os.path.exists(movement_dir):
                snb_info = {}
                for struct_info in movement_data['structures']:
                    struct_key = f"concerted_{movement_idx}_{struct_info['structure_type']}"
                    if struct_key in self.shakenbreak_results.get('analysis', {}):
                        snb_info[struct_info['structure_type']] = self.shakenbreak_results['analysis'][struct_key]
                
                if snb_info:
                    with open(os.path.join(movement_dir, "shakenbreak_info.json"), 'w') as f:
                        json.dump(snb_info, f, indent=2)

    def extract_vacancy_info_from_pathway(self, finder, pathway, step_index=None):
        """
        Extract comprehensive vacancy information from a pathway.
        Args:
            finder: Migration pathway finder instance
            pathway: Individual pathway data
            step_index: Optional specific step to analyze (0 = initial, -1 = final, etc.)
                    If None, returns general pathway vacancy information
            
        Returns:
            Dictionary with vacancy information:
            - If step_index is specified: detailed info for that specific step
            - If step_index is None: general pathway vacancy information
        """
        node_sequence = pathway['node_sequence']
        oxygen_index = pathway['oxygen_index']
        
        # Get oxygen's initial position
        oxygen_initial_coords = finder.structure[oxygen_index].frac_coords
        
        # If no specific step requested, return general pathway info
        if step_index is None:
            vacancy_info = {
                'has_vacancy': False,
                'vacancy_nodes': [],
                'vacancy_coords': [],
                'oxygen_index': oxygen_index,
                'oxygen_initial_coords': oxygen_initial_coords.tolist()
            }
            
            # Check each node in the pathway
            for i, node in enumerate(node_sequence):
                node_data = finder.migration_graph.nodes[node]
                if 'vacancy' in node_data.get('site_type', ''):
                    vacancy_info['has_vacancy'] = True
                    vacancy_info['vacancy_nodes'].append({
                        'node': node,
                        'step': i,
                        'coords': node_data['coords'],
                        'site_type': node_data['site_type']
                    })
                    vacancy_info['vacancy_coords'].append(node_data['coords'])
            
            # Also check if finder has stored vacancy information
            if hasattr(finder, 'all_vacancies'):
                # Find oxygen vacancies near the pathway
                for vac in finder.all_vacancies.get('oxygen_vacancies', []):
                    vac_coords = vac['coords']
                    # Check if this vacancy is part of the pathway
                    for node in node_sequence:
                        node_coords = finder.migration_graph.nodes[node]['coords']
                        if np.allclose(vac_coords, node_coords, atol=0.01):
                            vacancy_info['finder_vacancy'] = vac
                            break
            
            return vacancy_info
        
        # If specific step requested, return detailed step information
        vacancy_info = {
            'step': step_index,
            'vacancy_coords': None,
            'vacancy_type': None,
            'oxygen_position': None,
            'node_info': None
        }
        
        # Normalize step index
        if step_index < 0:
            step_index = len(node_sequence) + step_index
        
        if step_index == 0:  # Initial state
            # Find the vacancy node in the sequence
            for i, node in enumerate(node_sequence):
                node_data = finder.migration_graph.nodes[node]
                if 'vacancy' in node_data['site_type']:
                    vacancy_info['vacancy_coords'] = node_data['coords']
                    vacancy_info['vacancy_type'] = node_data['site_type']
                    vacancy_info['oxygen_position'] = oxygen_initial_coords.tolist()
                    vacancy_info['node_info'] = {
                        'node': node,
                        'index': i,
                        'coords': node_data['coords'],
                        'site_type': node_data['site_type']
                    }
                    break
        
        elif step_index == len(node_sequence) - 1:  # Final state
            # Vacancy is now where the oxygen started
            vacancy_info['vacancy_coords'] = oxygen_initial_coords.tolist()
            vacancy_info['vacancy_type'] = 'oxygen_vacancy'
            # Oxygen is at the last node in sequence
            final_node = node_sequence[-1]
            final_node_data = finder.migration_graph.nodes[final_node]
            vacancy_info['oxygen_position'] = final_node_data['coords']
            vacancy_info['node_info'] = {
                'node': final_node,
                'index': len(node_sequence) - 1,
                'coords': final_node_data['coords'],
                'site_type': final_node_data['site_type']
            }
        
        else:  # Intermediate step
            # For multi-hop pathways, track where the vacancy is at each step
            current_oxygen_pos = oxygen_initial_coords.tolist()
            current_vacancy_pos = None
            
            # Find initial vacancy
            for node in node_sequence:
                node_data = finder.migration_graph.nodes[node]
                if 'vacancy' in node_data['site_type']:
                    current_vacancy_pos = node_data['coords']
                    break
            
            # Simulate hops up to the requested step
            for i in range(1, min(step_index + 1, len(node_sequence))):
                next_node = node_sequence[i]
                next_coords = finder.migration_graph.nodes[next_node]['coords']
                
                # Oxygen moves to next position, vacancy moves to oxygen's previous position
                current_vacancy_pos = current_oxygen_pos
                current_oxygen_pos = next_coords
            
            vacancy_info['vacancy_coords'] = current_vacancy_pos
            vacancy_info['oxygen_position'] = current_oxygen_pos
            
            # Add node information for the current step
            if step_index < len(node_sequence):
                current_node = node_sequence[step_index]
                current_node_data = finder.migration_graph.nodes[current_node]
                vacancy_info['node_info'] = {
                    'node': current_node,
                    'index': step_index,
                    'coords': current_node_data['coords'],
                    'site_type': current_node_data['site_type']
                }
        
        return vacancy_info

    def extract_vacancy_coordinates_from_concerted(self, finder, movement, step='initial'):
        """
        Extract vacancy coordinates for concerted movements
        
        Args:
            finder: Migration pathway finder instance
            movement: Concerted movement data
            step: 'initial' or 'final'
            
        Returns:
            Dictionary with vacancy information
        """
        movement_type = movement['movement_type']
        vacancy_info = {
            'step': step,
            'vacancy_coords': None,
            'vacancy_type': None,
            'moved_atoms': []
        }
        
        if movement_type == 'swaps':
            # Direct swaps typically don't involve vacancies
            o1_idx = movement['oxygen1']
            o2_idx = movement['oxygen2']
            
            o1_coords = finder.structure[o1_idx].frac_coords
            o2_coords = finder.structure[o2_idx].frac_coords
            
            if step == 'initial':
                vacancy_info['moved_atoms'] = [
                    {'atom': f'O{o1_idx}', 'position': o1_coords},
                    {'atom': f'O{o2_idx}', 'position': o2_coords}
                ]
            else:  # final
                vacancy_info['moved_atoms'] = [
                    {'atom': f'O{o1_idx}', 'position': o2_coords},
                    {'atom': f'O{o2_idx}', 'position': o1_coords}
                ]
        
        elif movement_type == 'rings':
            cycle_nodes = movement['cycle_nodes']
            
            # Find vacancy node and oxygen nodes
            vacancy_node = None
            oxygen_nodes = []
            
            for node in cycle_nodes:
                node_data = finder.migration_graph.nodes[node]
                if 'vacancy' in node_data['site_type']:
                    vacancy_node = node
                    vacancy_info['vacancy_type'] = node_data['site_type']
                else:
                    # Find corresponding oxygen
                    for o_idx in finder.oxygen_indices:
                        o_coords = finder.structure[o_idx].frac_coords
                        if np.allclose(o_coords, node_data['coords'], atol=0.01):
                            oxygen_nodes.append((node, o_idx))
                            break
            
            if vacancy_node is not None:
                vacancy_idx = cycle_nodes.index(vacancy_node)
                
                if step == 'initial':
                    # Initial vacancy position
                    vacancy_info['vacancy_coords'] = finder.migration_graph.nodes[vacancy_node]['coords']
                else:  # final
                    # Vacancy moves to previous position in ring (reverse direction)
                    prev_idx = (vacancy_idx - 1) % len(cycle_nodes)
                    prev_node = cycle_nodes[prev_idx]
                    vacancy_info['vacancy_coords'] = finder.migration_graph.nodes[prev_node]['coords']
        
        elif movement_type == 'chains':
            chain_nodes = movement['chain_nodes']
            
            # Typically, last node is vacancy
            last_node = chain_nodes[-1]
            last_node_data = finder.migration_graph.nodes[last_node]
            
            if 'vacancy' in last_node_data['site_type']:
                if step == 'initial':
                    vacancy_info['vacancy_coords'] = last_node_data['coords']
                    vacancy_info['vacancy_type'] = last_node_data['site_type']
                else:  # final
                    # Vacancy is now at the first position of the chain
                    first_node = chain_nodes[0]
                    vacancy_info['vacancy_coords'] = finder.migration_graph.nodes[first_node]['coords']
        
        return vacancy_info

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

    def generate_enhanced_shakenbreak_report(self):
        """Generate comprehensive report including ShakeNBreak findings"""
        
        # Generate base report from parent class
        self.generate_enhanced_report()
        
        with open(os.path.join(self.output_dir, "WORKFLOW_COMPLETE.txt"), 'w') as f:
            f.write("OXYGEN MIGRATION WORKFLOW WITH SHAKENBREAK - COMPLETE\n")
            f.write("="*70 + "\n\n")
            f.write(f"Completion time: {datetime.now()}\n\n")
            
            if self.shakenbreak_results:
                analysis = self.shakenbreak_results.get('analysis', {})
                total_analyzed = len(analysis)
                energy_lowering_found = sum(
                    1 for r in analysis.values()
                    if r.get('found_lower_energy', False)
                )
                
                f.write("Key Results:\n")
                if energy_lowering_found > 0:
                    f.write(f"✓ Found energy-lowering distortions for {energy_lowering_found}/{total_analyzed} structures\n")
                    f.write("✓ Ground state structures identified and saved\n")
                else:
                    f.write("✓ All structures at global minimum\n")
            
            f.write("\nFor detailed ShakeNBreak analysis, see:\n")
            f.write("- shakenbreak_report_enhanced.txt\n")
            f.write("- shakenbreak_workflow_summary.md\n")
            f.write("- energy_report.txt (base energy analysis)\n")

    def step2_extract_all_structures_enhanced(self, finder, max_pathways=10):
        """
        Enhanced version of step2 that extracts vacancy information from pathways
        
        Args:
            finder: Migration pathway finder instance
            max_pathways: Maximum pathways to extract
            
        Returns:
            Enhanced structure plan with vacancy information
        """
        print("Extracting structures with vacancy information...")
            
        # Create a temporary directory inside the workflow output directory
        temp_dir = os.path.join(self.output_dir, "temp_structures")
        os.makedirs(temp_dir, exist_ok=True)
        
        structure_plan = {
            'individual_pathways': [],
            'concerted_movements': []
        }
        
        # Extract individual pathways with vacancy info
        num_individual = min(max_pathways, len(finder.migration_pathways))
        
        for i in range(num_individual):
            pathway = finder.migration_pathways[i]
            
            # Get detailed pathway information including vacancy positions
            pathway_data = {
                'pathway_index': i,
                'oxygen_index': pathway['oxygen_index'],
                'total_score': pathway['total_score'],
                'structures': []
            }
            
            # Extract vacancy information from the pathway
            vacancy_info = self.extract_vacancy_info_from_pathway(finder, pathway)
            pathway_data['vacancy_info'] = vacancy_info
            
            # Get node sequence information
            node_sequence = pathway['node_sequence']
            node_info = []
            
            for node in node_sequence:
                node_data = finder.migration_graph.nodes[node]
                node_info.append({
                    'node': node,
                    'coords': node_data['coords'],
                    'site_type': node_data['site_type']
                })
            
            pathway_data['node_info'] = node_info
            

            # Extract structures along the pathway
            # Use full path for output prefix
            temp_prefix = os.path.join(temp_dir, f"pathway_{i}")
            structures = finder.execute_migration_pathway(
                pathway_index=i,
                output_prefix=temp_prefix
            )
            
                        
            # Add detailed vacancy tracking for each step:
            for j, (structure, node_data) in enumerate(zip(structures, node_info)):
                # Get vacancy position for this specific step
                vacancy_info_step = self.extract_vacancy_info_from_pathway(finder, pathway, step_index=j)
                
                struct_info = {
                    'structure': structure,
                    'structure_type': f'step_{j}',
                    'node_data': node_data,
                    'vacancy_coords': vacancy_info_step.get('vacancy_coords'),
                    'vacancy_type': vacancy_info_step.get('vacancy_type'),
                    'oxygen_position': vacancy_info_step.get('oxygen_position')
                }
                
                pathway_data['structures'].append(struct_info)
            
            structure_plan['individual_pathways'].append(pathway_data)
        
        # Extract concerted movements with vacancy info
        if hasattr(finder, 'concerted_pathways') and finder.concerted_pathways['all']:
            num_concerted = min(max_pathways, len(finder.concerted_pathways['all']))
            
            for i in range(num_concerted):
                movement = finder.concerted_pathways['all'][i]
                
                movement_data = {
                    'movement_index': i,
                    'movement_type': movement['movement_type'],
                    'score': movement.get('score', 0),
                    'movement_info': movement,
                    'structures': []
                }
                
                # Extract vacancy information for concerted movements
                vacancy_info = self.extract_vacancy_coordinates_from_concerted(finder, movement, step='initial')
                vacancy_info = {
                'has_vacancy': vacancy_info.get('vacancy_coords') is not None,
                'vacancy_type': vacancy_info.get('vacancy_type'),
                'coords': vacancy_info.get('vacancy_coords')
}
                movement_data['vacancy_info'] = vacancy_info
                # Get structures with full path
                temp_prefix = os.path.join(temp_dir, f"concerted_{i}")
                structures = finder.execute_concerted_movement(
                    movement_index=i,
                    output_prefix=temp_prefix
                )
                
                for j, structure in enumerate(structures):
                    # Get vacancy info for this specific step (need to implement this method)
                    step_vacancy_info = self.extract_vacancy_info_from_concerted_step(
                        finder, movement, step_index=j
                    )
                    
                    struct_info = {
                        'structure': structure,
                        'structure_type': f'step_{j}',
                        'vacancy_coords': step_vacancy_info.get('vacancy_coords'),
                        'vacancy_type': step_vacancy_info.get('vacancy_type')
                    }
                    
                    movement_data['structures'].append(struct_info)
                
                structure_plan['concerted_movements'].append(movement_data)
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass
        
        return structure_plan

    def extract_vacancy_info_from_concerted_step(self, finder, movement, step_index):
        """
        Extract vacancy information for a specific step in a concerted movement
        
        This is a wrapper that calls the appropriate method for concerted movements
        instead of the pathway method.
        
        Args:
            finder: Migration pathway finder instance
            movement: Concerted movement data
            step_index: Step index (0 = initial, -1 = final, etc.)
            
        Returns:
            Dictionary with vacancy information for the step
        """
        # For concerted movements, we need different logic than for pathways
        if step_index == 0:
            step = 'initial'
        elif step_index == 1 or step_index == -1:
            step = 'final'
        else:
            # For intermediate steps, default to initial
            step = 'initial'
        
        return self.extract_vacancy_coordinates_from_concerted(finder, movement, step)



def run_shakenbreak_workflow(structure_file: str,
                           config: Dict,
                           reference_file: str = None,
                           output_dir: str = None,
                           calc_type: str = "single_point",
                           max_pathways: int = 10,
                           energy_tolerance: float = 0.1,
                           test_mode: bool = False,
                           apply_shakenbreak: bool = True,
                           shakenbreak_calc_type: str = "relax",
                           bond_distortions: List[float] = None,
                           min_e_diff: float = 0.05,
                           apply_dimer: bool = True,
                           apply_local_rattle: bool = False,
                           stdev: float = 0.25,
                           d_min: float = 2.25,
                           num_nearest_neighbours: int = 2):
    """
    Run oxygen migration analysis with ShakeNBreak integration
    
    Args:
        structure_file: POSCAR or CIF file with vacancy
        config: Configuration dictionary for VASP calculations
        reference_file: Optional reference structure
        output_dir: Output directory
        calc_type: Calculation type for migration barriers
        max_pathways: Maximum pathways to evaluate
        energy_tolerance: Energy tolerance for viable pathways (eV)
        test_mode: If True, skip VASP calculations for testing
        apply_shakenbreak: Whether to apply ShakeNBreak distortions
        shakenbreak_calc_type: Calculation type for ShakeNBreak
        bond_distortions: List of distortion factors for ShakeNBreak
        min_e_diff: Minimum energy difference for ShakeNBreak (eV)
        apply_dimer: Whether to apply dimer distortions
        apply_local_rattle: Whether to use local rattling (displacements tail off from defect)
        stdev: Standard deviation for rattling (Angstroms)
        d_min: Minimum interatomic distance (Angstroms)
        num_nearest_neighbours: Number of neighbours to distort in bond distortions
    
    Returns:
        List of viable pathways with ground state structures
    """
    workflow = ShakeNBreakOxygenWorkflow(config)
    viable_pathways = workflow.run_shakenbreak_enhanced_analysis(
        structure_file=structure_file,
        reference_file=reference_file,
        output_dir=output_dir,
        calc_type=calc_type,
        max_pathways=max_pathways,
        energy_tolerance=energy_tolerance,
        test_mode=test_mode,
        apply_shakenbreak=apply_shakenbreak,
        shakenbreak_calc_type=shakenbreak_calc_type,
        bond_distortions=bond_distortions,
        min_e_diff=min_e_diff,
        apply_dimer=apply_dimer,
        apply_local_rattle=apply_local_rattle,
        stdev=stdev,
        d_min=d_min,
        num_nearest_neighbours=num_nearest_neighbours
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
    
    print("Starting oxygen migration analysis with ShakeNBreak...")
    print("This workflow will:")
    print("1. Generate oxygen migration pathways")
    print("2. Extract structures along pathways")
    print("3. Apply ShakeNBreak distortions to find global minima")
    print("4. Calculate migration barriers using ground state structures")
    
    viable_pathways = run_shakenbreak_workflow(
        structure_file="POSCAR-defect",
        config=config,
        reference_file="POSCAR-perfect",
        calc_type="quick_relax",
        max_pathways=5,
        energy_tolerance=0.1,
        test_mode=False,
        apply_shakenbreak=True,
        shakenbreak_calc_type="relax",
        bond_distortions=[-0.4, -0.2, 0.2, 0.4],  # Exclude 0.0 as it's redundant with Unperturbed
        min_e_diff=0.05
    )
    # Update the ShakeNBreak manager initialization to store reference structure

    print("\nWorkflow complete!")
    if viable_pathways:
        best = viable_pathways[0]
        print(f"Best pathway: {best['type']} {best['index']}")
        print(f"  Class: {best['pathway_class']}")
        print(f"  Migration score: {best['migration_score']:.3f}")
        print(f"  Forward barrier: {best['forward_barrier']:.3f} eV")
        print(f"  Energy change: {best['delta_energy']:+.3f} eV")
