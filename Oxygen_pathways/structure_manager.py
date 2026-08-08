#!/usr/bin/env python3
"""
Structure Management Module
==========================
Handles structure comparison, storage, and metadata management
"""

import os
import json
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher


class StructureManager:
    """Manages structure comparison, storage, and metadata"""
    
    def __init__(self, ltol: float = 0.3, stol: float = 0.05, angle_tol: float = 10):
        """
        Initialize structure manager with matching tolerances
        
        Args:
            ltol: Lattice tolerance
            stol: Site tolerance  
            angle_tol: Angle tolerance in degrees
        """
        self.matcher = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=False,
            scale=False,
            attempt_supercell=False
        )
    
    @staticmethod
    def get_structure_hash(structure: Structure) -> str:
        """
        Quick hash based on composition and volume for pre-filtering
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            String hash for quick comparison
        """
        comp_hash = str(structure.composition.reduced_formula)
        vol_hash = f"{structure.volume:.1f}"
        return f"{comp_hash}_{vol_hash}"
    
    def structures_match(self, struct1: Structure, struct2: Structure) -> bool:
        """
        Check if two structures are the same using StructureMatcher
        
        Args:
            struct1: First structure
            struct2: Second structure
            
        Returns:
            True if structures match within tolerances
        """
        return self.matcher.fit(struct1, struct2)
    
    def find_unique_structures(self, structure_plan: Dict) -> Tuple[Dict, Dict]:
        """
        Identify unique structures across all pathways to avoid redundant calculations
        
        Args:
            structure_plan: Dictionary with 'individual_pathways' and 'concerted_movements'
            
        Returns:
            Tuple of (unique_structures dict, structure_to_calc mapping)
        """
        unique_structures = {}
        structure_to_calc = {}
        structure_hashes = {}
        calc_index = 0
        
        print("Identifying unique structures across all pathways...")
        
        # Process all structures from individual pathways
        for pathway_data in structure_plan.get('individual_pathways', []):
            for struct_info in pathway_data['structures']:
                structure = struct_info['structure']
                struct_key = f"pathway_{pathway_data['pathway_index']}_{struct_info['structure_type']}"
                
                # Quick hash check first
                struct_hash = self.get_structure_hash(structure)
                
                # Check if this structure is unique
                is_unique = True
                
                if struct_hash in structure_hashes:
                    for unique_key in structure_hashes[struct_hash]:
                        if self.structures_match(structure, unique_structures[unique_key]):
                            structure_to_calc[struct_key] = unique_key
                            is_unique = False
                            break
                
                if is_unique:
                    unique_key = f"calc_{calc_index:04d}"
                    unique_structures[unique_key] = structure
                    structure_to_calc[struct_key] = unique_key
                    
                    if struct_hash not in structure_hashes:
                        structure_hashes[struct_hash] = []
                    structure_hashes[struct_hash].append(unique_key)
                    
                    calc_index += 1
        
        # Process all structures from concerted movements
        for movement_data in structure_plan.get('concerted_movements', []):
            for struct_info in movement_data['structures']:
                structure = struct_info['structure']
                struct_key = f"concerted_{movement_data['movement_index']}_{struct_info['structure_type']}"
                
                struct_hash = self.get_structure_hash(structure)
                is_unique = True
                
                if struct_hash in structure_hashes:
                    for unique_key in structure_hashes[struct_hash]:
                        if self.structures_match(structure, unique_structures[unique_key]):
                            structure_to_calc[struct_key] = unique_key
                            is_unique = False
                            break
                
                if is_unique:
                    unique_key = f"calc_{calc_index:04d}"
                    unique_structures[unique_key] = structure
                    structure_to_calc[struct_key] = unique_key
                    
                    if struct_hash not in structure_hashes:
                        structure_hashes[struct_hash] = []
                    structure_hashes[struct_hash].append(unique_key)
                    
                    calc_index += 1
        
        print(f"✓ Found {len(unique_structures)} unique structures")
        print(f"✓ Computational savings: {len(structure_to_calc) - len(unique_structures)} redundant calculations avoided")
        
        return unique_structures, structure_to_calc
    
    @staticmethod
    def save_structure_sequence(structures: List[Dict], output_dir: str, sequence_name: str = "structure_sequence.json"):
        """
        Save structure sequence for NEB interpolation
        
        Args:
            structures: List of structure info dictionaries
            output_dir: Directory to save the sequence file
            sequence_name: Name of the sequence file
        """
        sequence_file = os.path.join(output_dir, sequence_name)
        with open(sequence_file, 'w') as f:
            json.dump(structures, f, indent=2)
    
    @staticmethod
    def save_pathway_metadata(pathway_data: Dict, analyzed_pathway: Optional[Dict], 
                            output_dir: str, pathway_type: str = "individual"):
        """
        Save comprehensive pathway metadata including classification and energy profile
        
        Args:
            pathway_data: Basic pathway structure data
            analyzed_pathway: Analysis results with classification and scores
            output_dir: Directory to save metadata
            pathway_type: 'individual' or 'concerted'
        """
        # Prepare pathway info
        pathway_info = {
            'pathway_type': pathway_type,
            'pathway_index': pathway_data.get('pathway_index', pathway_data.get('movement_index', -1)),
            'total_steps': len(pathway_data['structures']),
            'pathway_class': analyzed_pathway['pathway_class'] if analyzed_pathway else 'Not analyzed',
            'migration_score': analyzed_pathway.get('migration_score', float('inf')) if analyzed_pathway else float('inf'),
            'energy_profile': analyzed_pathway.get('energy_profile', []) if analyzed_pathway else []
        }
        
        # Add pathway-specific info
        if pathway_type == 'individual':
            pathway_info['moving_atoms'] = [f"O{pathway_data.get('oxygen_index', -1)}"]
            pathway_info['num_hops'] = pathway_data.get('num_hops', 0)
        else:  # concerted
            movement_info = pathway_data.get('movement_info', {})
            pathway_info['movement_type'] = movement_info.get('movement_type', 'unknown')
            
            # Extract moving atoms based on movement type
            moving_atoms = []
            if 'participating_oxygens' in movement_info:
                moving_atoms = [f"O{idx}" for idx in movement_info['participating_oxygens']]
            elif movement_info.get('movement_type') == 'swaps':
                if 'oxygen1' in movement_info and 'oxygen2' in movement_info:
                    moving_atoms = [f"O{movement_info['oxygen1']}", f"O{movement_info['oxygen2']}"]
            
            pathway_info['moving_atoms'] = moving_atoms
        
        # Save pathway info
        pathway_info_file = os.path.join(output_dir, "pathway_info.json")
        with open(pathway_info_file, 'w') as f:
            json.dump(pathway_info, f, indent=2)
    
    @staticmethod
    def copy_relaxed_structure(source_path: str, dest_dir: str, structure_name: str) -> bool:
        """
        Copy relaxed structure (CONTCAR) to destination with new name
        
        Args:
            source_path: Path to CONTCAR file
            dest_dir: Destination directory
            structure_name: New name for the structure file
            
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(source_path) and os.path.getsize(source_path) > 0:
            dest_path = os.path.join(dest_dir, structure_name)
            try:
                shutil.copy2(source_path, dest_path)
                return True
            except Exception as e:
                print(f"Failed to copy {source_path}: {e}")
        return False