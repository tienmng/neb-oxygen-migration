#!/usr/bin/env python3
"""
Atom Movement Tracking Module
============================
Tracks and documents atom movements in migration pathways
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple


class AtomMovementTracker:
    """Tracks and documents atom movements in oxygen migration pathways"""
    
    @staticmethod
    def extract_pathway_movements(finder, pathway_index: int, oxygen_index: int) -> Dict:
        """
        Extract detailed movement information for an individual pathway
        
        Args:
            finder: OxygenMigrationFinder object
            pathway_index: Index of the pathway
            oxygen_index: Index of the moving oxygen atom
            
        Returns:
            Dictionary with movement details
        """
        movement_data = {
            'pathway_index': pathway_index,
            'oxygen_index': oxygen_index,
            'atom_movements': []
        }
        
        try:
            pathway = finder.migration_pathways[pathway_index]
            
            # Track movement through the pathway
            for i, node in enumerate(pathway['node_sequence']):
                node_data = finder.migration_graph.nodes[node]
                
                movement_step = {
                    'step': i,
                    'node': node,
                    'position': node_data['coords'].tolist(),
                    'site_type': node_data['site_type']
                }
                
                if i > 0:
                    prev_node = pathway['node_sequence'][i-1]
                    prev_coords = finder.migration_graph.nodes[prev_node]['coords']
                    
                    # Calculate hop vector with PBC
                    hop_vector = node_data['coords'] - prev_coords
                    hop_vector = hop_vector - np.round(hop_vector)
                    
                    distance = finder.structure.lattice.get_distance_and_image(
                        prev_coords, node_data['coords'])[0]
                    
                    movement_step['hop_distance'] = distance
                    movement_step['hop_vector'] = hop_vector.tolist()
                    
                    # Get edge data if available
                    if finder.migration_graph.has_edge(prev_node, node):
                        edge_data = finder.migration_graph[prev_node][node]
                        movement_step['energy_barrier'] = edge_data.get('energy_barrier', None)
                        movement_step['hop_type'] = edge_data.get('hop_type', 'Unknown')
                
                movement_data['atom_movements'].append(movement_step)
                
        except Exception as e:
            print(f"Could not extract detailed movement for pathway {pathway_index}: {e}")
        
        return movement_data
    
    @staticmethod
    def extract_concerted_movements(finder, movement_index: int, movement_type: str) -> Dict:
        """
        Extract detailed movement information for concerted movements
        
        Args:
            finder: OxygenMigrationFinder object
            movement_index: Index of the concerted movement
            movement_type: Type of movement ('swaps', 'rings', 'chains')
            
        Returns:
            Dictionary with movement details
        """
        movement_data = {
            'movement_index': movement_index,
            'movement_type': movement_type,
            'moving_atoms': [],
            'atom_movements': []
        }
        
        try:
            concerted_movement = finder.concerted_pathways['all'][movement_index]
            
            if movement_type == 'swaps':
                # Extract swap information
                o1_idx = concerted_movement.get('oxygen1', -1)
                o2_idx = concerted_movement.get('oxygen2', -1)
                
                if o1_idx >= 0 and o2_idx >= 0:
                    o1_start = finder.structure[o1_idx].frac_coords
                    o2_start = finder.structure[o2_idx].frac_coords
                    
                    movement_data['moving_atoms'] = [f"O{o1_idx}", f"O{o2_idx}"]
                    movement_data['atom_movements'] = [
                        {
                            'atom': f"O{o1_idx}", 
                            'start': o1_start.tolist(), 
                            'end': o2_start.tolist(),
                            'description': f'swaps with O{o2_idx}'
                        },
                        {
                            'atom': f"O{o2_idx}", 
                            'start': o2_start.tolist(), 
                            'end': o1_start.tolist(),
                            'description': f'swaps with O{o1_idx}'
                        }
                    ]
            
            elif movement_type == 'rings':
                cycle_nodes = concerted_movement.get('cycle_nodes', [])
                
                for i, node in enumerate(cycle_nodes):
                    node_data = finder.migration_graph.nodes[node]
                    next_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
                    next_data = finder.migration_graph.nodes[next_node]
                    
                    # Find oxygen at this position
                    for o_idx in finder.oxygen_indices:
                        if np.allclose(finder.structure[o_idx].frac_coords, 
                                     node_data['coords'], atol=0.01):
                            movement_data['moving_atoms'].append(f"O{o_idx}")
                            movement_data['atom_movements'].append({
                                'atom': f"O{o_idx}",
                                'start': node_data['coords'].tolist(),
                                'end': next_data['coords'].tolist(),
                                'description': f'participates in {len(cycle_nodes)}-member ring'
                            })
                            break
            
            elif movement_type == 'chains':
                chain_nodes = concerted_movement.get('chain_nodes', [])
                
                for i in range(len(chain_nodes) - 1):
                    node = chain_nodes[i]
                    next_node = chain_nodes[i + 1]
                    node_data = finder.migration_graph.nodes[node]
                    next_data = finder.migration_graph.nodes[next_node]
                    
                    # Find oxygen at this position
                    for o_idx in finder.oxygen_indices:
                        if np.allclose(finder.structure[o_idx].frac_coords,
                                     node_data['coords'], atol=0.01):
                            movement_data['moving_atoms'].append(f"O{o_idx}")
                            movement_data['atom_movements'].append({
                                'atom': f"O{o_idx}",
                                'start': node_data['coords'].tolist(),
                                'end': next_data['coords'].tolist(),
                                'description': 'participates in chain movement'
                            })
                            break
                            
        except Exception as e:
            print(f"Could not extract concerted movement details: {e}")
        
        return movement_data
    
    @staticmethod
    def save_movement_info(movement_data: Dict, output_file: str):
        """
        Save atom movement information to JSON file
        
        Args:
            movement_data: Dictionary with movement information
            output_file: Path to output JSON file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(movement_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save movement info: {e}")
    
    @staticmethod
    def write_movement_report(movement_data: Dict, pathway_analysis: Optional[Dict], 
                            output_file: str, pathway_type: str = "individual"):
        """
        Write human-readable movement report
        
        Args:
            movement_data: Movement tracking data
            pathway_analysis: Energy analysis results
            output_file: Path to output text file
            pathway_type: 'individual' or 'concerted'
        """
        with open(output_file, 'w') as f:
            if pathway_type == "individual":
                f.write("ATOM MOVEMENT INFORMATION\n")
                f.write("="*50 + "\n")
                f.write(f"Pathway index: {movement_data.get('pathway_index', -1)}\n")
                f.write(f"Moving oxygen atom: O{movement_data.get('oxygen_index', 'Unknown')}\n")
                
                if pathway_analysis:
                    f.write(f"\nPathway classification: {pathway_analysis['pathway_class']}\n")
                    f.write(f"Migration score: {pathway_analysis['migration_score']:.3f}\n")
                    f.write(f"Forward barrier: {pathway_analysis['forward_barrier']:.3f} eV\n")
                    f.write(f"Energy change: {pathway_analysis['delta_energy']:.3f} eV\n")
                
                f.write("\n" + "-"*50 + "\n")
                f.write("DETAILED MOVEMENT SEQUENCE:\n")
                f.write("-"*50 + "\n\n")
                
                for i, step in enumerate(movement_data.get('atom_movements', [])):
                    if i == 0:
                        f.write(f"Starting position: {step['position']} ({step['site_type']})\n")
                    else:
                        f.write(f"Step {i}: Hop to {step['site_type']}\n")
                        f.write(f"  New position: {step['position']}\n")
                        if 'hop_distance' in step:
                            f.write(f"  Hop distance: {step['hop_distance']:.3f} Å\n")
                        if 'hop_vector' in step:
                            f.write(f"  Hop vector: {step['hop_vector']}\n")
                        if 'energy_barrier' in step:
                            f.write(f"  Energy barrier: {step['energy_barrier']} eV\n")
                    f.write("\n")
            
            else:  # concerted
                f.write("CONCERTED MOVEMENT INFORMATION\n")
                f.write("="*50 + "\n")
                f.write(f"Movement index: {movement_data.get('movement_index', -1)}\n")
                f.write(f"Movement type: {movement_data.get('movement_type', 'Unknown')}\n")
                
                if pathway_analysis:
                    f.write(f"\nMovement classification: {pathway_analysis['pathway_class']}\n")
                    f.write(f"Migration score: {pathway_analysis['migration_score']:.3f}\n")
                    f.write(f"Effective barrier: {pathway_analysis.get('effective_barrier', pathway_analysis.get('forward_barrier', 0)):.3f} eV\n")
                    f.write(f"Energy change: {pathway_analysis['delta_energy']:.3f} eV\n")
                
                f.write("\n" + "-"*50 + "\n")
                f.write("ATOMS INVOLVED:\n")
                f.write("-"*50 + "\n\n")
                
                for atom in movement_data.get('moving_atoms', []):
                    f.write(f"{atom} participates in {movement_data.get('movement_type', 'unknown')} movement\n")
                
                if movement_data.get('atom_movements'):
                    f.write("\nDetailed movements:\n")
                    for movement in movement_data['atom_movements']:
                        f.write(f"{movement['atom']}: {movement.get('description', 'moves')}\n")
                        if 'start' in movement and 'end' in movement:
                            f.write(f"  From: {movement['start']}\n")
                            f.write(f"  To: {movement['end']}\n")