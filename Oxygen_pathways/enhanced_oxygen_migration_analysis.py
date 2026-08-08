#!/usr/bin/env python3
"""
Simple script to find oxygen migration pathways and create visualizations
Combines oxygen_hopping_vacancy_pathway_generation.py and oxygen_pathway_visualization.py
Enhanced with detailed pathway information and vacancy-focused structure generation
"""

import os
import sys
import numpy as np
from datetime import datetime
sys.path.append('/nfs/home/6/nguyenm/pymatgen-packages/Oxygen_pathways')  # UPDATE THIS PATH

# Import the pathway finder and visualizer modules
try:
    from oxygen_hopping_vacancy_pathway_generation import UniversalVacancyOxygenMigrationFinder
    from oxygen_pathway_visualization import OxygenPathwayVisualizer, visualize_oxygen_pathways
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure both modules are in the same directory or in your Python path")
    sys.exit(1)


def get_detailed_pathway_info(finder, pathway):
    """Extract detailed information about a pathway with proper occupancy tracking"""
    details = {
        'steps': [],
        'score_breakdown': {},
        'vacancy_involvement': False,
        'physical_warnings': []
    }
    
    # Track current occupancy state
    node_occupancy = {}
    for node, data in finder.migration_graph.nodes(data=True):
        node_occupancy[node] = data['site_type']
    
    # Track which nodes are currently occupied by oxygen atoms
    occupied_nodes = set()
    for o_idx in finder.oxygen_indices:
        o_coords = finder.structure[o_idx].frac_coords
        for node, data in finder.migration_graph.nodes(data=True):
            if np.allclose(data['coords'], o_coords, atol=0.01):
                occupied_nodes.add(node)
                break
    
    # Analyze each hop in the pathway
    node_sequence = pathway['node_sequence']
    moving_oxygen_node = node_sequence[0] if node_sequence else None
    
    for i in range(len(node_sequence)):
        node = node_sequence[i]
        node_data = finder.migration_graph.nodes[node]
        
        # Determine actual site type based on current occupancy
        if i == 0:
            current_site_type = node_data['site_type']
        else:
            # Check if this node is currently occupied
            if node in occupied_nodes and node != moving_oxygen_node:
                current_site_type = 'ERROR_occupied_by_other_oxygen'
                details['physical_warnings'].append(
                    f"Step {i}: Cannot hop to site occupied by another oxygen at {node_data['coords']}"
                )
            else:
                current_site_type = node_data['site_type']
        
        step_info = {
            'node_index': node,
            'coords': node_data['coords'],
            'site_type': current_site_type,
            'original_site_type': node_data['site_type'],
            'cart_coords': finder.structure.lattice.get_cartesian_coords(node_data['coords'])
        }
        
        # Check if this involves a vacancy
        if 'vacancy' in node_data['site_type']:
            details['vacancy_involvement'] = True
        
        # For hops (not the starting position)
        if i > 0:
            prev_node = node_sequence[i-1]
            if finder.migration_graph.has_edge(prev_node, node):
                edge_data = finder.migration_graph[prev_node][node]
                step_info['hop_distance'] = edge_data['distance']
                step_info['hop_score'] = edge_data.get('score', 0)
                step_info['energy_barrier'] = edge_data.get('energy_barrier', 1.0)
                step_info['hop_type'] = edge_data.get('hop_type', 'unknown')
                
                # Update occupancy tracking
                if moving_oxygen_node in occupied_nodes:
                    occupied_nodes.remove(moving_oxygen_node)
                occupied_nodes.add(node)
                moving_oxygen_node = node
        
        details['steps'].append(step_info)
    
    # Calculate score breakdown with better categorization
    total_score = 0
    distance_bonus = 0
    vacancy_bonus = 0
    site_type_bonus = 0
    base_hop_scores = 0
    
    for i in range(1, len(node_sequence)):
        if finder.migration_graph.has_edge(node_sequence[i-1], node_sequence[i]):
            edge_data = finder.migration_graph[node_sequence[i-1]][node_sequence[i]]
            hop_score = edge_data.get('score', 0)
            total_score += hop_score
            
            # Base score per hop (typically 10)
            base_hop_scores += 10
            
            # Distance bonus
            distance = edge_data['distance']
            if 2.0 <= distance <= 3.0:
                distance_bonus += 5
            elif 1.8 <= distance < 2.0 or 3.0 < distance <= 3.5:
                distance_bonus += 2
            
            # Hop type bonuses
            hop_type = edge_data.get('hop_type', '')
            if 'oxygen_vacancy' in hop_type:
                vacancy_bonus += 15  # Primary vacancy bonus
            elif 'vacancy' in hop_type:
                vacancy_bonus += 10  # Other vacancy types
                
            if 'cation_influenced' in hop_type:
                site_type_bonus += 8
            elif 'alternative_site' in hop_type:
                site_type_bonus += 5
    
    details['score_breakdown'] = {
        'total': total_score,
        'base_hop_scores': base_hop_scores,
        'distance_contribution': distance_bonus,
        'vacancy_contribution': vacancy_bonus,
        'site_type_contribution': site_type_bonus,
        'other_contributions': max(0, total_score - base_hop_scores - distance_bonus - vacancy_bonus - site_type_bonus)
    }
    
    return details


def get_detailed_concerted_info(finder, movement):
    """Extract detailed information about a concerted movement"""
    details = {
        'atoms_involved': [],
        'movement_sequence': [],
        'score_factors': {},
        'vacancy_involvement': False
    }
    
    movement_type = movement['movement_type']
    
    if movement_type == 'swaps':
        # Direct swap between two oxygen atoms
        o1_idx = movement['oxygen1']
        o2_idx = movement['oxygen2']
        
        o1_coords = finder.structure[o1_idx].frac_coords
        o2_coords = finder.structure[o2_idx].frac_coords
        
        details['atoms_involved'] = [
            {'atom': f'O{o1_idx}', 'start': o1_coords, 'end': o2_coords},
            {'atom': f'O{o2_idx}', 'start': o2_coords, 'end': o1_coords}
        ]
        
        details['movement_sequence'] = [
            f"O{o1_idx} at {o1_coords} swaps with O{o2_idx} at {o2_coords}",
            f"Distance between atoms: {movement['distance']:.3f} Å"
        ]
        
        # Score factors
        score = movement.get('score', 0)
        details['score_factors'] = {
            'base_swap_score': 20,
            'distance_factor': f"Distance {movement['distance']:.2f} Å",
            'distance_penalty': -5 if movement['distance'] > 3.5 else 0,
            'distance_bonus': 10 if 2.0 <= movement['distance'] <= 3.0 else 0,
            'coordination_preservation': 5,  # Swaps preserve coordination
            'total_score': score
        }
        
    elif movement_type == 'rings':
        # Ring exchange
        cycle_nodes = movement['cycle_nodes']
        details['movement_sequence'].append(f"Ring exchange involving {len(cycle_nodes)} sites:")
        
        for i, node in enumerate(cycle_nodes):
            node_data = finder.migration_graph.nodes[node]
            next_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
            next_data = finder.migration_graph.nodes[next_node]
            
            if 'vacancy' in node_data['site_type']:
                details['vacancy_involvement'] = True
                details['movement_sequence'].append(
                    f"  Site {i+1}: Vacancy at {node_data['coords']}"
                )
            else:
                # Find corresponding oxygen
                for o_idx in finder.oxygen_indices:
                    if np.allclose(finder.structure[o_idx].frac_coords, node_data['coords'], atol=0.01):
                        details['atoms_involved'].append({
                            'atom': f'O{o_idx}',
                            'start': node_data['coords'],
                            'end': next_data['coords']
                        })
                        details['movement_sequence'].append(
                            f"  Site {i+1}: O{o_idx} moves from {node_data['coords']} to {next_data['coords']}"
                        )
                        break
        
        # Score factors for rings
        details['score_factors'] = {
            'base_ring_score': 25 - len(cycle_nodes) * 2,
            'connectivity_bonus': movement.get('score', 0) - (25 - len(cycle_nodes) * 2),
            'size_factor': f"Smaller rings preferred ({len(cycle_nodes)}-member ring)",
            'total_score': movement.get('score', 0)
        }
        
    elif movement_type == 'chains':
        # Chain movement
        chain_nodes = movement['chain_nodes']
        details['movement_sequence'].append(f"Chain movement with {len(chain_nodes)} sites:")
        
        for i in range(len(chain_nodes) - 1):
            node = chain_nodes[i]
            next_node = chain_nodes[i + 1]
            node_data = finder.migration_graph.nodes[node]
            next_data = finder.migration_graph.nodes[next_node]
            
            if 'vacancy' in next_data['site_type']:
                details['vacancy_involvement'] = True
            
            # Find oxygen atom at this position
            for o_idx in finder.oxygen_indices:
                if np.allclose(finder.structure[o_idx].frac_coords, node_data['coords'], atol=0.01):
                    details['atoms_involved'].append({
                        'atom': f'O{o_idx}',
                        'start': node_data['coords'],
                        'end': next_data['coords']
                    })
                    details['movement_sequence'].append(
                        f"  Step {i+1}: O{o_idx} moves from {node_data['coords']} to {next_data['coords']} ({next_data['site_type']})"
                    )
                    break
        
        # Score factors for chains
        details['score_factors'] = {
            'base_chain_score': 15 + len(chain_nodes) * 2,
            'length_factor': f"Longer chains can transport further ({len(chain_nodes)} sites)",
            'connectivity_score': movement.get('score', 0) - (15 + len(chain_nodes) * 2),
            'total_score': movement.get('score', 0)
        }
    
    return details


def find_vacancy_involved_pathways(finder):
    """Find pathways that involve vacancies"""
    vacancy_pathways = {
        'individual': [],
        'concerted': []
    }
    
    # Find individual pathways with vacancies
    for i, pathway in enumerate(finder.migration_pathways):
        details = get_detailed_pathway_info(finder, pathway)
        if details['vacancy_involvement']:
            vacancy_pathways['individual'].append((i, pathway))
    
    # Find concerted movements with vacancies
    if hasattr(finder, 'concerted_pathways'):
        for i, movement in enumerate(finder.concerted_pathways['all']):
            details = get_detailed_concerted_info(finder, movement)
            if details['vacancy_involvement']:
                vacancy_pathways['concerted'].append((i, movement))
    
    return vacancy_pathways


def write_detailed_pathway_summary(finder, output_dir):
    """Write detailed pathway summary with hop-by-hop information and validation"""
    pathway_file = os.path.join(output_dir, "pathway_summary_detailed.txt")
    
    with open(pathway_file, "w") as f:
        f.write("DETAILED OXYGEN MIGRATION PATHWAY ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis timestamp: {datetime.now()}\n")
        f.write(f"Total oxygen atoms: {len(finder.oxygen_indices)}\n")
        f.write(f"Total vacancies: {sum(len(v) for v in finder.all_vacancies.values())}\n")
        f.write("\n")
        
        # Scoring explanation
        f.write("SCORING SYSTEM EXPLANATION:\n")
        f.write("-" * 40 + "\n")
        f.write("Each pathway is scored based on:\n")
        f.write("  - Base hop score: 10 points per hop (connectivity reward)\n")
        f.write("  - Distance bonus: +5 for optimal distances (2.0-3.0 Å)\n")
        f.write("  - Vacancy bonus: +15 for oxygen vacancies, +10 for other vacancies\n")
        f.write("  - Site type bonus: +8 for cation-influenced sites, +5 for alternative sites\n")
        f.write("  - Other factors: Network connectivity, symmetry considerations\n")
        f.write("\n")
        
        # Vacancy breakdown
        f.write("Vacancy Distribution:\n")
        f.write("-" * 40 + "\n")
        for vac_type, vacancies in finder.all_vacancies.items():
            f.write(f"  {vac_type}: {len(vacancies)}\n")
            if len(vacancies) > 0 and len(vacancies) <= 5:
                for v in vacancies[:5]:
                    f.write(f"    - Position: {v['coords']}, Expected: {v.get('expected_species', 'Unknown')}\n")
        f.write("\n")
        
        # Individual pathways
        f.write("\n" + "=" * 80 + "\n")
        f.write("INDIVIDUAL OXYGEN MIGRATION PATHWAYS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, pathway in enumerate(finder.migration_pathways[:10]):
            details = get_detailed_pathway_info(finder, pathway)
            
            f.write(f"\nPATHWAY {i+1}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Oxygen atom: O{pathway['oxygen_index']}\n")
            f.write(f"Total hops: {pathway['pathway_length'] - 1}\n")
            f.write(f"Total distance: {pathway['total_distance']:.3f} Å\n")
            f.write(f"Total score: {pathway['total_score']:.1f}\n")
            f.write(f"Involves vacancy: {'YES' if details['vacancy_involvement'] else 'NO'}\n")
            
            # Physical warnings
            if details['physical_warnings']:
                f.write(f"⚠️  WARNINGS: \n")
                for warning in details['physical_warnings']:
                    f.write(f"   - {warning}\n")
            
            # Score breakdown with explanations
            f.write("\nScore Composition:\n")
            sb = details['score_breakdown']
            f.write(f"  Base hop scores: {sb['base_hop_scores']:.0f} ({pathway['pathway_length']-1} hops × 10 points)\n")
            f.write(f"  Distance bonuses: {sb['distance_contribution']:.0f} (optimal hop distances)\n")
            f.write(f"  Vacancy bonuses: {sb['vacancy_contribution']:.0f} (vacancy site involvement)\n")
            f.write(f"  Site type bonuses: {sb['site_type_contribution']:.0f} (special site types)\n")
            f.write(f"  Other factors: {sb['other_contributions']:.0f}\n")
            f.write(f"  TOTAL SCORE: {sb['total']:.0f}\n")
            
            # Step-by-step details
            f.write("\nMigration Steps:\n")
            for j, step in enumerate(details['steps']):
                if j == 0:
                    f.write(f"  Start: {step['site_type']} at {step['coords']} ")
                    f.write(f"(x={step['cart_coords'][0]:.3f}, y={step['cart_coords'][1]:.3f}, z={step['cart_coords'][2]:.3f} Å)\n")
                else:
                    site_type_display = step['site_type']
                    if 'ERROR' in site_type_display:
                        site_type_display = f"⚠️  {site_type_display}"
                    
                    f.write(f"  Hop {j}: → {site_type_display} at {step['coords']} ")
                    f.write(f"(x={step['cart_coords'][0]:.3f}, y={step['cart_coords'][1]:.3f}, z={step['cart_coords'][2]:.3f} Å)\n")
                    f.write(f"         Distance: {step.get('hop_distance', 0):.3f} Å, ")
                    f.write(f"Score: {step.get('hop_score', 0):.1f}, ")
                    f.write(f"Barrier: {step.get('energy_barrier', 0):.2f} eV\n")
                    f.write(f"         Type: {step.get('hop_type', 'unknown')}\n")
                    if step['site_type'] != step['original_site_type']:
                        f.write(f"         Note: Original site type was {step['original_site_type']}\n")
            
            f.write("\nWhy this pathway scores well:\n")
            if sb['vacancy_contribution'] > 0:
                f.write("  ✓ Direct involvement of vacancy sites enables oxygen movement\n")
            if sb['distance_contribution'] > 0:
                f.write("  ✓ Optimal hop distances (2.0-3.0 Å) balance energy barriers and probability\n")
            if sb['site_type_contribution'] > 0:
                f.write("  ✓ Enhanced by cationic vacancies reducing electrostatic barriers\n")
            if sb['base_hop_scores'] >= 30:
                f.write("  ✓ Multiple connected hops allow long-range transport\n")
            if details['physical_warnings']:
                f.write("  ✗ Physical consistency issues detected - pathway may not be realistic\n")
        
        # Concerted movements
        if hasattr(finder, 'concerted_pathways'):
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("CONCERTED OXYGEN MOVEMENTS\n")
            f.write("=" * 80 + "\n")
            
            f.write("\nCONCERTED MOVEMENT SCORING:\n")
            f.write("-" * 40 + "\n")
            f.write("  Direct swaps: Base 20 points, +10 for distances 2-3 Å, -5 for >3.5 Å\n")
            f.write("  Ring exchanges: Base (25 - 2×ring_size), connectivity bonuses\n")
            f.write("  Chain movements: Base (15 + 2×chain_length), connectivity bonuses\n")
            f.write("\n")
            
            for i, movement in enumerate(finder.concerted_pathways['all'][:10]):
                details = get_detailed_concerted_info(finder, movement)
                
                f.write(f"\nCONCERTED MOVEMENT {i+1}:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Type: {movement['movement_type']}\n")
                f.write(f"Score: {movement.get('score', 0):.1f}\n")
                f.write(f"Involves vacancy: {'YES' if details['vacancy_involvement'] else 'NO'}\n")
                
                # Score factors with explanations
                f.write("\nScore Factors:\n")
                for key, value in details['score_factors'].items():
                    if isinstance(value, str):
                        f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {key}: {value:.1f}\n")
                
                # Movement details
                f.write("\nMovement Details:\n")
                for seq in details['movement_sequence']:
                    f.write(f"  {seq}\n")
                
                # Atoms involved
                if details['atoms_involved']:
                    f.write("\nAtoms Involved:\n")
                    for atom_info in details['atoms_involved']:
                        start = atom_info['start']
                        end = atom_info['end']
                        f.write(f"  {atom_info['atom']}: [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}] → ")
                        f.write(f"[{end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f}]\n")
                
                f.write("\nWhy this movement is favorable:\n")
                if movement['movement_type'] == 'swaps':
                    f.write("  ✓ Direct exchange minimizes structural disruption\n")
                    if movement['distance'] <= 3.0:
                        f.write("  ✓ Short distance (≤3.0 Å) allows easy exchange\n")
                    f.write("  ✓ Simultaneous movement preserves local stoichiometry\n")
                elif movement['movement_type'] == 'rings':
                    f.write("  ✓ Cyclic exchange preserves local coordination\n")
                    if len(movement['cycle_nodes']) == 3:
                        f.write("  ✓ 3-member rings have minimal strain\n")
                    elif len(movement['cycle_nodes']) == 4:
                        f.write("  ✓ 4-member rings balance strain and flexibility\n")
                    if details['vacancy_involvement']:
                        f.write("  ✓ Vacancy provides free space for rotation\n")
                    f.write("  ✓ All atoms move simultaneously, no intermediate high-energy states\n")
                elif movement['movement_type'] == 'chains':
                    f.write("  ✓ Sequential movement allows long-range transport\n")
                    if details['vacancy_involvement']:
                        f.write("  ✓ Vacancy at chain end enables movement without displacement\n")
                    f.write("  ✓ Domino effect reduces overall activation barrier\n")
        
        # Summary statistics
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        
        # Count pathways with issues
        pathways_with_warnings = 0
        for pathway in finder.migration_pathways[:10]:
            details = get_detailed_pathway_info(finder, pathway)
            if details['physical_warnings']:
                pathways_with_warnings += 1
        
        f.write(f"\nPathway Quality:\n")
        f.write(f"  Total pathways analyzed: {min(10, len(finder.migration_pathways))}\n")
        f.write(f"  Pathways with physical warnings: {pathways_with_warnings}\n")
        f.write(f"  Pathways involving vacancies: {len([p for p in finder.migration_pathways[:10] if get_detailed_pathway_info(finder, p)['vacancy_involvement']])}\n")
        
        if hasattr(finder, 'concerted_pathways'):
            f.write(f"\nConcerted Movements:\n")
            f.write(f"  Total movements analyzed: {min(10, len(finder.concerted_pathways['all']))}\n")
            f.write(f"  Movements involving vacancies: {len([m for m in finder.concerted_pathways['all'][:10] if get_detailed_concerted_info(finder, m)['vacancy_involvement']])}\n")
    
    print(f"Detailed pathway summary saved to: {pathway_file}")
    return pathway_file
def generate_structures_with_vacancy_pathways(finder, output_dir, num_pathways=10):
    """Generate structures for top N pathways of each type in organized folders"""
    struct_dir = os.path.join(output_dir, "structures")
    if not os.path.exists(struct_dir):
        os.makedirs(struct_dir)
    
    # Find vacancy-involved pathways
    vacancy_pathways = find_vacancy_involved_pathways(finder)
    
    print("\nGenerating migration structures...")
    print(f"Found {len(vacancy_pathways['individual'])} individual pathways with vacancies")
    print(f"Found {len(vacancy_pathways['concerted'])} concerted movements with vacancies")
    
    # 1. Generate top individual pathways
    individual_dir = os.path.join(struct_dir, "individual_pathways")
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)
    
    print(f"\n=== Generating top {num_pathways} individual pathways ===")
    num_to_generate = min(num_pathways, len(finder.migration_pathways))
    
    for i in range(num_to_generate):
        pathway = finder.migration_pathways[i]
        details = get_detailed_pathway_info(finder, pathway)
        
        # Create descriptive folder name
        folder_name = f"pathway_{i+1:02d}_O{pathway['oxygen_index']}"
        if details['vacancy_involvement']:
            folder_name += "_vacancy"
        
        pathway_dir = os.path.join(individual_dir, folder_name)
        if not os.path.exists(pathway_dir):
            os.makedirs(pathway_dir)
        
        print(f"\n{i+1}. Generating pathway for O{pathway['oxygen_index']} (score: {pathway['total_score']:.1f})")
        structures = finder.execute_migration_pathway(
            pathway_index=i,
            output_prefix=os.path.join(pathway_dir, "step")
        )
    
    # 2. Generate top vacancy-involved individual pathways (if different from top 10)
    if vacancy_pathways['individual']:
        vacancy_individual_dir = os.path.join(struct_dir, "vacancy_individual_pathways")
        if not os.path.exists(vacancy_individual_dir):
            os.makedirs(vacancy_individual_dir)
        
        print(f"\n=== Generating top vacancy-involved individual pathways ===")
        generated_indices = set(range(num_to_generate))  # Already generated in top 10
        vacancy_count = 0
        
        for vac_idx, vac_pathway in vacancy_pathways['individual']:
            if vacancy_count >= num_pathways:
                break
                
            if vac_idx not in generated_indices:
                folder_name = f"vacancy_pathway_{vacancy_count+1:02d}_O{vac_pathway['oxygen_index']}"
                pathway_dir = os.path.join(vacancy_individual_dir, folder_name)
                if not os.path.exists(pathway_dir):
                    os.makedirs(pathway_dir)
                
                print(f"\n{vacancy_count+1}. Generating vacancy pathway for O{vac_pathway['oxygen_index']} (score: {vac_pathway['total_score']:.1f})")
                structures = finder.execute_migration_pathway(
                    pathway_index=vac_idx,
                    output_prefix=os.path.join(pathway_dir, "step")
                )
                vacancy_count += 1
    
    # 3. Generate top concerted movements
    if hasattr(finder, 'concerted_pathways') and finder.concerted_pathways['all']:
        concerted_dir = os.path.join(struct_dir, "concerted_movements")
        if not os.path.exists(concerted_dir):
            os.makedirs(concerted_dir)
        
        print(f"\n=== Generating top {num_pathways} concerted movements ===")
        num_concerted = min(num_pathways, len(finder.concerted_pathways['all']))
        
        for i in range(num_concerted):
            movement = finder.concerted_pathways['all'][i]
            details = get_detailed_concerted_info(finder, movement)
            
            # Create descriptive folder name with atoms involved
            if movement['movement_type'] == 'swaps':
                folder_name = f"concerted_{i+1:02d}_swap_O{movement['oxygen1']}_O{movement['oxygen2']}"
            elif movement['movement_type'] == 'rings':
                # Get oxygen indices involved in the ring
                oxygen_indices = []
                for node in movement['cycle_nodes']:
                    node_data = finder.migration_graph.nodes[node]
                    if node_data['site_type'] == 'occupied_oxygen':
                        # Find corresponding oxygen index
                        for o_idx in finder.oxygen_indices:
                            if np.allclose(finder.structure[o_idx].frac_coords, node_data['coords'], atol=0.01):
                                oxygen_indices.append(o_idx)
                                break
                oxygen_str = "_".join([f"O{idx}" for idx in sorted(oxygen_indices)])
                folder_name = f"concerted_{i+1:02d}_ring_{len(movement['cycle_nodes'])}member_{oxygen_str}"
            elif movement['movement_type'] == 'chains':
                # Get oxygen indices in the chain
                oxygen_indices = []
                for node in movement['chain_nodes'][:-1]:  # Exclude last node if it's vacancy
                    node_data = finder.migration_graph.nodes[node]
                    if node_data['site_type'] == 'occupied_oxygen':
                        for o_idx in finder.oxygen_indices:
                            if np.allclose(finder.structure[o_idx].frac_coords, node_data['coords'], atol=0.01):
                                oxygen_indices.append(o_idx)
                                break
                oxygen_str = "_".join([f"O{idx}" for idx in oxygen_indices])
                folder_name = f"concerted_{i+1:02d}_chain_{oxygen_str}"
            else:
                folder_name = f"concerted_{i+1:02d}_{movement['movement_type']}"
            
            if details['vacancy_involvement']:
                folder_name += "_vacancy"
            
            movement_dir = os.path.join(concerted_dir, folder_name)
            if not os.path.exists(movement_dir):
                os.makedirs(movement_dir)
            
            print(f"\n{i+1}. Generating {movement['movement_type']} (score: {movement.get('score', 0):.1f})")
            structures = finder.execute_concerted_movement(
                movement_index=i,
                output_prefix=os.path.join(movement_dir, "step")
            )
    
    # 4. Generate top vacancy-involved concerted movements (if different)
    if vacancy_pathways['concerted']:
        vacancy_concerted_dir = os.path.join(struct_dir, "vacancy_concerted_movements")
        if not os.path.exists(vacancy_concerted_dir):
            os.makedirs(vacancy_concerted_dir)
        
        print(f"\n=== Generating top vacancy-involved concerted movements ===")
        generated_concerted_indices = set(range(num_concerted))
        vacancy_concerted_count = 0
        
        for vac_idx, vac_movement in vacancy_pathways['concerted']:
            if vacancy_concerted_count >= num_pathways:
                break
                
            if vac_idx not in generated_concerted_indices:
                # Create descriptive folder name with atoms involved
                if vac_movement['movement_type'] == 'swaps':
                    folder_name = f"vacancy_concerted_{vacancy_concerted_count+1:02d}_swap_O{vac_movement['oxygen1']}_O{vac_movement['oxygen2']}"
                elif vac_movement['movement_type'] == 'rings':
                    # Get oxygen indices involved in the ring
                    oxygen_indices = []
                    for node in vac_movement['cycle_nodes']:
                        node_data = finder.migration_graph.nodes[node]
                        if node_data['site_type'] == 'occupied_oxygen':
                            for o_idx in finder.oxygen_indices:
                                if np.allclose(finder.structure[o_idx].frac_coords, node_data['coords'], atol=0.01):
                                    oxygen_indices.append(o_idx)
                                    break
                    oxygen_str = "_".join([f"O{idx}" for idx in sorted(oxygen_indices)])
                    folder_name = f"vacancy_concerted_{vacancy_concerted_count+1:02d}_ring_{len(vac_movement['cycle_nodes'])}member_{oxygen_str}"
                elif vac_movement['movement_type'] == 'chains':
                    # Get oxygen indices in the chain
                    oxygen_indices = []
                    for node in vac_movement['chain_nodes'][:-1]:  # Exclude last node if it's vacancy
                        node_data = finder.migration_graph.nodes[node]
                        if node_data['site_type'] == 'occupied_oxygen':
                            for o_idx in finder.oxygen_indices:
                                if np.allclose(finder.structure[o_idx].frac_coords, node_data['coords'], atol=0.01):
                                    oxygen_indices.append(o_idx)
                                    break
                    oxygen_str = "_".join([f"O{idx}" for idx in oxygen_indices])
                    folder_name = f"vacancy_concerted_{vacancy_concerted_count+1:02d}_chain_{oxygen_str}"
                else:
                    folder_name = f"vacancy_concerted_{vacancy_concerted_count+1:02d}_{vac_movement['movement_type']}"
                movement_dir = os.path.join(vacancy_concerted_dir, folder_name)
                if not os.path.exists(movement_dir):
                    os.makedirs(movement_dir)
                
                print(f"\n{vacancy_concerted_count+1}. Generating vacancy {vac_movement['movement_type']} (score: {vac_movement.get('score', 0):.1f})")
                structures = finder.execute_concerted_movement(
                    movement_index=vac_idx,
                    output_prefix=os.path.join(movement_dir, "step")
                )
                vacancy_concerted_count += 1
    
    # Print summary
    print("\n" + "="*70)
    print("STRUCTURE GENERATION SUMMARY")
    print("="*70)
    print(f"Generated structures in: {struct_dir}/")
    print("\nFolder structure:")
    print("  individual_pathways/       - Top 10 individual oxygen migration pathways")
    if vacancy_pathways['individual']:
        print("  vacancy_individual_pathways/ - Additional vacancy-involved pathways")
    if hasattr(finder, 'concerted_pathways') and finder.concerted_pathways['all']:
        print("  concerted_movements/       - Top 10 concerted movements")
    if vacancy_pathways['concerted']:
        print("  vacancy_concerted_movements/ - Additional vacancy-involved concerted movements")
    
    return struct_dir

def analyze_oxygen_migration(structure_file="POSCAR", 
                                    reference_file=None,
                                    output_dir=None,
                                    max_hop_distance=4.0,
                                    max_pathway_length=6,
                                    create_animations=True,
                                    verbose=True):
    """Enhanced oxygen migration analysis with detailed pathway information"""
    
    # Create output directory with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"oxygen_migration_analysis_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save analysis parameters
    with open(os.path.join(output_dir, "analysis_parameters.txt"), "w") as f:
        f.write(f"Structure file: {structure_file}\n")
        f.write(f"Reference file: {reference_file}\n")
        f.write(f"Max hop distance: {max_hop_distance} Å\n")
        f.write(f"Max pathway length: {max_pathway_length}\n")
        f.write(f"Analysis timestamp: {datetime.now()}\n")
    
    print("="*70)
    print("ENHANCED OXYGEN MIGRATION ANALYSIS")
    print("="*70)
    
    # Step 1: Find migration pathways
    print("\n1. Finding oxygen migration pathways...")
    print("-"*50)
    
    finder = UniversalVacancyOxygenMigrationFinder(
        structure_file=structure_file,
        reference_structure_file=reference_file,
        max_hop_distance=max_hop_distance,
        max_pathway_length=max_pathway_length
    )
    
    # Step 2: Write detailed pathway summary
    print("\n2. Writing detailed pathway analysis...")
    print("-"*50)
    detailed_summary = write_detailed_pathway_summary(finder, output_dir)
    
    # Step 3: Generate structures with vacancy pathways
    print("\n3. Generating migration structures (including vacancy pathways)...")
    print("-"*50)
    struct_dir = generate_structures_with_vacancy_pathways(finder, output_dir)
    
    # Step 4: Create visualizations
    print("\n4. Creating visualizations...")
    print("-"*50)
    
    plot_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Initialize visualizer
    viz = OxygenPathwayVisualizer(structure_file, finder)
    
    # Create visualizations (same as before but with enhanced pathway info)
    if finder.migration_pathways:
        # Find vacancy pathways for visualization
        vacancy_pathways = find_vacancy_involved_pathways(finder)
        
        # Create main pathway plot including vacancy pathways
        pathways_to_plot = finder.migration_pathways[:3]
        if vacancy_pathways['individual']:
            vac_idx, vac_pathway = vacancy_pathways['individual'][0]
            if vac_pathway not in pathways_to_plot:
                pathways_to_plot.append(vac_pathway)
        
        fig, ax = viz.create_pathway_plot(
            pathways=pathways_to_plot,
            concerted_movements=finder.concerted_pathways['all'][:3] if hasattr(finder, 'concerted_pathways') else None,
            show_atoms=True,
            show_vacancies=True,
            show_lattice=True,
            save_path=os.path.join(plot_dir, "main_pathways_with_vacancies.png")
        )
    
    # Create animations if requested
    if create_animations:
        print("\n5. Creating animations...")
        print("-"*50)
        
        anim_dir = os.path.join(output_dir, "animations")
        if not os.path.exists(anim_dir):
            os.makedirs(anim_dir)
        
        # Animate top pathways including vacancy ones
        vacancy_pathways = find_vacancy_involved_pathways(finder)
        
        # Animate best overall pathway
        if finder.migration_pathways:
            print("Creating animation for best overall pathway...")
            viz.create_pathway_animation(
                finder.migration_pathways[0],
                output_file=os.path.join(anim_dir, "best_overall_pathway.gif"),
                fps=5,
                n_frames=20
            )
        
        # Animate best vacancy pathway
        if vacancy_pathways['individual']:
            vac_idx, vac_pathway = vacancy_pathways['individual'][0]
            print(f"Creating animation for best vacancy pathway (pathway {vac_idx + 1})...")
            viz.create_pathway_animation(
                vac_pathway,
                output_file=os.path.join(anim_dir, "best_vacancy_pathway.gif"),
                fps=5,
                n_frames=20
            )
    
    # Final summary
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey outputs:")
    print(f"  - {detailed_summary} (detailed pathway analysis)")
    print(f"  - {struct_dir}/ (includes vacancy-involved pathways)")
    print(f"  - {os.path.join(output_dir, 'visualizations/')} (plots)")
    if create_animations:
        print(f"  - {os.path.join(output_dir, 'animations/')} (includes vacancy pathway animations)")
    
    return finder, viz


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced oxygen migration analysis')
    parser.add_argument('structure', nargs='?', default='POSCAR',
                       help='Structure file (POSCAR or CIF format)')
    parser.add_argument('--reference', '-r', default=None,
                       help='Reference structure file without vacancies')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory name')
    parser.add_argument('--max-hop', '-d', type=float, default=4.0,
                       help='Maximum hop distance in Angstroms (default: 4.0)')
    parser.add_argument('--max-length', '-l', type=int, default=6,
                       help='Maximum pathway length in hops (default: 6)')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation creation')
    
    args = parser.parse_args()
    
    # Run enhanced analysis
    finder, viz = analyze_oxygen_migration(
        structure_file=args.structure,
        reference_file=args.reference,
        output_dir=args.output,
        max_hop_distance=args.max_hop,
        max_pathway_length=args.max_length,
        create_animations=not args.no_animation,
        verbose=True
    )
