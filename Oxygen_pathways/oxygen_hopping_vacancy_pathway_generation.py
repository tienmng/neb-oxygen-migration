#!/usr/bin/env python3
"""
Universal Vacancy-Mediated Oxygen Migration Finder
Handles oxygen migration through all types of vacancy networks (oxygen, cationic, anionic)
WITH CONCERTED MOVEMENT DETECTION
"""

import numpy as np
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.io.vasp import Poscar
from pymatgen.core import Element, Structure, PeriodicSite, Species
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffusion.neb.pathfinder import DistinctPathFinder
import itertools
from collections import defaultdict, deque
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class UniversalVacancyOxygenMigrationFinder:
    def __init__(self, structure_file="POSCAR", reference_structure_file=None, max_hop_distance=4.0, max_pathway_length=8):
        """
        Initialize universal vacancy-mediated oxygen migration finder
        
        Args:
            structure_file: Path to structure file (can contain any type of vacancy)
            reference_structure_file: Optional path to reference structure without vacancies
            max_hop_distance: Maximum distance for a single oxygen hop (Angstroms)
            max_pathway_length: Maximum number of hops in a migration pathway
        """
        self.structure_file = structure_file
        self.reference_structure_file = reference_structure_file
        self.max_hop_distance = max_hop_distance
        self.max_pathway_length = max_pathway_length
        
        # Universal vacancy parameters
        self.vacancy_params = {
            "vacancy_detection_radius": 2.5,    # Search radius for vacancy detection
            "min_hop_distance": 1.5,           # Minimum meaningful hop distance
            "max_hop_distance": 4.0,           # Maximum single hop distance
            "coordination_tolerance": 1,        # Tolerance for coordination changes
            "pathway_energy_cutoff": 50.0,     # Energy cutoff for viable pathways
            "clustering_tolerance": 1.5,       # Vacancy clustering tolerance
            "oxygen_radius": 1.4,              # Effective oxygen radius
            "cation_influence_radius": 4.0,    # How far cationic vacancies affect O migration
        }
        
        self.load_structures()
        self.detect_all_vacancies()
        self.analyze_vacancy_effects_on_oxygen()
        self.build_universal_migration_network()
        self.find_migration_pathways()
        self.find_concerted_movement_pathways()  # NEW: Add concerted movements
    
    def load_structures(self):
        """Load main structure and optional reference structure"""
        # Load main structure (may contain vacancies)
        if self.structure_file.endswith('.cif'):
            self.structure = Structure.from_file(self.structure_file)
        else:
            self.structure = Poscar.from_file(self.structure_file).structure
        
        # Load reference structure if provided
        if self.reference_structure_file:
            if self.reference_structure_file.endswith('.cif'):
                self.reference_structure = Structure.from_file(self.reference_structure_file)
            else:
                self.reference_structure = Poscar.from_file(self.reference_structure_file).structure
        else:
            # Try to generate reference structure from symmetry
            self.reference_structure = self.generate_reference_structure()
        
        # Find oxygen atoms in current structure
        self.oxygen_indices = []
        for i, site in enumerate(self.structure):
            if site.species_string == "O":
                self.oxygen_indices.append(i)
        
        if not self.oxygen_indices:
            raise ValueError("No oxygen atoms found in the structure!")
        
        print(f"Loaded structure: {self.structure.composition}")
        print(f"Reference structure: {self.reference_structure.composition}")
        print(f"Found {len(self.oxygen_indices)} oxygen atoms in current structure")
    
    def generate_reference_structure(self):
        """Generate a reference structure by filling potential vacancy sites"""
        print("Generating reference structure from symmetry...")
        
        # Use symmetry analysis
        sga = SpacegroupAnalyzer(self.structure, symprec=0.1)
        symmetric_structure = sga.get_symmetrized_structure()
        
        # Get conventional structure which often has fewer vacancies
        try:
            reference = sga.get_conventional_standard_structure()
            print(f"Generated reference from conventional structure")
            return reference
        except:
            print("Using original structure as reference")
            return self.structure.copy()
    
    # Utility function for consistent PBC distance calculation
    def calculate_pbc_distance(self, coords1, coords2):
        """
        Utility function for consistent PBC distance calculation
        
        Args:
            coords1, coords2: fractional coordinates (numpy arrays)
        
        Returns:
            float: minimum distance considering periodic images
        """
        return self.structure.lattice.get_distance_and_image(coords1, coords2)[0]

    def detect_all_vacancies(self):
        """Detect all types of vacancies (oxygen, cationic, anionic)"""
        print("\nDetecting all types of vacancies...")
        
        self.all_vacancies = {
            'oxygen_vacancies': [],
            'cationic_vacancies': [],
            'anionic_vacancies': [],
            'unknown_vacancies': []
        }
        
        # Method 1: Compare with reference structure
        if self.reference_structure.num_sites > self.structure.num_sites:
            self.detect_vacancies_by_comparison()
        
        # Method 2: Use Voronoi analysis for additional sites
        self.detect_vacancies_by_voronoi()
        
        # Method 3: Detect based on coordination analysis
        self.detect_vacancies_by_coordination()
        
        # Summarize findings
        total_vacancies = sum(len(v) for v in self.all_vacancies.values())
        print(f"\nVacancy Detection Summary:")
        print(f"  Oxygen vacancies: {len(self.all_vacancies['oxygen_vacancies'])}")
        print(f"  Cationic vacancies: {len(self.all_vacancies['cationic_vacancies'])}")
        print(f"  Anionic vacancies: {len(self.all_vacancies['anionic_vacancies'])}")
        print(f"  Unknown vacancies: {len(self.all_vacancies['unknown_vacancies'])}")
        print(f"  Total vacancies: {total_vacancies}")
    
    def detect_vacancies_by_comparison(self):
        """Fixed vacancy detection with proper PBC distance calculation"""
        print("Detecting vacancies by comparison with reference structure...")
        
        # For each site in reference, check if occupied in current structure
        for ref_idx, ref_site in enumerate(self.reference_structure):
            is_occupied = False
            
            for curr_idx, curr_site in enumerate(self.structure):
                # FIX: Use proper distance calculation
                distance = self.structure.lattice.get_distance_and_image(
                    ref_site.frac_coords, curr_site.frac_coords)[0]
                
                if distance < self.vacancy_params["clustering_tolerance"]:
                    is_occupied = True
                    break
            
            if not is_occupied:
                vacancy_info = {
                    'coords': ref_site.frac_coords,
                    'expected_species': ref_site.species_string,
                    'detection_method': 'reference_comparison',
                    'reference_index': ref_idx
                }
                
                # Classify vacancy type
                if ref_site.species_string == "O":
                    self.all_vacancies['oxygen_vacancies'].append(vacancy_info)
                else:
                    # Get the element to check if it's a metal
                    element_symbol = ref_site.species_string.split('+')[0].split('-')[0]
                    element = Element(element_symbol)
                    if element.is_metal:
                        self.all_vacancies['cationic_vacancies'].append(vacancy_info)
                    else:
                        self.all_vacancies['anionic_vacancies'].append(vacancy_info)
    def detect_vacancies_by_voronoi(self):
        """Use Voronoi analysis to find additional vacancy sites"""
        print("Detecting vacancies using Voronoi analysis...")
        
        try:
            generator = VoronoiInterstitialGenerator(
                clustering_tol=self.vacancy_params["clustering_tolerance"],
                min_dist=1.5,
            )
            
            # Generate interstitial sites
            for interstitial in generator.generate(self.structure, "H"):
                for site in interstitial.equivalent_sites[:2]:
                    # Check if this is truly vacant
                    is_vacant = True
                    for curr_site in self.structure:
                        distance = self.structure.lattice.get_distance_and_image(
                            site.frac_coords, curr_site.frac_coords)[0]
                        if distance < 1.0:  # Too close to existing atom
                            is_vacant = False
                            break
                    
                    if is_vacant:
                        vacancy_info = {
                            'coords': site.frac_coords,
                            'expected_species': 'Unknown',
                            'detection_method': 'voronoi',
                            'multiplicity': interstitial.multiplicity
                        }
                        self.all_vacancies['unknown_vacancies'].append(vacancy_info)
        
        except Exception as e:
            print(f"Voronoi analysis failed: {e}")
    
    def detect_vacancies_by_coordination(self):
        """Detect vacancies based on unusually low coordination environments"""
        print("Detecting vacancies from coordination analysis...")
        
        nn = CrystalNN()
        
        # Analyze each atom's coordination
        for i, site in enumerate(self.structure):
            try:
                neighbors = nn.get_nn_info(self.structure, i)
                coord_num = len(neighbors)
                
                # Check for unusually low coordination (potential nearby vacancy)
                expected_coord = self.get_expected_coordination(site.species_string)
                
                if coord_num < expected_coord - 2:  # Significantly undercoordinated
                    # Look for potential vacancy sites around this atom
                    vacancy_sites = self.find_vacancy_sites_around_atom(i, neighbors)
                    
                    for vacancy_coords in vacancy_sites:
                        vacancy_info = {
                            'coords': vacancy_coords,
                            'expected_species': 'Inferred',
                            'detection_method': 'coordination_analysis',
                            'associated_atom': i,
                            'coordination_deficiency': expected_coord - coord_num
                        }
                        self.all_vacancies['unknown_vacancies'].append(vacancy_info)
            
            except Exception as e:
                continue
    
    def get_expected_coordination(self, species_string):
        """Get expected coordination number for a species"""
        expected_coords = {
            'O': 4,   # Typical for oxygen
            'Li': 6,  'Na': 6,  'K': 8,   # Alkali metals
            'Mg': 6,  'Ca': 8,  'Sr': 8,  # Alkaline earth
            'Al': 6,  'Si': 4,  'P': 4,   # Common elements
            'Ti': 6,  'V': 6,   'Cr': 6,  # Transition metals
            'Mn': 6,  'Fe': 6,  'Co': 6,  'Ni': 6,
            'Cu': 4,  'Zn': 4, 'La': 6
        }
        
        # Extract element symbol
        element = species_string.split('+')[0].split('-')[0]
        return expected_coords.get(element, 6)  # Default to 6
    
    def find_vacancy_sites_around_atom(self, atom_index, neighbors):
        """Find potential vacancy sites around an undercoordinated atom"""
        atom_site = self.structure[atom_index]
        vacancy_sites = []
        
        # Calculate average neighbor distance
        if neighbors:
            avg_distance = np.mean([n['weight'] for n in neighbors])
            
            # Look for gaps in coordination sphere
            neighbor_vectors = []
            for neighbor in neighbors:
                vector = neighbor['site'].coords - atom_site.coords
                neighbor_vectors.append(vector / np.linalg.norm(vector))
            
            # Generate candidate positions based on typical coordination geometries
            if len(neighbors) < 6:  # Room for more neighbors
                # Add some candidate positions
                candidates = self.generate_coordination_candidates(atom_site.coords, avg_distance)
                
                for candidate in candidates:
                    # Check if this position is truly vacant
                    is_vacant = True
                    candidate_frac = self.structure.lattice.get_fractional_coords(candidate)
                    
                    for existing_site in self.structure:
                        distance = self.structure.lattice.get_distance_and_image(
                            candidate_frac, existing_site.frac_coords)[0]
                        if distance < 1.5:
                            is_vacant = False
                            break
                    
                    if is_vacant:
                        vacancy_sites.append(candidate_frac)
        
        return vacancy_sites
    
    def generate_coordination_candidates(self, center_coords, typical_distance):
        """Generate candidate coordination positions around a center"""
        candidates = []
        
        # Octahedral positions
        directions = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], 
            [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]
        
        for direction in directions:
            candidate = center_coords + np.array(direction) * typical_distance
            candidates.append(candidate)
        
        return candidates
    
    def analyze_vacancy_effects_on_oxygen(self):
        """Analyze how different types of vacancies affect oxygen migration"""
        print("\nAnalyzing vacancy effects on oxygen migration...")
        
        self.vacancy_oxygen_interactions = {}
        
        # Analyze each oxygen atom's environment with respect to all vacancies
        for o_idx in self.oxygen_indices:
            oxygen_site = self.structure[o_idx]
            interactions = {
                'nearby_oxygen_vacancies': [],
                'nearby_cationic_vacancies': [],
                'nearby_anionic_vacancies': [],
                'nearby_unknown_vacancies': [],
                'migration_enhancement': 0,
                'coordination_effects': {}
            }
            
            # Check proximity to different vacancy types
            for vacancy_type, vacancies in self.all_vacancies.items():
                for vacancy in vacancies:
                    distance = self.structure.lattice.get_distance_and_image(
                        oxygen_site.frac_coords, vacancy['coords'])[0]
                    
                    if distance <= self.vacancy_params["cation_influence_radius"]:
                        vacancy_effect = self.calculate_vacancy_effect_on_oxygen(
                            vacancy_type, vacancy, distance)
                        
                        interactions[f'nearby_{vacancy_type}'].append({
                            'vacancy': vacancy,
                            'distance': distance,
                            'effect': vacancy_effect
                        })
                        
                        interactions['migration_enhancement'] += vacancy_effect['enhancement_score']
            
            self.vacancy_oxygen_interactions[o_idx] = interactions
            
            print(f"O{o_idx}: Enhancement score = {interactions['migration_enhancement']:.1f}")
    
    def calculate_vacancy_effect_on_oxygen(self, vacancy_type, vacancy, distance):
        """Calculate how a specific vacancy affects oxygen migration"""
        effect = {
            'enhancement_score': 0,
            'pathway_modification': 'none',
            'energy_barrier_change': 0
        }
        
        if vacancy_type == 'oxygen_vacancies':
            # Oxygen vacancies directly provide migration sites
            if distance <= self.max_hop_distance:
                effect['enhancement_score'] = 20 / (distance + 0.1)
                effect['pathway_modification'] = 'direct_pathway'
        
        elif vacancy_type == 'cationic_vacancies':
            # Cationic vacancies reduce local electrostatic attraction
            if distance <= 3.0:
                effect['enhancement_score'] = 15 / (distance + 0.5)
                effect['pathway_modification'] = 'reduced_barriers'
                effect['energy_barrier_change'] = -0.2  # Lower barriers
        
        elif vacancy_type == 'anionic_vacancies':
            # Other anionic vacancies may create alternative pathways
            if distance <= 2.5:
                effect['enhancement_score'] = 10 / (distance + 0.5)
                effect['pathway_modification'] = 'alternative_sites'
        
        else:  # Unknown vacancies
            if distance <= 2.0:
                effect['enhancement_score'] = 5 / (distance + 0.5)
        
        return effect
    
    def build_universal_migration_network(self):
        """Build migration network considering all vacancy types"""
        print("\nBuilding universal migration network...")
        
        self.migration_graph = nx.Graph()
        
        # Add all potential migration sites
        migration_sites = []
        site_types = []
        
        # 1. Current oxygen positions
        for o_idx in self.oxygen_indices:
            migration_sites.append(self.structure[o_idx].frac_coords)
            site_types.append('occupied_oxygen')
        
        # 2. Oxygen vacancy sites (primary migration targets)
        for vacancy in self.all_vacancies['oxygen_vacancies']:
            migration_sites.append(vacancy['coords'])
            site_types.append('oxygen_vacancy')
        
        # 3. Sites influenced by cationic vacancies
        for vacancy in self.all_vacancies['cationic_vacancies']:
            influenced_sites = self.find_oxygen_sites_near_cationic_vacancy(vacancy)
            for site in influenced_sites:
                migration_sites.append(site)
                site_types.append('cation_influenced')
        
        # 4. Alternative sites from other vacancies
        for vacancy in self.all_vacancies['anionic_vacancies'] + self.all_vacancies['unknown_vacancies']:
            if self.is_suitable_for_oxygen(vacancy):
                migration_sites.append(vacancy['coords'])
                site_types.append('alternative_site')
        
        # Add nodes to graph
        for i, (site, site_type) in enumerate(zip(migration_sites, site_types)):
            self.migration_graph.add_node(i, coords=site, site_type=site_type)
        
        # Add edges between compatible sites
        for i in range(len(migration_sites)):
            for j in range(i + 1, len(migration_sites)):
                distance = self.structure.lattice.get_distance_and_image(
                    migration_sites[i], migration_sites[j])[0]
                
                if (self.vacancy_params["min_hop_distance"] <= distance <= 
                    self.vacancy_params["max_hop_distance"]):
                    
                    hop_viability = self.evaluate_universal_hop_viability(
                        migration_sites[i], migration_sites[j], 
                        site_types[i], site_types[j], distance)
                    
                    if hop_viability['score'] > 0:
                        self.migration_graph.add_edge(i, j, **hop_viability)
        
        print(f"Migration network: {self.migration_graph.number_of_nodes()} sites, "
              f"{self.migration_graph.number_of_edges()} connections")
        
        # Analyze network connectivity
        self.analyze_migration_network()
    
    def find_oxygen_sites_near_cationic_vacancy(self, cationic_vacancy):
        """Fixed method with proper PBC coordinate handling"""
        influenced_sites = []
        vacancy_coords = cationic_vacancy['coords']
        
        # Generate potential oxygen positions around the cationic vacancy
        directions = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
        
        for direction in directions:
            for distance_angstrom in [2.0, 2.5, 3.0]:  # Typical M-O distances
                # Convert Angstrom distance to fractional coordinates
                direction_cart = np.array(direction) * distance_angstrom
                direction_frac = self.structure.lattice.get_fractional_coords(direction_cart)
                
                # Add to vacancy position and wrap coordinates properly
                candidate_frac = vacancy_coords + direction_frac
                candidate_frac = candidate_frac % 1.0  # Wrap to [0,1) range
                
                # Check if site is reasonable using proper PBC distance
                if self.is_reasonable_oxygen_site_pbc(candidate_frac):
                    influenced_sites.append(candidate_frac)
        
        return influenced_sites
    
    def is_suitable_for_oxygen(self, vacancy):
        """Check if a vacancy site could accommodate oxygen"""
        # Check coordination environment that would be suitable for oxygen
        try:
            temp_structure = self.structure.copy()
            temp_structure.append(Element("O"), vacancy['coords'], coords_are_cartesian=False)
            
            nn = CrystalNN()
            neighbors = nn.get_nn_info(temp_structure, len(temp_structure)-1)
            
            # Oxygen typically wants 2-6 neighbors
            if 2 <= len(neighbors) <= 6:
                # Check if neighbor distances are reasonable for O
                avg_distance = np.mean([n['weight'] for n in neighbors])
                if 1.5 <= avg_distance <= 3.0:
                    return True
        except:
            pass
        
        return False
    

    def is_reasonable_oxygen_site_pbc(self, coords):
        """Check if coordinates represent a reasonable oxygen site with proper PBC"""
        # Ensure coordinates are in [0,1) range
        coords = np.array(coords) % 1.0
        
        # Check if too close to existing atoms using PBC distance
        for site in self.structure:
            distance = self.structure.lattice.get_distance_and_image(
                coords, site.frac_coords)[0]
            if distance < 1.2:  # Too close
                return False
        
        return True
    
    def evaluate_universal_hop_viability(self, site1, site2, type1, type2, distance):
        """Evaluate hop viability considering vacancy effects"""
        base_score = 10
        energy_barrier = 1.0  # Base energy barrier
        
        # Distance preferences
        if 2.0 <= distance <= 3.0:
            base_score += 5
        elif distance > 3.5:
            base_score -= 3
        
        # Site type preferences
        type_bonuses = {
            ('occupied_oxygen', 'oxygen_vacancy'): 15,      # Direct O to O vacancy
            ('occupied_oxygen', 'cation_influenced'): 10,   # Enhanced by cation vacancy
            ('occupied_oxygen', 'alternative_site'): 5,     # Alternative pathway
            ('oxygen_vacancy', 'oxygen_vacancy'): 12,       # Vacancy to vacancy
        }
        
        hop_key = (type1, type2) if type1 <= type2 else (type2, type1)
        base_score += type_bonuses.get(hop_key, 0)
        
        # Energy barrier modifications
        if 'cation_influenced' in [type1, type2]:
            energy_barrier *= 0.8  # Lower barrier due to cationic vacancy
        
        if 'oxygen_vacancy' in [type1, type2]:
            energy_barrier *= 0.9  # Slightly lower barrier
        
        return {
            'score': base_score,
            'distance': distance,
            'energy_barrier': energy_barrier,
            'hop_type': f"{type1}->{type2}"
        }
    
    def analyze_migration_network(self):
        """Analyze the migration network connectivity"""
        # Find connected components
        components = list(nx.connected_components(self.migration_graph))
        print(f"Found {len(components)} connected migration clusters")
        
        for i, component in enumerate(components):
            oxygen_nodes = 0
            vacancy_nodes = 0
            
            for node in component:
                node_data = self.migration_graph.nodes[node]
                if node_data['site_type'] == 'occupied_oxygen':
                    oxygen_nodes += 1
                elif 'vacancy' in node_data['site_type']:
                    vacancy_nodes += 1
            
            print(f"  Cluster {i+1}: {len(component)} sites ({oxygen_nodes} O atoms, {vacancy_nodes} vacancies)")
    
    def find_migration_pathways(self):
        """Find migration pathways through the universal network"""
        print("\nFinding universal migration pathways...")
        
        self.migration_pathways = []
        
        # Find pathways for each oxygen atom
        for o_idx in self.oxygen_indices:
            # Find corresponding node in migration graph
            oxygen_coords = self.structure[o_idx].frac_coords
            start_node = None
            
            for node, data in self.migration_graph.nodes(data=True):
                if (data['site_type'] == 'occupied_oxygen' and 
                    np.allclose(data['coords'], oxygen_coords, atol=0.01)):
                    start_node = node
                    break
            
            if start_node is not None:
                # Use modified pathway finding
                pathways = self.find_valid_pathways_from_node(start_node, o_idx)
                self.migration_pathways.extend(pathways)
        
        # Sort pathways by quality
        self.migration_pathways.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"Found {len(self.migration_pathways)} valid migration pathways")
        
        # Report top pathways
        for i, pathway in enumerate(self.migration_pathways[:5]):
            print(f"  Pathway {i+1}: O{pathway['oxygen_index']} → {len(pathway['node_sequence'])} hops, "
                f"score={pathway['total_score']:.1f}, types={pathway.get('pathway_types', 'mixed')}")


    def find_valid_pathways_from_node(self, start_node, oxygen_index):
        """Find only physically valid pathways from a node"""
        pathways = []
        
        # Get positions of all other oxygen atoms
        other_oxygen_positions = []
        for o_idx in self.oxygen_indices:
            if o_idx != oxygen_index:
                other_oxygen_positions.append(self.structure[o_idx].frac_coords)
        
        # Use BFS to explore pathways
        queue = deque([(start_node, [start_node], 0, 0)])
        visited_sequences = set()
        
        while queue:
            current_node, path, total_distance, total_score = queue.popleft()
            
            # Create pathway entry
            if len(path) > 1:  # At least one hop
                pathway = {
                    'oxygen_index': oxygen_index,
                    'node_sequence': path.copy(),
                    'total_distance': total_distance,
                    'total_score': total_score,
                    'pathway_length': len(path),
                    'pathway_types': self.get_pathway_types(path)
                }
                pathways.append(pathway)
            
            # Continue exploring if within limits
            if len(path) < self.max_pathway_length:
                for neighbor in self.migration_graph.neighbors(current_node):
                    if neighbor not in path:  # Avoid cycles
                        # Check if neighbor is occupied by another oxygen
                        neighbor_coords = self.migration_graph.nodes[neighbor]['coords']
                        
                        is_occupied = False
                        for other_o_coords in other_oxygen_positions:
                            dist = self.structure.lattice.get_distance_and_image(
                                neighbor_coords, other_o_coords)[0]
                            if dist < 0.5:  # Too close - same site
                                is_occupied = True
                                break
                        
                        # Only add valid paths
                        if not is_occupied:
                            edge_data = self.migration_graph[current_node][neighbor]
                            new_distance = total_distance + edge_data['distance']
                            new_score = total_score + edge_data['score']
                            
                            new_path = path + [neighbor]
                            path_key = tuple(sorted(new_path))
                            
                            if path_key not in visited_sequences and new_score > 0:
                                visited_sequences.add(path_key)
                                queue.append((neighbor, new_path, new_distance, new_score))
        
        return pathways
    
    def get_pathway_types(self, node_sequence):
        """Get string describing the types of sites in the pathway"""
        types = []
        for node in node_sequence:
            site_type = self.migration_graph.nodes[node]['site_type']
            if site_type not in types:
                types.append(site_type)
        return ','.join(types)
    
    # NEW: CONCERTED MOVEMENT DETECTION
    def find_concerted_movement_pathways(self):
        """Find concerted movement pathways (swaps, rings, chains)"""
        print("\nFinding concerted movement pathways...")
        
        self.concerted_pathways = {
            'swaps': self.find_direct_swaps(),
            'rings': self.find_ring_exchanges(),
            'chains': self.find_chain_movements()
        }
        
        # Combine and rank all concerted pathways
        all_concerted = []
        for movement_type, pathways in self.concerted_pathways.items():
            for pathway in pathways:
                pathway['movement_type'] = movement_type
                all_concerted.append(pathway)
        
        # Sort by score
        all_concerted.sort(key=lambda x: x.get('score', 0), reverse=True)
        self.concerted_pathways['all'] = all_concerted
        
        print(f"Found {len(self.concerted_pathways['swaps'])} swap opportunities")
        print(f"Found {len(self.concerted_pathways['rings'])} ring exchanges")
        print(f"Found {len(self.concerted_pathways['chains'])} chain movements")
    
    def find_direct_swaps(self):
        """Find direct swap opportunities between oxygen atoms"""
        swaps = []
        
        # Check all pairs of oxygen atoms
        for i, o1_idx in enumerate(self.oxygen_indices):
            for o2_idx in self.oxygen_indices[i+1:]:
                # Check if they can swap positions
                o1_coords = self.structure[o1_idx].frac_coords
                o2_coords = self.structure[o2_idx].frac_coords
                
                distance = self.structure.lattice.get_distance_and_image(o1_coords, o2_coords)[0]
                
                if 2.0 <= distance <= self.max_hop_distance:
                    swap_score = self.evaluate_swap_quality(o1_idx, o2_idx)
                    
                    if swap_score > 10:  # Only good swaps
                        swaps.append({
                            'oxygen1': o1_idx,
                            'oxygen2': o2_idx,
                            'distance': distance,
                            'score': swap_score,
                            'type': 'direct_swap'
                        })
        
        return swaps
    def find_ring_exchanges(self):
        """Find ring exchange opportunities including mixed oxygen-vacancy rings"""
        rings = []
        
        # Get all nodes (both oxygen and vacancy)
        all_nodes = []
        for node, data in self.migration_graph.nodes(data=True):
            if data['site_type'] in ['occupied_oxygen', 'oxygen_vacancy']:
                all_nodes.append(node)
        
        # Look for 3 and 4-member cycles
        for cycle_length in [3, 4]:
            cycles = self.find_cycles_of_length(all_nodes, cycle_length)
            
            for cycle in cycles:
                # Check that cycle has at least one oxygen
                has_oxygen = any(self.migration_graph.nodes[node]['site_type'] == 'occupied_oxygen' 
                            for node in cycle)
                
                if has_oxygen:
                    ring_score = self.evaluate_ring_quality(cycle)
                    
                    if ring_score > 15:
                        rings.append({
                            'cycle_nodes': cycle,
                            'cycle_length': len(cycle),
                            'score': ring_score,
                            'type': 'ring_exchange'
                        })
        
        return rings
    
    def find_chain_movements(self):
        """Find chain movement opportunities (A→B→C→vacancy)"""
        chains = []
        
        # Look for chains ending at vacancy sites
        oxygen_nodes = []
        vacancy_nodes = []
        
        for node, data in self.migration_graph.nodes(data=True):
            if data['site_type'] == 'occupied_oxygen':
                oxygen_nodes.append(node)
            elif 'vacancy' in data['site_type']:
                vacancy_nodes.append(node)
        
        # Find chains of length 2-4
        for vacancy_node in vacancy_nodes:
            chains_to_vacancy = self.find_chains_to_node(vacancy_node, max_length=4)
            
            for chain in chains_to_vacancy:
                if len(chain) >= 2:  # At least 2 oxygen moves
                    chain_score = self.evaluate_chain_quality(chain)
                    
                    if chain_score > 12:  # Only good chains
                        chains.append({
                            'chain_nodes': chain,
                            'chain_length': len(chain),
                            'score': chain_score,
                            'type': 'chain_movement'
                        })
        
        return chains

    def evaluate_swap_quality(self, o1_idx, o2_idx):
        """Fixed swap evaluation with proper PBC handling"""
        try:
            # Get positions using proper PBC distance
            o1_pos = self.structure[o1_idx].frac_coords.copy()
            o2_pos = self.structure[o2_idx].frac_coords.copy()
            
            # Calculate distance properly
            distance = self.structure.lattice.get_distance_and_image(o1_pos, o2_pos)[0]
            
            # Create test structure with swapped positions
            test_structure = self.structure.copy()
            test_structure.replace(o1_idx, Element("O"), o2_pos, coords_are_cartesian=False)
            test_structure.replace(o2_idx, Element("O"), o1_pos, coords_are_cartesian=False)
            
            # Continue with rest of evaluation...
            nn = CrystalNN()
            o1_new_neighbors = nn.get_nn_info(test_structure, o1_idx)
            o2_new_neighbors = nn.get_nn_info(test_structure, o2_idx)
            
            score = 20  # Base score
            
            # Add distance-based scoring
            if 2.0 <= distance <= 3.5:
                score += 10
            elif distance > 4.0:
                score -= 5
                
            return score
            
        except Exception as e:
            print(f"Error in swap evaluation: {e}")
            return 0

    
    def evaluate_ring_quality(self, cycle_nodes):
        """Evaluate the quality of a ring exchange"""
        # Simple scoring based on cycle length and connectivity
        base_score = 25 - len(cycle_nodes) * 2  # Prefer smaller rings
        
        # Check if all nodes are well connected
        total_edges = 0
        for i in range(len(cycle_nodes)):
            node1 = cycle_nodes[i]
            node2 = cycle_nodes[(i + 1) % len(cycle_nodes)]
            
            if self.migration_graph.has_edge(node1, node2):
                edge_data = self.migration_graph[node1][node2]
                total_edges += 1
                base_score += edge_data.get('score', 0) / len(cycle_nodes)
        
        # Must be a complete cycle
        if total_edges == len(cycle_nodes):
            return base_score
        else:
            return 0
    
    def evaluate_chain_quality(self, chain_nodes):
        """Evaluate the quality of a chain movement"""
        base_score = 15 + len(chain_nodes) * 2  # Longer chains can be good
        
        # Check connectivity along chain
        for i in range(len(chain_nodes) - 1):
            node1 = chain_nodes[i]
            node2 = chain_nodes[i + 1]
            
            if self.migration_graph.has_edge(node1, node2):
                edge_data = self.migration_graph[node1][node2]
                base_score += edge_data.get('score', 0) / len(chain_nodes)
            else:
                return 0  # Chain is broken
        
        return base_score
    
    def find_cycles_of_length(self, nodes, length):
        """Find cycles of specific length in the migration graph"""
        cycles = []
        
        # Simple cycle detection for small cycles
        if length == 3:
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    for node3 in nodes:
                        if (node3 != node1 and node3 != node2 and
                            self.migration_graph.has_edge(node1, node2) and
                            self.migration_graph.has_edge(node2, node3) and
                            self.migration_graph.has_edge(node3, node1)):
                            cycles.append([node1, node2, node3])
        
        elif length == 4:
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    for node3 in nodes:
                        for node4 in nodes:
                            if (len(set([node1, node2, node3, node4])) == 4 and
                                self.migration_graph.has_edge(node1, node2) and
                                self.migration_graph.has_edge(node2, node3) and
                                self.migration_graph.has_edge(node3, node4) and
                                self.migration_graph.has_edge(node4, node1)):
                                cycles.append([node1, node2, node3, node4])
        
        return cycles
    
    def find_chains_to_node(self, target_node, max_length=4):
        """Find chains of oxygen nodes leading to target node"""
        chains = []
        
        # BFS from target backwards
        oxygen_nodes = [node for node, data in self.migration_graph.nodes(data=True) 
                       if data['site_type'] == 'occupied_oxygen']
        
        for start_node in oxygen_nodes:
            if self.migration_graph.has_edge(start_node, target_node):
                # Found direct connection, now look for chains
                paths = self.find_paths_between_nodes(start_node, target_node, max_length)
                chains.extend(paths)
        
        return chains
    
    def find_paths_between_nodes(self, start, end, max_length):
        """Find all paths between two nodes up to max_length"""
        paths = []
        
        # Simple DFS for short paths
        def dfs(current, path, target, max_len):
            if len(path) > max_len:
                return
            if current == target and len(path) > 1:
                paths.append(path.copy())
                return
            
            for neighbor in self.migration_graph.neighbors(current):
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, path, target, max_len)
                    path.pop()
        
        dfs(start, [start], end, max_length)
        return paths
    
    def execute_migration_pathway(self, pathway_index=0, output_prefix="universal_migration"):
        """Execute a migration pathway and generate outputs"""
        if pathway_index >= len(self.migration_pathways):
            print(f"Pathway index {pathway_index} not available.")
            return None
        
        pathway = self.migration_pathways[pathway_index]
        
        print(f"\nExecuting universal migration pathway {pathway_index + 1}")
        print("="*60)
        print(f"Pathway types: {pathway.get('pathway_types', 'mixed')}")
        
        # Simulate the migration
        structures = []
        structures.append(self.structure.copy())  # Initial
        
        current_structure = self.structure.copy()
        oxygen_index = pathway['oxygen_index']
        
        for step, node in enumerate(pathway['node_sequence'][1:], 1):
            # Get target coordinates
            target_coords = self.migration_graph.nodes[node]['coords']
            site_type = self.migration_graph.nodes[node]['site_type']
            
            # Move oxygen
            current_structure.replace(oxygen_index, Element("O"), target_coords, coords_are_cartesian=False)
            structures.append(current_structure.copy())
            
            print(f"  Step {step}: O{oxygen_index} → {site_type}")
        
        # Save structures
        for i, structure in enumerate(structures):
            step_prefix = f"{output_prefix}_step_{i:02d}"
            poscar = Poscar(structure)
            poscar.write_file(f"{step_prefix}.POSCAR")
            
            cif_writer = CifWriter(structure)
            cif_writer.write_file(f"{step_prefix}.cif")
        
        print(f"Generated {len(structures)} structure files")
        return structures


    def get_pbc_interpolated_coords(self, start_coords, end_coords, fraction):
        start_coords = np.array(start_coords)
        end_coords = np.array(end_coords)
        
        # Calculate difference considering periodic boundaries
        diff = end_coords - start_coords
        diff = diff - np.round(diff)  # Wrap to [-0.5, 0.5] for shortest path
        
        return (start_coords + fraction * diff) % 1.0
    
    def create_swap_steps(self, movement):
        """Create steps for direct swap - just initial and final"""
        o1_idx = movement['oxygen1']
        o2_idx = movement['oxygen2']
        
        o1_start = self.structure[o1_idx].frac_coords.copy()
        o2_start = self.structure[o2_idx].frac_coords.copy()
        
        # Just create the final swapped structure
        final_structure = self.structure.copy()
        final_structure.replace(o1_idx, Element("O"), o2_start, coords_are_cartesian=False)
        final_structure.replace(o2_idx, Element("O"), o1_start, coords_are_cartesian=False)
        
        steps = [final_structure]
        
        print(f"  Direct swap: O{o1_idx} ↔ O{o2_idx} (complete exchange)")
        
        return steps
    
    def create_ring_steps(self, movement, sequential=False):
        """
        Create steps for ring exchange
        
        Args:
            movement: Ring movement data
            sequential: If True, show sequential hops; if False, show only initial+final
        """
        cycle_nodes = movement['cycle_nodes']
        
        # Map nodes to oxygen indices and positions
        node_info = []
        for i, node in enumerate(cycle_nodes):
            node_data = self.migration_graph.nodes[node]
            if node_data['site_type'] == 'occupied_oxygen':
                # Find corresponding oxygen index
                for o_idx in self.oxygen_indices:
                    o_coords = self.structure[o_idx].frac_coords
                    dist = self.calculate_pbc_distance(node_data['coords'], o_coords)
                    if dist < 0.01:
                        node_info.append({
                            'type': 'oxygen',
                            'index': o_idx,
                            'coords': node_data['coords'],
                            'position': i
                        })
                        break
            else:
                node_info.append({
                    'type': 'vacancy',
                    'index': None,
                    'coords': node_data['coords'],
                    'position': i
                })
        
        steps = []
        
        if not sequential or len(cycle_nodes) == 3:
            # For 3-member rings or non-sequential mode: just show final state
            final_structure = self.structure.copy()
            
            for info in node_info:
                if info['type'] == 'oxygen':
                    # This oxygen moves to the next position in the ring
                    next_pos = (info['position'] + 1) % len(node_info)
                    next_coords = node_info[next_pos]['coords']
                    final_structure.replace(info['index'], Element("O"), next_coords, coords_are_cartesian=False)
            
            steps.append(final_structure)
            print(f"  Ring exchange: Complete {len(node_info)}-member ring rotation")
            
        else:
            # For larger rings in sequential mode: show step-by-step
            current_structure = self.structure.copy()
            
            # Find starting point (often where vacancy is)
            vacancy_pos = next(i for i, info in enumerate(node_info) if info['type'] == 'vacancy')
            
            # Move atoms sequentially into the vacancy
            for step in range(len(cycle_nodes) - 1):
                # Determine which atom moves into current vacancy position
                source_pos = (vacancy_pos - 1) % len(cycle_nodes)
                
                # Find oxygen at source position
                source_info = next(info for info in node_info if info['position'] == source_pos and info['type'] == 'oxygen')
                
                if source_info:
                    # Move this oxygen to vacancy position
                    vacancy_coords = node_info[vacancy_pos]['coords']
                    current_structure.replace(source_info['index'], Element("O"), vacancy_coords, coords_are_cartesian=False)
                    
                    # Update vacancy position for next iteration
                    vacancy_pos = source_pos
                    
                    steps.append(current_structure.copy())
                    print(f"  Step {step + 1}: O{source_info['index']} → vacancy position")
        
        return steps


    def create_chain_steps(self, movement, sequential=True):
        """
        Create steps for chain movement
        
        Args:
            movement: Chain movement data
            sequential: If True, show each hop; if False, show only initial+final
        """
        chain_nodes = movement['chain_nodes']
        
        # Map nodes to positions and find oxygen atoms
        node_positions = []
        oxygen_at_node = {}
        
        for i, node in enumerate(chain_nodes):
            node_data = self.migration_graph.nodes[node]
            node_positions.append(node_data['coords'])
            
            if node_data['site_type'] == 'occupied_oxygen':
                # Find oxygen at this position
                for o_idx in self.oxygen_indices:
                    o_coords = self.structure[o_idx].frac_coords
                    dist = self.calculate_pbc_distance(node_data['coords'], o_coords)
                    if dist < 0.01:
                        oxygen_at_node[i] = o_idx
                        break
        
        steps = []
        
        if not sequential:
            # Just show final state
            final_structure = self.structure.copy()
            
            for node_idx, o_idx in oxygen_at_node.items():
                if node_idx < len(chain_nodes) - 1:
                    next_coords = node_positions[node_idx + 1]
                    final_structure.replace(o_idx, Element("O"), next_coords, coords_are_cartesian=False)
            
            steps.append(final_structure)
            print(f"  Chain movement: All {len(oxygen_at_node)} atoms moved")
            
        else:
            # Show sequential hops from end to beginning
            current_structure = self.structure.copy()
            
            # Move atoms starting from the one closest to vacancy
            for step_num in range(len(chain_nodes) - 1, 0, -1):
                if step_num - 1 in oxygen_at_node:
                    o_idx = oxygen_at_node[step_num - 1]
                    next_coords = node_positions[step_num]
                    current_structure.replace(o_idx, Element("O"), next_coords, coords_are_cartesian=False)
                    steps.append(current_structure.copy())
                    print(f"  Step {len(chain_nodes) - step_num}: O{o_idx} → position {step_num}")
        
        return steps


    def execute_concerted_movement(self, movement_index=0, output_prefix="concerted_migration", sequential_mode=None):
        """
        Execute a concerted movement
        
        Args:
            movement_index: Index of movement to execute
            output_prefix: Prefix for output files
            sequential_mode: None (auto), True (force sequential), or False (force simultaneous)
        """
        if not hasattr(self, 'concerted_pathways') or not self.concerted_pathways['all']:
            print("No concerted pathways available.")
            return None
            
        if movement_index >= len(self.concerted_pathways['all']):
            print(f"Concerted movement index {movement_index} not available.")
            return None
        
        movement = self.concerted_pathways['all'][movement_index]
        
        print(f"\nExecuting concerted movement {movement_index + 1}")
        print("="*60)
        print(f"Movement type: {movement['movement_type']}")
        print(f"Score: {movement['score']:.1f}")
        
        # Determine sequential mode
        if sequential_mode is None:
            # Auto-decide based on movement type
            if movement['movement_type'] == 'chains':
                sequential = True  # Chains are naturally sequential
            elif movement['movement_type'] == 'rings' and len(movement.get('cycle_nodes', [])) >3: #3
                sequential = True  # Large rings can be sequential
            else:
                sequential = False  # Swaps and 3-member rings are simultaneous
        else:
            sequential = sequential_mode
        
        # Generate structures
        structures = []
        structures.append(self.structure.copy())  # Initial structure
        
        if movement['movement_type'] == 'swaps':
            structures.extend(self.create_swap_steps(movement))
            
        elif movement['movement_type'] == 'rings':
            structures.extend(self.create_ring_steps(movement, sequential=sequential))
            
        elif movement['movement_type'] == 'chains':
            structures.extend(self.create_chain_steps(movement, sequential=sequential))
        
        else:
            print(f"WARNING: Unknown movement type: {movement['movement_type']}")
            return structures
        
        # Save structures
        for i, structure in enumerate(structures):
            step_prefix = f"{output_prefix}_step_{i:02d}"
            poscar = Poscar(structure)
            poscar.write_file(f"{step_prefix}.POSCAR")
            
            cif_writer = CifWriter(structure)
            cif_writer.write_file(f"{step_prefix}.cif")
        
        mode_str = "sequential" if sequential and len(structures) > 2 else "simultaneous"
        print(f"Generated {len(structures)} structures ({mode_str} mode)")
        
        return structures


def main():
    """Example usage with universal vacancy detection and concerted movements"""
    print("Universal Vacancy-Mediated Oxygen Migration Finder with Concerted Movements")
    print("===========================================================================")
    
    # Initialize with optional reference structure
    finder = UniversalVacancyOxygenMigrationFinder(
        structure_file="POSCAR",
        reference_structure_file=None,  # Set to reference POSCAR if available
        max_hop_distance=4.0,
        max_pathway_length=6
    )
    
    if finder.migration_pathways:
        print(f"\nFound {len(finder.migration_pathways)} individual migration pathways")
        
        # Execute best individual pathway
        finder.execute_migration_pathway(pathway_index=0, output_prefix="best_individual_migration")
    
    # Check concerted movements
    if hasattr(finder, 'concerted_pathways') and finder.concerted_pathways['all']:
        print(f"\nFound {len(finder.concerted_pathways['all'])} concerted movement opportunities")
        
        # Show top 3 concerted movements
        for i, movement in enumerate(finder.concerted_pathways['all'][:3]):
            print(f"  {i+1}. {movement['movement_type']}: score={movement['score']:.1f}")
        
        # Execute best concerted movement
        finder.execute_concerted_movement(movement_index=0, output_prefix="best_concerted_migration")
    else:
        print("No concerted movement opportunities found")


if __name__ == "__main__":
    main()