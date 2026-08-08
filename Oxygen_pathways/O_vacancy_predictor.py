#!/usr/bin/env python3
"""
Smart Oxygen Vacancy Pre-screener
Ranks O sites by physical/chemical factors BEFORE expensive DFT calculations
"""

import numpy as np
from pymatgen.core import Structure, Element, Specie
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN, MinimumDistanceNN
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class SmartVacancyPrescreener:
    """
    Pre-screen and rank oxygen sites for vacancy formation
    based on physical/chemical factors before DFT
    """
    
    def __init__(self, structure_file="POSCAR"):
        """Initialize with structure"""
        if structure_file.endswith('.cif'):
            self.structure = Structure.from_file(structure_file)
        else:
            self.structure = Poscar.from_file(structure_file).structure
        
        # Initialize analyzers
        self.nn_analyzer = CrystalNN()
        self.voronoi_nn = VoronoiNN()
        self.bv_analyzer = BVAnalyzer()
        
        # Scoring weights (can be tuned based on material type)
        self.weights = {
            'coordination': 0.25,      # Undercoordination favorability
            'bond_strength': 0.20,     # Weak bonds favor vacancy
            'covalency': 0.15,         # Covalent character
            'strain': 0.15,            # Local distortion/strain
            'symmetry': 0.10,          # Low symmetry sites
            'polyhedra': 0.10,         # Edge-sharing penalty
            'oxidation': 0.05          # Cation oxidation states
        }
        
        # Material-specific parameters
        self._identify_material_type()
        
    def _identify_material_type(self):
        """Identify material type to adjust parameters"""
        # Check for common oxygen-conducting materials
        elements = [str(el) for el in self.structure.composition.elements]
        
        self.material_type = 'generic'
        if 'La' in elements and ('Ni' in elements or 'Co' in elements):
            self.material_type = 'perovskite_related'
        elif 'Ce' in elements:
            self.material_type = 'fluorite_related'
        elif all(Element(el).is_metal for el in elements if el != 'O'):
            self.material_type = 'oxide'
            
        print(f"Identified material type: {self.material_type}")
    
    def rank_oxygen_sites(self, top_n=15):
        """
        Rank all oxygen sites by vacancy formation favorability
        
        Returns:
            DataFrame with ranked sites and detailed scores
        """
        print("Analyzing oxygen sites for vacancy formation favorability...\n")
        
        # Find oxygen sites
        oxygen_sites = [(i, site) for i, site in enumerate(self.structure) 
                       if site.species_string == "O"]
        
        if not oxygen_sites:
            raise ValueError("No oxygen sites found in structure!")
        
        print(f"Found {len(oxygen_sites)} oxygen sites\n")
        
        # Analyze each site
        site_analyses = []
        
        for idx, (i, site) in enumerate(oxygen_sites):
            print(f"\rAnalyzing site {idx+1}/{len(oxygen_sites)}...", end='')
            
            analysis = {
                'site_index': i,
                'position': site.frac_coords,
                'cart_position': site.coords
            }
            
            # 1. Local bonding environment
            bonding = self._analyze_bonding_environment(i)
            analysis.update(bonding)
            
            # 2. Electronic factors
            electronic = self._analyze_electronic_factors(i)
            analysis.update(electronic)
            
            # 3. Structural factors
            structural = self._analyze_structural_factors(i)
            analysis.update(structural)
            
            # 4. Chemical environment
            chemical = self._analyze_chemical_environment(i)
            analysis.update(chemical)
            
            # 5. Special site identification
            special = self._identify_special_sites(i)
            analysis.update(special)
            
            # Calculate composite score
            analysis['total_score'] = self._calculate_composite_score(analysis)
            
            site_analyses.append(analysis)
        
        print("\n")
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(site_analyses)
        
        # Sort by total score (higher = more favorable for vacancy)
        df = df.sort_values('total_score', ascending=False)
        
        # Find symmetrically unique sites
        unique_indices = self._find_symmetrically_unique_sites(df)
        df['is_unique'] = df['site_index'].isin(unique_indices)
        
        # Generate report
        self._generate_prescreening_report(df, top_n)
        
        return df
    
    def _analyze_bonding_environment(self, site_idx):
        """Analyze local bonding environment"""
        try:
            nn_info = self.nn_analyzer.get_nn_info(self.structure, site_idx)
            
            # Coordination number
            coord_num = len(nn_info)
            
            # Bond lengths and strengths
            bond_lengths = []
            bond_strengths = []
            neighbor_elements = []
            
            for neighbor in nn_info:
                bond_lengths.append(neighbor['weight'])
                neighbor_elements.append(neighbor['site'].specie.symbol)
                
                # Estimate bond strength using Pauling's equation
                bond_strength = self._estimate_bond_strength(
                    self.structure[site_idx].specie,
                    neighbor['site'].specie,
                    neighbor['weight']
                )
                bond_strengths.append(bond_strength)
            
            # Calculate bond valence sum
            try:
                bv_sum = self.bv_analyzer.get_valences(self.structure)[site_idx]
            except:
                bv_sum = 2.0  # Default for oxygen
            
            # Analyze covalency
            covalency = self._calculate_covalency(site_idx, nn_info)
            
            return {
                'coordination_number': coord_num,
                'avg_bond_length': np.mean(bond_lengths),
                'bond_length_variance': np.var(bond_lengths),
                'avg_bond_strength': np.mean(bond_strengths),
                'min_bond_strength': min(bond_strengths) if bond_strengths else 0,
                'bond_valence_sum': bv_sum,
                'bond_valence_deviation': abs(bv_sum - 2.0),  # Deviation from ideal O2-
                'covalency_factor': covalency,
                'neighbor_types': dict(pd.Series(neighbor_elements).value_counts())
            }
            
        except Exception as e:
            print(f"\nError analyzing site {site_idx}: {e}")
            return {
                'coordination_number': 0,
                'avg_bond_length': 0,
                'bond_length_variance': 0,
                'avg_bond_strength': 0,
                'min_bond_strength': 0,
                'bond_valence_sum': 0,
                'bond_valence_deviation': 0,
                'covalency_factor': 0,
                'neighbor_types': {}
            }
    
    def _analyze_electronic_factors(self, site_idx):
        """Analyze electronic factors"""
        electronic = {
            'avg_cation_oxidation': 0,
            'oxidation_variance': 0,
            'charge_transfer_factor': 0,
            'electronegativity_diff': 0
        }
        
        try:
            nn_info = self.nn_analyzer.get_nn_info(self.structure, site_idx)
            
            oxidation_states = []
            electroneg_diffs = []
            
            for neighbor in nn_info:
                neighbor_element = neighbor['site'].specie
                
                # Get oxidation state if available
                if isinstance(neighbor_element, Specie) and neighbor_element.oxi_state:
                    oxidation_states.append(neighbor_element.oxi_state)
                
                # Electronegativity difference
                if hasattr(neighbor_element, 'X'):
                    en_diff = abs(Element('O').X - neighbor_element.X)
                    electroneg_diffs.append(en_diff)
            
            if oxidation_states:
                electronic['avg_cation_oxidation'] = np.mean(oxidation_states)
                electronic['oxidation_variance'] = np.var(oxidation_states)
            
            if electroneg_diffs:
                electronic['electronegativity_diff'] = np.mean(electroneg_diffs)
                # Charge transfer factor (higher EN diff = more ionic = easier vacancy)
                electronic['charge_transfer_factor'] = np.mean(electroneg_diffs) / 3.5
                
        except:
            pass
        
        return electronic
    
    def _analyze_structural_factors(self, site_idx):
        """Analyze structural factors"""
        structural = {
            'local_distortion': 0,
            'site_symmetry': 0,
            'voronoi_volume': 0,
            'angular_distortion': 0,
            'octahedral_distortion': 0
        }
        
        try:
            nn_info = self.nn_analyzer.get_nn_info(self.structure, site_idx)
            
            if len(nn_info) >= 3:
                # Calculate polyhedral distortion
                central_pos = self.structure[site_idx].coords
                neighbor_positions = [n['site'].coords for n in nn_info]
                
                # Bond length distortion
                bond_lengths = [n['weight'] for n in nn_info]
                structural['local_distortion'] = np.std(bond_lengths) / np.mean(bond_lengths)
                
                # Angular distortion
                angles = []
                for i in range(len(neighbor_positions)):
                    for j in range(i+1, len(neighbor_positions)):
                        v1 = neighbor_positions[i] - central_pos
                        v2 = neighbor_positions[j] - central_pos
                        angle = np.arccos(np.clip(
                            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                            -1, 1
                        ))
                        angles.append(np.degrees(angle))
                
                # For octahedral coordination, check deviation from 90 and 180 degrees
                if len(nn_info) == 6:
                    ideal_angles = []
                    for angle in angles:
                        if 80 < angle < 100:
                            ideal_angles.append(abs(angle - 90))
                        elif 170 < angle < 190:
                            ideal_angles.append(abs(angle - 180))
                    
                    if ideal_angles:
                        structural['octahedral_distortion'] = np.mean(ideal_angles)
                
                structural['angular_distortion'] = np.std(angles)
            
            # Voronoi volume
            try:
                vcf = VoronoiCoordFinder(self.structure)
                voronoi = vcf.get_voronoi_polyhedra(site_idx)
                structural['voronoi_volume'] = sum(v['volume'] for v in voronoi.values())
            except:
                pass
                
        except:
            pass
        
        return structural
    
    def _analyze_chemical_environment(self, site_idx):
        """Analyze chemical environment"""
        chemical = {
            'nearest_O_distance': float('inf'),
            'O_O_coordination': 0,
            'cation_variety': 0,
            'local_density': 0
        }
        
        try:
            site_pos = self.structure[site_idx].coords
            
            # Find nearest oxygen-oxygen distances
            o_distances = []
            for i, other_site in enumerate(self.structure):
                if i != site_idx and other_site.species_string == "O":
                    dist = np.linalg.norm(other_site.coords - site_pos)
                    o_distances.append(dist)
            
            if o_distances:
                chemical['nearest_O_distance'] = min(o_distances)
                chemical['O_O_coordination'] = sum(1 for d in o_distances if d < 3.5)
            
            # Cation variety in local environment
            nn_info = self.nn_analyzer.get_nn_info(self.structure, site_idx)
            unique_neighbors = set(n['site'].specie.symbol for n in nn_info)
            chemical['cation_variety'] = len(unique_neighbors)
            
            # Local density (inverse of average distance to neighbors)
            if nn_info:
                avg_dist = np.mean([n['weight'] for n in nn_info])
                chemical['local_density'] = 1.0 / avg_dist
                
        except:
            pass
        
        return chemical
    
    def _identify_special_sites(self, site_idx):
        """Identify special crystallographic sites"""
        special = {
            'is_edge_sharing': False,
            'is_corner_sharing': True,  # Default assumption
            'is_undercoordinated': False,
            'is_distorted': False,
            'polyhedra_type': 'unknown'
        }
        
        try:
            nn_info = self.nn_analyzer.get_nn_info(self.structure, site_idx)
            coord_num = len(nn_info)
            
            # Undercoordinated check
            expected_coord = {'oxide': 4, 'perovskite_related': 6, 'fluorite_related': 4}
            if coord_num < expected_coord.get(self.material_type, 4):
                special['is_undercoordinated'] = True
            
            # Edge-sharing check
            # Look for shared oxygen neighbors between cations
            for i, neighbor1 in enumerate(nn_info):
                for j, neighbor2 in enumerate(nn_info[i+1:], i+1):
                    # Check if these two cations share other oxygens
                    n1_idx = neighbor1['site_index']
                    n2_idx = neighbor2['site_index']
                    
                    if self._check_edge_sharing(n1_idx, n2_idx):
                        special['is_edge_sharing'] = True
                        special['is_corner_sharing'] = False
                        break
            
            # Distortion check
            if coord_num >= 4:
                bond_lengths = [n['weight'] for n in nn_info]
                if np.std(bond_lengths) / np.mean(bond_lengths) > 0.1:
                    special['is_distorted'] = True
            
            # Identify polyhedra type
            if coord_num == 4:
                special['polyhedra_type'] = 'tetrahedral'
            elif coord_num == 6:
                special['polyhedra_type'] = 'octahedral'
            elif coord_num == 5:
                special['polyhedra_type'] = 'pyramidal'
                
        except:
            pass
        
        return special
    
    def _check_edge_sharing(self, cation1_idx, cation2_idx):
        """Check if two cations share an edge (2 oxygens)"""
        try:
            # Get oxygen neighbors of both cations
            c1_neighbors = []
            c2_neighbors = []
            
            nn1 = self.nn_analyzer.get_nn_info(self.structure, cation1_idx)
            nn2 = self.nn_analyzer.get_nn_info(self.structure, cation2_idx)
            
            for n in nn1:
                if n['site'].species_string == "O":
                    c1_neighbors.append(n['site_index'])
            
            for n in nn2:
                if n['site'].species_string == "O":
                    c2_neighbors.append(n['site_index'])
            
            # Count shared oxygens
            shared = len(set(c1_neighbors) & set(c2_neighbors))
            
            return shared >= 2  # Edge-sharing if 2 or more shared oxygens
            
        except:
            return False
    
    def _calculate_covalency(self, site_idx, nn_info):
        """Calculate covalency factor based on electronegativity"""
        covalency_scores = []
        
        for neighbor in nn_info:
            element = Element(neighbor['site'].species_string.split('+')[0].split('-')[0])
            if hasattr(element, 'X'):
                # Pauling electronegativity difference
                en_diff = abs(Element('O').X - element.X)
                # Convert to covalency (0 = ionic, 1 = covalent)
                covalency = 1.0 - min(en_diff / 2.0, 1.0)
                covalency_scores.append(covalency)
        
        return np.mean(covalency_scores) if covalency_scores else 0
    
    def _estimate_bond_strength(self, species1, species2, bond_length):
        """Estimate bond strength using empirical relations"""
        # Simple inverse relationship with bond length
        # Can be refined with Pauling bond order or other methods
        typical_length = 2.0  # Typical M-O bond
        return typical_length / bond_length
    
    def _calculate_composite_score(self, analysis):
        """Calculate composite score for vacancy formation favorability"""
        score = 0
        
        # 1. Coordination score (undercoordinated is favorable)
        ideal_coord = 4 if self.material_type == 'oxide' else 6
        coord_score = 1.0 - min(analysis['coordination_number'] / ideal_coord, 1.0)
        score += self.weights['coordination'] * coord_score
        
        # 2. Bond strength score (weak bonds are favorable)
        if analysis['avg_bond_strength'] > 0:
            bond_score = 1.0 / (1.0 + analysis['avg_bond_strength'])
            score += self.weights['bond_strength'] * bond_score
        
        # 3. Covalency score (more covalent is often favorable)
        score += self.weights['covalency'] * analysis['covalency_factor']
        
        # 4. Strain score (high strain is favorable)
        strain_score = analysis['local_distortion'] + analysis['angular_distortion'] / 90.0
        score += self.weights['strain'] * min(strain_score, 1.0)
        
        # 5. Symmetry score (low symmetry is favorable)
        if analysis['is_distorted']:
            score += self.weights['symmetry'] * 0.5
        
        # 6. Polyhedra score
        if analysis['is_edge_sharing']:
            score += self.weights['polyhedra'] * 1.0  # Edge-sharing favorable
        elif analysis['is_undercoordinated']:
            score += self.weights['polyhedra'] * 0.8
        
        # 7. Oxidation state effects
        if analysis['oxidation_variance'] > 0:
            score += self.weights['oxidation'] * min(analysis['oxidation_variance'] / 2.0, 1.0)
        
        # Bonus factors
        if analysis['bond_valence_deviation'] > 0.3:
            score += 0.1  # Bonus for high BV deviation
        
        if analysis['is_undercoordinated'] and analysis['is_edge_sharing']:
            score += 0.15  # Extra bonus for combination
        
        return score
    
    def _get_detailed_neighbors(self, site_idx):
        """Get detailed neighbor information for reporting"""
        try:
            nn_info = self.nn_analyzer.get_nn_info(self.structure, site_idx)
            
            neighbors = []
            for n in nn_info:
                neighbor_data = {
                    'element': n['site'].species_string,
                    'distance': n['weight'],
                    'site_index': n['site_index'],
                    'coords': n['site'].coords
                }
                neighbors.append(neighbor_data)
            
            # Sort by distance
            neighbors.sort(key=lambda x: x['distance'])
            return neighbors
            
        except:
            return []
        """Find symmetrically unique oxygen sites"""
        try:
            sga = SpacegroupAnalyzer(self.structure, symprec=0.1)
            symm_struct = sga.get_symmetrized_structure()
            
            unique_indices = []
            seen_positions = set()
            
            for _, row in df.iterrows():
                site_idx = row['site_index']
                site = self.structure[site_idx]
                
                # Get equivalent sites
                equiv_sites = symm_struct.find_equivalent_sites(site)
                
                # Check if we've seen this position family
                position_key = tuple(np.round(site.frac_coords, 4))
                
                if position_key not in seen_positions:
                    unique_indices.append(site_idx)
                    
                    # Mark all equivalent positions as seen
                    for equiv in equiv_sites:
                        equiv_key = tuple(np.round(equiv.frac_coords, 4))
                        seen_positions.add(equiv_key)
            
            return unique_indices
            
        except:
            # If symmetry analysis fails, return all sites
            return df['site_index'].tolist()
    
    def _generate_prescreening_report(self, df, top_n):
        """Generate detailed prescreening report"""
        report_file = "vacancy_prescreening_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("OXYGEN VACANCY FORMATION PRE-SCREENING REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Structure: {self.structure.composition}\n")
            f.write(f"Material type: {self.material_type}\n")
            f.write(f"Total oxygen sites: {len(df)}\n")
            f.write(f"Symmetrically unique sites: {df['is_unique'].sum()}\n\n")
            
            # Scoring weights
            f.write("Scoring Weights:\n")
            for factor, weight in self.weights.items():
                f.write(f"  {factor}: {weight:.2f}\n")
            f.write("\n")
            
            # Top unique sites
            f.write(f"TOP {top_n} UNIQUE SITES FOR DFT CALCULATION:\n")
            f.write("-"*70 + "\n")
            f.write("Rank | Site | Score | Coord | Edge-sharing | Distorted | BV Dev\n")
            f.write("-"*70 + "\n")
            
            unique_df = df[df['is_unique']]
            for i, (_, row) in enumerate(unique_df.head(top_n).iterrows()):
                f.write(f"{i+1:4d} | {row['site_index']:4d} | "
                       f"{row['total_score']:5.3f} | "
                       f"{row['coordination_number']:5d} | "
                       f"{'Yes' if row['is_edge_sharing'] else 'No':12s} | "
                       f"{'Yes' if row['is_distorted'] else 'No':9s} | "
                       f"{row['bond_valence_deviation']:6.3f}\n")
            
            # Detailed analysis of top 5
            f.write("\n\nDETAILED ANALYSIS OF TOP 5 SITES:\n")
            f.write("="*70 + "\n")
            
            for i, (_, row) in enumerate(unique_df.head(5).iterrows()):
                site_idx = row['site_index']
                
                f.write(f"\n{'='*60}\n")
                f.write(f"RANK {i+1}: Oxygen Site {site_idx}\n")
                f.write(f"{'='*60}\n")
                
                # Basic info
                f.write(f"\nBASIC INFORMATION:\n")
                f.write(f"  Position (fractional): [{row['position'][0]:.4f}, {row['position'][1]:.4f}, {row['position'][2]:.4f}]\n")
                f.write(f"  Position (Cartesian): [{row['cart_position'][0]:.3f}, {row['cart_position'][1]:.3f}, {row['cart_position'][2]:.3f}] Å\n")
                f.write(f"  Total favorability score: {row['total_score']:.3f}\n")
                f.write(f"  Polyhedra type: {row['polyhedra_type']}\n")
                
                # Detailed coordination environment
                f.write(f"\nCOORDINATION ENVIRONMENT:\n")
                f.write(f"  Number of neighbors: {row['coordination_number']}")
                expected_coord = 4 if self.material_type == 'oxide' else 6
                if row['coordination_number'] < expected_coord:
                    f.write(f" (UNDERCOORDINATED by {expected_coord - row['coordination_number']})")
                elif row['coordination_number'] > expected_coord:
                    f.write(f" (overcoordinated by {row['coordination_number'] - expected_coord})")
                f.write("\n")
                
                # List each neighbor with details
                f.write(f"\n  Neighboring atoms:\n")
                nn_info = self._get_detailed_neighbors(site_idx)
                for j, neighbor in enumerate(nn_info):
                    f.write(f"    {j+1}. {neighbor['element']} at distance {neighbor['distance']:.3f} Å")
                    if neighbor['distance'] > 2.2:
                        f.write(" (LONG BOND)")
                    elif neighbor['distance'] < 1.8:
                        f.write(" (SHORT BOND)")
                    f.write(f"\n")
                
                # Bond analysis
                f.write(f"\n  Bond length statistics:\n")
                f.write(f"    Average: {row['avg_bond_length']:.3f} Å\n")
                f.write(f"    Variance: {row['bond_length_variance']:.4f}\n")
                f.write(f"    Min-Max spread: {nn_info[0]['distance']:.3f} - {nn_info[-1]['distance']:.3f} Å\n")
                
                # Why is it strained?
                f.write(f"\nSTRAIN ANALYSIS:\n")
                if row['local_distortion'] > 0.1:
                    f.write(f"  ⚠️  HIGH LOCAL DISTORTION ({row['local_distortion']:.3f})\n")
                    f.write(f"  Reasons for strain:\n")
                    
                    # Analyze bond length distribution
                    bond_lengths = [n['distance'] for n in nn_info]
                    if max(bond_lengths) - min(bond_lengths) > 0.3:
                        f.write(f"    • Large bond length variation ({max(bond_lengths)-min(bond_lengths):.3f} Å spread)\n")
                        f.write(f"      Suggests asymmetric environment or Jahn-Teller distortion\n")
                    
                    if row['angular_distortion'] > 10:
                        f.write(f"    • High angular distortion ({row['angular_distortion']:.1f}°)\n")
                        f.write(f"      Bond angles deviate significantly from ideal geometry\n")
                    
                    if row['octahedral_distortion'] > 5:
                        f.write(f"    • Octahedral distortion ({row['octahedral_distortion']:.1f}°)\n")
                        f.write(f"      O-M-O angles deviate from ideal 90°/180°\n")
                
                # Electronic factors
                f.write(f"\nELECTRONIC FACTORS:\n")
                f.write(f"  Bond valence sum: {row['bond_valence_sum']:.3f} ")
                if abs(row['bond_valence_sum'] - 2.0) > 0.1:
                    f.write(f"(deviates from ideal 2.0 by {row['bond_valence_deviation']:.3f})\n")
                    if row['bond_valence_sum'] < 1.9:
                        f.write("    → Oxygen is UNDERBONDED (weakly held)\n")
                    else:
                        f.write("    → Oxygen is OVERBONDED (strongly held but strained)\n")
                else:
                    f.write("(close to ideal)\n")
                
                f.write(f"  Covalency factor: {row['covalency_factor']:.3f}")
                if row['covalency_factor'] > 0.3:
                    f.write(" (significant covalent character → directional bonding)\n")
                else:
                    f.write(" (mostly ionic)\n")
                
                if row['avg_cation_oxidation'] > 0:
                    f.write(f"  Average neighbor oxidation state: +{row['avg_cation_oxidation']:.1f}\n")
                    if row['oxidation_variance'] > 0.5:
                        f.write(f"    → Mixed valence environment (variance: {row['oxidation_variance']:.2f})\n")
                
                # Structural features
                f.write(f"\nSTRUCTURAL FEATURES:\n")
                if row['is_edge_sharing']:
                    f.write("  ✓ EDGE-SHARING polyhedra detected\n")
                    f.write("    → Shared edges create structural flexibility\n")
                    f.write("    → Lower activation energy for O migration\n")
                    f.write("    → Common in materials with good ionic conductivity\n")
                
                if row['is_undercoordinated']:
                    f.write("  ✓ UNDERCOORDINATED site\n")
                    f.write("    → Fewer bonds to break during vacancy formation\n")
                    f.write("    → Lower formation energy expected\n")
                
                if row['voronoi_volume'] > 0:
                    f.write(f"  Voronoi volume: {row['voronoi_volume']:.3f} Ų\n")
                    avg_voronoi = df['voronoi_volume'].mean()
                    if row['voronoi_volume'] > avg_voronoi * 1.1:
                        f.write(f"    → LARGER than average ({avg_voronoi:.3f} Ų)\n")
                        f.write("    → More space available, easier to accommodate vacancy\n")
                
                # Chemical environment
                f.write(f"\nCHEMICAL ENVIRONMENT:\n")
                f.write(f"  Nearest O-O distance: {row['nearest_O_distance']:.3f} Å\n")
                if row['nearest_O_distance'] < 2.5:
                    f.write("    → Close O-O interaction may destabilize this site\n")
                f.write(f"  O-O coordination: {row['O_O_coordination']} oxygen neighbors within 3.5 Å\n")
                f.write(f"  Cation variety: {row['cation_variety']} different cation types\n")
                if row['cation_variety'] > 2:
                    f.write("    → Complex chemical environment\n")
                    f.write("    → Multiple migration pathways possible\n")
                
                # Overall assessment
                f.write(f"\nOVERALL ASSESSMENT:\n")
                f.write("  Key favorable factors:\n")
                
                # Rank factors by contribution
                factor_contributions = []
                if row['coordination_number'] < expected_coord:
                    contrib = self.weights['coordination'] * (1.0 - row['coordination_number']/expected_coord)
                    factor_contributions.append((contrib, f"Undercoordination (contributes {contrib:.3f} to score)"))
                
                if row['is_edge_sharing']:
                    contrib = self.weights['polyhedra'] * 1.0
                    factor_contributions.append((contrib, f"Edge-sharing polyhedra (contributes {contrib:.3f} to score)"))
                
                if row['local_distortion'] > 0.1:
                    contrib = self.weights['strain'] * min(row['local_distortion'], 1.0)
                    factor_contributions.append((contrib, f"High local strain (contributes {contrib:.3f} to score)"))
                
                if row['bond_valence_deviation'] > 0.2:
                    factor_contributions.append((0.1, f"Non-ideal bond valence (bonus: 0.100)"))
                
                if row['covalency_factor'] > 0.3:
                    contrib = self.weights['covalency'] * row['covalency_factor']
                    factor_contributions.append((contrib, f"Covalent character (contributes {contrib:.3f} to score)"))
                
                # Sort by contribution
                factor_contributions.sort(reverse=True)
                for contrib, description in factor_contributions:
                    f.write(f"    • {description}\n")
                
                # Predicted behavior
                f.write(f"\n  Predicted vacancy behavior:\n")
                if row['total_score'] > 0.7:
                    f.write("    → EXCELLENT candidate - expect low formation energy\n")
                    f.write("    → Should form easily under reducing conditions\n")
                elif row['total_score'] > 0.5:
                    f.write("    → GOOD candidate - moderate formation energy expected\n")
                    f.write("    → May require mild reducing conditions\n")
                else:
                    f.write("    → FAIR candidate - higher formation energy possible\n")
                    f.write("    → May require strong reducing conditions\n")
                
                # Specific predictions based on features
                if row['is_edge_sharing'] and row['is_undercoordinated']:
                    f.write("    → Combination of edge-sharing + undercoordination is particularly favorable\n")
                    f.write("    → Expect enhanced oxygen mobility through this site\n")
                
                if row['bond_valence_sum'] < 1.8:
                    f.write("    → Very underbonded oxygen - may be unstable even without vacancy\n")
                    f.write("    → Consider this site for intrinsic vacancy formation\n")
        
        print(f"\nReport saved to: {report_file}")
        
        # Also save to CSV for further analysis
        csv_file = "vacancy_prescreening_results.csv"
        unique_df.head(top_n).to_csv(csv_file, index=False)
        print(f"Results saved to: {csv_file}")
    
    def export_top_sites_for_vasp(self, df, top_n=10, output_dir="vasp_calculations"):
        """Export top sites as separate POSCAR files for VASP"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        unique_df = df[df['is_unique']]
        
        for i, (_, row) in enumerate(unique_df.head(top_n).iterrows()):
            # Create defect structure
            defect_structure = self.structure.copy()
            defect_structure.remove_sites([row['site_index']])
            
            # Save POSCAR
            output_file = os.path.join(output_dir, f"POSCAR_vacancy_rank{i+1}_site{row['site_index']}")
            Poscar(defect_structure).write_file(output_file)
        
        print(f"\nExported {min(top_n, len(unique_df))} structures to {output_dir}/")
        
        # Create a setup script
        with open(os.path.join(output_dir, "setup_vasp.sh"), 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Setup VASP calculations for top vacancy candidates\n\n")
            
            for i in range(min(top_n, len(unique_df))):
                f.write(f"mkdir -p rank{i+1}_calculation\n")
                f.write(f"cp POSCAR_vacancy_rank{i+1}_* rank{i+1}_calculation/POSCAR\n")
                f.write(f"cp INCAR KPOINTS POTCAR rank{i+1}_calculation/\n")
                f.write(f"cd rank{i+1}_calculation && sbatch job.sh && cd ..\n\n")
        
        print("Created setup_vasp.sh script for batch calculations")


def smart_vacancy_workflow(structure_file="POSCAR", 
                          top_prescreening=15,
                          top_for_dft=5,
                          vasp_config=None):
    """
    Complete smart workflow for vacancy formation analysis
    
    1. Pre-screen all O sites based on physical/chemical factors
    2. Select top candidates for DFT
    3. Calculate actual formation energies
    4. Re-rank based on DFT results
    """
    
    print("SMART OXYGEN VACANCY FORMATION WORKFLOW")
    print("="*70 + "\n")
    
    # Step 1: Pre-screening
    print("STEP 1: Pre-screening based on physical/chemical factors")
    print("-"*70)
    
    prescreener = SmartVacancyPrescreener(structure_file)
    df_ranked = prescreener.rank_oxygen_sites(top_n=top_prescreening)
    
    # Export top sites for VASP
    prescreener.export_top_sites_for_vasp(df_ranked, top_n=top_for_dft)
    
    print(f"\nPre-screening complete!")
    print(f"Top {top_for_dft} sites selected for DFT calculations\n")
    
    # Display summary
    print("Summary of top candidates:")
    print("-"*50)
    unique_df = df_ranked[df_ranked['is_unique']]
    for i, (_, row) in enumerate(unique_df.head(top_for_dft).iterrows()):
        print(f"{i+1}. Site {row['site_index']}: "
              f"Score={row['total_score']:.3f}, "
              f"Coord={row['coordination_number']}, "
              f"Edge-sharing={'Y' if row['is_edge_sharing'] else 'N'}")
    
    if vasp_config:
        print("\nSTEP 2: DFT calculations")
        print("-"*70)
        print("Run VASP calculations using the exported structures")
        print("Then use VacancyFormationEnergyCalculator to analyze results")
    
    return df_ranked


# Example usage
if __name__ == "__main__":
    # Run smart pre-screening
    results = smart_vacancy_workflow(
        structure_file="POSCAR",
        top_prescreening=15,  # Analyze top 15 from pre-screening
        top_for_dft=5         # Calculate DFT for top 5
    )
    
    print("\nNext steps:")
    print("1. Run VASP calculations in vasp_calculations/ directory")
    print("2. Use VacancyFormationEnergyCalculator to get formation energies")
    print("3. Re-rank based on actual DFT energies")
    print("4. Analyze migration for the best DFT-confirmed sites")