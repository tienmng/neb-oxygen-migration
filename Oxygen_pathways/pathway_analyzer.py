#!/usr/bin/env python3
"""
Pathway Analysis Module
======================
Handles energy profile analysis and pathway classification
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class PathwayAnalyzer:
    """Analyzes energy profiles and classifies pathways"""
    
    @staticmethod
    def calculate_migration_score(forward_barrier: float, delta_E: float, 
                                max_acceptable_barrier: float = 1.5) -> float:
        """
        Calculate multi-criteria migration score
        Lower score = better pathway
        """
        # Kinetic penalty (barrier height)
        kinetic_score = forward_barrier
        
        # Thermodynamic penalty (only penalize uphill paths)
        thermo_penalty = max(0, delta_E) * 0.5  # Weight thermodynamics less than kinetics
        
        # Bonus for downhill paths
        thermo_bonus = min(0, delta_E) * 0.3  # Small bonus for favorable thermodynamics
        
        # Total score
        total_score = kinetic_score + thermo_penalty + thermo_bonus
        
        # Additional penalty for very high barriers
        if forward_barrier > max_acceptable_barrier:
            total_score += (forward_barrier - max_acceptable_barrier) * 2
        
        return total_score
    
    @staticmethod
    def classify_pathway(forward_barrier: float, delta_E: float) -> str:
        """
        Classify pathway into categories A-D based on barrier and thermodynamics
        """
        if forward_barrier < 0.5 and delta_E < -0.1:
            return "A"  # Spontaneous - low barrier AND downhill
        elif forward_barrier < 0.5 or delta_E < -0.1:
            return "B"  # Favorable - low barrier OR downhill
        elif forward_barrier < 1.0:
            return "C"  # Accessible - moderate barrier
        else:
            return "D"  # Difficult - high barrier AND uphill
    
    @staticmethod
    def analyze_energy_profile(energy_profile: List[Dict], energy_tolerance: float = 0.1) -> Dict:
        """
        Analyze a complete energy profile and extract key metrics
        """
        if len(energy_profile) < 2:
            raise ValueError("Energy profile must have at least 2 points")
        
        # Extract energies
        all_energies = [ep['energy'] for ep in energy_profile]
        start_energy = all_energies[0]
        end_energy = all_energies[-1]
        
        # Find extrema
        min_energy = min(all_energies)
        max_energy = max(all_energies)
        min_index = all_energies.index(min_energy)
        max_index = all_energies.index(max_energy)
        
        # Calculate barriers
        forward_barrier = max_energy - start_energy
        reverse_barrier = max_energy - end_energy
        delta_energy = end_energy - start_energy
        
        # Determine pathway characteristics
        is_downhill = end_energy < start_energy - energy_tolerance
        is_uphill = end_energy > start_energy + energy_tolerance
        is_reversible = abs(end_energy - start_energy) <= energy_tolerance
        
        # Calculate driving force
        if is_downhill:
            driving_force = -delta_energy
        elif is_uphill:
            driving_force = delta_energy
        else:
            driving_force = 0
        
        # Check for intermediate minima
        has_intermediate_minimum = min_index not in [0, len(energy_profile) - 1]
        
        return {
            'start_energy': start_energy,
            'end_energy': end_energy,
            'min_energy': min_energy,
            'max_energy': max_energy,
            'min_index': min_index,
            'max_index': max_index,
            'delta_energy': delta_energy,
            'forward_barrier': forward_barrier,
            'reverse_barrier': reverse_barrier,
            'effective_barrier': min(forward_barrier, reverse_barrier),
            'is_downhill': is_downhill,
            'is_uphill': is_uphill,
            'is_reversible': is_reversible,
            'driving_force': driving_force,
            'has_intermediate_minimum': has_intermediate_minimum
        }
    
    @staticmethod
    def suggest_pathway_segments(energy_profile: List[Dict]) -> List[Dict]:
        """Suggest optimal pathway segments based on energy profile"""
        segments = []
        
        # Find all local minima
        for i in range(1, len(energy_profile) - 1):
            if (energy_profile[i]['energy'] < energy_profile[i-1]['energy'] and 
                energy_profile[i]['energy'] < energy_profile[i+1]['energy']):
                
                # Suggest segments from this minimum
                segments.append({
                    'type': 'local_minimum',
                    'position': energy_profile[i].get('position', f'step_{i}'),
                    'index': i,
                    'energy': energy_profile[i]['energy']
                })
        
        return segments