#!/usr/bin/env python3
"""
Report Generation Module
=======================
Generates comprehensive analysis reports for oxygen migration pathways
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class ReportGenerator:
    """Generates comprehensive reports for pathway analysis"""
    
    def generate_energy_report(self, viable_pathways: List[Dict], report_path: str, 
                              finder: Optional[object] = None):
        """
        Generate comprehensive energy report with pathway classification
        
        Args:
            viable_pathways: List of analyzed pathways
            report_path: Path to save the report
            finder: Optional OxygenMigrationFinder for additional details
        """
        with open(report_path, 'w') as f:
            self._write_header(f)
            self._write_pathway_classification(f, viable_pathways)
            self._write_thermodynamic_analysis(f, viable_pathways)
            self._write_statistics(f, viable_pathways)
            self._write_recommendations(f, viable_pathways)
            
            if finder:
                self._write_system_details(f, finder)
    
    def _write_header(self, f):
        """Write report header"""
        f.write("ENERGY-DRIVEN OXYGEN MIGRATION ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Analysis version: 2.0 (Improved)\n\n")
    
    def _write_pathway_classification(self, f, pathways: List[Dict]):
        """Write pathway classification section"""
        # Group pathways by class
        pathways_by_class = {'A': [], 'B': [], 'C': [], 'D': []}
        for pathway in pathways:
            pathways_by_class[pathway['pathway_class']].append(pathway)
        
        # Class A - Spontaneous
        f.write("="*70 + "\n")
        f.write("PATHWAY CLASSIFICATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("Class A - SPONTANEOUS PATHWAYS (Low barrier + Downhill):\n")
        f.write("-"*60 + "\n")
        if pathways_by_class['A']:
            self._write_pathway_details(f, pathways_by_class['A'])
        else:
            f.write("  [None found in this system]\n")
        
        # Class B - Favorable
        f.write("\n\nClass B - FAVORABLE PATHWAYS (Low barrier OR Downhill):\n")
        f.write("-"*60 + "\n")
        if pathways_by_class['B']:
            self._write_pathway_details(f, pathways_by_class['B'])
        else:
            f.write("  [None found]\n")
        
        # Class C - Accessible
        f.write("\n\nClass C - ACCESSIBLE PATHWAYS (Moderate barrier):\n")
        f.write("-"*60 + "\n")
        if pathways_by_class['C']:
            self._write_pathway_details(f, pathways_by_class['C'][:5])
        else:
            f.write("  [None found]\n")
        
        # Class D - Difficult
        f.write("\n\nClass D - DIFFICULT PATHWAYS (High barrier):\n")
        f.write("-"*60 + "\n")
        if pathways_by_class['D']:
            f.write(f"  {len(pathways_by_class['D'])} pathways with high barriers (>1.0 eV)\n")
            f.write("  [Details omitted - not favorable for migration]\n")
        else:
            f.write("  [None found]\n")
    
    def _write_pathway_details(self, f, pathways: List[Dict], limit: int = 10):
        """Write detailed information for pathways"""
        for i, pathway in enumerate(pathways[:limit]):
            f.write(f"\n{i+1}. {pathway['type'].upper()} {pathway['index']}")
            
            # Add atom info for better identification
            if pathway['type'] == 'individual':
                oxygen_idx = pathway.get('oxygen_index', -1)
                if oxygen_idx >= 0:
                    f.write(f" (O{oxygen_idx})")
            else:  # concerted
                movement_type = pathway.get('movement_info', {}).get('movement_type', 'unknown')
                f.write(f" ({movement_type})")
            
            f.write("\n")
            f.write(f"   Migration score: {pathway['migration_score']:.3f}\n")
            f.write(f"   Forward barrier: {pathway['forward_barrier']:.3f} eV\n")
            f.write(f"   Reverse barrier: {pathway['reverse_barrier']:.3f} eV\n")
            f.write(f"   Overall ΔE: {pathway['delta_energy']:+.3f} eV\n")
            
            # Warnings
            if pathway.get('no_intermediates'):
                f.write("   ⚠️  No intermediate states - barrier estimate uncertain\n")
            
            if pathway.get('has_intermediate_minimum'):
                f.write("   📍 Has intermediate minimum - consider segmented approach\n")
            
            # Energy profile
            f.write("   Profile: ")
            profile = pathway['energy_profile']
            for j, ep in enumerate(profile):
                rel_energy = ep['energy'] - pathway['start_energy']
                if j == 0:
                    f.write(f"0.00")
                else:
                    f.write(f" → {rel_energy:+.2f}")
                
                # Mark extrema
                if ep.get('hop_index', ep.get('step_index', j)) == pathway.get('max_index', -1):
                    f.write("*")
                elif ep.get('hop_index', ep.get('step_index', j)) == pathway.get('min_index', -1):
                    f.write("°")
            f.write(" eV\n")
            
            # Assessment
            f.write("   Assessment: ")
            if pathway['forward_barrier'] < 0.3:
                f.write("Excellent - spontaneous at room temperature")
            elif pathway['forward_barrier'] < 0.6:
                f.write("Good - thermally activated at moderate temperatures")
            elif pathway['forward_barrier'] < 1.0:
                f.write("Moderate - requires elevated temperatures")
            else:
                f.write("Difficult - high activation energy")
            
            if pathway['is_downhill']:
                f.write(" (thermodynamically favorable)")
            elif pathway['is_uphill']:
                f.write(" (thermodynamically unfavorable)")
            else:
                f.write(" (thermodynamically neutral)")
            f.write("\n")
    
    def _write_thermodynamic_analysis(self, f, pathways: List[Dict]):
        """Write thermodynamic analysis section"""
        f.write("\n\n" + "="*70 + "\n")
        f.write("THERMODYNAMIC ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        downhill_count = sum(1 for p in pathways if p['is_downhill'])
        uphill_count = sum(1 for p in pathways if p['is_uphill'])
        reversible_count = sum(1 for p in pathways if p['is_reversible'])
        
        f.write(f"Thermodynamic distribution:\n")
        f.write(f"  Downhill pathways: {downhill_count} ({downhill_count/len(pathways)*100:.1f}%)\n")
        f.write(f"  Uphill pathways: {uphill_count} ({uphill_count/len(pathways)*100:.1f}%)\n")
        f.write(f"  Reversible pathways: {reversible_count} ({reversible_count/len(pathways)*100:.1f}%)\n")
        
        if downhill_count == 0:
            f.write("\n⚠️  WARNING: All pathways are thermodynamically uphill\n")
            f.write("   - Consider reverse direction migration\n")
            f.write("   - May indicate oxygen-poor conditions favor vacancy migration\n")
            f.write("   - Actual migration may require external driving force\n")
        else:
            f.write(f"\n✓ Found {downhill_count} thermodynamically favorable pathways\n")
            
            # Find most favorable
            most_favorable = min((p for p in pathways if p['is_downhill']), 
                               key=lambda x: x['delta_energy'])
            f.write(f"   Most favorable: {most_favorable['type']} {most_favorable['index']} ")
            f.write(f"(ΔE = {most_favorable['delta_energy']:.3f} eV)\n")
    
    def _write_statistics(self, f, pathways: List[Dict]):
        """Write statistical analysis"""
        f.write("\n\n" + "="*70 + "\n")
        f.write("STATISTICAL ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        if not pathways:
            f.write("No pathways to analyze\n")
            return
        
        # Barrier statistics
        barriers = [p['forward_barrier'] for p in pathways]
        f.write("Forward barrier statistics:\n")
        f.write(f"  Minimum: {min(barriers):.3f} eV\n")
        f.write(f"  Maximum: {max(barriers):.3f} eV\n")
        f.write(f"  Average: {np.mean(barriers):.3f} eV\n")
        f.write(f"  Median: {np.median(barriers):.3f} eV\n")
        f.write(f"  Std dev: {np.std(barriers):.3f} eV\n")
        
        # Barrier distribution
        f.write("\nBarrier distribution:\n")
        f.write(f"  < 0.3 eV: {sum(1 for b in barriers if b < 0.3)} pathways\n")
        f.write(f"  0.3-0.5 eV: {sum(1 for b in barriers if 0.3 <= b < 0.5)} pathways\n")
        f.write(f"  0.5-1.0 eV: {sum(1 for b in barriers if 0.5 <= b < 1.0)} pathways\n")
        f.write(f"  > 1.0 eV: {sum(1 for b in barriers if b >= 1.0)} pathways\n")
        
        # Migration score statistics
        scores = [p['migration_score'] for p in pathways]
        f.write("\nMigration score statistics (lower is better):\n")
        f.write(f"  Best: {min(scores):.3f}\n")
        f.write(f"  Worst: {max(scores):.3f}\n")
        f.write(f"  Average: {np.mean(scores):.3f}\n")
        
        # Pathway type breakdown
        individual_count = sum(1 for p in pathways if p['type'] == 'individual')
        concerted_count = sum(1 for p in pathways if p['type'] == 'concerted')
        
        f.write("\nPathway type distribution:\n")
        f.write(f"  Individual: {individual_count} pathways\n")
        f.write(f"  Concerted: {concerted_count} pathways\n")
        
        if concerted_count > 0:
            # Concerted movement types
            movement_types = {}
            for p in pathways:
                if p['type'] == 'concerted':
                    mtype = p.get('movement_info', {}).get('movement_type', 'unknown')
                    movement_types[mtype] = movement_types.get(mtype, 0) + 1
            
            f.write("\nConcerted movement types:\n")
            for mtype, count in movement_types.items():
                f.write(f"  {mtype}: {count}\n")
        
        # Warnings
        no_intermediate_count = sum(1 for p in pathways if p.get('no_intermediates', False))
        if no_intermediate_count > 0:
            f.write(f"\n⚠️  {no_intermediate_count} pathways lack intermediate states\n")
            f.write("   Barriers for these pathways cannot be accurately assessed\n")
    
    def _write_recommendations(self, f, pathways: List[Dict]):
        """Write recommendations section"""
        f.write("\n\n" + "="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        if not pathways:
            f.write("⚠️  No viable pathways found\n")
            f.write("   - Check structure for errors\n")
            f.write("   - Consider different vacancy positions\n")
            f.write("   - Verify calculation parameters\n")
            return
        
        # Best pathway
        best = pathways[0]  # Already sorted by migration score
        f.write(f"BEST PATHWAY: {best['type'].upper()} {best['index']}")
        
        if best['type'] == 'individual':
            oxygen_idx = best.get('oxygen_index', -1)
            if oxygen_idx >= 0:
                f.write(f" (O{oxygen_idx} migration)")
        else:
            movement_type = best.get('movement_info', {}).get('movement_type', 'unknown')
            f.write(f" ({movement_type} movement)")
        
        f.write(f"\n  Classification: Class {best['pathway_class']}\n")
        f.write(f"  Migration score: {best['migration_score']:.3f}\n")
        f.write(f"  Forward barrier: {best['forward_barrier']:.3f} eV\n")
        f.write(f"  Energy change: {best['delta_energy']:+.3f} eV\n")
        
        # Pathway-specific recommendations
        f.write("\nPathway characteristics:\n")
        if best['pathway_class'] == 'A':
            f.write("  ✓ Excellent pathway - spontaneous at room temperature\n")
            f.write("  ✓ No thermal activation required\n")
            f.write("  ✓ Ideal for sensor applications\n")
        elif best['pathway_class'] == 'B':
            f.write("  ✓ Good pathway - favorable migration conditions\n")
            f.write("  ✓ Moderate thermal activation sufficient\n")
            f.write("  ✓ Suitable for most applications\n")
        elif best['pathway_class'] == 'C':
            f.write("  ⚗ Viable pathway - requires moderate thermal activation\n")
            f.write("  ⚗ Operating temperature > 500K recommended\n")
            f.write("  ⚗ May limit response times\n")
        else:
            f.write("  ✗ Challenging pathway - high activation energy required\n")
            f.write("  ✗ Operating temperature > 800K likely needed\n")
            f.write("  ✗ Consider alternative materials or dopants\n")
        
        # Application-specific recommendations
        f.write("\nAPPLICATION-SPECIFIC RECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        
        # Oxygen sensors
        f.write("\nFor oxygen sensor applications:\n")
        class_a_b = sum(1 for p in pathways if p['pathway_class'] in ['A', 'B'])
        if class_a_b > 0:
            f.write(f"  ✓ {class_a_b} favorable pathways available\n")
            f.write("  ✓ Good oxygen mobility expected\n")
            f.write("  ✓ Should show reasonable response times\n")
            
            # Operating temperature recommendation
            min_barrier = min(p['forward_barrier'] for p in pathways)
            if min_barrier < 0.3:
                f.write("  ✓ Room temperature operation possible\n")
            elif min_barrier < 0.5:
                f.write("  ⚗ Optimal operation at 400-600K\n")
            else:
                f.write("  ⚗ Requires operation above 600K\n")
        else:
            f.write("  ⚗ Limited oxygen mobility\n")
            f.write("  ⚗ High operating temperatures required\n")
            f.write("  ⚗ Consider material optimization\n")
        
        # Solid oxide fuel cells
        f.write("\nFor SOFC applications:\n")
        if best['forward_barrier'] < 0.5:
            f.write("  ✓ Excellent ionic conductor potential\n")
            f.write("  ✓ Low polarization resistance expected\n")
        elif best['forward_barrier'] < 1.0:
            f.write("  ⚗ Moderate ionic conductivity\n")
            f.write("  ⚗ Suitable for high-temperature operation\n")
        else:
            f.write("  ✗ Poor ionic conductor\n")
            f.write("  ✗ High polarization losses expected\n")
        
        # Oxygen separation membranes
        f.write("\nFor oxygen separation membranes:\n")
        downhill_paths = [p for p in pathways if p['is_downhill']]
        if downhill_paths and best['forward_barrier'] < 0.7:
            f.write("  ✓ Good oxygen permeation expected\n")
            f.write(f"  ✓ {len(downhill_paths)} thermodynamically favorable paths\n")
        else:
            f.write("  ⚗ Limited oxygen permeation\n")
            f.write("  ⚗ May require pressure differential\n")
        
        # NEB calculation recommendations
        f.write("\n\nNEB CALCULATION SETUP:\n")
        f.write("-"*40 + "\n")
        
        if best.get('has_intermediate_minimum'):
            f.write("⚠️  Intermediate minimum detected\n")
            f.write("   - Use climbing image NEB (CI-NEB)\n")
            f.write("   - Consider increased image density\n")
            f.write("   - May need segmented calculation\n")
        
        # Image number recommendation
        profile_points = len(best['energy_profile'])
        if profile_points == 2:
            f.write("⚠️  No intermediate structures available\n")
            f.write("   - Use linear interpolation with 5-7 images\n")
            f.write("   - Monitor convergence carefully\n")
        else:
            f.write(f"✓ {profile_points} structures available for interpolation\n")
            if profile_points >= 5:
                f.write("   - Can use provided intermediates directly\n")
            else:
                f.write(f"   - Interpolate to 5-7 images total\n")
        
        # Convergence recommendations
        f.write("\nConvergence recommendations:\n")
        f.write("  - EDIFFG = -0.05 eV/Å for initial convergence\n")
        f.write("  - EDIFFG = -0.03 eV/Å for final convergence\n")
        f.write("  - Use VTST optimizers (IOPT = 1 or 2)\n")
        
        # Material optimization suggestions
        if best['forward_barrier'] > 0.7:
            f.write("\n\nMATERIAL OPTIMIZATION SUGGESTIONS:\n")
            f.write("-"*40 + "\n")
            f.write("Consider the following to reduce migration barriers:\n")
            f.write("  - Increase oxygen vacancy concentration\n")
            f.write("  - Try aliovalent doping (e.g., Sr, Ca)\n")
            f.write("  - Explore different crystal orientations\n")
            f.write("  - Consider strain engineering\n")
    
    def _write_system_details(self, f, finder):
        """Write additional system details if finder is available"""
        f.write("\n\n" + "="*70 + "\n")
        f.write("SYSTEM DETAILS\n")
        f.write("="*70 + "\n\n")
        
        if hasattr(finder, 'structure'):
            structure = finder.structure
            f.write(f"Formula: {structure.composition.reduced_formula}\n")
            f.write(f"Space group: {structure.get_space_group_info()[0]}\n")
            f.write(f"Volume: {structure.volume:.2f} Ų\n")
            f.write(f"Density: {structure.density:.2f} g/cm³\n")
            
            if hasattr(finder, 'vacancy_site'):
                f.write(f"\nVacancy position: {finder.vacancy_site}\n")
            
            if hasattr(finder, 'oxygen_indices'):
                f.write(f"Number of oxygen atoms: {len(finder.oxygen_indices)}\n")
            
            if hasattr(finder, 'migration_graph'):
                f.write(f"\nMigration network:\n")
                f.write(f"  Nodes (sites): {finder.migration_graph.number_of_nodes()}\n")
                f.write(f"  Edges (hops): {finder.migration_graph.number_of_edges()}\n")
    
    def generate_summary_json(self, viable_pathways: List[Dict], output_path: str):
        """
        Generate machine-readable summary in JSON format
        
        Args:
            viable_pathways: List of analyzed pathways
            output_path: Path to save JSON summary
        """
        import json
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pathways': len(viable_pathways),
            'pathway_classes': {},
            'best_pathway': None,
            'statistics': {},
            'pathways': []
        }
        
        # Count by class
        for pathway in viable_pathways:
            pclass = pathway['pathway_class']
            summary['pathway_classes'][pclass] = summary['pathway_classes'].get(pclass, 0) + 1
        
        # Best pathway
        if viable_pathways:
            best = viable_pathways[0]
            summary['best_pathway'] = {
                'type': best['type'],
                'index': best['index'],
                'class': best['pathway_class'],
                'migration_score': best['migration_score'],
                'forward_barrier': best['forward_barrier'],
                'delta_energy': best['delta_energy']
            }
            
            # Statistics
            barriers = [p['forward_barrier'] for p in viable_pathways]
            summary['statistics'] = {
                'min_barrier': min(barriers),
                'max_barrier': max(barriers),
                'mean_barrier': np.mean(barriers),
                'std_barrier': np.std(barriers)
            }
        
        # Simplified pathway data
        for pathway in viable_pathways:
            summary['pathways'].append({
                'type': pathway['type'],
                'index': pathway['index'],
                'class': pathway['pathway_class'],
                'migration_score': pathway['migration_score'],
                'forward_barrier': pathway['forward_barrier'],
                'reverse_barrier': pathway['reverse_barrier'],
                'delta_energy': pathway['delta_energy'],
                'is_downhill': pathway['is_downhill'],
                'profile_length': len(pathway['energy_profile'])
            })
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def generate_csv_summary(self, viable_pathways: List[Dict], output_path: str):
        """
        Generate CSV summary for easy analysis
        
        Args:
            viable_pathways: List of analyzed pathways
            output_path: Path to save CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = [
                'pathway_type', 'pathway_index', 'pathway_class', 'migration_score',
                'forward_barrier', 'reverse_barrier', 'delta_energy', 'effective_barrier',
                'is_downhill', 'is_uphill', 'is_reversible', 'profile_points',
                'has_intermediate_minimum', 'oxygen_index', 'movement_type'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pathway in viable_pathways:
                row = {
                    'pathway_type': pathway['type'],
                    'pathway_index': pathway['index'],
                    'pathway_class': pathway['pathway_class'],
                    'migration_score': pathway['migration_score'],
                    'forward_barrier': pathway['forward_barrier'],
                    'reverse_barrier': pathway['reverse_barrier'],
                    'delta_energy': pathway['delta_energy'],
                    'effective_barrier': pathway['effective_barrier'],
                    'is_downhill': pathway['is_downhill'],
                    'is_uphill': pathway['is_uphill'],
                    'is_reversible': pathway['is_reversible'],
                    'profile_points': len(pathway['energy_profile']),
                    'has_intermediate_minimum': pathway.get('has_intermediate_minimum', False),
                    'oxygen_index': pathway.get('oxygen_index', -1) if pathway['type'] == 'individual' else '',
                    'movement_type': pathway.get('movement_info', {}).get('movement_type', '') if pathway['type'] == 'concerted' else ''
                }
                writer.writerow(row)