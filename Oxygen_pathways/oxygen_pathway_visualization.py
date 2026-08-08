#!/usr/bin/env python3
"""
Oxygen Migration Pathway Visualization Module
Plots hopping pathways created by oxygen_hopping_vacancy_pathway_generation.py
Adapted from trajectory.py visualization capabilities
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Add this before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.local_env import CrystalNN
import networkx as nx
from PIL import Image
import os
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class Arrow3D(FancyArrowPatch):
    """3D arrow for pathway visualization"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)


class OxygenPathwayVisualizer:
    def __init__(self, structure_file="POSCAR", finder_object=None):
        """
        Initialize the oxygen pathway visualizer
        
        Args:
            structure_file: Path to structure file (POSCAR or CIF)
            finder_object: Optional UniversalVacancyOxygenMigrationFinder object
        """
        # Load structure
        if structure_file.endswith('.cif'):
            self.structure = Structure.from_file(structure_file)
        else:
            self.structure = Poscar.from_file(structure_file).structure
        
        self.lattice_matrix = self.structure.lattice.matrix
        
        # Store finder object if provided
        self.finder = finder_object
        
        # Color schemes
        self.colors = {
            'oxygen': '#FF0000',           # Red for oxygen atoms
            'oxygen_vacancy': '#FFD700',   # Gold for oxygen vacancies
            'cation': '#0000FF',           # Blue for cations
            'cation_vacancy': '#87CEEB',   # Sky blue for cation vacancies
            'anion': '#00FF00',            # Green for other anions
            'anion_vacancy': '#90EE90',    # Light green for anion vacancies
            'unknown_vacancy': '#808080',   # Gray for unknown vacancies
            'pathway_normal': '#FF6347',    # Tomato for normal pathways
            'pathway_enhanced': '#FF1493',  # Deep pink for enhanced pathways
            'pathway_concerted': '#9370DB', # Medium purple for concerted movements
        }
        
        # Pathway styles
        self.pathway_styles = {
            'normal': {'linewidth': 2, 'alpha': 0.7, 'arrowstyle': '-|>'},
            'enhanced': {'linewidth': 3, 'alpha': 0.8, 'arrowstyle': '-|>'},
            'concerted': {'linewidth': 2.5, 'alpha': 0.9, 'arrowstyle': '<->'},
            'swap': {'linewidth': 3, 'alpha': 0.8, 'arrowstyle': '<->'},
            'ring': {'linewidth': 2.5, 'alpha': 0.85, 'arrowstyle': '-|>', 'linestyle': '--'},
            'chain': {'linewidth': 2.5, 'alpha': 0.8, 'arrowstyle': '-|>', 'linestyle': ':'}
        }
        
        # Extract atom positions
        self._extract_atom_positions()
    
    def _extract_atom_positions(self):
        """Extract positions of different atom types"""
        self.oxygen_positions = []
        self.oxygen_indices = []
        self.cation_positions = []
        self.anion_positions = []
        
        for i, site in enumerate(self.structure):
            if site.species_string == "O":
                self.oxygen_positions.append(site.coords)
                self.oxygen_indices.append(i)
            else:
                # Get the element from the species string
                element_symbol = site.species_string.split('+')[0].split('-')[0]
                from pymatgen.core import Element
                element = Element(element_symbol)
                
                if element.is_metal:
                    self.cation_positions.append(site.coords)
                else:
                    self.anion_positions.append(site.coords)
        
        self.oxygen_positions = np.array(self.oxygen_positions)
        self.cation_positions = np.array(self.cation_positions) if self.cation_positions else np.array([]).reshape(0,3)
        self.anion_positions = np.array(self.anion_positions) if self.anion_positions else np.array([]).reshape(0,3)
    
    def plot_lattice(self, ax, style='full', label=False):
        """
        Plot the lattice framework
        
        Args:
            ax: matplotlib 3D axis
            style: 'full' for complete box, 'minimal' for just axes
            label: whether to label lattice points
        """
        origin = np.zeros(3)
        a, b, c = self.lattice_matrix
        
        if style == 'full':
            # All edges of the unit cell
            edges = [
                (origin, a), (origin, b), (origin, c),
                (a, a+b), (a, a+c), (b, a+b), (b, b+c),
                (c, a+c), (c, b+c), (a+b, a+b+c),
                (a+c, a+b+c), (b+c, a+b+c)
            ]
        else:  # minimal
            edges = [(origin, a), (origin, b), (origin, c)]
        
        for start, end in edges:
            edge_points = np.array([start, end]).T
            ax.plot(edge_points[0], edge_points[1], edge_points[2], 
                   'k-', linewidth=0.5, alpha=0.3)
        
        # Add axis labels
        if label:
            ax.text(*(a/2), 'a', fontsize=8)
            ax.text(*(b/2), 'b', fontsize=8)
            ax.text(*(c/2), 'c', fontsize=8)

    def get_bonds(self, max_bond_length=3.0):
        """
        Detect bonds in the structure using CrystalNN
        
        Returns:
            List of tuples (site1_idx, site2_idx, distance)
        """
        bonds = []
        nn = CrystalNN()
        
        for i in range(len(self.structure)):
            try:
                neighbors = nn.get_nn_info(self.structure, i)
                for neighbor in neighbors:
                    j = neighbor['site_index']
                    if i < j:  # Avoid duplicates
                        distance = neighbor['weight']
                        if distance <= max_bond_length:
                            bonds.append((i, j, distance))
            except:
                continue
        
        return bonds

    def plot_bonds(self, ax, bonds=None, bond_color='gray', bond_width=0.5, bond_alpha=0.3):
        """Plot bonds between atoms"""
        if bonds is None:
            bonds = self.get_bonds()
        
        for bond in bonds:
            i, j, distance = bond
            pos1 = self.structure[i].coords
            pos2 = self.structure[j].coords
            
            # Check for PBC and get minimum image
            pos2_wrapped = self.get_pbc_minimum_image_coords(pos1, pos2)
            
            # Draw bond
            ax.plot([pos1[0], pos2_wrapped[0]], 
                    [pos1[1], pos2_wrapped[1]], 
                    [pos1[2], pos2_wrapped[2]], 
                    color=bond_color, 
                    linewidth=bond_width, 
                    alpha=bond_alpha)

    def plot_atoms_with_labels(self, ax, show_oxygen=True, show_cations=True, 
                            show_anions=True, atom_size=100, label_offset=0.1,
                            label_fontsize=10, show_indices=False):
        """Plot atoms with element labels"""
        # Track element counts for labeling
        element_counts = {}
        
        for i, site in enumerate(self.structure):
            coords = site.coords
            
            # Get element symbol (remove charge)
            element = site.species_string.split('+')[0].split('-')[0]
            
            # Count elements for unique labeling
            if element not in element_counts:
                element_counts[element] = 0
            element_counts[element] += 1
            
            # Determine if we should plot this atom
            should_plot = False
            color = 'gray'
            size = atom_size
            
            if element == "O" and show_oxygen:
                should_plot = True
                color = self.colors['oxygen']
                size = atom_size
            elif element != "O":
                from pymatgen.core import Element
                try:
                    elem_obj = Element(element)
                    if elem_obj.is_metal and show_cations:
                        should_plot = True
                        color = self.colors['cation']
                        size = atom_size * 0.8
                    elif not elem_obj.is_metal and show_anions:
                        should_plot = True
                        color = self.colors['anion']
                        size = atom_size * 0.8
                except:
                    pass
            
            if should_plot:
                # Plot atom
                ax.scatter(coords[0], coords[1], coords[2],
                        c=color, s=size,
                        edgecolors='black', linewidth=0.5,
                        alpha=0.9)
                
                # Add label
                if show_indices:
                    label = f"{element}{i}"
                else:
                    label = element
                
                ax.text(coords[0] + label_offset, 
                    coords[1] + label_offset, 
                    coords[2] + label_offset,
                    label, 
                    fontsize=label_fontsize,
                    weight='bold',
                    color='black',
                    ha='left',
                    va='bottom')
    def plot_atoms(self, ax, show_oxygen=True, show_cations=True, 
                   show_anions=True, atom_size=100, label_atoms=False):
        """Plot atoms in the structure"""
        if show_oxygen and len(self.oxygen_positions) > 0:
            ax.scatter(self.oxygen_positions[:, 0],
                      self.oxygen_positions[:, 1],
                      self.oxygen_positions[:, 2],
                      c=self.colors['oxygen'],
                      s=atom_size,
                      edgecolors='black',
                      linewidth=0.5,
                      label='Oxygen',
                      alpha=0.9)
            
            if label_atoms:
                for i, (pos, idx) in enumerate(zip(self.oxygen_positions, self.oxygen_indices)):
                    ax.text(pos[0], pos[1], pos[2], f'O{idx}', 
                           fontsize=6, ha='center')
        
        if show_cations and len(self.cation_positions) > 0:
            ax.scatter(self.cation_positions[:, 0],
                      self.cation_positions[:, 1],
                      self.cation_positions[:, 2],
                      c=self.colors['cation'],
                      s=atom_size*0.8,
                      edgecolors='black',
                      linewidth=0.5,
                      label='Cations',
                      alpha=0.7)
        
        if show_anions and len(self.anion_positions) > 0:
            ax.scatter(self.anion_positions[:, 0],
                      self.anion_positions[:, 1],
                      self.anion_positions[:, 2],
                      c=self.colors['anion'],
                      s=atom_size*0.8,
                      edgecolors='black',
                      linewidth=0.5,
                      label='Other anions',
                      alpha=0.7)
    
    def plot_vacancies(self, ax, vacancy_dict, vacancy_size=150):
        """Plot vacancy sites from finder object"""
        if not vacancy_dict:
            return
        
        for vac_type, vacancies in vacancy_dict.items():
            if not vacancies:
                continue
                
            # Convert fractional to Cartesian coordinates
            vac_coords = []
            for vac in vacancies:
                cart_coords = self.structure.lattice.get_cartesian_coords(vac['coords'])
                vac_coords.append(cart_coords)
            
            if not vac_coords:
                continue
                
            vac_coords = np.array(vac_coords)
            
            # Determine color and label
            if vac_type == 'oxygen_vacancies':
                color = self.colors['oxygen_vacancy']
                label = 'O vacancies'
            elif vac_type == 'cationic_vacancies':
                color = self.colors['cation_vacancy']
                label = 'Cation vacancies'
            elif vac_type == 'anionic_vacancies':
                color = self.colors['anion_vacancy']
                label = 'Anion vacancies'
            else:
                color = self.colors['unknown_vacancy']
                label = 'Unknown vacancies'
            
            ax.scatter(vac_coords[:, 0],
                      vac_coords[:, 1],
                      vac_coords[:, 2],
                      c=color,
                      s=vacancy_size,
                      marker='s',
                      edgecolors='black',
                      linewidth=1,
                      label=label,
                      alpha=0.6)
    def plot_migration_pathway(self, ax, pathway, style='normal', 
                            show_intermediates=True, arrow_scale=15):
        """Plot a single migration pathway with PBC consideration"""
        node_sequence = pathway['node_sequence']
        
        if len(node_sequence) < 2:
            return
        
        # Get coordinates for each node
        coords = []
        for node in node_sequence:
            node_data = self.finder.migration_graph.nodes[node]
            cart_coords = self.structure.lattice.get_cartesian_coords(node_data['coords'])
            coords.append(cart_coords)
        
        coords = np.array(coords)
        
        # Determine pathway color based on type
        if 'pathway_types' in pathway:
            if 'cation_influenced' in pathway['pathway_types']:
                color = self.colors['pathway_enhanced']
            else:
                color = self.colors['pathway_normal']
        else:
            color = self.colors['pathway_normal']
        
        # Plot pathway segments with PBC
        for i in range(len(coords) - 1):
            start = coords[i]
            
            # Get minimum image position for end point
            end_wrapped = self.get_pbc_minimum_image_coords(start, coords[i + 1])
            
            # Check if this hop crosses a boundary
            distance_direct = np.linalg.norm(coords[i + 1] - start)
            distance_wrapped = np.linalg.norm(end_wrapped - start)
            
            # Create arrow properties FIRST
            arrow_props = self.pathway_styles.get(style, self.pathway_styles['normal']).copy()
            arrow_props['color'] = color
            arrow_props['mutation_scale'] = arrow_scale
            
            # Then modify alpha based on whether it crosses boundary
            if distance_wrapped < distance_direct * 0.9:  # Crosses boundary
                # Use wrapped coordinates and mark as boundary crossing
                end = end_wrapped
                arrow_props['alpha'] = arrow_props.get('alpha', 0.7) * 0.7  # Make slightly transparent
            else:
                end = coords[i + 1]
                # Keep original alpha
            
            # Create 3D arrow
            arrow = Arrow3D([start[0], end[0]], 
                        [start[1], end[1]], 
                        [start[2], end[2]], 
                        **arrow_props)
            ax.add_artist(arrow)
            
            # Show intermediate positions
            if show_intermediates and i > 0:
                ax.scatter(start[0], start[1], start[2], 
                        c='white', s=50, edgecolors=color, 
                        linewidth=2, marker='o', alpha=0.8)
            
            # Add boundary crossing indicator
            if distance_wrapped < distance_direct * 0.9:
                mid_point = (start + end) / 2
                ax.text(mid_point[0], mid_point[1], mid_point[2], 
                    'PBC', fontsize=6, color='red', ha='center')
    
    def plot_concerted_movement(self, ax, movement, show_steps=False):
        """Plot concerted movements (swaps, rings, chains)"""
        movement_type = movement['movement_type']
        
        # FIX: Update to match the actual movement type values
        if movement_type == 'swaps':  # Changed from 'direct_swap'
            self._plot_swap(ax, movement)
        elif movement_type == 'rings':  # Changed from 'ring_exchange'
            self._plot_ring(ax, movement)
        elif movement_type == 'chains':  # Changed from 'chain_movement'
            self._plot_chain(ax, movement)
        else:
            print(f"WARNING: Unknown movement type: {movement_type}")
    
    def _plot_swap(self, ax, movement):
        """Plot direct swap movement"""
        o1_idx = movement['oxygen1']
        o2_idx = movement['oxygen2']
        
        pos1 = self.structure[o1_idx].coords
        pos2 = self.structure[o2_idx].coords
        
        # Bidirectional arrow
        arrow_props = self.pathway_styles['swap'].copy()
        arrow_props['color'] = self.colors['pathway_concerted']
        arrow_props['mutation_scale'] = 20
        
        arrow = Arrow3D([pos1[0], pos2[0]], 
                       [pos1[1], pos2[1]], 
                       [pos1[2], pos2[2]], 
                       **arrow_props)
        ax.add_artist(arrow)
        
        # Label the swap
        mid_point = (pos1 + pos2) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2], 
               'swap', fontsize=8, ha='center')
    
    def _plot_ring(self, ax, movement):
        """Plot ring exchange movement"""
        cycle_nodes = movement['cycle_nodes']
        
        # Get positions
        positions = []
        for node in cycle_nodes:
            node_data = self.finder.migration_graph.nodes[node]
            cart_coords = self.structure.lattice.get_cartesian_coords(node_data['coords'])
            positions.append(cart_coords)
        
        # Plot ring segments
        for i in range(len(positions)):
            start = positions[i]
            end = positions[(i + 1) % len(positions)]
            
            arrow_props = self.pathway_styles['ring'].copy()
            arrow_props['color'] = self.colors['pathway_concerted']
            arrow_props['mutation_scale'] = 15
            
            arrow = Arrow3D([start[0], end[0]], 
                          [start[1], end[1]], 
                          [start[2], end[2]], 
                          **arrow_props)
            ax.add_artist(arrow)
    
    def _plot_chain(self, ax, movement):
        """Plot chain movement"""
        chain_nodes = movement['chain_nodes']
        
        # Get positions
        positions = []
        for node in chain_nodes:
            node_data = self.finder.migration_graph.nodes[node]
            cart_coords = self.structure.lattice.get_cartesian_coords(node_data['coords'])
            positions.append(cart_coords)
        
        # Plot chain segments
        for i in range(len(positions) - 1):
            start = positions[i]
            end = positions[i + 1]
            
            arrow_props = self.pathway_styles['chain'].copy()
            arrow_props['color'] = self.colors['pathway_concerted']
            arrow_props['mutation_scale'] = 15
            
            arrow = Arrow3D([start[0], end[0]], 
                          [start[1], end[1]], 
                          [start[2], end[2]], 
                          **arrow_props)
            ax.add_artist(arrow)
    def create_concerted_movement_animation(self, movement, output_file='concerted_animation.gif',
                                        fps=5, figsize=(10, 8)):
        """
        Create an animation for concerted movements
        
        Args:
            movement: Concerted movement dictionary
            output_file: Output filename for GIF
            fps: Frames per second
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get movement type and create appropriate frames
        movement_type = movement['movement_type']
        
        if movement_type == 'swaps':
            frames = self._create_swap_frames(movement)
        elif movement_type == 'rings':
            frames = self._create_ring_frames(movement)
        elif movement_type == 'chains':
            frames = self._create_chain_frames(movement)
        else:
            print(f"Unknown movement type: {movement_type}")
            return None
        
        def update(frame_idx):
            ax.clear()
            
            # Plot lattice
            self.plot_lattice(ax, style='minimal')
            
            # Plot static atoms (except those involved in movement)
            moving_indices = frames['moving_indices']
            
            # Plot non-moving oxygen atoms
            for i, pos in enumerate(self.oxygen_positions):
                if self.oxygen_indices[i] not in moving_indices:
                    ax.scatter(pos[0], pos[1], pos[2],
                            c=self.colors['oxygen'],
                            s=80,
                            edgecolors='black',
                            linewidth=0.5,
                            alpha=0.7)
            
            # Plot other atoms
            if len(self.cation_positions) > 0:
                ax.scatter(self.cation_positions[:, 0],
                        self.cation_positions[:, 1],
                        self.cation_positions[:, 2],
                        c=self.colors['cation'],
                        s=60,
                        alpha=0.5)
            
            # Plot vacancies
            if self.finder:
                self.plot_vacancies(ax, self.finder.all_vacancies, vacancy_size=100)
            
            # Plot moving atoms at interpolated positions
            current_positions = frames['positions'][frame_idx]
            for i, (idx, pos) in enumerate(zip(moving_indices, current_positions)):
                ax.scatter(pos[0], pos[1], pos[2],
                        c='yellow',
                        s=150,
                        edgecolors='red',
                        linewidth=2,
                        label=f'O{idx}' if i == 0 else "")
            
            # Add movement arrows/traces
            if frame_idx > 0:
                for i in range(len(moving_indices)):
                    trace = np.array([frames['positions'][j][i] for j in range(frame_idx + 1)])
                    ax.plot(trace[:, 0], trace[:, 1], trace[:, 2],
                        'r--', linewidth=2, alpha=0.5)
            
            ax.set_xlabel('x (Å)')
            ax.set_ylabel('y (Å)')
            ax.set_zlabel('z (Å)')
            ax.view_init(elev=30, azim=45)
            
            # Add title and frame counter
            ax.set_title(f'{movement_type.title()} Movement Animation', fontsize=12)
            ax.text2D(0.05, 0.95, f'Frame: {frame_idx+1}/{len(frames["positions"])}', 
                    transform=ax.transAxes, fontsize=10)
            
            if frame_idx == 0:
                ax.legend(loc='upper right')
            
            return ax,
        
        anim = FuncAnimation(fig, update, frames=len(frames['positions']),
                        interval=1000/fps, blit=False)
        
        # Save animation
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"Concerted movement animation saved to {output_file}")
        
        plt.close()
        return anim
    def _create_swap_frames(self, movement, n_frames=20):
        """Create interpolated frames for swap movement with PBC"""
        o1_idx = movement['oxygen1']
        o2_idx = movement['oxygen2']
        
        o1_start = self.structure[o1_idx].coords
        o2_start = self.structure[o2_idx].coords
        
        # Get PBC-aware paths
        path1to2 = self.get_pbc_interpolated_path(o1_start, o2_start, n_frames + 1)
        path2to1 = self.get_pbc_interpolated_path(o2_start, o1_start, n_frames + 1)
        
        positions = []
        for i in range(n_frames + 1):
            positions.append([path1to2[i], path2to1[i]])
        
        return {
            'moving_indices': [o1_idx, o2_idx],
            'positions': positions
        }

    def _create_ring_frames(self, movement, n_frames=30):
        """Create interpolated frames for ring movement"""
        cycle_nodes = movement['cycle_nodes']
        
        # Get oxygen indices and positions involved in the ring
        moving_indices = []
        start_positions = []
        
        for node in cycle_nodes:
            node_data = self.finder.migration_graph.nodes[node]
            if node_data['site_type'] == 'occupied_oxygen':
                # Find corresponding oxygen index
                node_coords = node_data['coords']
                for o_idx in self.oxygen_indices:
                    o_coords = self.structure[o_idx].frac_coords
                    if np.allclose(node_coords, o_coords, atol=0.01):
                        moving_indices.append(o_idx)
                        start_positions.append(self.structure[o_idx].coords)
                        break
        
        # Create circular motion frames
        positions = []
        for frame in range(n_frames + 1):
            t = frame / n_frames
            frame_positions = []
            
            for i, start_pos in enumerate(start_positions):
                # Next position in the ring
                next_idx = (i + 1) % len(start_positions)
                end_pos = start_positions[next_idx]
                
                # Interpolate
                current_pos = start_pos + t * (end_pos - start_pos)
                frame_positions.append(current_pos)
            
            positions.append(frame_positions)
        
        return {
            'moving_indices': moving_indices,
            'positions': positions
        }

    def _create_chain_frames(self, movement, n_frames=25):
        """Create interpolated frames for chain movement"""
        chain_nodes = movement['chain_nodes']
        
        # Similar to ring but linear motion
        moving_indices = []
        positions_sequence = []
        
        # Get positions along the chain
        for node in chain_nodes:
            node_data = self.finder.migration_graph.nodes[node]
            cart_coords = self.structure.lattice.get_cartesian_coords(node_data['coords'])
            positions_sequence.append(cart_coords)
        
        # Find oxygen atoms in the chain
        for i in range(len(chain_nodes) - 1):
            node_data = self.finder.migration_graph.nodes[chain_nodes[i]]
            if node_data['site_type'] == 'occupied_oxygen':
                # Find corresponding oxygen index
                for o_idx in self.oxygen_indices:
                    o_coords = self.structure[o_idx].frac_coords
                    if np.allclose(node_data['coords'], o_coords, atol=0.01):
                        moving_indices.append(o_idx)
                        break
        
        # Create frames for chain motion
        positions = []
        for frame in range(n_frames + 1):
            t = frame / n_frames
            frame_positions = []
            
            for i, o_idx in enumerate(moving_indices):
                start_pos = positions_sequence[i]
                end_pos = positions_sequence[i + 1]
                current_pos = start_pos + t * (end_pos - start_pos)
                frame_positions.append(current_pos)
            
            positions.append(frame_positions)
        
        return {
            'moving_indices': moving_indices,
            'positions': positions
        }
    
    def create_enhanced_pathway_plot(self, pathways=None, concerted_movements=None,
                               show_bonds=True, show_labels=True, 
                               show_indices=False, figsize=(14, 10),
                               view_angles=(30, 45), save_path=None):
        """
        Create enhanced pathway visualization with bonds and labels
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot lattice
        self.plot_lattice(ax, style='minimal', label=True)
        
        # Plot bonds first (so they appear behind atoms)
        if show_bonds:
            bonds = self.get_bonds()
            self.plot_bonds(ax, bonds)
        
        # Plot atoms with labels
        if show_labels:
            self.plot_atoms_with_labels(ax, show_indices=show_indices)
        else:
            self.plot_atoms(ax)
        
        # Plot vacancies with labels
        if self.finder and self.finder.all_vacancies:
            for vac_type, vacancies in self.finder.all_vacancies.items():
                if vacancies:
                    for i, vac in enumerate(vacancies):
                        cart_coords = self.structure.lattice.get_cartesian_coords(vac['coords'])
                        
                        # Determine vacancy appearance
                        if vac_type == 'oxygen_vacancies':
                            color = self.colors['oxygen_vacancy']
                            label = 'Vo'
                            size = 150
                        elif vac_type == 'cationic_vacancies':
                            color = self.colors['cation_vacancy']
                            label = f'V{vac.get("expected_species", "M")}'
                            size = 140
                        else:
                            color = self.colors['unknown_vacancy']
                            label = 'V'
                            size = 130
                        
                        # Plot vacancy
                        ax.scatter(cart_coords[0], cart_coords[1], cart_coords[2],
                                c=color, s=size, marker='s',
                                edgecolors='black', linewidth=1,
                                alpha=0.6)
                        
                        # Add vacancy label
                        if show_labels:
                            ax.text(cart_coords[0] + 0.1, 
                                cart_coords[1] + 0.1, 
                                cart_coords[2] + 0.1,
                                label, 
                                fontsize=9,
                                style='italic',
                                color='darkred',
                                ha='left',
                                va='bottom')
        
        # Plot migration pathways
        if pathways:
            for i, pathway in enumerate(pathways):
                self.plot_migration_pathway(ax, pathway, style='normal')
        
        # Plot concerted movements
        if concerted_movements:
            for movement in concerted_movements:
                self.plot_concerted_movement(ax, movement)
        
        # Set view and labels
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.set_xlabel('x (Å)', fontsize=12, weight='bold')
        ax.set_ylabel('y (Å)', fontsize=12, weight='bold')
        ax.set_zlabel('z (Å)', fontsize=12, weight='bold')
        
        # Create custom legend
        legend_elements = []
        
        # Add atom types to legend
        for site in self.structure:
            element = site.species_string.split('+')[0].split('-')[0]
            if element == "O":
                if not any(e.get_label() == f'{element} atoms' for e in legend_elements):
                    legend_elements.append(plt.scatter([], [], c=self.colors['oxygen'], 
                                                    s=100, label=f'{element} atoms'))
            else:
                from pymatgen.core import Element
                elem_obj = Element(element)
                if elem_obj.is_metal:
                    if not any(e.get_label() == f'{element} atoms' for e in legend_elements):
                        legend_elements.append(plt.scatter([], [], c=self.colors['cation'], 
                                                        s=80, label=f'{element} atoms'))
        
        # Add vacancies to legend
        if self.finder:
            if self.finder.all_vacancies.get('oxygen_vacancies'):
                legend_elements.append(plt.scatter([], [], c=self.colors['oxygen_vacancy'], 
                                                s=100, marker='s', label='O vacancies'))
            if self.finder.all_vacancies.get('cationic_vacancies'):
                legend_elements.append(plt.scatter([], [], c=self.colors['cation_vacancy'], 
                                                s=100, marker='s', label='Cation vacancies'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title('Enhanced Oxygen Migration Pathways', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced plot saved to {save_path}")
        
        plt.show()
        return fig, ax

    def create_enhanced_animation(self, pathway=None, movement=None, 
                                output_file='enhanced_animation.gif',
                                show_bonds=True, show_labels=True,
                                fps=5, n_frames=20, figsize=(12, 9)):
        """
        Create enhanced animation with bonds and labels
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine if this is a pathway or concerted movement
        if pathway is not None:
            oxygen_idx = pathway['oxygen_index']
            node_sequence = pathway['node_sequence']
            coords = []
            for node in node_sequence:
                node_data = self.finder.migration_graph.nodes[node]
                cart_coords = self.structure.lattice.get_cartesian_coords(node_data['coords'])
                coords.append(cart_coords)
            
            # Create interpolated positions
            oxygen_positions = []
            for i in range(len(coords) - 1):
                segment_path = self.get_pbc_interpolated_path(coords[i], coords[i+1], 
                                                            n_frames // (len(coords) - 1))
                oxygen_positions.extend(segment_path[:-1])
            oxygen_positions.append(coords[-1])
            
            moving_indices = [oxygen_idx]
            
        elif movement is not None:
            # Handle concerted movement
            if movement['movement_type'] == 'swaps':
                frames_data = self._create_swap_frames(movement, n_frames)
            elif movement['movement_type'] == 'rings':
                frames_data = self._create_ring_frames(movement, n_frames)
            else:
                frames_data = self._create_chain_frames(movement, n_frames)
            
            moving_indices = frames_data['moving_indices']
            oxygen_positions = frames_data['positions']
        
        # Get bonds once
        bonds = self.get_bonds() if show_bonds else None
        
        def update(frame_idx):
            ax.clear()
            
            # Plot lattice
            self.plot_lattice(ax, style='minimal', label=True)
            
            # Plot bonds
            if bonds:
                self.plot_bonds(ax, bonds)
            
            # Plot non-moving atoms with labels
            for i, site in enumerate(self.structure):
                if pathway and i == moving_indices[0]:
                    continue  # Skip moving atom in pathway
                elif movement and i in moving_indices:
                    continue  # Skip moving atoms in concerted movement
                
                coords = site.coords
                element = site.species_string.split('+')[0].split('-')[0]
                
                # Determine color
                if element == "O":
                    color = self.colors['oxygen']
                    size = 100
                else:
                    from pymatgen.core import Element
                    elem_obj = Element(element)
                    if elem_obj.is_metal:
                        color = self.colors['cation']
                    else:
                        color = self.colors['anion']
                    size = 80
                
                # Plot atom
                ax.scatter(coords[0], coords[1], coords[2],
                        c=color, s=size,
                        edgecolors='black', linewidth=0.5,
                        alpha=0.7)
                
                # Add label
                if show_labels:
                    ax.text(coords[0] + 0.1, coords[1] + 0.1, coords[2] + 0.1,
                        element, fontsize=9, weight='bold', color='black')
            
            # Plot moving atoms
            if pathway:
                # Single moving oxygen
                current_pos = oxygen_positions[min(frame_idx, len(oxygen_positions)-1)]
                ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                        c='yellow', s=200, edgecolors='red', linewidth=3)
                
                if show_labels:
                    ax.text(current_pos[0] + 0.15, current_pos[1] + 0.15, current_pos[2] + 0.15,
                        'O*', fontsize=12, weight='bold', color='red')
                
                # Plot trace
                if frame_idx > 0:
                    trace = np.array(oxygen_positions[:frame_idx+1])
                    ax.plot(trace[:, 0], trace[:, 1], trace[:, 2],
                        'r--', linewidth=2, alpha=0.6)
            
            elif movement:
                # Multiple moving atoms
                current_positions = oxygen_positions[frame_idx]
                for idx, pos in zip(moving_indices, current_positions):
                    ax.scatter(pos[0], pos[1], pos[2],
                            c='yellow', s=180, edgecolors='red', linewidth=2.5)
                    
                    if show_labels:
                        ax.text(pos[0] + 0.15, pos[1] + 0.15, pos[2] + 0.15,
                            f'O{idx}*', fontsize=10, weight='bold', color='red')
            
            # Plot vacancies
            if self.finder:
                self.plot_vacancies(ax, self.finder.all_vacancies, vacancy_size=120)
            
            ax.set_xlabel('x (Å)', fontsize=11, weight='bold')
            ax.set_ylabel('y (Å)', fontsize=11, weight='bold')
            ax.set_zlabel('z (Å)', fontsize=11, weight='bold')
            ax.view_init(elev=30, azim=45)
            
            # Add title with frame info
            if pathway:
                ax.set_title(f'O Migration Animation (Frame {frame_idx+1}/{len(oxygen_positions)})', 
                            fontsize=12, weight='bold')
            else:
                ax.set_title(f'{movement["movement_type"].title()} Movement (Frame {frame_idx+1})', 
                            fontsize=12, weight='bold')
            
            return ax,
        
        # Determine number of frames
        if pathway:
            n_frames_actual = len(oxygen_positions)
        else:
            n_frames_actual = len(oxygen_positions)
        
        anim = FuncAnimation(fig, update, frames=n_frames_actual,
                        interval=1000/fps, blit=False)
        
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"Enhanced animation saved to {output_file}")
        
        plt.close()
        return anim

    def create_pathway_plot(self, pathways=None, concerted_movements=None,
                          show_atoms=True, show_vacancies=True, 
                          show_lattice=True, figsize=(12, 10),
                          view_angles=(30, 45), save_path=None):
        """
        Create a comprehensive pathway visualization
        
        Args:
            pathways: List of migration pathways to plot
            concerted_movements: List of concerted movements to plot
            show_atoms: Whether to show atoms
            show_vacancies: Whether to show vacancy sites
            show_lattice: Whether to show lattice framework
            figsize: Figure size
            view_angles: (elev, azim) viewing angles
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot lattice
        if show_lattice:
            self.plot_lattice(ax, style='full')
        
        # Plot atoms
        if show_atoms:
            self.plot_atoms(ax, show_oxygen=True, show_cations=True, 
                          show_anions=False, label_atoms=True)
        
        # Plot vacancies
        if show_vacancies and self.finder:
            self.plot_vacancies(ax, self.finder.all_vacancies)
        
        # Plot migration pathways
        if pathways:
            for i, pathway in enumerate(pathways):
                self.plot_migration_pathway(ax, pathway, style='normal')
        
        # Plot concerted movements
        if concerted_movements:
            for movement in concerted_movements:
                self.plot_concerted_movement(ax, movement)
        
        # Set view and labels
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å)')
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title('Oxygen Migration Pathways', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return fig, ax
    def create_pathway_animation(self, pathway, output_file='pathway_animation.gif',
                            fps=5, n_frames=20, figsize=(10, 8)):
        """Create an animation of oxygen migration along a pathway with PBC"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get pathway coordinates
        node_sequence = pathway['node_sequence']
        coords = []
        for node in node_sequence:
            node_data = self.finder.migration_graph.nodes[node]
            cart_coords = self.structure.lattice.get_cartesian_coords(node_data['coords'])
            coords.append(cart_coords)
        coords = np.array(coords)
        
        # Create PBC-aware interpolated positions
        oxygen_idx = pathway['oxygen_index']
        oxygen_positions = []
        
        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]
            
            # Get PBC-aware interpolated path
            segment_path = self.get_pbc_interpolated_path(start, end, 
                                                        n_frames // (len(coords) - 1))
            oxygen_positions.extend(segment_path[:-1])  # Avoid duplicates
        
        oxygen_positions.append(coords[-1])  # Add final position
        
        
        def update(frame):
            ax.clear()
            
            # Plot lattice
            self.plot_lattice(ax, style='minimal')
            
            # Plot static atoms
            self.plot_atoms(ax, show_oxygen=True, show_cations=True, 
                          show_anions=False, atom_size=80)
            
            # Plot vacancies
            if self.finder:
                self.plot_vacancies(ax, self.finder.all_vacancies, vacancy_size=120)
            
            # Plot moving oxygen
            current_pos = oxygen_positions[min(frame, len(oxygen_positions)-1)]
            ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                      c='yellow', s=150, edgecolors='red', linewidth=2,
                      label=f'O{oxygen_idx} (migrating)')
            
            # Plot pathway trace
            if frame > 0:
                trace_end = min(frame + 1, len(oxygen_positions))
                trace = np.array(oxygen_positions[:trace_end])
                ax.plot(trace[:, 0], trace[:, 1], trace[:, 2],
                       'r--', linewidth=2, alpha=0.5)
            
            ax.set_xlabel('x (Å)')
            ax.set_ylabel('y (Å)')
            ax.set_zlabel('z (Å)')
            ax.view_init(elev=30, azim=45)
            ax.legend(loc='upper right')
            
            # Add frame counter
            ax.text2D(0.05, 0.95, f'Frame: {frame+1}/{len(oxygen_positions)}', 
                     transform=ax.transAxes, fontsize=10)
            
            return ax,
        
        anim = FuncAnimation(fig, update, frames=len(oxygen_positions),
                           interval=1000/fps, blit=False)
        
        # Save animation
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
        
        plt.close()
        return anim
    
    def plot_atoms_with_ghosts(self, ax, show_oxygen=True, show_cations=True, 
                          show_anions=True, atom_size=100, ghost_alpha=0.3):
        """Plot atoms including ghost images near boundaries"""
        # Plot main atoms
        self.plot_atoms(ax, show_oxygen, show_cations, show_anions, atom_size)
        
        # Add ghost atoms near boundaries
        for axis in range(3):
            for sign in [-1, 1]:
                shift = np.zeros(3)
                shift[axis] = sign
                
                # Shift in fractional coordinates
                for i, site in enumerate(self.structure):
                    frac_coords = site.frac_coords
                    shifted_frac = frac_coords + shift
                    
                    # Check if ghost would be visible (near boundary)
                    if (0 <= shifted_frac[axis] <= 0.1) or (0.9 <= shifted_frac[axis] <= 1.0):
                        ghost_cart = self.structure.lattice.get_cartesian_coords(shifted_frac)
                        
                        if site.species_string == "O" and show_oxygen:
                            ax.scatter(ghost_cart[0], ghost_cart[1], ghost_cart[2],
                                    c=self.colors['oxygen'], s=atom_size,
                                    alpha=ghost_alpha, edgecolors='none')

    def create_energy_landscape_plot(self, pathways, save_path=None):
        """Create an energy landscape visualization for pathways"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Pathway scores
        pathway_names = []
        scores = []
        distances = []
        
        for i, pathway in enumerate(pathways[:10]):  # Top 10 pathways
            pathway_names.append(f"Path {i+1}")
            scores.append(pathway.get('total_score', 0))
            distances.append(pathway.get('total_distance', 0))
        
        x = np.arange(len(pathway_names))
        
        ax1.bar(x, scores, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Pathway')
        ax1.set_ylabel('Total Score')
        ax1.set_title('Pathway Quality Scores')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pathway_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distance vs Score scatter
        ax2.scatter(distances, scores, s=100, alpha=0.6, c=range(len(distances)), cmap='viridis')
        ax2.set_xlabel('Total Distance (Å)')
        ax2.set_ylabel('Total Score')
        ax2.set_title('Pathway Distance vs Quality')
        ax2.grid(True, alpha=0.3)
        
        # Add pathway numbers
        for i, (d, s) in enumerate(zip(distances, scores)):
            ax2.annotate(f'{i+1}', (d, s), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy landscape saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_network_visualization(self, save_path=None):
        """Create a 2D network visualization of the migration graph"""
        if not self.finder or not hasattr(self.finder, 'migration_graph'):
            print("No migration graph available")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create layout
        pos = nx.spring_layout(self.finder.migration_graph, k=2, iterations=50)
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        for node in self.finder.migration_graph.nodes():
            node_data = self.finder.migration_graph.nodes[node]
            site_type = node_data['site_type']
            
            if site_type == 'occupied_oxygen':
                node_colors.append(self.colors['oxygen'])
                node_sizes.append(300)
            elif site_type == 'oxygen_vacancy':
                node_colors.append(self.colors['oxygen_vacancy'])
                node_sizes.append(400)
            elif site_type == 'cation_influenced':
                node_colors.append(self.colors['pathway_enhanced'])
                node_sizes.append(350)
            else:
                node_colors.append(self.colors['unknown_vacancy'])
                node_sizes.append(250)
        
        # Draw network
        nx.draw_networkx_nodes(self.finder.migration_graph, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.8, ax=ax)
        
        # Draw edges with weights
        edges = self.finder.migration_graph.edges()
        weights = [self.finder.migration_graph[u][v].get('score', 1) for u, v in edges]
        
        nx.draw_networkx_edges(self.finder.migration_graph, pos,
                              width=[w/5 for w in weights],
                              alpha=0.5, ax=ax)
        
        # Add labels for key nodes
        labels = {}
        for node in self.finder.migration_graph.nodes():
            node_data = self.finder.migration_graph.nodes[node]
            if node_data['site_type'] in ['occupied_oxygen', 'oxygen_vacancy']:
                labels[node] = f"{node}"
        
        nx.draw_networkx_labels(self.finder.migration_graph, pos, labels,
                               font_size=8, ax=ax)
        
        # Create legend
        legend_elements = [
            plt.scatter([], [], c=self.colors['oxygen'], s=100, label='O atoms'),
            plt.scatter([], [], c=self.colors['oxygen_vacancy'], s=100, marker='s', label='O vacancies'),
            plt.scatter([], [], c=self.colors['pathway_enhanced'], s=100, label='Enhanced sites'),
            plt.scatter([], [], c=self.colors['unknown_vacancy'], s=100, label='Other sites')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title('Oxygen Migration Network Graph', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_summary_figure(self, max_pathways=5, save_path=None):
        """Create a comprehensive summary figure with multiple views"""
        fig = plt.figure(figsize=(16, 12))
        
        # 3D structure view
        ax1 = fig.add_subplot(221, projection='3d')
        self.plot_lattice(ax1)
        self.plot_atoms(ax1)
        if self.finder:
            self.plot_vacancies(ax1, self.finder.all_vacancies)
            # Plot top pathways
            for pathway in self.finder.migration_pathways[:max_pathways]:
                self.plot_migration_pathway(ax1, pathway)
        ax1.set_title('3D Structure and Pathways')
        ax1.view_init(elev=30, azim=45)
        
        # Different angle view
        ax2 = fig.add_subplot(222, projection='3d')
        self.plot_lattice(ax2, style='minimal')
        self.plot_atoms(ax2, show_cations=False)
        if self.finder:
            self.plot_vacancies(ax2, self.finder.all_vacancies)
            # Show concerted movements if available
            if hasattr(self.finder, 'concerted_pathways'):
                for movement in self.finder.concerted_pathways['all'][:3]:
                    self.plot_concerted_movement(ax2, movement)
        ax2.set_title('Concerted Movements')
        ax2.view_init(elev=0, azim=90)
        
        # Pathway statistics
        ax3 = fig.add_subplot(223)
        if self.finder and hasattr(self.finder, 'migration_pathways'):
            pathway_types = defaultdict(int)
            for pathway in self.finder.migration_pathways:
                types = pathway.get('pathway_types', 'unknown').split(',')
                for t in types:
                    pathway_types[t] += 1
            
            if pathway_types:
                labels = list(pathway_types.keys())
                values = list(pathway_types.values())
                ax3.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Pathway Type Distribution')
        else:
            ax3.text(0.5, 0.5, 'No pathway data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Pathway Type Distribution')
        
        # Vacancy summary
        ax4 = fig.add_subplot(224)
        if self.finder and hasattr(self.finder, 'all_vacancies'):
            vac_counts = {k: len(v) for k, v in self.finder.all_vacancies.items()}
            if any(vac_counts.values()):
                bars = ax4.bar(range(len(vac_counts)), list(vac_counts.values()))
                ax4.set_xticks(range(len(vac_counts)))
                ax4.set_xticklabels([k.replace('_', '\n') for k in vac_counts.keys()], 
                                   rotation=45, ha='right')
                ax4.set_ylabel('Count')
                ax4.set_title('Vacancy Type Summary')
                
                # Color bars
                bar_colors = {
                    'oxygen_vacancies': self.colors['oxygen_vacancy'],
                    'cationic_vacancies': self.colors['cation_vacancy'],
                    'anionic_vacancies': self.colors['anion_vacancy'],
                    'unknown_vacancies': self.colors['unknown_vacancy']
                }
                for bar, (vac_type, _) in zip(bars, vac_counts.items()):
                    bar.set_color(bar_colors.get(vac_type, 'gray'))
        else:
            ax4.text(0.5, 0.5, 'No vacancy data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Vacancy Type Summary')
        
        plt.suptitle('Oxygen Migration Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary figure saved to {save_path}")
        
        plt.show()
        return fig

    def get_pbc_minimum_image_coords(self, pos1, pos2):
        """
        Get the minimum image vector between two positions considering PBC
        
        Args:
            pos1, pos2: Cartesian coordinates
        
        Returns:
            pos2_wrapped: Cartesian coordinates of pos2 in minimum image
        """
        # Convert to fractional
        frac1 = self.structure.lattice.get_fractional_coords(pos1)
        frac2 = self.structure.lattice.get_fractional_coords(pos2)
        
        # Get minimum image difference
        frac_diff = frac2 - frac1
        frac_diff = frac_diff - np.round(frac_diff)  # Wrap to [-0.5, 0.5]
        
        # Get wrapped position
        frac2_wrapped = frac1 + frac_diff
        pos2_wrapped = self.structure.lattice.get_cartesian_coords(frac2_wrapped)
        
        return pos2_wrapped

    def get_pbc_interpolated_path(self, pos1, pos2, n_points=20):
        """
        Get interpolated path between two positions considering PBC
        
        Args:
            pos1, pos2: Cartesian coordinates
            n_points: Number of interpolation points
        
        Returns:
            path: Array of Cartesian coordinates along the path
        """
        # Convert to fractional
        frac1 = self.structure.lattice.get_fractional_coords(pos1)
        frac2 = self.structure.lattice.get_fractional_coords(pos2)
        
        # Get minimum image difference
        frac_diff = frac2 - frac1
        frac_diff = frac_diff - np.round(frac_diff)  # Wrap to [-0.5, 0.5]
        
        # Interpolate in fractional coordinates
        path_frac = []
        for i in range(n_points):
            t = i / (n_points - 1)
            frac_interp = frac1 + t * frac_diff
            path_frac.append(frac_interp)
        
        # Convert back to Cartesian
        path_cart = [self.structure.lattice.get_cartesian_coords(frac) for frac in path_frac]
        
        return np.array(path_cart)


def visualize_oxygen_pathways(structure_file="POSCAR", finder_object=None, 
                            output_dir="oxygen_pathway_plots"):
    """
    Convenience function to create all visualizations
    
    Args:
        structure_file: Path to structure file
        finder_object: UniversalVacancyOxygenMigrationFinder object
        output_dir: Directory to save plots
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize visualizer
    viz = OxygenPathwayVisualizer(structure_file, finder_object)
    
    # Create various plots
    print("Creating pathway visualizations...")
    
    # 1. Main pathway plot
    if finder_object and hasattr(finder_object, 'migration_pathways'):
        viz.create_pathway_plot(
            pathways=finder_object.migration_pathways[:5],
            save_path=os.path.join(output_dir, "main_pathways.png")
        )
    
    # 2. Concerted movements
    if finder_object and hasattr(finder_object, 'concerted_pathways'):
        viz.create_pathway_plot(
            concerted_movements=finder_object.concerted_pathways['all'][:3],
            save_path=os.path.join(output_dir, "concerted_movements.png")
        )
    
    # 3. Network visualization
    viz.create_network_visualization(
        save_path=os.path.join(output_dir, "migration_network.png")
    )
    
    # 4. Energy landscape
    if finder_object and hasattr(finder_object, 'migration_pathways'):
        viz.create_energy_landscape_plot(
            finder_object.migration_pathways,
            save_path=os.path.join(output_dir, "energy_landscape.png")
        )
    
    # 5. Summary figure
    viz.create_summary_figure(
        save_path=os.path.join(output_dir, "summary.png")
    )
    
    # 6. Animation for best pathway
    if finder_object and hasattr(finder_object, 'migration_pathways') and finder_object.migration_pathways:
        viz.create_pathway_animation(
            finder_object.migration_pathways[0],
            output_file=os.path.join(output_dir, "best_pathway_animation.gif")
        )
    
    print(f"All visualizations saved to {output_dir}/")
    
    return viz


# Example usage
if __name__ == "__main__":
    # This would typically be called after running oxygen_hopping_vacancy_pathway_generation.py
    print("Oxygen Migration Pathway Visualization Module")
    print("=" * 50)
    print("Usage:")
    print("  from oxygen_pathway_visualization import visualize_oxygen_pathways")
    print("  viz = visualize_oxygen_pathways('POSCAR', finder_object)")
