from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree

def get_unique_combinations(poscar_file, r):
    """
    Returns the unique combinations of selecting r atoms from n atoms in a periodic structure,
    accounting for symmetry equivalences.
    """
    # Load the structure and analyze its symmetry
    structure = Structure.from_file(poscar_file)
    analyzer = SpacegroupAnalyzer(structure)
    symm_ops = analyzer.get_symmetry_operations()
    
    # Build a KDTree for fast nearest-neighbor search
    coords = np.array([site.coords for site in structure])
    kdtree = cKDTree(coords)
    
    # Generate all possible combinations of r atoms
    n = len(structure)
    all_combinations = list(combinations(range(n), r))
    
    # Reduce combinations based on symmetry equivalences
    unique_configs = set()
    unique_combinations = []
    
    for combo in all_combinations:
        combo_set = frozenset(combo)
        is_unique = True
        
        for op in symm_ops:
            transformed_indices = apply_symmetry_operation(op, combo, structure, kdtree)
            if frozenset(transformed_indices) in unique_configs:
                is_unique = False
                break
        
        if is_unique:
            unique_configs.add(combo_set)
            unique_combinations.append(tuple(sorted(combo)))
    
    return unique_combinations

def apply_symmetry_operation(op, combo, structure, kdtree):
    """
    Applies a symmetry operation to a combination of atom indices and returns the transformed indices.
    """
    transformed_coords = [op.operate(structure[i].coords) for i in combo]
    transformed_indices = kdtree.query(transformed_coords)[1]  # Find nearest indices
    return transformed_indices

def add_hydrogens_and_write_poscars(poscar_file, unique_combinations):
    """
    Adds hydrogen atoms on top of selected unique positions and writes POSCAR files.
    """
    structure = Structure.from_file(poscar_file)
    for i, combo in enumerate(unique_combinations):
        new_structure = structure.copy()
        for index in combo:
            atom_position = new_structure[index].coords + np.array([0, 0, 1.36])  # Offset along z-axis
            new_structure.append(Element("H"), atom_position, coords_are_cartesian=True)
        new_structure.to(fmt="poscar", filename=f"POSCARalpha_hydrogen_{i+1}.vasp")

if __name__ == "__main__":
    poscar_file = "POSCARalpha.vasp"
    r = 2  # Number of atoms to select
    unique_combinations = get_unique_combinations(poscar_file, r)
    
    print(f"Number of unique combinations for r={r}: {len(unique_combinations)}")
    print("Unique combinations:")
    for combo in unique_combinations:
        print(combo)
    
    # Add hydrogen atoms and create POSCAR files
    add_hydrogens_and_write_poscars(poscar_file, unique_combinations)
