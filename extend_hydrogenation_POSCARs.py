from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
import os

def detect_previously_hydrogenated(structure):
    """Returns indices of atoms that already have a hydrogen atom vertically nearby (1.1 < dz < 1.6 Å)."""
    heavy_indices = [i for i, site in enumerate(structure) if site.specie.symbol != "H"]
    hydrogen_coords = [site.coords for site in structure if site.specie.symbol == "H"]
    previously_hydrogenated = set()

    for i in heavy_indices:
        atom_coord = structure[i].coords
        for h_coord in hydrogen_coords:
            dz = h_coord[2] - atom_coord[2]
            dx_dy = np.linalg.norm(h_coord[:2] - atom_coord[:2])
            if dx_dy < 0.3 and 1.1 < dz < 1.6:
                previously_hydrogenated.add(i)
                break
    return previously_hydrogenated

def generate_unique_hydrogenated_structures(structure, n, dz=1.36):
    """Generate unique structures with n new hydrogens added to unhydrogenated atoms."""
    already_hydrogenated = detect_previously_hydrogenated(structure)
    candidates = [i for i, site in enumerate(structure) if site.specie.symbol != "H" and i not in already_hydrogenated]

    print(f"Total candidate sites: {len(candidates)} | Already hydrogenated: {sorted(already_hydrogenated)}")

    all_combos = list(combinations(candidates, n))
    unique_structures = []
    fingerprints = []

    for combo in all_combos:
        new_structure = structure.copy()
        for idx in combo:
            h_coord = new_structure[idx].coords + np.array([0, 0, dz])
            new_structure.append(Element("H"), h_coord, coords_are_cartesian=True)

        # Symmetry reduction: hash symmetry-invariant structure (site types + frac coords)
        analyzer = SpacegroupAnalyzer(new_structure, symprec=0.01)
        sym_struct = analyzer.get_symmetrized_structure()
        sorted_sites = sorted([(site.specie.symbol, tuple(np.round(site.frac_coords, 3))) for site in sym_struct])

        fingerprint = tuple(sorted_sites)

        if fingerprint not in fingerprints:
            fingerprints.append(fingerprint)
            unique_structures.append((combo, new_structure))

    return unique_structures

def main():
    poscar_file = "POSCAR_292"
    structure = Structure.from_file(poscar_file)

    n = int(input("Enter number of new hydrogens to add (n): ").strip())

    unique_structures = generate_unique_hydrogenated_structures(structure, n)

    print(f"\nGenerated {len(unique_structures)} unique hydrogenated structures with n = {n}\n")

    for i, (combo, new_struct) in enumerate(unique_structures, 1):
        fname = f"{os.path.splitext(poscar_file)[0]}_add{n}H_{i}.vasp"
        new_struct.to(fmt="poscar", filename=fname)
        print(f"Saved: {fname} | Added H on atoms: {combo}")

if __name__ == "__main__":
    main()
