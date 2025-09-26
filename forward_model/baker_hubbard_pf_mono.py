from __future__ import print_function
import matplotlib.pyplot as plt
import itertools
import mdtraj as md
import mdtraj.testing
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch

#function to load a pdb file using mdtraj
def load_pdb(pdb_file):
    return md.load_pdb(pdb_file)

#load a pdb file using biopython 
def load_pdb_bio(pdb_filename):
    # Load the PDB file
    parser = PDBParser(QUIET=True)
    #print(f"Trying to load PDB file [load_pdb_bio]: {pdb_filename}")
    try:
        structure = parser.get_structure('protein', pdb_filename)
        #print(f"Successfully loaded PDB file [load_pdb 2]: {pdb_filename}")
        return structure
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# extract residues from hydrogen bonds that takes in a trajectory and hydrogen bonds
def get_residues(t, hbond):
    res1 = t.topology.atom(hbond[0]).residue.index
    res2 = t.topology.atom(hbond[2]).residue.index
    return [res1, res2]

#avoids double counting hydrogen bonds
def drop_duplicate_hbonds(hbonds):
    unique_hbonds = []
    seen_pairs = set()
    
    for hbond in hbonds:
        donor_acceptor_pair = (hbond[0], hbond[2])
        if donor_acceptor_pair not in seen_pairs:
            unique_hbonds.append(hbond)
            seen_pairs.add(donor_acceptor_pair)
    
    return unique_hbonds

#calculate the number of hbonds using the baker hubbard method
def calculate_hbond_number(path_to_pdb):
    #load trajectory
    t = md.load_pdb(path_to_pdb)
    #calculate hydrogen bonds
    hbonds = md.baker_hubbard(t, periodic=False)
    
    #select first chain only 
    chain = t.topology.chain(0)
    residue_counts = {}
    chain_residues = list(chain.residues)
    chain_residue_indices = [residue.index for residue in chain_residues]

    for hbond in hbonds:
        #donor atom index (first element in the hbond tuple)
        donor_atom_index = hbond[0]
        #residue index of the donor atom
        donor_residue = t.topology.atom(donor_atom_index).residue.index
        if donor_residue in chain_residue_indices:
            if donor_residue not in residue_counts:
                residue_counts[donor_residue] = 0
            residue_counts[donor_residue] += 1

    # add zeros for residues without hydrogen bonds
    all_residues = set(chain_residue_indices)
    residues_with_hbonds = set(residue_counts.keys())
    residues_without_hbonds = all_residues - residues_with_hbonds
    for res in residues_without_hbonds:
        residue_counts[res] = 0

    #add 1 to all of the keys to match the residue numbering in the PDB file
    residue_counts = {k + 1: v for k, v in residue_counts.items()}

    return residue_counts

# def calculate_center_of_mass(atoms):
#     """Calculate the center of mass of a list of atoms."""
#     total_mass = sum(atom.mass for atom in atoms)
#     center_of_mass = np.sum([atom.coord * atom.mass for atom in atoms], axis=0) / total_mass
#     return center_of_mass

#sigmoid function
def sigmoid_12_6(distance, k=1.0, d0=0.0):
    return 1 / (1 + np.exp(-k * (distance - d0)))

#count heavy atom contacts using a sigmoid function
def count_heavy_atom_contacts_sigmoid(structure, k=1, d0=0.0, distance_threshold=6.5):
    """
    Count heavy atom contacts for each residue using a sigmoid-weighted scheme.
    Skips non-standard residues, Proline, and residues missing backbone N.
    Returns: {residue_number: contact_count}
    """
    # Remove hydrogens and non-standard residues from neighbor search
    atom_list = [
        atom for atom in structure.get_atoms()
        if atom.element != 'H' and atom.get_parent().id[0] == ' '
    ]
    ns = NeighborSearch(atom_list)
    contact_counts = {}

    for chain in structure.get_chains():
        residues = list(chain)

        for i, residue in enumerate(residues):
            # Skip non-standard residues
            if residue.id[0] != ' ':
                continue

            # Skip Proline (no backbone amide H)
            if residue.get_resname().upper() == 'PRO':
                continue

            # Skip residues with no heavy atoms
            residue_atoms = [atom for atom in residue if atom.element != 'H']
            if not residue_atoms:
                continue

            # Skip residues missing backbone N atom
            if 'N' not in residue:
                continue
            residue_N = residue['N']

            contacts = 0
            neighbors = ns.search(residue_N.coord, level='A', radius=distance_threshold)

            for neighbor in neighbors:
                neighbor_residue = neighbor.get_parent()
                if neighbor_residue not in residues:
                    continue

                neighbor_index = residues.index(neighbor_residue)

                # Exclude self and i±1, i±2 neighbors
                if neighbor_residue != residue and abs(neighbor_index - i) > 2:
                    distance = np.linalg.norm(neighbor.coord - residue_N.coord)
                    contacts += sigmoid_12_6(distance, k, d0)

            contact_counts[residue.id[1]] = contacts

    return contact_counts

def calculate_protection_factors(contacts, hbonds, bh = 0.35, bc = 2):
    protection_factors = {}
    for residue in contacts:
        protection_factors[residue] = bh*contacts[residue] + bc*hbonds[residue]
    return protection_factors

def estimate_protection_factors_sigmoid(h_bonds, contact_counts, bc=0.35, bh=2.0):
    """Estimate protection factors using the Best and Vendruscolo method."""
    protection_factors = {}
    h_bond_counts = {residue: 0 for residue in contact_counts.keys()}
    for donor_residue, donor_atom, acceptor_residue, acceptor_atom in h_bonds:
        h_bond_counts[donor_residue] += 1
        h_bond_counts[acceptor_residue] += 1

    for residue, h_bond_count in h_bond_counts.items():
        heavy_atom_count = contact_counts[residue]
        protection_factor = bh * h_bond_count + bc * heavy_atom_count
        protection_factors[residue] = (h_bond_count, heavy_atom_count, protection_factor)

    return protection_factors

def estimate_protection_factors(file_path_or_list, distance_threshold=5):
    """
    Estimate hydrogen bonds (HBonds) and heavy atom contacts (HAC) per residue
    for one or more PDB files.

    Parameters
    ----------
    file_path_or_list : str or list
        - A single PDB filename,
        - A file containing a list of PDB filenames,
        - OR a list of PDB filenames.
    distance_threshold : float
        Distance cutoff for heavy atom contacts.

    Returns
    -------
    dict
        { residue_index: {"hbonds": int, "hac": int} }
    """
    # ---- 1. Load PDB file paths ----
    if isinstance(file_path_or_list, str):
        if file_path_or_list.lower().endswith('.pdb'):
            pdb_files = [file_path_or_list]
        else:
            with open(file_path_or_list, 'r') as f:
                pdb_files = [line.strip() for line in f.readlines() if line.strip()]
    elif isinstance(file_path_or_list, list):
        pdb_files = file_path_or_list
    else:
        raise ValueError("Input must be a PDB filename, a file containing a list of PDB filenames, or a list of PDB filenames.")

    # ---- 2. Initialize storage ----
    residue_hbond_sums = {}
    residue_hac_sums = {}
    residue_counts = {}
    loaded_structures = {}

    # ---- 3. Process each PDB ----
    for pdb_file in pdb_files:
        if pdb_file not in loaded_structures:
            structure = load_pdb_bio(pdb_file)
            loaded_structures[pdb_file] = structure
        else:
            structure = loaded_structures[pdb_file]

        # Calculate raw counts
        contact_counts = count_heavy_atom_contacts_sigmoid(structure, distance_threshold)
        h_bond_counts = calculate_hbond_number(pdb_file)

        # ---- 4. Aggregate counts ----
        for residue in contact_counts:
            hbonds = h_bond_counts.get(residue, 0)
            hac = contact_counts.get(residue, 0)

            if residue not in residue_hbond_sums:
                residue_hbond_sums[residue] = 0
                residue_hac_sums[residue] = 0
                residue_counts[residue] = 0

            residue_hbond_sums[residue] += hbonds
            residue_hac_sums[residue] += hac
            residue_counts[residue] += 1

    # ---- 5. Compute average counts per residue ----
    result = {}
    for residue in residue_counts:
        result[residue] = {
            "hbonds": residue_hbond_sums[residue] / residue_counts[residue],
            "hac": residue_hac_sums[residue] / residue_counts[residue]
        }

    return result