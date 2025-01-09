from __future__ import print_function
import os
import matplotlib.pyplot as plt
import itertools
import mdtraj as md
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch

# Function to load a pdb file using mdtraj
def load_pdb(pdb_file):
    return md.load_pdb(pdb_file)

# Function to load a pdb file using biopython 
def load_pdb_bio(pdb_filename):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_filename)
    return structure

# Extract residues from hydrogen bonds that takes in a trajectory and hydrogen bonds
def get_residues(t, hbond):
    res1 = t.topology.atom(hbond[0]).residue.index
    res2 = t.topology.atom(hbond[2]).residue.index
    return [res1, res2]

# Avoids double counting hydrogen bonds
def drop_duplicate_hbonds(hbonds):
    unique_hbonds = []
    seen_pairs = set()
    
    for hbond in hbonds:
        donor_acceptor_pair = (hbond[0], hbond[2])
        if donor_acceptor_pair not in seen_pairs:
            unique_hbonds.append(hbond)
            seen_pairs.add(donor_acceptor_pair)
    
    return unique_hbonds

# Calculate the number of hbonds using the baker hubbard method
def calculate_hbond_number(path_to_pdb):
    # Load trajectory
    t = md.load_pdb(path_to_pdb)
    # Calculate hydrogen bonds
    hbonds = md.baker_hubbard(t, periodic=False)
    
    # Select first chain only 
    chain = t.topology.chain(0)
    residue_counts = {}
    chain_residues = list(chain.residues)
    chain_residue_indices = [residue.index for residue in chain_residues]

    for hbond in hbonds:
        # Donor atom index (first element in the hbond tuple)
        donor_atom_index = hbond[0]
        # Residue index of the donor atom
        donor_residue = t.topology.atom(donor_atom_index).residue.index
        if donor_residue in chain_residue_indices:
            if donor_residue not in residue_counts:
                residue_counts[donor_residue] = 0
            residue_counts[donor_residue] += 1

    # Add zeros for residues without hydrogen bonds
    all_residues = set(chain_residue_indices)
    residues_with_hbonds = set(residue_counts.keys())
    residues_without_hbonds = all_residues - residues_with_hbonds
    for res in residues_without_hbonds:
        residue_counts[res] = 0

    # Add 1 to all of the keys to match the residue numbering in the PDB file
    residue_counts = {k + 1: v for k, v in residue_counts.items()}

    return residue_counts

# Sigmoid function
def sigmoid_12_6(distance, k=1.0, d0=0.0):
    return 1 / (1 + np.exp(-k * (distance - d0)))

# Count heavy atom contacts using a sigmoid function
def count_heavy_atom_contacts_sigmoid(structure, k=1, d0=0.0, distance_threshold=6.5):
    
    # Remove hydrogen atoms
    atom_list = [atom for atom in structure.get_atoms() if atom.element not in ['H']]
    
    # Remove heteroatoms
    atom_list = [atom for atom in atom_list if atom.get_parent().id[0] == ' ']
    
    ns = NeighborSearch(atom_list)
    
    contact_counts = {}
    
    for chain in structure.get_chains():
        residues = list(chain)  
        
        for i, residue in enumerate(residues):
            
            if residue.id[0] != ' ': 
                continue
            
            residue_atoms = [atom for atom in residue if atom.element not in ['H']]
            if not residue_atoms:  # Skip residue if it has no heavy atoms
                continue
            
            contacts = 0
            # Get the amide nitrogen atoms of the residue 
            residue_N = residue['N']
            
            # Find atoms within the distance threshold of the amide nitrogen atom
            neighbors = ns.search(residue_N.coord, level='A', radius=distance_threshold)
            
            for neighbor in neighbors:
                neighbor_residue = neighbor.get_parent()
                neighbor_index = residues.index(neighbor_residue)
                
                # Exclude the same residue and residues i-1, i-2, i+1, i+2
                if neighbor_residue != residue and abs(neighbor_index - i) > 2:
                    distance = np.linalg.norm(neighbor.coord - residue_N.coord)
                    # Apply the sigmoid to the heavy atom contact count
                    contacts += sigmoid_12_6(distance, k, d0)
            
            # Store the contact count for the residue
            contact_counts[residue] = contacts
    
    # Contact counts is a dictionary of residue number and contact count
    contact_counts = {residue.id[1]: count for residue, count in contact_counts.items()}
    
    return contact_counts

def calculate_protection_factors(contacts, hbonds, bh = 0.35, bc = 2):
    protection_factors = {}
    for residue in contacts:
        protection_factors[residue] = bh*contacts[residue] + bc*hbonds[residue]
    return protection_factors

# Function to return the protection factors. Input is a pdb file and the bc and bh values, output is a dictionary of residue number and protection factor 
def estimate_protection_factors(file_path, bc=0.35, bh=2.0, distance_threshold=5):
    
    with open(file_path, 'r') as f:
        pdb_files = [line.strip() for line in f]

    residue_protection_sums = {}
    residue_counts = {}

    for pdb_file in pdb_files:
        structure = load_pdb_bio(pdb_file)
        contact_counts = count_heavy_atom_contacts_sigmoid(structure, distance_threshold=distance_threshold)
        h_bond_counts = calculate_hbond_number(pdb_file)
        
        for residue in contact_counts:
            h_bond_count = h_bond_counts.get(residue, 0)
            heavy_atom_count = contact_counts[residue]
            protection_factor = bh * h_bond_count + bc * heavy_atom_count
            
            if residue not in residue_protection_sums:
                residue_protection_sums[residue] = 0
                residue_counts[residue] = 0
            
            residue_protection_sums[residue] += protection_factor
            residue_counts[residue] += 1

    average_protection_factors = {residue: residue_protection_sums[residue] / residue_counts[residue] for residue in residue_protection_sums}

    return average_protection_factors

def estimate_protection_factors(file_path, bc=0.35, bh=2.0, distance_threshold=5):
    with open(file_path, 'r') as f:
        pdb_files = [line.strip() for line in f]

    residue_protection_sums = {}
    residue_counts = {}

    for pdb_file in pdb_files:
        if not os.path.isfile(pdb_file):
            print(f"File not found: {pdb_file}")
            continue
        
        structure = load_pdb_bio(pdb_file)
        contact_counts = count_heavy_atom_contacts_sigmoid(structure, distance_threshold=distance_threshold)
        h_bond_counts = calculate_hbond_number(pdb_file)
        
        for residue in contact_counts:
            h_bond_count = h_bond_counts.get(residue, 0)
            heavy_atom_count = contact_counts[residue]
            protection_factor = bh * h_bond_count + bc * heavy_atom_count
            
            if residue not in residue_protection_sums:
                residue_protection_sums[residue] = 0
                residue_counts[residue] = 0
            
            residue_protection_sums[residue] += protection_factor
            residue_counts[residue] += 1

    average_protection_factors = {residue: residue_protection_sums[residue] / residue_counts[residue] for residue in residue_protection_sums}

    return average_protection_factors