from __future__ import print_function   #this has to be at the start
import numpy as np
    #import baker_hubbard_pf as bh
import tryptic_peptides as tp
import os
import pandas as pd
import mdtraj as md
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio.PDB import *
from Bio.SeqUtils import seq1


############################################################################################################
# the following portion of the code is for the former baker_hubbard_pf.py file


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
def estimate_protection_factors(file_path, bc=0.35, bh=2.0, distance_threshold=6.5):
    
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

#def estimate_protection_factors(file_path, bc=0.35, bh=2.0, distance_threshold=5):
#    with open(file_path, 'r') as f:
#        pdb_files = [line.strip() for line in f]

#    residue_protection_sums = {}
#    residue_counts = {}

#    for pdb_file in pdb_files:
#        if not os.path.isfile(pdb_file):
#            print(f"File not found: {pdb_file}")
#            continue
        
#        structure = load_pdb_bio(pdb_file)
#        contact_counts = count_heavy_atom_contacts_sigmoid(structure, distance_threshold=distance_threshold)
#        h_bond_counts = calculate_hbond_number(pdb_file)
        
#        for residue in contact_counts:
#            h_bond_count = h_bond_counts.get(residue, 0)
#            heavy_atom_count = contact_counts[residue]
#            protection_factor = bh * h_bond_count + bc * heavy_atom_count
            
#            if residue not in residue_protection_sums:
#                residue_protection_sums[residue] = 0
#                residue_counts[residue] = 0
            
#            residue_protection_sums[residue] += protection_factor
#            residue_counts[residue] += 1

#    average_protection_factors = {residue: residue_protection_sums[residue] / residue_counts[residue] for residue in residue_protection_sums}

#    return average_protection_factors

############################################################################################################
# the following portion of the code is for the former tryptic_peptides.py file

#get protein sequences from a pdb file using biopython
def get_protein_sequences_from_pdb(path_to_pdb: str):
    parser = PDBParser()
    structure = parser.get_structure('protein', path_to_pdb)
    sequences = []
    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if residue.id[0] == ' ':
                    three_letter_code = residue.resname
                    one_letter_code = seq1(three_letter_code)
                    sequence += one_letter_code
            sequences.append(sequence)
    return sequences

#finds cut sites in a protein sequence for trypsin
def find_cut_sites(sequence):
    cut_sites = []
    for i in range(len(sequence) - 1):
        if sequence[i] in 'KR' and (i + 1 >= len(sequence) or sequence[i + 1] != 'P'):
            cut_sites.append(i + 1)  # Cut after the K or R
    return cut_sites

# generate all possible tryptic peptides from a protein sequence, but it is not efficient for large proteins so I didn't use it
def generate_fragments(path_to_pdb: str):
    sequence = get_protein_sequences_from_pdb(path_to_pdb)[0]
    cut_sites = find_cut_sites(sequence)
    n = len(cut_sites)
    fragments = []

    # Generate all possible combinations of cut sites
    for i in range(n + 1):  # +1 to include the case with no cuts
        for combo in combinations(cut_sites, i):
            start = 0
            fragment = []
            for cut in combo:
                if cut - start > 3:  # make the site more than 3 residues long
                    fragment.append(sequence[start:cut])
                    start = cut
            if len(sequence) - start > 3:
                fragment.append(sequence[start:])  # Add the last fragment
            fragments.extend([frag for frag in fragment if 5 <= len(frag) <= 20])
    
    # Remove all redundant fragments
    return set(fragments)

# def generate_fragments(path_to_pdb: str):
#     sequence = get_protein_sequences_from_pdb(path_to_pdb)[0]
#     cut_sites = find_cut_sites(sequence)
#     fragments = set()

#     # Generate fragments directly
#     start = 0
#     for cut in cut_sites:
#         fragment = sequence[start:cut]
#         if 5 <= len(fragment) <= 20:
#             fragments.add(fragment)
#         start = cut

#     # Add the last fragment
#     fragment = sequence[start:]
#     if 5 <= len(fragment) <= 20:
#         fragments.add(fragment)

#     return fragments

#takes in a list of peptide sequences and a protein sequence and outputs the start and end indices of the peptides in the protein sequence
def find_peptide_indices(peptides, protein_sequence):
    indices = []
    for peptide in peptides:
        start = protein_sequence.find(peptide)
        end = start + len(peptide) - 1
        indices.append((start, end))
    return indices

#takes in a single peptide sequence and a protein sequence and outputs the start and end indices of the peptide in the protein sequence
def find_peptide_indices_single(peptide, protein_sequence):
    indices = []
    start = protein_sequence.find(peptide)
    end = start + len(peptide) - 1
    indices.append((start, end))
    return indices

def peptide_df(sequence,peptides):
    peptide_df = pd.DataFrame(peptides, columns=['Peptide'])
    peptide_df['Length'] = peptide_df['Peptide'].apply(len)
    peptide_df['Start'] = peptide_df['Peptide'].apply(lambda x: sequence.find(x))
    peptide_df['End'] = peptide_df['Start'] + peptide_df['Length'] - 1
    return peptide_df

# this fucntion takes in a native protein protein sequence and a list of peptide sequences and outputs a plot showing the overlap of the peptides with the protein sequence
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#for visualization of petide coverage 
def plot_peptide_overlaps(path_to_pdb:str):
    # Get the protein sequence from the PDB file
    sequence = get_protein_sequences_from_pdb(path_to_pdb)[0]
    #need to uncomment the generate_fragments function to use this
    fragments = generate_fragments(path_to_pdb)
    peptides = find_peptide_indices(fragments, sequence)
    
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.set_xlim(0, len(sequence))
    ax.set_ylim(0, 1)  # Set y-axis limit to 1 since we will plot non-overlapping fragments at the same height
    ax.set_yticks([])
    ax.set_xticks(range(len(sequence)))
    ax.set_xticklabels(sequence)  # Set x-tick labels to the amino acid sequence
    
    # Plot the native sequence as a line
    ax.plot(range(len(sequence)), [0.05] * len(sequence), color='black', lw=4)
    
    # Track the y-coordinates for each peptide
    y_positions = [0.1]  # Start with the first y-coordinate
    peptide_y_positions = []  # List to store y-coordinates of plotted peptides
    
    for start, end in peptides:
        # Find a y-coordinate that does not overlap with existing peptides
        for y in y_positions:
            if all(not (start < e and end > s) for (s, e), y_pos in zip(peptides, peptide_y_positions) if y == y_pos):
                break
        else:
            y = y_positions[-1] + 0.1  # Increment y-coordinate if no non-overlapping position is found
            y_positions.append(y)
        
        peptide_y_positions.append(y)
        rect = patches.Rectangle((start, y), end - start, 0.05, linewidth=1, edgecolor='r', facecolor='r', alpha=0.5)
        ax.add_patch(rect)
    
    plt.title('Peptide Overlaps with Native Sequence')
    plt.xlabel('Sequence')
    plt.ylabel('Unique Tryptic Peptides')


############################################################################################################
# the following portion of the code is for the former forward_model.py file


# mapping for edge case named residues
nonstandard_to_standard = {
    'HIE': 'HIS','HID':'HIS','HIP':'HIS','HEZ': 'HIS','HDZ': 'HIS','CYM': 'CYS','CYZ': 'CYS','CYX': 'CYS','ASH': 'ASP','GLH': 'GLU',
}

def map_nonstandard_residues(resname: str) -> str:
    """Map nonstandard residue names to standard ones."""
    return nonstandard_to_standard.get(resname, resname)

#this function calculates the number of observable amides in a peptide
def calc_observable_amides(peptide: str): 
    #if proline is in the first two residues, then there are no observable amides
    length = len(peptide)
    #count the number of prolines 
    num_prolines = peptide.count('P') 
    observable_amides = length - 2 - num_prolines
    return observable_amides

#returns a list of booleans that indicate whether an amide is observable or not
def is_observable_amide(peptide: str):
    observable_amides = []
    for i, residue in enumerate(peptide):
        if i < 2 or residue == 'P':
            observable_amides.append(False)
        else:
            observable_amides.append(True)
    return observable_amides

#from salzberg 2016
def get_residue_neighbor_effects(AA, pDcorr, T):
    # For each residue, a tuple containing:
    # 0:Milne acid lambda
    # 1:Milne acid rho
    # 2:Milne base lambda
    # 3:Milne base rho

    R = 1.987

    # Calculate Temp dependent pKa's
    pK_D = -1 * np.log10(
        10**(-1*4.48)*np.exp(-1*1000*((1.0/T-1.0/278)/R)))
    pK_E = -1 * np.log10(
        10**(-1*4.93)*np.exp(-1*1083*((1.0/T-1.0/278)/R)))
    pK_H = -1 * np.log10(
        10**(-1*7.42)*np.exp(-1*7500*((1.0/T-1.0/278)/R)))

    eff_dict = {"A": (0.0, 0.0, 0.0, 0.0),
                "R": (-0.59, -0.32, 0.07671225, 0.22),
                "N": (-0.58, -0.13, 0.49, 0.32),
                "C": (-0.54, -0.46, 0.62, 0.55),
                "Q": (-0.47, -0.27, 0.06, 0.20),
                "G": (-0.22, 0.21817605, 0.26725157, 0.17),
                "I": (-0.91, -0.59, -0.73, -0.23),
                "L": (-0.57, -0.13, -0.57625273, -0.21),
                "K": (-0.56, -0.29, -0.04, 0.12),
                "M": (-0.64, -0.28, -0.00895484, 0.11),
                "F": (-0.52, -0.43, -0.23585946, 0.06313159),
                "P": ("", -0.19477347, "", -0.24),
                "S": (-0.43799228, -0.38851893, 0.37, 0.29955029),
                "T": (-0.79, -0.46807313, -0.06625798, 0.20),
                "W": (-0.40, -0.44, -0.41, -0.11),
                "Y": (-0.41, -0.37, -0.27, 0.05),
                "V": (-0.73902227, -0.30, -0.70193448, -0.14),
                "NT": ("", -1.32, "", 1.62)}

    # Ionizable AA data from
    # Susumu Mori, Peter C.M. van Zijl, and David Shortle
    # PROTEINS: Structure, Function, and Genetics 28:325-332 (1997)

    if AA == "D":
        ne0 = np.log10(
            10**(-0.9-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(0.9-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne1 = np.log10(
            10**(-0.12-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(0.58-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne2 = np.log10(
            10**(0.69-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(0.1-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne3 = np.log10(
            10**(0.6-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(-0.18-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
    elif AA == "E":
        ne0 = np.log10(
            10**(-0.6-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(-0.9-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne1 = np.log10(
            10**(-0.27-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(0.31-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne2 = np.log10(
            10**(0.24-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(-0.11-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne3 = np.log10(
            10**(0.39-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(-0.15-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
    elif AA == "H":
        ne0 = np.log10(
            10**(-0.8-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(0-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne1 = np.log10(
            10**(-0.51-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(0-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne2 = np.log10(
            10**(0.8-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(-0.1-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne3 = np.log10(
            10**(0.83-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(0.14-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
    elif AA == "CT":
        ne0 = np.log10(
            10**(0.05-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(0.96-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne1 = ""
        ne2 = -1.8
        ne3 = ""
    else:
        (ne0, ne1, ne2, ne3) = eff_dict.get(AA)

    # print(AA, pDcorr, (ne0, ne1, ne2, ne3))

    return (ne0, ne1, ne2, ne3)


ResidueChemicalContent = {
    # Tuple containing number of atoms of
    #  (Carbon, Hydrogen, Nitrogen, Oxygen, Sulfur)
    #  for a free amino aicd
    "A": (3, 5, 1, 1, 0),
    "R": (6, 12, 4, 1, 0),
    "N": (4, 6, 2, 2, 0),
    "D": (4, 5, 1, 3, 0),
    "C": (3, 5, 1, 1, 1),
    "Q": (5, 8, 2, 2, 0),
    "E": (5, 7, 1, 3.0),
    "G": (2, 3, 1, 1, 0),
    "H": (6, 7, 3, 1, 0),
    "I": (6, 11, 1, 1, 0),
    "L": (6, 11, 1, 1, 0),
    "K": (6, 12, 2, 1, 0),
    "M": (5, 9, 1, 1, 1),
    "F": (9, 9, 1, 1, 0),
    "P": (5, 7, 1, 1, 0),
    "S": (3, 5, 1, 2, 0),
    "T": (4, 7, 1, 2, 0),
    "W": (11, 10, 2, 1, 0),
    "Y": (9, 9, 1, 2, 0),
    "V": (5, 9, 1, 1, 0),
    "CT": (0, 1, 0, 1, 0),
    "NT": (0, 1, 0, 0, 0)
    # Eventually add AA modifications
}

ElementMasses = {
    # List of tuples containing the mass and abundance of atoms
    # in biological peptides (C, H, N, O, S)
    #
    # Data from
    # https://www.ncsu.edu/chemistry/msf/pdf/IsotopicMass_NaturalAbundance.pdf
    #
    # Original references:
    # The isotopic mass data is from:
    #   G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65
    #   G. Audi, A. H. Wapstra Nucl. Phys A. 1995, 595, 409-480.
    # The percent natural abundance data is from the 1997 report of the
    # IUPAC Subcommittee for Isotopic Abundance Measurements by
    # K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607.

    "C": [(12.000000, 98.93), (13.003355, 1.07)],
    "H": [(1.007825, 99.9885), (2.14101, 0.0115)],
    "N": [(14.0030764, 99.632), (15.000109, 0.368)],
    "O": [(15.994915, 99.757), (16.999132, 0.038), (17.999160, 0.205)],
    "S": [(31.972071, 94.93), (32.97158, 0.76),
          (33.967867, 4.29), (35.967081, 0.02)]
}
#from saltzberg 2016 

def calc_intrinsic_rate(Laa, Raa, pH, T, La2="A", Ra2="A", log=False):
    ''' Calculates random coil hydrogen exchange rate for amide cooresponding
    to side chain Raa
    @param Laa - amino acid letter code N-terminal to amide
    @param Raa - amino acid letter of amide
    @param pH - The pH of the experiment
    @param T - Temperature of the experiment

    Equation and constants derived from Bai, Englander (1980)
    '''

    if Raa == "P" or Raa == "CT" or Raa == "NT" or Laa == "NT":
        return 0
    # Constants
    pKD20c = 15.05
    ka = 0.694782306
    kb = 187003075.7
    kw = 0.000527046
    R = 1.987
    EaA = 14000
    EaB = 17000
    EaW = 19000
    # the pD is different than the pH by +0.4 units
    pDcorr = pH+0.4

    inv_dTR = (1./T-1./293)/R

    FTa = np.exp(-1*EaA*inv_dTR)
    FTb = np.exp(-1*EaB*inv_dTR)
    FTw = np.exp(-1*EaW*inv_dTR)
    Dplus = 10**(-1*pDcorr)
    ODminus = 10**(pDcorr-pKD20c)

    # Get residue-specific effect factors

    L_ne = get_residue_neighbor_effects(Laa, pDcorr, T)
    R_ne = get_residue_neighbor_effects(Raa, pDcorr, T)

    Fa = L_ne[1]+R_ne[0]
    Fb = L_ne[3]+R_ne[2]

    if La2 == "NT":
        Fa += get_residue_neighbor_effects(La2, pDcorr, T)[1]
        Fb += get_residue_neighbor_effects(La2, pDcorr, T)[3]
    if Ra2 == "CT":
        Fa += get_residue_neighbor_effects(Ra2, pDcorr, T)[0]
        Fb += get_residue_neighbor_effects(Ra2, pDcorr, T)[2]

    Fa = 10**(Fa)
    Fb = 10**(Fb)

    krc = Fa*Dplus*ka*FTa + Fb*ODminus*kb*FTb + Fb*kw*FTw

    return krc

#from saltzberg 2016 
def get_sequence_intrinsic_rates(seq, pH, T, log=False):
    i_rates = np.zeros(len(seq))
    i_rates[0] = calc_intrinsic_rate("NT", seq[0], pH, T)
    i_rates[1] = calc_intrinsic_rate(seq[0], seq[1], pH, T, La2="NT")
    for n in range(2, len(seq)-1):
        # print(n, seq[n],seq[n+1])
        L = seq[n-1]
        R = seq[n]
        i_rates[n] = calc_intrinsic_rate(L, R, pH, T)

    i_rates[-1] = calc_intrinsic_rate(seq[-2], seq[-1], pH, T, Ra2="CT")
    if log:
        # Suppress divide by zero error.
        with np.errstate(divide='ignore'):
            i_rates = np.log10(i_rates)

        # print("LOG", seq, i_rates)
        return i_rates
    else:
        return i_rates 

#getting the full amino acid sequence from a pdb file using biopython 
def get_amino_acid_sequence(path_to_pdb: str):
    structure = load_pdb_bio(path_to_pdb)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    # map nonstandard names to standard ones
                    standard_resname = map_nonstandard_residues(residue.resname)
                        #three_letter_code = residue.resname
                        #one_letter_code = seq1(three_letter_code)
                    one_letter_code = seq1(standard_resname)
                    sequence += one_letter_code
    return sequence

# returns the indices of a peptide in the full sequence
def find_peptide_in_full_sequence(peptide: str, sequence: str):
    full_sequence = sequence
    start_index = full_sequence.find(peptide)
    if start_index == -1:
        raise ValueError("Peptide sequence not found in the full sequence.")
    end_index = start_index + len(peptide) - 1
    return start_index + 1, end_index + 1

#returns protection factors for a peptide
def filter_protection_factors(peptide_indices: tuple, protection_factors: dict):
    start, end = peptide_indices
    filtered_pfs = {residue: pf for residue, pf in protection_factors.items() if start <= residue <= end}
    return filtered_pfs

#takes in peptide and path to pdb file and returns a dictionary of protection factors for the peptide
def get_peptide_protection_factors(peptide: str, path_to_pdb: str):
    peptide_indices = find_peptide_in_full_sequence(peptide, path_to_pdb)
    all_pfs = estimate_protection_factors(path_to_pdb)
    peptide_pfs = filter_protection_factors(peptide_indices, all_pfs)
    #reset the keys to start at 0
    pf = {key - peptide_indices[0] : value for key, value in peptide_pfs.items()}
    return pf

#only the summation component of the forward model 
def forward_model_sum(peptide: str, time: float, pH: float, temperature: float):
    total_sum = 0
    intrinsic_rates = get_sequence_intrinsic_rates(peptide, pH, temperature)
    observed = is_observable_amide(peptide)
    for i in range(len(peptide)):
        n = observed[i]
        if n == True:
            intrinsic_rate = intrinsic_rates[i]
            #pfs is a dictionary, so get the value of the key corresponding to i + 1
            log_protection_factor = pfs.get(i)
            protection_factor = np.exp(log_protection_factor)
            k_obs = intrinsic_rate / protection_factor
            total_sum += np.exp(-k_obs * time)
        else:
            total_sum += 0
    return total_sum

#this is not needed, only to compare our forward model with HDX results 
def forward_model_sum_hdxer(peptide: str, protection_factors: dict,  time: float, pH: float, temperature: float, path_to_pdb: str):
    total_sum = 0
    intrinsic_rates = get_sequence_intrinsic_rates(peptide, pH, temperature)
    #create a subdictionary called pfs that only contains the protection factors for the peptide
    #get the indices of the peptide in the full sequence
    peptide_indices = find_peptide_in_full_sequence(peptide, path_to_pdb)
    peptide_pfs = filter_protection_factors(peptide_indices, protection_factors)
    peptide_pf = {key - peptide_indices[0] : value for key, value in peptide_pfs.items()}
    observed = is_observable_amide(peptide)
    for i in range(len(peptide)):
        n = observed[i]
        if n == True:
            intrinsic_rate = intrinsic_rates[i]
            log_protection_factor = peptide_pf.get(i)
            protection_factor = np.exp(log_protection_factor)
            k_obs = intrinsic_rate / protection_factor
            total_sum += np.exp(-k_obs * time)
        else:
            total_sum += 0
    return total_sum

def calc_percentage_deuterium_per_peptide(peptide: str, deuteration_fraction: float, time: float, pH: float, temperature: float, path_to_pdb: str):
    observable_amides = is_observable_amide(peptide)
    num_observable_amides = sum(observable_amides)
    deuteration_fraction = deuteration_fraction
    forward_sum = forward_model_sum(peptide, time, pH, temperature, path_to_pdb)
    deuteration_fraction = deuteration_fraction * (num_observable_amides - forward_sum)
    return deuteration_fraction

def get_peptide_protection_factors(peptide: str, path_to_pdb: str):
    peptide_indices = find_peptide_in_full_sequence(peptide, path_to_pdb)
    all_pfs = estimate_protection_factors(path_to_pdb)
    peptide_pfs = filter_protection_factors(peptide_indices, all_pfs)
    #reset the keys to start at 0
    pf = {key - peptide_indices[0] : value for key, value in peptide_pfs.items()}
    return pf


#this is the main forward model function. Currently, it takes in a list of peptides and then searches the full sequence for the indices of that peptides 
# which may be slow, so maybe we change this to take in a list of indices instead.
def calc_incorporated_deuterium(peptide_list: str, deuterium_fraction: float, time_points: list, pH: float, temperature: float, file_path: str):
    """
    Calculates %D for all peptides at multiple time points.

    Parameters:
    - deuterium_fraction: Fraction of deuterium incorporated
    - time_points: List of time points (float)
    - pH: pH value for intrinsic rate calculation
    - temperature: Temperature for intrinsic rate calculation
    - path_to_pdb: Path to the PDB file
    - peptide_list: Text file containing list of peptides

    Returns:
    - Pandas dataframe of peptide and %D at each time point
    """ 
    #open the file path and store the pdb paths in a list called path_list 
    with open(file_path, 'r') as f:
        path_list = [line.strip() for line in f]
     
    #select the first item of path_list 
    path_to_pdb = path_list[0]
    with open(peptide_list, 'r') as f:
        all_peptides = [line.strip() for line in f]

    all_pfs = estimate_protection_factors(file_path)
    full_sequence = get_amino_acid_sequence(path_to_pdb)
    
    #dictionary to store deuteration values for each time point
    deuteration_dict = {}
    deuteration_fraction = deuterium_fraction

    #iterate over the time points
    for time in time_points:
        peptide_deuteration_dict = {}

        #calculate forward model for each peptide for the current time point and add to dictionary 
        for peptide in all_peptides:
            try:
                #get intrinsic rates, peptide indices, and protection factors
                intrinsic_rates = get_sequence_intrinsic_rates(peptide, pH, temperature)
                peptide_indices = find_peptide_in_full_sequence(peptide, full_sequence)
                peptide_pf = filter_protection_factors(peptide_indices, all_pfs)

                #adjusting indexing
                pfs = {key - peptide_indices[0]: value for key, value in peptide_pf.items()}

                #check observable amides and calculate deuteration fraction
                observable_amides = is_observable_amide(peptide)
                num_observable_amides = sum(observable_amides)
                total_sum = 0

                for i in range(len(peptide)):
                    if observable_amides[i]:
                        intrinsic_rate = intrinsic_rates[i]
                        log_protection_factor = pfs.get(i, 0)  # Default to 0 if not found
                        protection_factor = np.exp(log_protection_factor) if log_protection_factor is not None else 1
                        #observed rate is kint divided by protection factor
                        k_obs = intrinsic_rate / protection_factor
                        total_sum += np.exp(-k_obs * time)

                #calculate deuteration fraction for the peptide at the current time point
                peptide_deuteration_fraction = deuteration_fraction * (num_observable_amides - total_sum)
                peptide_deuteration_dict[peptide] = peptide_deuteration_fraction
            
            #print error if peptide isn't found in the full sequence, but continue to the next peptide
            except Exception as e:
                print(f"Error processing peptide {peptide}: {e}")
                continue

        #add the peptide deuteration dictionary for the current time point to the main dictionary
        deuteration_dict[time] = peptide_deuteration_dict
        
    #create a pandas dataframe with the peptide and the deuteration fraction at each time point as columns
    df = pd.DataFrame(deuteration_dict)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Peptide'}, inplace=True)
    
    #for each time point, calculate percentage deuterium incorporated by dividing each number by the length of the peptide
    for time in time_points:
        df[f'{time}_percent'] = (df[time] / df['Peptide'].apply(len)) * 100
    
    return df