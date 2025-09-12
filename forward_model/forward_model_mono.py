import numpy
import baker_hubbard_pf_mono as bh
import tryptic_peptide_mono as tp
from Bio.PDB import *
from Bio.SeqUtils import seq1
import pandas as pd

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
    pK_D = -1 * numpy.log10(
        10**(-1*4.48)*numpy.exp(-1*1000*((1.0/T-1.0/278)/R)))
    pK_E = -1 * numpy.log10(
        10**(-1*4.93)*numpy.exp(-1*1083*((1.0/T-1.0/278)/R)))
    pK_H = -1 * numpy.log10(
        10**(-1*7.42)*numpy.exp(-1*7500*((1.0/T-1.0/278)/R)))

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
        ne0 = numpy.log10(
            10**(-0.9-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(0.9-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne1 = numpy.log10(
            10**(-0.12-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(0.58-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne2 = numpy.log10(
            10**(0.69-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(0.1-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne3 = numpy.log10(
            10**(0.6-pDcorr)/(10**(-pK_D)+10**(-pDcorr))
            + 10**(-0.18-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
    elif AA == "E":
        ne0 = numpy.log10(
            10**(-0.6-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(-0.9-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne1 = numpy.log10(
            10**(-0.27-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(0.31-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne2 = numpy.log10(
            10**(0.24-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(-0.11-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne3 = numpy.log10(
            10**(0.39-pDcorr)/(10**(-pK_E)+10**(-pDcorr))
            + 10**(-0.15-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
    elif AA == "H":
        ne0 = numpy.log10(
            10**(-0.8-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(0-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne1 = numpy.log10(
            10**(-0.51-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(0-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne2 = numpy.log10(
            10**(0.8-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(-0.1-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne3 = numpy.log10(
            10**(0.83-pDcorr)/(10**(-pK_H)+10**(-pDcorr))
            + 10**(0.14-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
    elif AA == "CT":
        ne0 = numpy.log10(
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

    FTa = numpy.exp(-1*EaA*inv_dTR)
    FTb = numpy.exp(-1*EaB*inv_dTR)
    FTw = numpy.exp(-1*EaW*inv_dTR)
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
    i_rates = numpy.zeros(len(seq))
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
        with numpy.errstate(divide='ignore'):
            i_rates = numpy.log10(i_rates)

        # print("LOG", seq, i_rates)
        return i_rates
    else:
        return i_rates 

#getting the full amino acid sequence from a pdb file using biopython- may neeed to update to handle modified residues
def get_amino_acid_sequence(path_to_pdb: str):
    MOD_MAP = {
        'CYZ': 'CYS',
        'CYM': 'CYS',
        'HEZ': 'HIS',
        'HDZ': 'HIS',
        'HIP': 'HIS',
        'HIE': 'HIS',
        'HID': 'HIS',
        'MSE': 'MET',
        'SEP': 'SER',
        'TPO': 'THR',
        'PTR': 'TYR',
    }

    structure = bh.load_pdb_bio(path_to_pdb)
    seq = []
    unknown_residues = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    resname = MOD_MAP.get(residue.resname, residue.resname)
                    try:
                        one_letter = seq1(resname)
                    except KeyError:
                        unknown_residues.add(resname)
                        one_letter = 'X'
                    seq.append(one_letter)

    full = ''.join(seq).upper()
    full = ''.join([c for c in full if 'A' <= c <= 'Z'])

    print(f"\n[get_amino_acid_sequence] Loaded '{path_to_pdb}':")
    print(f"  length={len(full)} aa: {full}\n")

    if unknown_residues:
        print("Unknown residues causing X:", unknown_residues)

    return full

def find_peptide_in_full_sequence(peptide: str, sequence: str):
    """
    Locate `peptide` in `sequence`, treating any 'X' in the sequence as a wildcard
    that matches any single residue.
    Returns 1-based (start, end).
    """
    pep = peptide.upper().strip()
    seq = sequence.upper()
    L   = len(pep)

    # slide a window of length L across seq
    for i in range(len(seq) - L + 1):
        window = seq[i:i+L]
        # allow X in the PDB‐sequence to match anything
        if all(s == p or s == 'X' for s, p in zip(window, pep)):
            return i+1, i+L

    raise ValueError(f"Peptide '{peptide}' not found (with X→wildcard) in sequence.")


#returns protection factors for a peptide
def filter_protection_factors(peptide_indices: tuple, protection_factors: dict):
    start, end = peptide_indices
    filtered_pfs = {residue: pf for residue, pf in protection_factors.items() if start <= residue <= end}
    return filtered_pfs

#takes in peptide and path to pdb file and returns a dictionary of protection factors for the peptide
def get_peptide_protection_factors(peptide: str, path_to_pdb: str):
    peptide_indices = find_peptide_in_full_sequence(peptide, path_to_pdb)
    all_pfs = bh.estimate_protection_factors(path_to_pdb)
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
            total_sum += nuempy.exp(-k_obs * time)
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

def calc_incorporated_deuterium_with_weights(
    peptide_list,
    deuterium_fraction: float,
    time_points: list,
    pH: float,
    temperature: float,
    file_path: str,
    weights: list = None
):
    """
    calculates %D for all peptides at multiple time points, handling multiple PDBs with weights if provided.
    Uses get_amino_acid_sequence and find_peptide_in_full_sequence for modified residues.
    """
    # Read PDB file paths
    with open(file_path, 'r') as f:
        path_list = [line.strip() for line in f]

    num_pdbs = len(path_list)
    if weights is None:
        weights = [1.0] * num_pdbs
    elif len(weights) != num_pdbs:
        raise ValueError("Number of weights must match PDB paths.")
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    if isinstance(peptide_list, str):
        with open(peptide_list, 'r') as f:
            all_peptides = [line.strip().upper() for line in f if line.strip()]
    else:
        all_peptides = [pep.upper().strip() for pep in peptide_list]

    deuteration_dict = {time: {} for time in time_points}

    for path_to_pdb, weight in zip(path_list, weights):
        all_pfs = bh.estimate_protection_factors(path_to_pdb)
        full_sequence = get_amino_acid_sequence(path_to_pdb)

        for time in time_points:
            for peptide in all_peptides:
                try:
                    intrinsic_rates = get_sequence_intrinsic_rates(peptide, pH, temperature)
                    start, end = find_peptide_in_full_sequence(peptide, full_sequence)
                    peptide_pf = filter_protection_factors((start, end), all_pfs)
                    pfs = {idx - start: pf for idx, pf in peptide_pf.items()}
                    obs = is_observable_amide(peptide)
                    total_sum = 0.0
                    for i, rate in enumerate(intrinsic_rates):
                        if obs[i]:
                            pf = np.exp(pfs.get(i, 0))
                            k_obs = rate / pf
                            total_sum += np.exp(-k_obs * time)
                    num_obs = sum(obs)
                    frag = weight * deuterium_fraction * (num_obs - total_sum)
                    deuteration_dict[time][peptide] = deuteration_dict[time].get(peptide, 0.0) + frag
                except Exception as e:
                    print(f"Error processing peptide {peptide} for {path_to_pdb}: {e}")

    df = pd.DataFrame(deuteration_dict)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Peptide'}, inplace=True)

    for time in time_points:
        t_int = int(time)
        out_col = f"{t_int}.0_percent"
        df[out_col] = (df[time] / df['Peptide'].str.len()) * 100

    return df