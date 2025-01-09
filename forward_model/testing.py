import os
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# Custom mapping for nonstandard residues
nonstandard_to_standard = {
    'HIE': 'HIS',
    'HID': 'HIS',
    'HIP': 'HIS',
    'HEZ': 'HIS',
    'HDZ': 'HIS',
    'CYM': 'CYS',
    'CYZ': 'CYS',
    'CYX': 'CYS',
    # Add more mappings if needed
}

def map_nonstandard_residues(resname: str) -> str:
    """Map nonstandard residue names to standard ones."""
    return nonstandard_to_standard.get(resname, resname)

def load_pdb_bio(pdb_filename: str):
    """Load a PDB file and return the structure.

    Args:
        pdb_filename (str): Path to the PDB file.

    Returns:
        structure: The structure object parsed from the PDB file.
    """
    if not os.path.exists(pdb_filename):
        raise FileNotFoundError(f"The file {pdb_filename} does not exist.")
    
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_filename)
    except Exception as e:
        raise RuntimeError(f"An error occurred while parsing the PDB file: {e}")
    return structure

def get_amino_acid_sequence(path_to_pdb: str) -> str:
    """Extract the amino acid sequence from a PDB file.

    Args:
        path_to_pdb (str): Path to the PDB file.

    Returns:
        str: The amino acid sequence.
    """
    structure = load_pdb_bio(path_to_pdb)
    sequence = ""
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue is a standard amino acid (not a heteroatom or water)
                if residue.id[0] == ' ':
                    # Map nonstandard residues to standard ones
                    standard_resname = map_nonstandard_residues(residue.resname)
                    # Convert the three-letter code to a one-letter code
                    one_letter_code = seq1(standard_resname)
                    # Append the one-letter code to the sequence
                    sequence += one_letter_code
    
    if not sequence:
        raise ValueError("No valid amino acid sequence found in the PDB file.")
    
    return sequence

def find_peptide_in_full_sequence(peptide: str, sequence: str):
    """Find the indices of a peptide in the full sequence.

    Args:
        peptide (str): The peptide sequence to find.
        sequence (str): The full amino acid sequence.

    Returns:
        tuple: The 1-based start and end indices of the peptide in the sequence.
    """
    start_index = sequence.find(peptide)
    if start_index == -1:
        raise ValueError("Peptide sequence not found in the full sequence.")
    end_index = start_index + len(peptide) - 1
    return start_index + 1, end_index + 1

# Example usage
if __name__ == "__main__":
    pdb_filename = '../nsp2.pdb'
    peptide = 'FIDTKRGVYCCREHEHEIAWY'
    
    try:
        # Get the full amino acid sequence from the PDB file
        full_sequence = get_amino_acid_sequence(pdb_filename)
        print(f"Full sequence: {full_sequence}")
        
        # Find the peptide in the full sequence
        start, end = find_peptide_in_full_sequence(peptide, full_sequence)
        print(f"Peptide found at positions {start} to {end} in the full sequence.")
    except Exception as e:
        print(e)