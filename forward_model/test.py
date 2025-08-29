from Bio.SeqUtils import seq1
import biotite.structure as struc
import biotite.structure.io as strucio

# Load PDB (adjust your loader as needed)
path_to_pdb = "/Users/kehuang/Documents/projects/nsp2/analysis/base/cluster_all.c0.pdb"
structure = strucio.load_structure(path_to_pdb)

MOD_MAP = {
    'CYZ': 'CYS',
    'CYM': 'CYS',
    'HEZ': 'HIS',
    'HDZ': 'HIS',
    'HIP': 'HIS',
    'MSE': 'MET',
    'SEP': 'SER',
    'TPO': 'THR',
    'PTR': 'TYR',
}

unknown_residues = set()

for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] == ' ':
                resname = MOD_MAP.get(residue.resname, residue.resname)
                try:
                    seq1(resname)
                except KeyError:
                    unknown_residues.add(resname)

if unknown_residues:
    print("Unknown residues found in PDB (not in standard AA or MOD_MAP):")
    for r in sorted(unknown_residues):
        print(f"  {r}")
else:
    print("All residues recognized.")

