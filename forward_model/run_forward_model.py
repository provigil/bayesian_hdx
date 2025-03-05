import argparse
import numpy as np
import pandas as pd
from forward_model import calc_incorporated_deuterium, calc_incorporated_deuterium_weighted, get_amino_acid_sequence
import baker_hubbard_pf as bh
import tryptic_peptides as tp
from hdx_likelihood_function import calculate_sigma, total_likelihood, add_noised_data


def parse_arguments():
    """
    Parse command line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Calculate incorporated deuterium at multiple time points.")
    
    # Add arguments 
    parser.add_argument('-d', '--deuterium_fraction', type=float, required=True, help="Fraction of deuterium incorporated (float)")
    parser.add_argument('-t', '--time_points', type=float, nargs='+', required=True, help="List of time points (comma-separated floats)")
    parser.add_argument('-p', '--pH', type=float, required=True, help="pH value for intrinsic rate calculation")
    parser.add_argument('-temp', '--temperature', type=float, required=True, help="Temperature for intrinsic rate calculation")
    parser.add_argument('-f', '--file_path', type=str, required=True, help="Path to the text file containing PDB paths")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file path to save results")
    parser.add_argument('-l', '--peptide_list', type=str, help="Text file containing list of peptides (optional)")
    parser.add_argument('-w', '--weights', type=str, help="Text file containing list of weights for each structure (optional)")
    
    return parser.parse_args()

def main():
    # Parse the arguments
    args = parse_arguments()

    # fork if peptide list is not provided, generate tryptic peptides
    if args.peptide_list:
        with open(args.peptide_list, 'r') as f:
            peptide_list = [line.strip() for line in f]
    else:
        # make tryptic peptides from the PDB file
        with open(args.file_path, 'r') as f:
            path_list = [line.strip() for line in f]
        path_to_pdb = path_list[0]
        full_sequence = get_amino_acid_sequence(path_to_pdb)
        peptide_list = tp.generate_tryptic_peptides(full_sequence)

    # Filter out peptides that are too short
    peptide_list = [pep for pep in peptide_list if len(pep) > 1]

    # Read the PDB paths from the input file
    with open(args.file_path, 'r') as f:
        pdb_paths = [line.strip() for line in f]

    # Read the weights from the weights file if provided
    weights = None
    if args.weights:
        with open(args.weights, 'r') as f:
            weights = [float(line.strip()) for line in f]
            if len(weights) != len(pdb_paths):
                raise ValueError("The number of weights must match the number of PDB paths.")

    deuteration_dfs = []

    for i, pdb_path in enumerate(pdb_paths):
        if weights:
            weight = weights[i]
        else:
            weight = 1

        deuteration_df = calc_incorporated_deuterium(
            peptide_list=peptide_list,
            deuterium_fraction=args.deuterium_fraction,
            time_points=args.time_points,
            pH=args.pH,
            temperature=args.temperature,
            file_path=pdb_path
        )

        deuteration_df['Weight'] = weight
        deuteration_dfs.append(deuteration_df)
    
    # Concatenate all DataFrames
    final_deuteration_df = pd.concat(deuteration_dfs, ignore_index=True)

    # Add synthetic 'noised' data to the DataFrame
    final_deuteration_df = add_noised_data(final_deuteration_df, args.time_points)

    print(final_deuteration_df)

    # Calculate the total likelihood for each time point and add to DataFrame
    peptide_avg_likelihoods, overall_likelihood = total_likelihood(final_deuteration_df)
    
    # Add the average likelihoods to the DataFrame
    for peptide, avg_likelihood in peptide_avg_likelihoods.items():
        final_deuteration_df.loc[final_deuteration_df['Peptide'] == peptide, 'Avg_Likelihood'] = avg_likelihood
    
    # Add the overall likelihood to the DataFrame
    final_deuteration_df['Overall_Likelihood'] = overall_likelihood

    # Save the dataframe to a CSV file
    try:
        final_deuteration_df.to_csv(args.output, index=False)
        print(f"Results have been saved to {args.output}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")    

if __name__ == "__main__":
    main()