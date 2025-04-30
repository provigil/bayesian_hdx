import argparse
import numpy as np
import pandas as pd
from forward_model_mono import calc_incorporated_deuterium_with_weights, get_amino_acid_sequence
import baker_hubbard_pf_mono as bh
import tryptic_peptide_mono as tp
from hdx_likelihood_function_mono import calculate_sigma, total_likelihood, add_noised_data, total_likelihood_benchmark, likelihood, total_likelihood_test

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)

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
    #parser.add_argument('-f', '--file_path', type=str, required=True, help="Path to the text file containing PDB paths")
    #try this
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--single_file', type=str, help="Path to the text file containing PDB paths for both exp and model (uses same data set)")
    group.add_argument('--dual_file', nargs=2, metavar=('exp_file', 'model_file'), help="Paths to two text files containing PDB paths: first for experimental data, second for model predictions")
    #
    parser.add_argument('-w', '--weights', type=float, nargs='+', help="List of weights for the structures (optional, required if multiple PDB paths are provided)")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file path to save results")
    parser.add_argument('-l', '--peptide_list', type=str, help="Text file containing list of peptides (optional)")

    return parser.parse_args()

def generate_deuteration_df(file_path, peptide_list, deuterium_fraction, time_points, pH, temperature, weights):
    # If a peptide list is provided use it; otherwise generate tryptic peptides from the first PDB
    if peptide_list:
        with open(peptide_list, 'r') as f:
            peptides = [line.strip() for line in f]
    else:
        with open(file_path, 'r') as f:
            path_list = [line.strip() for line in f]
        # Use the first PDB path to get the full amino acid sequence
        path_to_pdb = path_list[0]
        full_sequence = get_amino_acid_sequence(path_to_pdb)
        peptides = tp.generate_tryptic_peptides(full_sequence)
    
    # Filter out very short peptides
    peptides = [pep for pep in peptides if len(pep) > 1]
    
    # Ensure that if multiple PDB files are provided, the corresponding weights are given
    with open(file_path, 'r') as f:
        path_list = [line.strip() for line in f]
    num_pdb_files = len(path_list)

    # Default to equal weights if none provided
    if weights is None:
        weights = [1.0] * num_pdb_files
    elif len(weights) != num_pdb_files:
        raise ValueError("The number of weights must match the number of file paths.")

    
    # Return the DataFrame computed using your forward model function
    return calc_incorporated_deuterium_with_weights(
        peptide_list=peptides,
        deuterium_fraction=deuterium_fraction,
        time_points=time_points,
        pH=pH,
        temperature=temperature,
        file_path=file_path,
        weights=weights
    )

def main():
    args = parse_arguments()

    # load one or two files
    if args.single_file:
        print("Single‐file mode (exp & model = same)")
        df_exp   = generate_deuteration_df(
                       args.single_file, args.peptide_list,
                       args.deuterium_fraction, args.time_points,
                       args.pH, args.temperature, args.weights)
        df_model = df_exp.copy()
    else:
        exp_file, model_file = args.dual_file
        print("Dual‐file mode (exp vs. model)")
        df_exp   = generate_deuteration_df(
                       exp_file, args.peptide_list,
                       args.deuterium_fraction, args.time_points,
                       args.pH, args.temperature, args.weights)
        df_model = generate_deuteration_df(
                       model_file, args.peptide_list,
                       args.deuterium_fraction, args.time_points,
                       args.pH, args.temperature, args.weights)

    print("Experimental DataFrame:")
    print(df_exp)
    print("Model DataFrame (head):")
    print(df_model)

    # run the likelihood test
    peptide_lkhd, overall_time_lkhd, total_lkhd, avg_lkhd = \
        total_likelihood_test(df_exp, df_model)

    # map per-time likelihoods back into df_exp
    for t, lkhd_map in peptide_lkhd.items():
        df_exp[f"LogLKHD_{t}s"] = df_exp['Peptide'].map(lkhd_map)

    # add overall per-time summary
    df_exp['Total_LKHD_per_time'] = df_exp['Peptide'].map(lambda pep: 
        sum(peptide_lkhd[t].get(pep, 0.0) for t in peptide_lkhd)
    )

    # save
    df_exp.to_csv(args.output, index=False)
    print(f"Results written to {args.output!r}") 

if __name__ == "__main__":
    main()

# def main():
#     # Parse the arguments
#     args = parse_arguments()

#     # If peptide list is not provided, generate tryptic peptides
#     if args.peptide_list:
#         with open(args.peptide_list, 'r') as f:
#             peptide_list = [line.strip() for line in f]
#     else:
#         # Generate tryptic peptides from the PDB file
#         with open(args.file_path, 'r') as f:
#             path_list = [line.strip() for line in f]
#         path_to_pdb = path_list[0]
#         #print(f"Call get_amino_acid_sequence 123")
#         full_sequence = get_amino_acid_sequence(path_to_pdb)
#         peptide_list = tp.generate_tryptic_peptides(full_sequence)

#     # Filter out peptides that are too short
#     peptide_list = [pep for pep in peptide_list if len(pep) > 1]

#     # Print the number of PDB file paths and the number of weights
#     num_pdb_files = count_lines_in_file(args.file_path)
#     #print(f"PDB file paths input [run_forward]: {num_pdb_files}")
#     num_weights = len(args.weights) if args.weights else 0
#     #print(f"Number of weights [run_forward]: {num_weights}")

#     # Define which function to use
#     if num_pdb_files > 1:
#         # Ensure the number of weights matches the number of file paths
#         if num_weights != num_pdb_files:
#             raise ValueError("The number of weights must match the number of file paths.")
        
#         # Use provided weights
#         weights = args.weights
#     elif num_pdb_files == 1:
#         # Use a default weight of 1.0 if there is only one PDB file
#         weights = [1.0]

#     #print(f"moving to calc_incorp")


#     # Call the new function to handle both single and multitotal_likelihood_benchmarkple PDBs
#     deuteration_df = calc_incorporated_deuterium_with_weights(
#         peptide_list=peptide_list,
#         deuterium_fraction=args.deuterium_fraction,
#         time_points=args.time_points,
#         pH=args.pH,
#         temperature=args.temperature,
#         file_path=args.file_path,
#         weights=args.weights
#     )

#     print(deuteration_df)

#     # Add synthetic 'noised' data to the DataFrame
#     #deuteration_df = add_noised_data(deuteration_df, args.time_points)
#     # benchmarking behavior
#     #deuteration_df = add_noised_data_benchmark(deuteration_df, args.time_points)

#     #print(deuteration_df)

#     # Calculate the total likelihood for each time point and add to DataFrame
#     #peptide_avg_likelihoods, overall_likelihood = total_likelihood(deuteration_df)
#     # benchmarking behavior- make sure to toggle this!
#     #peptide_avg_likelihoods, overall_likelihood, overall_avg_likelihood = total_likelihood_benchmark(deuteration_df)
#     peptide_avg_likelihoods, overall_likelihood, overall_avg_likelihood = total_likelihood_benchmark(deuteration_df)
    
#     # Add the average likelihoods to the DataFrame
#     for peptide, avg_likelihood in peptide_avg_likelihoods.items():
#         deuteration_df.loc[deuteration_df['Peptide'] == peptide, 'Avg_Likelihood'] = avg_likelihood
    
#     # Add the overall likelihood to the DataFrame
#     deuteration_df['Overall_Likelihood'] = overall_likelihood

#     # Save the dataframe to a CSV file
#     try:
#         deuteration_df.to_csv(args.output, index=False)
#         print(f"Results have been saved to {args.output}")
#     except Exception as e:
#         print(f"Error saving results to CSV: {e}")    

# if __name__ == "__main__":
#     main()