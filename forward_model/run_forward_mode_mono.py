import argparse
import numpy as np
import pandas as pd
from forward_model_mono import calc_incorporated_deuterium_with_weights, get_amino_acid_sequence, find_peptide_in_full_sequence
import baker_hubbard_pf_mono as bh
import tryptic_peptide_mono as tp
from hdx_likelihood_function_mono import calculate_sigma, total_likelihood, add_noised_data, total_likelihood_benchmark, likelihood, total_likelihood_test

#usage: run_forward_mode_mono.py [-h] -d DEUTERIUM_FRACTION -t TIME_POINTS [TIME_POINTS ...] -p PH -temp TEMPERATURE (--single_file SINGLE_FILE | --dual_file exp_file model_file) [-w WEIGHTS [WEIGHTS ...]] -o OUTPUT [-l PEPTIDE_LIST] run_forward_mode_mono.py: error: the following arguments are required: -d/--deuterium_fraction, -t/--time_points, -p/--pH, -temp/--temperature, -o/--output
#the way this runs right now is as follows:
    #DEUTERIUM_FRACTION = 0.5; this is the fraction of deuterium incorporated
    #TIME_POINTS = [0.0, 30.0, 60.0, 120.0, 240.0]; these are the time points at which the deuterium incorporation is measured
    #PH = 7.0; this is the pH value for intrinsic rate calculation
    #TEMPERATURE = 298.15; this is the temperature for intrinsic rate calculation

#    --single_file SINGLE_FILE; this is the path to the text file containing PDB paths for both experimental and model data (uses same data set)
#    --dual_file exp_file model_file; this is the path to the text file containing PDB paths for experimental data and model predictions
#    -w WEIGHTS [WEIGHTS ...]; this is the list of weights for the structures (optional, required if multiple PDB paths are provided)
#    -o OUTPUT; this is the output file path to save results
#    -l PEPTIDE_LIST; this is the text file containing list of peptides (optional)

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)
    
def load_experimental_csv(csv_path, time_points, peptide_list=None):
    """
    Load a CSV with columns:
        peptide, <t1>, <t2>, ..., <tn>
    Optionally filter to only peptides in `peptide_list`.
    Returns a DataFrame with:
        Peptide, t1.0_percent, …, tn.0_percent
    """
    df = pd.read_csv(csv_path)
    if 'peptide' not in df.columns:
        raise KeyError("CSV must have a 'peptide' column")
    df = df.rename(columns={'peptide': 'Peptide'})

    # — optionally filter by peptide_list —
    if peptide_list:
        # read the list of peptides from file if a path was provided
        with open(peptide_list, 'r') as f:
            allowed = {line.strip() for line in f}
        df = df[df['Peptide'].isin(allowed)]

    new_cols = []
    for t in time_points:
        raw_col = str(t)          # matches your CSV headers "0.0", "30.0", …
        if raw_col not in df.columns:
            raise KeyError(f"Expected CSV column '{raw_col}' not found in {csv_path}")
        t_int   = int(float(t))
        out_col = f"{t_int}.0_percent"
        # if your CSV is already in percent, skip the *100; otherwise multiply
        df[out_col] = df[raw_col]  
        new_cols.append(out_col)

    return df[['Peptide'] + new_cols]

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
        # — normalize your provided list to all‑caps, no extra whitespace —
        with open(peptide_list, 'r') as f:
            peptides = [
                line.strip().upper()
                for line in f
                if line.strip()
            ]
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
        # if the experimental data is already a CSV, load it directly
        if exp_file.lower().endswith('.csv'):
            df_exp = load_experimental_csv(
                     exp_file,
                     args.time_points,
                     peptide_list=args.peptide_list
                )
        else:
            df_exp = generate_deuteration_df(
                         exp_file, args.peptide_list,
                         args.deuterium_fraction, args.time_points,
                         args.pH, args.temperature, args.weights)

        # model is still generated from structure(s)
        df_model = generate_deuteration_df(
                   model_file, args.peptide_list,
                   args.deuterium_fraction, args.time_points,
                   args.pH, args.temperature, args.weights)

    print("Experimental DataFrame:")
    print(df_exp)
    print("Model DataFrame (head):")
    print(df_model)

    # run the likelihood function
    peptide_lkhd, overall_time_lkhd, total_lkhd, avg_lkhd = total_likelihood_test(df_exp, df_model)

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

# #this was for the benchmarking case with total_likelihood_test
# def main_test():
#     args = parse_arguments()

#     # load one or two files
#     if args.single_file:
#         print("Single‐file mode (exp & model = same)")
#         df_exp   = generate_deuteration_df(
#                        args.single_file, args.peptide_list,
#                        args.deuterium_fraction, args.time_points,
#                        args.pH, args.temperature, args.weights)
#         df_model = df_exp.copy()
#     else:
#         exp_file, model_file = args.dual_file
#         print("Dual‐file mode (exp vs. model)")
#         df_exp   = generate_deuteration_df(
#                        exp_file, args.peptide_list,
#                        args.deuterium_fraction, args.time_points,
#                        args.pH, args.temperature, args.weights)
#         df_model = generate_deuteration_df(
#                        model_file, args.peptide_list,
#                        args.deuterium_fraction, args.time_points,
#                        args.pH, args.temperature, args.weights)

#     print("Experimental DataFrame:")
#     print(df_exp)
#     print("Model DataFrame (head):")
#     print(df_model)

#     # run the likelihood function
#     peptide_lkhd, overall_time_lkhd, total_lkhd, avg_lkhd = \
#         total_likelihood_test(df_exp, df_model)

#     # map per-time likelihoods back into df_exp
#     for t, lkhd_map in peptide_lkhd.items():
#         df_exp[f"LogLKHD_{t}s"] = df_exp['Peptide'].map(lkhd_map)

#     # add overall per-time summary
#     df_exp['Total_LKHD_per_time'] = df_exp['Peptide'].map(lambda pep: 
#         sum(peptide_lkhd[t].get(pep, 0.0) for t in peptide_lkhd)
#     )

#     # save
#     df_exp.to_csv(args.output, index=False)
#     print(f"Results written to {args.output!r}") 

if __name__ == "__main__":
    main()