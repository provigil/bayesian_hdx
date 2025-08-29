import argparse
import numpy as np
import pandas as pd
from forward_model_mono import calc_incorporated_deuterium_with_weights, get_amino_acid_sequence
import baker_hubbard_pf_mono as bh
import tryptic_peptide_mono as tp
from hdx_likelihood_function_mono import total_likelihood_test

def load_experimental_csv(csv_path, time_points, fill_value=-1.0):
    """
    Load experimental HDX data from CSV.

    - Numeric headers normalized to "<int>.0_percent".
    - Keeps all peptides in CSV.
    - Missing values filled with fill_value (-1.0 default).
    """
    df = pd.read_csv(csv_path)
    if 'peptide' not in df.columns:
        raise KeyError("CSV must have a 'peptide' column")
    df = df.rename(columns={'peptide': 'Peptide'})
    df['Peptide'] = df['Peptide'].str.upper()

    # Normalize time columns
    rename_map = {}
    for col in df.columns:
        try:
            val = float(col)
            rename_map[col] = f"{int(val)}.0_percent"
        except ValueError:
            pass
    df = df.rename(columns=rename_map)

    expected_cols = [f"{int(t)}.0_percent" for t in time_points]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors='coerce').fillna(fill_value)
    return df[['Peptide'] + expected_cols]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate incorporated deuterium at multiple time points.")
    parser.add_argument('-d', '--deuterium_fraction', type=float, required=True)
    parser.add_argument('-t', '--time_points', type=float, nargs='+', required=True)
    parser.add_argument('-p', '--pH', type=float, required=True)
    parser.add_argument('-temp', '--temperature', type=float, required=True)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--single_file', type=str, help="PDB paths for both exp & model")
    group.add_argument('--dual_file', nargs=2, metavar=('exp_file', 'model_file'), help="Experimental CSV & model PDB paths")

    parser.add_argument('-w', '--weights', type=float, nargs='+', help="Weights for multiple PDB files (optional)")
    parser.add_argument('-o', '--output', type=str, required=True)

    return parser.parse_args()

def generate_deuteration_df(file_path, peptide_list, deuterium_fraction, time_points, pH, temperature, weights):
    """
    Generate deuteration DataFrame from PDB structures using forward model.
    """
    # Use provided peptide list or generate from first PDB
    if peptide_list is not None:
        peptides = [p.strip().upper() for p in peptide_list if p.strip()]
    else:
        with open(file_path, 'r') as f:
            path_list = [line.strip() for line in f]
        full_sequence = get_amino_acid_sequence(path_list[0])
        peptides = tp.generate_tryptic_peptides(full_sequence)

    peptides = [pep for pep in peptides if len(pep) > 1]

    # Determine weights
    with open(file_path, 'r') as f:
        path_list = [line.strip() for line in f]
    num_pdb_files = len(path_list)
    if weights is None:
        weights = [1.0] * num_pdb_files
    elif len(weights) != num_pdb_files:
        raise ValueError("Number of weights must match number of PDB files.")

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

    if args.single_file:
        print("Single-file mode (exp & model same)")
        df_exp = generate_deuteration_df(
            args.single_file, None,
            args.deuterium_fraction, args.time_points,
            args.pH, args.temperature, args.weights
        )
        df_model = df_exp.copy()
    else:
        exp_file, model_file = args.dual_file
        print("Dual-file mode (exp vs. model)")

        # Load all peptides from experimental CSV
        if exp_file.lower().endswith('.csv'):
            df_exp = load_experimental_csv(exp_file, args.time_points)
            peptide_list = df_exp['Peptide'].tolist()
        else:
            raise ValueError("Experimental data must be provided as CSV")

        df_model = generate_deuteration_df(
            model_file, peptide_list,
            args.deuterium_fraction, args.time_points,
            args.pH, args.temperature, args.weights
        )

    print("Experimental DataFrame:")
    print(df_exp)
    print("Model DataFrame (head):")
    print(df_model.head())

    # Calculate likelihoods
    peptide_lkhd, overall_time_lkhd, total_lkhd, avg_lkhd = total_likelihood_test(df_exp, df_model)

    for t, lkhd_map in peptide_lkhd.items():
        df_exp[f"LogLKHD_{t}s"] = df_exp['Peptide'].map(lkhd_map)

    df_exp['Total_LKHD_per_time'] = df_exp['Peptide'].map(
        lambda pep: sum(peptide_lkhd[t].get(pep, 0.0) for t in peptide_lkhd)
    )

    df_exp.to_csv(args.output, index=False)
    print(f"Results written to {args.output!r}")

if __name__ == "__main__":
    main()

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