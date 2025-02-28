import argparse
import numpy as np
from forward_model import calc_incorporated_deuterium
import tryptic_peptides as tp

def parse_arguments():
    """
    Parse command line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Calculate incorporated deuterium at multiple time points.")
    
    #add arguments 
    parser.add_argument('-d', '--deuterium_fraction', type=float, required=True, help="Fraction of deuterium incorporated (float)")
    parser.add_argument('-t', '--time_points', type=float, nargs='+', required=True, help="List of time points (comma-separated floats)")
    parser.add_argument('-p', '--pH', type=float, required=True, help="pH value for intrinsic rate calculation")
    parser.add_argument('-temp', '--temperature', type=float, required=True, help="Temperature for intrinsic rate calculation")
    parser.add_argument('-f', '--file_path', type=str, required=True, help="Path to the text file containing PDB paths")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file path to save results")
    parser.add_argument('-l', '--peptide_list', type=str, required=False, help="Text file containing list of peptides")

    return parser.parse_args()

def main():
    #parse the arguments
    args = parse_arguments()

    # Check if peptide_list is provided
    if args.peptide_list:
        # If provided, pass the peptide list file path directly
        peptide_list = args.peptide_list
    else:
        # If not provided, generate the peptide list using a function from tryptic_peptides
        peptide_list = tp.generate_peptide_list(args.file_path)

    #call calc_incorporated_deuterium from forward_model.py
    deuteration_df = calc_incorporated_deuterium(
        peptide_list=peptide_list,
        deuterium_fraction=args.deuterium_fraction,
        time_points=args.time_points,
        pH=args.pH,
        temperature=args.temperature,
        file_path=args.file_path
    )
    

    #save the dataframe to a CSV file
    try:
        deuteration_df.to_csv(args.output, index=False)
        print(f"Results have been saved to {args.output}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")    





if __name__ == "__main__":
    main()