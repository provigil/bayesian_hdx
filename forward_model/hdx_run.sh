#!/bin/bash

# Create the Python script
cat > process_files.py <<EOF
import os
import subprocess
import glob
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process PDB files with user-defined parameters.")
parser.add_argument('-d', '--directory', type=str, required=True, help='Base directory to search for PDB files.')
parser.add_argument('-e', '--extension', type=str, default='*.pdb', help='File extension to search for.')
parser.add_argument('-l', '--list', type=str, required=True, help='Path to the residue list file (pep_list.txt).')
parser.add_argument('-o', '--output', type=str, required=True, help='Directory to print the output files.')
args = parser.parse_args()

# Define the base directory to search for files
base_directory = args.directory

# Define the file extension to search for
file_extension = args.extension

# Find all files with the specified extension in the base directory
input_files = glob.glob(os.path.join(base_directory, file_extension))

# Ensure the output directory exists
os.makedirs(args.output, exist_ok=True)

# Iterate through the list of files and perform your desired operations
for file_path in input_files:
    # Perform your desired operations with each file
    print(f"Processing file: {file_path}")

# Define the common arguments for the command
common_args = [
    'python', 'run_forward_model.py',
    '-d', '0.8',
    '-t', '0', '0.5', '1', '5', '15', '60', '240', '1440',
    '-p', '7',
    '-temp', '300',
    '-l', args.list
]

# Loop through each input file and run the command
for i, input_file in enumerate(input_files, start=1):
    # Define the specific input file for the -f argument
    example_file = os.path.join(args.output, f'example{i}.txt')
    
    # Write the full file path to the input file
    with open(example_file, 'w') as f:
        f.write(input_file + '\n')
    
    # Define the specific output file name
    output_file = os.path.join(args.output, f'out{i}.csv')
    
    # Construct the full command
    command = common_args + ['-f', example_file, '-o', output_file]
    
    # Print the command for debugging purposes
    print(f"Running command: {' '.join(command)}")
    
    # Run the command
    subprocess.run(command)
EOF

# Prompt user for input directory, residue list file, reference data file, and output directory
read -e -p "Enter the directory containing the PDB files: " base_directory
read -e -p "Enter the path to the residue list file (pep_list.txt): " residue_list
read -e -p "Path to the reference (experimental) data file: " refpath
read -e -p "Enter the directory to save the output files: " output_directory

# Run the Python script with the user-provided parameters
python process_files.py -d "$base_directory" -l "$residue_list" -o "$output_directory"

wait
# Part 2 starts here

# Define the range for the output files
start=1
end=$(ls ${output_directory}/out*.csv | wc -l)

# Clean up from previous step
rm ${output_directory}/example*.txt

# Process each file in the range
for i in $(seq $start $end); do
    
    # Iterate over the range of columns (11 to 17 in this example)
    for col in {11..17}; do
        # Create a temporary directory to store intermediate files
        mkdir -p "${output_directory}/out-${col}"
        # Extract the specified column and save it to a temporary file
        awk -F, -v col="$col" 'NR>1 {print $col}' ${output_directory}/out${i}.csv > ${output_directory}/out-${col}/out${i}.txt
        awk -F, -v col="$col" 'NR>1 {print $1}' ${output_directory}/out${i}.csv > ${output_directory}/out-${col}/peptide.dat

        # Compute the reference column index
        var=$((col - 9))
        
        # Extract reference data
        awk -F, -v var="$var" 'NR>1 {print $var}' $refpath > ${output_directory}/out-${col}/reference.txt
        awk -F, -v var="$var" 'NR>1 {print $1}' $refpath > ${output_directory}/out-${col}/reference_peptide.dat
    done
done

# Combine the columns side-by-side into the final output files and plot them
for j in {11..17}; do
    output_file="${output_directory}/out-${j}/combined_data_col${j}.txt"
    paste ${output_directory}/out-${j}/peptide.dat ${output_directory}/out-${j}/out*.txt > ${output_directory}/out-${j}/combined_data.txt

#cat > ${output_directory}/out-${j}/plot_col.py <<EOF
#import pandas as pd
#import matplotlib.pyplot as plt

# Read the combined data file
#df = pd.read_csv('${output_directory}/out-${j}/combined_data.txt', sep='\t', header=None)

# Check if DataFrame is empty
#if df.empty:
#    print("DataFrame is empty. Skipping plotting.")
#else:
    # Read the reference data
#    reference_df = pd.read_csv('${output_directory}/out-${j}/reference.txt', sep='\t', header=None)
#    reference_labels = pd.read_csv('${output_directory}/out-${j}/reference_peptide.dat', sep='\t', header=None)[0]

    # Extract the first column for labels
#    labels = df.iloc[:, 0]

    # Set the first column as the index (assuming it contains the labels)
#    df.set_index(df.columns[0], inplace=True)

    # Select only columns from 2 to 6
#    df = df.iloc[:, 1:6]

    # Plot each column (starting from the second column)
#    plt.figure(figsize=(10, 6))

    # Use a range based on the row count for the X-axis
#    x_values = range(len(df))

    # Ensure reference_df has the same length as x_values
#    min_length = min(len(x_values), len(reference_df[0]))
#    x_values = x_values[:min_length]
#    reference_data = reference_df[0][:min_length]

#    for col in df.columns:
#        plt.plot(x_values, df[col][:min_length], marker='o', label=col)

    # Plot reference data as circles
#    plt.plot(x_values, reference_data, 'o', label='Reference', markersize=5, color='red')

    # Annotate each point with the label from the first column
#    annotated_labels = set()
#    for i, label in enumerate(labels):
#        if label not in annotated_labels:
#            plt.annotate(label, (i, df.iloc[i, 0]), textcoords="offset points", xytext=(0,5), ha='center', rotation=90, fontsize='small')
#            annotated_labels.add(label)

    # Adding labels and title
#    plt.xlabel('Row Number')
#    plt.ylabel('Values')
#    plt.title('Line Graph with Multiple Columns (Columns 2 to 6)')

    # Show the plot
#    plt.show()
#EOF

cat > ${output_directory}/out-${j}/plot_col.py <<EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths to the input files
combined_data_path = '${output_directory}/out-${j}/combined_data.txt'
reference_data_path = '${output_directory}/out-${j}/reference.txt'

# Read the combined data file
df = pd.read_csv(combined_data_path, sep='\t', header=None)

# Check if DataFrame is empty
if df.empty:
    print("DataFrame is empty. Skipping plotting.")
else:
    # Read the reference data
    reference_df = pd.read_csv(reference_data_path, sep='\t', header=None)

    # Ensure reference_df has the same length as the combined data
    min_length = min(len(df), len(reference_df))
    df = df.iloc[:min_length, :]
    reference_data = reference_df.iloc[:min_length, 0]

    # Calculate correlations
    correlations = df.corrwith(reference_data)

    # Plot each column (starting from the second column)
    plt.figure(figsize=(10, 6))

    for col in df.columns[1:]:
        plt.plot(df.index, df[col], marker='o', label=f'Column {col} (corr={correlations[col]:.2f})')

    # Plot reference data as circles
    plt.plot(df.index, reference_data, 'o', label='Reference', markersize=5, color='red')

    # Annotate each point with the label from the first column
    labels = df.iloc[:, 0]
    annotated_labels = set()
    for i, label in enumerate(labels):
        if label not in annotated_labels:
            plt.annotate(label, (i, df.iloc[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center', rotation=90, fontsize='small')
            annotated_labels.add(label)

    # Adding labels and title
    plt.xlabel('Row Number')
    plt.ylabel('Values')
    plt.title('Line Graph with Multiple Columns and Correlations')

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()
EOF

    python ${output_directory}/out-${j}/plot_col.py
done

# Clean up the temporary directories if needed
# rm -r ${output_directory}/out-*