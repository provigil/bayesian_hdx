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
