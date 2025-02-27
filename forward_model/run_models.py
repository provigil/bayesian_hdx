import os
import subprocess
import glob

# Define the base directory to search for files
base_directory = '/Users/kehuang/Documents/GitHub/bayesian_hdx/forward_model/'

# Define the file extension to search for
file_extension = '*.pdb'

# Find all files with the specified extension in the base directory
input_files = glob.glob(os.path.join(base_directory, file_extension))

# Iterate through the list of files and perform your desired operations
for file_path in input_files:
    # Perform your desired operations with each file
    print(f"Processing file: {file_path}")
    # Example: Run a shell command on the file
    subprocess.run(['echo', file_path])  # Replace with your actual command

# Define the common arguments for the command
common_args = [
    'python', 'run_forward_model.py',
    '-d', '0.85',
    '-t', '0', '30', '60', '300', '900', '3600', '14400', '84600',
    '-p', '7',
    '-temp', '298',
    '-l', 'pep_list.txt'
]

# Loop through each input file and run the command
for i, input_file in enumerate(input_files, start=1):
    # Define the specific input file for the -f argument
    example_file = f'../example{i}.txt'
    
    # Write the full file path to the input file- this might be changed to make it smoother in the future, but seems to break if it's a file path for now?
    with open(example_file, 'w') as f:        f.write(input_file + '\n')
    
    # Define the specific output file name
    output_file = f'../out{i}.csv'
    
    # Construct the full command
    command = common_args + ['-f', example_file, '-o', output_file]
    
    # Print the command for debugging purposes
    print(f"Running command: {' '.join(command)}")
    
    # Run the command
    subprocess.run(command)