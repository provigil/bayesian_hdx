import os
import subprocess

# Define the base directory to append to each line
base_directory = '/Users/kehuang/Documents/projects/HDX/barnase/modeller/'

# Define the input file containing the list of file names
input_file_list = 'listref'

# Read the list of input files from the text file
with open(input_file_list, 'r') as file:
    input_files = file.read().splitlines()

# Define the common arguments for the command
common_args = [
    'python', 'run_forward_model.py',
    '-d', '0.8',
    '-t', '0', '15', '60', '180', '600', '1800', '3600', '7200',
    '-p', '7',
    '-temp', '300',
    '-l', '../pep_list.txt'
]

# Loop through each input file and run the command
for i, input_file in enumerate(input_files, start=1):
    # Construct the full file path
    full_file_path = os.path.join(base_directory, input_file)
    
    # Define the specific input file for the -f argument
    example_file = f'../example{i}.txt'
    
    # Write the full file path to the example file
    with open(example_file, 'w') as f:
        f.write(full_file_path + '\n')
    
    # Define the specific output file name
    output_file = f'../out{i}.csv'
    
    # Construct the full command
    command = common_args + ['-f', example_file, '-o', output_file]
    
    # Print the command for debugging purposes
    print(f"Running command: {' '.join(command)}")
    
    # Run the command
    subprocess.run(command)