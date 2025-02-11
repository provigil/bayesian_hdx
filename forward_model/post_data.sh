#!/bin/bash

# Define the input directory
input_dir="../"

# Clean up from previous step
#rm -f ${input_dir}/example*.txt

# Define the range for the output files
start=1
end=$(ls ${input_dir}/out*.csv | wc -l)

# Process each file in the range
for i in $(seq $start $end); do
    
    # Iterate over the range of columns (11 to 17 in this example)
    for col in {2..8}; do
        # Create a temporary directory to store intermediate files
        mkdir -p "${input_dir}/out-${col}"
        # Extract the specified column and save it to a temporary file
        awk -F, -v col="$col" 'NR>1 {print $col}' ${input_dir}/out${i}.csv > ${input_dir}/out-${col}/out${i}.txt
        awk -F, -v col="$col" 'NR>1 {print $1}' ${input_dir}/out${i}.csv > ${input_dir}/out-${col}/peptide.dat

        # Compute the reference column index
        #var=$((col - 9))
        var=$((col))
        
        # Extract reference data
        awk -F, -v var="$var" 'NR>1 {print $var}' ${input_dir}/reference.csv > ${input_dir}/out-${col}/reference.txt
        awk -F, -v var="$var" 'NR>1 {print $1}' ${input_dir}/reference.csv > ${input_dir}/out-${col}/reference_peptide.dat
    done
done

# Combine the columns side-by-side into the final output files and plot them
for j in {2..8}; do
    output_file="${input_dir}/out-${j}/combined_data_col${j}.txt"
    paste ${input_dir}/out-${j}/out*.txt > ${input_dir}/out-${j}/combined_data.txt

cat > ${input_dir}/out-${j}/plot_col.py <<EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Paths to the input files
calc_data = '${input_dir}/out-${j}/combined_data.txt'
ref_data = '${input_dir}/out-${j}/reference.txt'

# Read the files
data1 = pd.read_csv(calc_data, header=None, names=['Data1'])
data2 = pd.read_csv(ref_data, header=None, names=['Data2'])

# Ensure both datasets have the same length
min_length = min(len(data1), len(data2))
data1 = data1.iloc[:min_length]
data2 = data2.iloc[:min_length]

# Calculate correlation
correlation = data1['Data1'].corr(data2['Data2'])
print(f'Correlation coefficient: {correlation}')

# Calculate the R-squared value for the ideal line y = x
r_value = correlation
print(f'R: {r_value}')

# Define the range for the axes
x = np.linspace(0, 18, 25)

# Plot the perfect diagonal line (y = x)
plt.figure(figsize=(10, 10))
plt.plot(x, x, 'r', label='Perfect line (y = x)')

# Plot the data points
plt.scatter(data1['Data1'], data2['Data2'], label='Data points')

# Set the limits for the axes
plt.xlim(0, 18)
plt.ylim(0, 25)

# Annotate the differences
for i in range(len(data1)):
    plt.plot([data1['Data1'].iloc[i], data1['Data1'].iloc[i]], [data2['Data2'].iloc[i], data1['Data1'].iloc[i]], 'b--')

# Adding labels and title
plt.xlabel('Data1')
plt.ylabel('Data2')

# Show the plot
plt.grid(True)
plt.show()
EOF

    python ${input_dir}/out-${j}/plot_col.py
done

# Clean up the temporary directories if needed
# rm -r ${input_dir}/out-*