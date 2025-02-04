#!/bin/bash

# Define the range for the output files
start=1
end=$(ls ../out*.csv | wc -l)

# Define the input directory
input_dir="../"

#clean up from previous step
rm ../example*.txt

# Process each file in the range
for i in $(seq $start $end); do
    
    # Iterate over the range of columns (11 to 17 in this example)
    for col in {11..17}; do
        # Create a temporary directory to store intermediate files
        mkdir -p "../out-${col}"
        # Extract the specified column and save it to a temporary file
        awk -F, -v col="$col" 'NR>1 {print $col}' ../out${i}.csv > ../out-${col}/out${i}.txt
        awk -F, -v col="$col" 'NR>1 {print $1}' ../out${i}.csv > ../out-${col}/peptide.dat

        # Compute the reference column index
        var=$((col - 9))
        
        # Extract reference data
        awk -F, -v var="$var" 'NR>1 {print $var}' ../reference.csv > ../out-${col}/reference.txt
        awk -F, -v var="$var" 'NR>1 {print $1}' ../reference.csv > ../out-${col}/reference_peptide.dat

    done

done

# Combine the columns side-by-side into the final output files and plot them
for j in {11..17}; do
    output_file="${col_temp_dir}/combined_data_col${col}.txt"
    paste ../out-${j}/peptide.dat ../out-${j}/out*.txt > combined_data.txt

    cat > ../out-${j}/plot_col.py <<EOF
import pandas as pd
import matplotlib.pyplot as plt

# Read the combined data file
df = pd.read_csv('combined_data.txt', sep='\t', header=None)

# Read the reference data
reference_df = pd.read_csv('../out-${j}/reference.txt', sep='\t', header=None)
reference_labels = pd.read_csv('../out-${j}/reference_peptide.dat', sep='\t', header=None)[0]

# Extract the first column for labels
labels = df.iloc[:, 0]

# Set the first column as the index (assuming it contains the labels)
df.set_index(df.columns[0], inplace=True)

# Select only columns from 2 to 6
df = df.iloc[:, 1:6]

# Plot each column (starting from the second column)
plt.figure(figsize=(10, 6))

# Use a range based on the row count for the X-axis
x_values = range(len(df))

# Ensure reference_df has the same length as x_values
min_length = min(len(x_values), len(reference_df[0]))
x_values = x_values[:min_length]
reference_data = reference_df[0][:min_length]

for col in df.columns:
    plt.plot(x_values, df[col][:min_length], marker='o', label=col)

# Plot reference data as circles
plt.plot(x_values, reference_data, 'o', label='Reference', markersize=5, color='red')

# Annotate each point with the label from the first column
annotated_labels = set()
for i, label in enumerate(labels):
    if label not in annotated_labels:
        plt.annotate(label, (i, df.iloc[i, 0]), textcoords="offset points", xytext=(0,5), ha='center', rotation=90, fontsize='small')
        annotated_labels.add(label)

# Adding labels and title
plt.xlabel('Row Number')
plt.ylabel('Values')
plt.title('Line Graph with Multiple Columns (Columns 2 to 6)')

# Show the plot
plt.show()
EOF

    python ../out-${j}/plot_col.py
done

# Clean up the temporary directories if needed
# rm -r ../out-*