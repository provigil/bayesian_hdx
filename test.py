import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame, using the first row as the header
df = pd.read_csv('out10.csv')

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

for col in df.columns:
    plt.plot(x_values, df[col], marker='o', label=col)

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