import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Paths to the input files
calc_data = '..//out-7/combined_data.txt'
ref_data = '..//out-7/reference.txt'

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
