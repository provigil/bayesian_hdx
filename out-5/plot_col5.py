import pandas as pd
import matplotlib.pyplot as plt

# Read the combined data file
data = pd.read_csv('combined_data_col5.txt', sep='\t', header=None)

# Plot the combined data
plt.figure(figsize=(10, 6))
for column in data.columns:
    plt.plot(data.index, data[column], label=f'out{column}')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Combined Data Plot with Reference')
plt.grid(True)
plt.show()
