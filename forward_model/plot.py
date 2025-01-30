import pandas as pd
import matplotlib.pyplot as plt

# Read the combined data file
data = pd.read_csv('combined_data.txt', sep='\t', header=None)

# Plot each column individually
for column in data.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], label=f'out{column}', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Plot for Column {column}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'plot_column_{column}.png')
    plt.close()

print("Plots have been saved as PNG files.")
