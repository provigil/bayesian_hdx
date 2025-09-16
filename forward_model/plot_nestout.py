import matplotlib.pyplot as plt
import re

# ---------------------------
# Load and parse the data
# ---------------------------
file_path = "data_out.txt"

# Data containers
one_state_likelihoods = []
two_state_likelihoods = []
two_state_log_evidence = None

# Regex patterns
log_evidence_pattern = re.compile(r"2-state log_evidence\s*=\s*([\d\.]+)")
likelihood_pattern = re.compile(r"Likelihood:\s*([\d\.]+)")

section = None  # Track where we are in the file

with open(file_path, "r") as file:
    for line in file:
        line = line.strip()

        # Detect log evidence
        log_match = log_evidence_pattern.search(line)
        if log_match:
            two_state_log_evidence = float(log_match.group(1))
            continue

        # Identify section headers
        if "1-state models:" in line:
            section = "one_state"
            continue
        elif "2-state models:" in line:
            section = "two_state"
            continue

        # Extract likelihood values
        likelihood_match = likelihood_pattern.search(line)
        if likelihood_match:
            likelihood = float(likelihood_match.group(1))
            if section == "one_state":
                one_state_likelihoods.append(likelihood)
            elif section == "two_state":
                two_state_likelihoods.append(likelihood)

# ---------------------------
# Plotting
# ---------------------------
plt.figure(figsize=(12, 6))

# 1) Histogram of 2-state likelihoods
plt.hist(two_state_likelihoods,
         bins=50,                # Adjust bin count as needed
         color="blue",
         alpha=0.6,
         edgecolor="black",
         label="2-State Likelihoods")

# 2) Vertical dashed orange lines for the two 1-state likelihoods
for idx, likelihood in enumerate(one_state_likelihoods):
    plt.axvline(x=likelihood,
                color="orange",
                linestyle="--",
                linewidth=2,
                label="1-State Likelihood" if idx == 0 else "")

# ---------------------------
# Labels, Title, Legend
# ---------------------------
plt.xlabel("Likelihood", fontsize=20, fontweight='bold')   # Bold X-axis label
plt.ylabel("Population", fontsize=20, fontweight='bold')  # Bold Y-axis label

# Customize tick label sizes
plt.tick_params(axis='both', which='major', labelsize=16)  # major ticks
plt.tick_params(axis='both', which='minor', labelsize=16)  # minor ticks (if any)

plt.legend(fontsize=14)  # Adjust legend text size

# Full bounding box
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)

# Remove grid for clarity
plt.grid(False)

# Show the plot
plt.show()
