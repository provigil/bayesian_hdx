import re
import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- PARAMETERS ---
reference_pdb = "/Users/kehuang/Documents/projects/nsp2/analysis/models/expt_complete.pdb"
models_dir = "/Users/kehuang/Documents/projects/nsp2/analysis/models/"
time_points = [30, 60, 180, 600, 1800, 3600, 7200]
plot_time_points = [30, 60]  # Only plot these times

# --- READ AND PARSE data.txt ---
with open("data.txt") as f:
    raw_data = f.read()

samples = raw_data.strip().split("------------------------------")
data_by_time = {t: [] for t in time_points}
expt_values = {}  # logical expt_nsp2 log-likelihoods
model_loglikelihoods = {}

for sample in samples:
    sample = sample.strip()
    if not sample:
        continue
    lines = sample.split("\n")
    sample_name = lines[0].replace("Results for ", "").strip()

    for line in lines[1:]:
        match = re.search(r'Time\s*?(\d+).*?([0-9]+\.[0-9]+)', line)
        if match:
            time = int(match.group(1))
            value = float(match.group(2))
            if time in time_points:
                data_by_time[time].append(value)
                model_name = f"{sample_name}_{time}"
                model_loglikelihoods[model_name] = value
                if sample_name == "expt_nsp2":
                    expt_values[time] = value

# --- HISTOGRAM PLOTS FOR SELECTED TIMES ---
all_values = [v for t in plot_time_points for v in data_by_time[t]]
xmin, xmax = min(all_values), max(all_values)

fig, axs = plt.subplots(1, len(plot_time_points), figsize=(12, 5), sharey=True)

for i, t in enumerate(plot_time_points):
    values = data_by_time[t]
    counts, bins = np.histogram(values, bins=100, range=(xmin, xmax))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    axs[i].plot(bin_centers, counts, drawstyle="steps-mid", color="blue")
    axs[i].set_title(f'Time: {t}s', fontsize=20, fontweight='bold')
    axs[i].set_xlabel("Score", fontsize=18, fontweight='bold')
    if i == 0:
        axs[i].set_ylabel("Number of models", fontsize=18, fontweight='bold')
    axs[i].tick_params(axis='both', labelsize=12)

    if t in expt_values:
        axs[i].axvline(expt_values[t], color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()

# --- RMSD CALCULATION USING MDAnalysis + NUMPY ---
u_ref = mda.Universe(reference_pdb)
ref_CA = u_ref.select_atoms("name CA")
ref_coords = ref_CA.positions.copy()

def compute_rmsd(model_name):
    if model_name.startswith("expt_nsp2"):
        return None  # skip logical entries

    base_name = model_name.rsplit("_", 1)[0]

    # Fix double .pdb issue
    if not base_name.endswith(".pdb"):
        model_file = os.path.join(models_dir, f"{base_name}.pdb")
    else:
        model_file = os.path.join(models_dir, base_name)

    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None

    try:
        u_model = mda.Universe(model_file)
        model_CA = u_model.select_atoms("name CA")
        if len(model_CA) != len(ref_CA):
            print(f"CA mismatch: {model_file}")
            return None

        rmsd_value = rms.rmsd(model_CA.positions, ref_coords, superposition=True)
        return (model_name, rmsd_value)
    except Exception as e:
        print(f"Error processing {model_file}: {e}")
        return None

# --- PARALLEL RMSD COMPUTATION ---
model_rmsd = {}
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(compute_rmsd, mn) for mn in model_loglikelihoods.keys()]
    for future in as_completed(futures):
        result = future.result()
        if result:
            model_name, rmsd_value = result
            model_rmsd[model_name] = rmsd_value

# --- SCATTER PLOTS PER TIME SLICE (RMSD X, log-likelihood Y) ---
fig, axs = plt.subplots(1, len(plot_time_points), figsize=(12, 5), sharey=True)

for i, t in enumerate(plot_time_points):
    rmsds = []
    logliks = []
    for mn in model_loglikelihoods.keys():
        if mn.endswith(f"_{t}") and mn in model_rmsd:
            rmsds.append(model_rmsd[mn])
            logliks.append(model_loglikelihoods[mn])

    axs[i].scatter(rmsds, logliks, alpha=0.6, color='blue')
    axs[i].set_title(f'Time: {t}s', fontsize=20, fontweight='bold')
    axs[i].set_xlabel("RMSD (Å)", fontsize=18, fontweight='bold')
    if i == 0:
        axs[i].set_ylabel("Score", fontsize=18, fontweight='bold')
    axs[i].tick_params(axis='both', labelsize=12)

    # vertical line for expt_nsp2 if RMSD known
    expt_rmsd = model_rmsd.get(f"expt_nsp2_{t}")
    if expt_rmsd:
        axs[i].axvline(expt_rmsd, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()