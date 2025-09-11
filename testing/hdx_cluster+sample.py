import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from sklearn.cluster import DBSCAN
from collections import defaultdict
from joblib import Parallel, delayed
import argparse
import glob
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list
from scipy.spatial.distance import squareform

# Load coordinates
def load_heavy_atom_coords(pdb_files):
    """
    Load C-alpha atom coordinates from a list of PDB files.
    """
    coords_list = []
    atom_count = None

    for pdb_file in pdb_files:
        u = mda.Universe(pdb_file)
        ca_atoms = u.select_atoms("name CA")  # only C-alpha atoms
        # Ensure all structures have the same number of C-alpha atoms
        if atom_count is None:
            atom_count = len(ca_atoms)
        elif len(ca_atoms) != atom_count:
            raise ValueError(f"{pdb_file} has {len(ca_atoms)} CA atoms, expected {atom_count}.")

        coords_list.append(ca_atoms.positions.copy())

    return np.array(coords_list)


# Compute pairwise RMSD
def pairwise_rmsd_upper(coords_array, n_jobs=-1):
    """
    Compute the upper triangle of the pairwise RMSD matrix.
    """
    N = coords_array.shape[0]
    triu_indices = np.triu_indices(N, k=1)

    def compute_pair(i, j):
        return rms.rmsd(coords_array[i], coords_array[j], center=True)

    upper_rmsd = Parallel(n_jobs=n_jobs)(
        delayed(compute_pair)(i, j) for i, j in zip(*triu_indices)
    )
    return np.array(upper_rmsd)


def upper_to_full_matrix(upper_rmsd, N):
    """
    Convert an upper-triangle RMSD array to a full symmetric matrix.
    """
    full_matrix = np.zeros((N, N))
    triu_indices = np.triu_indices(N, k=1)
    full_matrix[triu_indices] = upper_rmsd
    full_matrix[(triu_indices[1], triu_indices[0])] = upper_rmsd
    return full_matrix

# Clustering
def cluster_structures(rmsd_matrix, rmsd_cutoff, min_samples=1):
    """
    Perform DBSCAN clustering on the RMSD matrix.
    Forces at least one structure per cluster by setting min_samples=1.
    """
    clustering = DBSCAN(eps=rmsd_cutoff, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(rmsd_matrix)

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            clusters["noise"].append(i)
        else:
            clusters[label].append(i)

    return dict(clusters), labels

# Cluster summary
def summarize_clusters(clusters, labels, n_structures):
    """
    Generate summary statistics for clustering results.
    """
    cluster_sizes = [len(indices) for key, indices in clusters.items() if key != "noise"]
    noise_count = len(clusters.get("noise", []))
    coverage = np.sum(labels != -1) / n_structures	#check how many 

    return {
        "n_clusters": len([k for k in clusters.keys() if k != "noise"]),
        "cluster_sizes": cluster_sizes,
        "noise": noise_count,
        "coverage": coverage
    }

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster PDB structures by pairwise RMSD using DBSCAN.")
    parser.add_argument("pdb_files", nargs="+", help="PDB files or wildcard patterns (e.g., *.pdb)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs for RMSD computation")
    parser.add_argument("--n_points", type=int, default=25, help="Number of cutoff points to scan")
    parser.add_argument("--save_hist", type=str, default="cluster_histogram.png", help="Path to save histogram image")
    args = parser.parse_args()

    # Expand wildcards and check for PDB files
    pdb_files = sorted(set(sum([glob.glob(p) for p in args.pdb_files], [])))
    if not pdb_files:
        raise ValueError("No PDB files found!")

    # Load coordinates
    coords_array = load_heavy_atom_coords(pdb_files)
    print("Coordinates shape:", coords_array.shape)

    # Compute RMSD matrix
    upper_rmsd = pairwise_rmsd_upper(coords_array, n_jobs=args.n_jobs)
    rmsd_matrix = upper_to_full_matrix(upper_rmsd, coords_array.shape[0])

    # Determine dynamic RMSD cutoff range
    mean_rmsd = np.mean(rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)])
    max_dynamic_rmsd = mean_rmsd
    print(f"Dynamic RMSD cutoff (mean): {max_dynamic_rmsd:.2f}")

    cutoffs = np.linspace(1.0, max_dynamic_rmsd, args.n_points)
    print(f"Testing cutoffs between 1.0 and {max_dynamic_rmsd:.2f}")

    best_cutoff = None
    best_clusters = None
    best_labels = None
    max_coverage = -1

    for cutoff in cutoffs:
        clusters, labels = cluster_structures(rmsd_matrix, rmsd_cutoff=cutoff, min_samples=1)
        summary = summarize_clusters(clusters, labels, len(coords_array))

        print(f"RMSD Cutoff: {cutoff:.2f}, Cluster # = {summary['n_clusters']}, Coverage: {summary['coverage']*100:.1f}%, Noise: {summary['noise']}")
        
# Hierarchical clustering
# -------------------------
# Convert RMSD matrix to condensed form for linkage (upper-triangle only)
condensed_rmsd = squareform(rmsd_matrix)  # Required for linkage

# Perform hierarchical clustering
Z = linkage(condensed_rmsd, method='average')  # 'average', 'single', 'complete' are options

# Get the leaf order from hierarchical clustering
leaf_order = leaves_list(Z)

# Reorder the RMSD matrix according to the hierarchical clustering
ordered_rmsd_matrix = rmsd_matrix[np.ix_(leaf_order, leaf_order)]


plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True, color_threshold=0)
plt.title("Pairwise RMSD", fontsize=20, fontweight='bold')
plt.xlabel("Structures", fontsize=18, fontweight='bold')
plt.ylabel("RMSD", fontsize=18, fontweight='bold')
plt.show()


# Reorder RMSD matrix according to hierarchical clustering
white_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

plt.figure(figsize=(10, 8))
sns.heatmap(
    ordered_rmsd_matrix,
    cmap=white_red,
    square=True,
    cbar_kws={'label': 'RMSD'},
    xticklabels=False,  # remove x-axis tick labels
    yticklabels=False   # remove y-axis tick labels
)

plt.xlabel("Structures", fontsize=18, fontweight='bold')
plt.ylabel("Structures", fontsize=18, fontweight='bold')
plt.title("Pairwise RMSD Heatmap", fontsize=20, fontweight='bold')

plt.show()