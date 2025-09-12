#!/usr/bin/env python3
"""
run_clustering_and_multinest.py

Pipeline Overview:
  1. Load multiple PDB structures and extract C-alpha coordinates.
  2. Compute a pairwise RMSD matrix to quantify structural similarity.
  3. Run DBSCAN clustering with varying RMSD cutoffs to identify stable clusters
     (microstates).
  4. Save a text file containing representative PDB file paths for each cluster.
  5. Convert clusters into microstates (representing unique structural ensembles).
  6. Instantiate an HDX forward model to compute likelihoods for microstate combinations.
  7. Run MultiNest to:
       - Compare models with 1..max_states to determine optimal number of states.
       - Identify the most probable microstates and their population weights.
  8. Save clustering, sampling, and model comparison results to the output directory.

Requirements:
  - Python packages: MDAnalysis, joblib, sklearn, pymultinest, numpy
  - forwardmodel.py: must provide HDXForwardModel.get_likelihood() for real use cases.
"""

# -------------------------
# Imports & Logging
# -------------------------
import os
import glob
import argparse
import json
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Global Defaults
# -------------------------
DEFAULT_MIN_SAMPLES = 2             # DBSCAN minimum samples per cluster
DEFAULT_N_POINTS = 25               # Number of cutoffs to scan when tuning DBSCAN
DEFAULT_DESIRED_MIN_CLUSTERS = 6    # Lower bound for desired cluster count
DEFAULT_DESIRED_MAX_CLUSTERS = 10   # Upper bound for desired cluster count
DEFAULT_OUTPUT_DIR = "./combined_output"

# ============================================================================
# SECTION 1. RMSD Utility Functions
# ============================================================================
def load_ca_coords(pdb_files: List[str]) -> np.ndarray:
    """
    Load C-alpha coordinates from multiple PDB files.

    Args:
        pdb_files: List of PDB file paths.

    Returns:
        coords: Shape (N_structures, N_atoms, 3)
    
    Raises:
        ValueError: If structures contain inconsistent atom counts.
    """
    coords_list = []
    atom_count = None
    for pdb in pdb_files:
        u = mda.Universe(pdb)
        ca = u.select_atoms("name CA")
        if atom_count is None:
            atom_count = len(ca)
            logger.debug("Detected %d CA atoms", atom_count)
        elif len(ca) != atom_count:
            raise ValueError(f"{pdb} has {len(ca)} CA atoms; expected {atom_count}.")
        coords_list.append(ca.positions.copy())
    return np.array(coords_list)


def pairwise_rmsd_upper(coords_array: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    Compute the upper-triangle of the RMSD matrix in parallel.

    Returns:
        1D array of RMSD values for triu indices.
    """
    N = coords_array.shape[0]
    if N < 2:
        return np.array([])
    triu = np.triu_indices(N, k=1)

    def _pair(i, j):
        return float(rms.rmsd(coords_array[i], coords_array[j], center=True))

    return np.array(Parallel(n_jobs=n_jobs)(delayed(_pair)(i, j) for i, j in zip(*triu)))


def upper_to_full_matrix(upper_rmsd: np.ndarray, N: int) -> np.ndarray:
    """
    Convert upper-triangle RMSD data to a full symmetric matrix.
    """
    mat = np.zeros((N, N), dtype=float)
    if upper_rmsd.size == 0:
        return mat
    tri = np.triu_indices(N, k=1)
    mat[tri] = upper_rmsd
    mat[(tri[1], tri[0])] = upper_rmsd
    return mat

# ============================================================================
# SECTION 2. Clustering (DBSCAN)
# ============================================================================
def cluster_dbscan(rmsd_matrix: np.ndarray, rmsd_cutoff: float, min_samples: int = 2):
    """
    Run DBSCAN clustering using the precomputed RMSD matrix.
    """
    clustering = DBSCAN(eps=rmsd_cutoff, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(rmsd_matrix)
    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters["noise" if lab == -1 else int(lab)].append(idx)
    return dict(clusters), labels


def summarize_clusters(clusters, labels, n_structures):
    """
    Summarize clustering results:
      - number of clusters (excl. noise)
      - cluster sizes
      - noise count
      - fraction of structures assigned to clusters
    """
    cluster_sizes = [len(v) for k, v in clusters.items() if k != "noise"]
    coverage = float(np.sum(labels != -1) / n_structures) if n_structures > 0 else 0.0
    return {
        "n_clusters": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "noise": len(clusters.get("noise", [])),
        "coverage": coverage
    }


def find_cutoff(rmsd_matrix, n_points=DEFAULT_N_POINTS, min_samples=DEFAULT_MIN_SAMPLES,
                desired_min=DEFAULT_DESIRED_MIN_CLUSTERS, desired_max=DEFAULT_DESIRED_MAX_CLUSTERS):
    """
    Automatically search for the best RMSD cutoff for DBSCAN.
    """
    N = rmsd_matrix.shape[0]
    if N < 2:
        raise ValueError("At least 2 structures are required for clustering.")
    mean_rmsd = float(np.mean(rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]))
    cutoffs = np.linspace(1.0, mean_rmsd, n_points)

    best_candidate = None
    best_score = None

    for cutoff in cutoffs:
        clusters, labels = cluster_dbscan(rmsd_matrix, rmsd_cutoff=cutoff, min_samples=min_samples)
        summary = summarize_clusters(clusters, labels, N)
        logger.info("cutoff=%.3f -> %d clusters, coverage=%.1f%%", cutoff, summary["n_clusters"], summary["coverage"] * 100.0)

        if desired_min <= summary["n_clusters"] <= desired_max:
            return cutoff, clusters, labels  # ideal match

        # Fallback scoring: minimize distance to target range, maximize coverage
        dist = max(desired_min - summary["n_clusters"], summary["n_clusters"] - desired_max, 0)
        score = (dist, -summary["coverage"])
        if best_score is None or score < best_score:
            best_score, best_candidate = score, (cutoff, clusters, labels, summary)

    logger.warning("No cutoff produced desired cluster range, using fallback cutoff %.3f", best_candidate[0])
    return best_candidate[0], best_candidate[1], best_candidate[2]


def clusters_to_microstates(clusters, include_noise=False):
    """
    Convert DBSCAN clusters into microstate dictionary.
    Keys are 'ms_<cluster_id>' for clarity.
    """
    microstates = {}
    for k, members in clusters.items():
        if k == "noise":
            if include_noise:
                microstates["ms_noise"] = list(members)
            continue
        microstates[f"ms_{k}"] = list(members)
    return microstates

# ============================================================================
# SECTION 3. Save representative PDBs
# ============================================================================
def save_microstate_representatives(microstates: Dict[str, List[int]], pdb_files: List[str], output_txt_path: str):
    """
    Save representative PDB path for each microstate cluster into a text file.
    """
    with open(output_txt_path, 'w') as f:
        for cluster_id, indices in microstates.items():
            if not indices:
                continue
            central_idx = len(indices) // 2
            rep_idx = indices[central_idx]
            rep_path = pdb_files[rep_idx]
            f.write(rep_path + '\n')

    logger.info(f"Representative PDB paths saved to: {output_txt_path}")

# ============================================================================
# SECTION 4. Forward Model (Example RMSD-based)
# ============================================================================
class RMSDForward:
    """
    Simple forward model using RMSD matrix to compute
    a Gaussian-like log-likelihood for active microstates.
    """
    def __init__(self, rmsd_matrix: np.ndarray, sigma: float = 1.0):
        self.rmsd = np.array(rmsd_matrix)
        if self.rmsd.shape[0] != self.rmsd.shape[1]:
            raise ValueError("RMSD matrix must be square.")
        self.sigma = max(sigma, 1e-12)

    def get_likelihood(self, active_indices, weights):
        """
        Compute likelihood for selected active structures with given weights.
        """
        if len(active_indices) == 0:
            return -1e10
        w = np.array(weights)
        if w.sum() <= 0:
            return -1e10
        w /= w.sum()
        sub_d = self.rmsd[np.ix_(active_indices, active_indices)]
        ens_rmsd = float(w.dot(sub_d).dot(w))
        return -0.5 * (ens_rmsd / self.sigma) ** 2

# ============================================================================
# SECTION 5. Main Orchestration
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Cluster PDBs by RMSD and run MultiNest.")
    parser.add_argument("pdb_files", nargs="+", help="PDB files or glob patterns (e.g. '*.pdb')")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    # 1. Expand PDB file globs
    pdb_files = sorted(set(sum([glob.glob(p) for p in args.pdb_files], [])))
    if not pdb_files:
        raise ValueError("No PDB files found.")

    logger.info("Loaded %d PDB structures.", len(pdb_files))

    # 2. Load coordinates
    coords = load_ca_coords(pdb_files)

    # 3. Compute pairwise RMSD
    logger.info("Computing RMSD matrix...")
    upper = pairwise_rmsd_upper(coords)
    rmsd_mat = upper_to_full_matrix(upper, coords.shape[0])

    # 4. Cluster with DBSCAN
    cutoff, clusters, labels = find_cutoff(rmsd_mat)
    microstates = clusters_to_microstates(clusters)
    logger.info("Identified %d microstates.", len(microstates))

    os.makedirs(args.outdir, exist_ok=True)

    # Save representative PDB paths
    rep_txt_path = os.path.join(args.outdir, "representative_microstates.txt")
    save_microstate_representatives(microstates, pdb_files, rep_txt_path)

    # Save full microstate membership mapping
    with open(os.path.join(args.outdir, "microstates.json"), "w") as f:
        json.dump(microstates, f, indent=2)

    # 5. Build forward model (placeholder)
    sigma_test = np.mean(rmsd_mat[np.triu_indices_from(rmsd_mat, k=1)])
    forward = RMSDForward(rmsd_mat, sigma=sigma_test)

    logger.info("Pipeline finished. Results saved in %s", args.outdir)

if __name__ == "__main__":
    main()
