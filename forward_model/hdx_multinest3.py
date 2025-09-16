import os
import json
import numpy as np
import logging
import pymultinest
import run_forward_mode_mono as rfm
import tempfile
from itertools import combinations

# ---------------------------------------------------------
# Load Microstates & Representative PDBs
# ---------------------------------------------------------
def load_microstates_and_representatives(microstate_json_path, representative_txt_path):
    """Load microstate definitions and representative PDBs."""
    with open(microstate_json_path, 'r') as f:
        microstates = json.load(f)

    with open(representative_txt_path, 'r') as f:
        representative_paths = [line.strip() for line in f if line.strip()]

    if len(representative_paths) != len(microstates):
        raise ValueError(
            f"Mismatch: {len(representative_paths)} representative structures "
            f"but {len(microstates)} microstates."
        )

    representative_pdbs = {
        ms_id: representative_paths[i] for i, ms_id in enumerate(microstates.keys())
    }
    return microstates, representative_pdbs


# ---------------------------------------------------------
# HDX Likelihood Wrapper
# ---------------------------------------------------------
class HDXMultiNestLikelihood:
    """Wraps HDX forward model for MultiNest."""

    def __init__(self, exp_csv, time_points, deuterium_fraction, pH, temperature, peptide_list=None):
        self.exp_csv = exp_csv
        self.time_points = time_points
        self.deuterium_fraction = deuterium_fraction
        self.pH = pH
        self.temperature = temperature
        self.peptide_list = peptide_list

    def get_likelihood(self, pdb_paths, weights):
        """Compute likelihood of HDX data given structures and weights."""
        # pdb_paths is a string (filename) with newline-separated PDBs
        df_model = rfm.generate_deuteration_df(
            file_path=pdb_paths,
            peptide_list=self.peptide_list,
            deuterium_fraction=self.deuterium_fraction,
            time_points=self.time_points,
            pH=self.pH,
            temperature=self.temperature,
            weights=weights
        )
        df_exp = rfm.load_experimental_csv(self.exp_csv, self.time_points)
        _, _, total_lkhd, _ = rfm.total_likelihood_test(df_exp, df_model)
        return total_lkhd


# ---------------------------------------------------------
# MultiNest Sampler
# ---------------------------------------------------------
class HDXMultiNestSampler:
    def __init__(self, microstates, representative_pdbs, hdx_likelihood_obj,
                 max_states=6, output_dir='./multinest_output/'):
        self.microstates = microstates
        self.microstate_structures = representative_pdbs
        self.hdx_likelihood = hdx_likelihood_obj
        self.max_states = max_states
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.microstate_ids = list(microstates.keys())
        self.n_microstates = len(self.microstate_ids)
        self.logger = logging.getLogger(__name__)
        self.eval_history = {}  # Store likelihood evaluations

    # -----------------------------------------------------
    # Run MultiNest for a given number of states
    # -----------------------------------------------------
    def run_nested_sampling(self, n_states, n_live_points=5):
        self.current_n_states = n_states
        n_dims = n_states
        output_prefix = os.path.join(self.output_dir, f'hdx_{n_states}states_')

        pymultinest.run(
            LogLikelihood=self._log_likelihood_wrapper,
            Prior=self._prior_transform_wrapper,
            n_dims=n_dims,
            outputfiles_basename=output_prefix,
            verbose=True,
            resume=False,
            n_live_points=n_live_points
        )
        return self._load_results(output_prefix)

    # -----------------------------------------------------
    # Log-Likelihood Function
    # -----------------------------------------------------
    def _log_likelihood_wrapper(self, params, n_dims, n_params):
        n_states = self.current_n_states

        # Ensure valid weights
        if n_dims == 1:
            weights = [1.0]
        else:
            clipped = np.clip(params[:n_dims], 1e-8, None)
            weights = (clipped / np.sum(clipped)).tolist()

        # Generate all combinations of n_states microstates
        active_combinations = list(combinations(self.microstate_ids, n_states))

        # Pick combination not yet evaluated
        for comb in active_combinations:
            comb_set = set(comb)
            already_eval = any(set(entry['microstates']) == comb_set 
                               for entry in self.eval_history.get(n_states, []))
            if not already_eval:
                active_microstate_ids = comb
                break
        else:
            active_microstate_ids = active_combinations[0]

        active_structures = [self.microstate_structures[mid] for mid in active_microstate_ids]

        # Save structures to a temp file
        tmpfile_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
                tmpfile_path = tmpfile.name
                for pdb_path in active_structures:
                    tmpfile.write(f"{pdb_path}\n")
                tmpfile.flush()

            lkhd = self.hdx_likelihood.get_likelihood(tmpfile_path, weights)

        finally:
            if tmpfile_path and os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)

        if not np.isfinite(lkhd):
            lkhd = -1e300

        # Store evaluation
        self.eval_history.setdefault(n_states, []).append({
            'microstates': list(active_microstate_ids),
            'weights': weights,
            'likelihood': lkhd
        })

        print(f"Evaluating {n_states}-state model: States={active_microstate_ids}, "
              f"Weights={weights}, Likelihood={lkhd:.3f}")
        return lkhd

    # -----------------------------------------------------
    # Prior Transform
    # -----------------------------------------------------
    def _prior_transform_wrapper(self, params, n_dims, n_params):
        if n_dims == 1:
            params[0] = 1.0
        else:
            clipped = np.clip(params[:n_dims], 1e-8, None)
            total = np.sum(clipped)
            for i in range(n_dims):
                params[i] = clipped[i] / total

    # -----------------------------------------------------
    # Load MultiNest Results
    # -----------------------------------------------------
    def _load_results(self, output_prefix):
        stats_file = output_prefix + 'stats.dat'
        with open(stats_file, 'r') as f:
            for line in f:
                if "Nested Sampling Global Log-Evidence" in line:
                    vals = line.split()
                    return {
                        'log_evidence': float(vals[5]),
                        'log_evidence_error': float(vals[7]),
                        'converged': True
                    }
        return {'log_evidence': -np.inf, 'converged': False}

    # -----------------------------------------------------
    # Compare models
    # -----------------------------------------------------
    def compare_models(self, max_states=None, n_live_points=5):
        if max_states is None:
            max_states = min(self.max_states, self.n_microstates)

        results, log_evidences = {}, {}

        # Single-state evaluations already done manually
        log_evidences[1] = max(entry['likelihood'] for entry in self.eval_history[1])

        # Multi-state models
        for n_states in range(2, max_states + 1):
            print(f"\nRunning {n_states}-state model...")
            res = self.run_nested_sampling(n_states, n_live_points)
            results[n_states] = res
            log_evidences[n_states] = res['log_evidence']
            print(f"{n_states}-state log_evidence = {res['log_evidence']:.3f}")

            # Print evaluation history
            print(f"--- Evaluation history for {n_states}-state model ---")
            for eval_info in self.eval_history.get(n_states, []):
                print(f"States: {eval_info['microstates']}, "
                      f"Weights: {eval_info['weights']}, "
                      f"Likelihood: {eval_info['likelihood']:.3f}")

        # Best model
        valid_models = {k: v for k, v in log_evidences.items() if v > -1e9}
        best_n_states = max(valid_models.keys(), key=lambda k: valid_models[k])
        bayes_factors = {
            f"{best_n_states}_vs_{n}": log_evidences[best_n_states] - log_evidences[n]
            for n in valid_models if n != best_n_states
        }

        # Summary
        print("\n=== Evaluation Summary ===")
        for n_states, evals in self.eval_history.items():
            print(f"\n{n_states}-state models:")
            for e in evals:
                print(f"States: {e['microstates']}, Likelihood: {e['likelihood']:.3f}, Weights: {e['weights']}")

        return {
            'results': results,
            'best_n_states': best_n_states,
            'log_evidences': log_evidences,
            'bayes_factors': bayes_factors,
            'evaluation_history': self.eval_history
        }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # Peptide list
    peptide_list_file = "/Users/kehuang/Documents/projects/nsp2/analysis/hdx/nsp2_pep.txt"
    with open(peptide_list_file) as f:
        peptide_list = [line.strip() for line in f if line.strip()]

    # Microstates & PDBs
    microstates_file = "combined_output/microstates.json"
    representatives_file = "combined_output/representative_microstates.txt"

    microstates, representative_pdbs = load_microstates_and_representatives(
        microstates_file,
        representatives_file
    )

    # HDX Likelihood
    hdx_likelihood = HDXMultiNestLikelihood(
        exp_csv="/Users/kehuang/Documents/projects/nsp2/analysis/hdx/nsp2_hdx.csv",
        time_points=[30, 60, 180, 600, 1800, 3600, 7200],
        deuterium_fraction=0.85,
        pH=7.0,
        temperature=300.0,
        peptide_list=peptide_list
    )

    # Sampler
    sampler = HDXMultiNestSampler(
        microstates=microstates,
        representative_pdbs=representative_pdbs,
        hdx_likelihood_obj=hdx_likelihood,
        max_states=6,
        output_dir="./multinest_output/"
    )

    # STEP 1: Single-state evaluations
    print("\nEvaluating each microstate individually (single-state likelihoods)...")
    for ms_id in sampler.microstate_ids:
        pdb_path = sampler.microstate_structures[ms_id]

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            tmpfile_path = tmpfile.name
            tmpfile.write(f"{pdb_path}\n")
            tmpfile.flush()

        try:
            lkhd = hdx_likelihood.get_likelihood(tmpfile_path, [1.0])
        finally:
            if os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)

        sampler.eval_history.setdefault(1, []).append({
            'microstates': [ms_id],
            'weights': [1.0],
            'likelihood': lkhd
        })
        print(f"Single-state evaluation: Microstate={ms_id}, Likelihood={lkhd:.3f}")

    # STEP 2: Multi-state comparison
    results = sampler.compare_models(max_states=2, n_live_points=5)
    print("Comparison Results:", results)

