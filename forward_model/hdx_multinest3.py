import os
import json
import numpy as np
import logging
import pymultinest
import run_forward_mode_mono as rfm  # Your HDX forward model
import tempfile

# Load Microstates & Representative PDBs
def load_microstates_and_representatives(microstate_json_path, representative_txt_path):
    with open(microstate_json_path, 'r') as f:
        microstates = json.load(f)

    with open(representative_txt_path, 'r') as f:
        representative_paths = [line.strip() for line in f if line.strip()]

    if len(representative_paths) != len(microstates):
        raise ValueError(
            f"Mismatch: {len(representative_paths)} representative structures "
            f"but {len(microstates)} microstates."
        )

    representative_pdbs = {ms_id: representative_paths[i] for i, ms_id in enumerate(microstates.keys())}
    return microstates, representative_pdbs

# HDX Likelihood
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

# MultiNest Sampler
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

    # Run MultiNest for n_states
    def run_nested_sampling(self, n_states, n_live_points=500):
        self.current_n_states = n_states
        n_dims = n_states + self.n_microstates  # weight + indicator variables
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

    # Log-Likelihood
    def _log_likelihood_wrapper(self, params, n_dims, n_params):
        n_states = self.current_n_states

        # Microstate indicators
        indicators = np.array([params[n_states + i] for i in range(self.n_microstates)])
        active_mask = indicators > 0.5
        active_microstate_ids = [self.microstate_ids[i] for i, flag in enumerate(active_mask) if flag]

        if len(active_microstate_ids) != n_states:
            return -1e10  # penalize invalid combinations

        active_structures = [self.microstate_structures[mid] for mid in active_microstate_ids]

        # Weights normalization
        unit_weights = [params[i] for i, flag in enumerate(active_mask) if flag]
        total = sum(unit_weights)
        if total > 0:
            weights = np.array([w / total for w in unit_weights])
        else:
            weights = np.array([1.0 / n_states] * n_states)

        print(f"Evaluating microstates: {active_microstate_ids} with weights: {weights}")

        # Compute likelihood
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as tmpfile:
            for pdb_path in active_structures:
                tmpfile.write(f"{pdb_path}\n")
            tmpfile.flush()
            lkhd = self.hdx_likelihood.get_likelihood(tmpfile.name, weights)

        print(f"Log-likelihood = {lkhd:.3f}\n")
        return lkhd

    # Prior Transform
    def _prior_transform_wrapper(self, params, n_dims, n_params):
        n_states = self.current_n_states
        n_microstates = self.n_microstates

        # Microstate indicators: select top n_states, but force at least 2
        indicators = [params[n_states + i] for i in range(n_microstates)]
        sorted_idx = np.argsort(indicators)
        n_active = max(n_states, 2)
    
        binary_indicators = np.zeros(n_microstates)
        for i in range(n_active):
            binary_indicators[sorted_idx[-(i+1)]] = 1.0
        for i in range(n_microstates):
            params[n_states + i] = float(binary_indicators[i])

        # Weight normalization
        unit_weights = [params[i] for i in range(n_states)]

        # Ensure first two weights are non-zero for multi-state start
        for i in range(min(2, n_states)):
            if unit_weights[i] == 0:
                unit_weights[i] = 0.1  # small non-zero starting weight

        total = sum(unit_weights)
        if total > 0:
            normalized_weights = [w / total for w in unit_weights]
        else:
            normalized_weights = [1.0 / n_states] * n_states

        for i in range(n_states):
            params[i] = float(normalized_weights[i])

    # Load MultiNest results
    def _load_results(self, output_prefix):
        stats_file = output_prefix + 'stats.dat'
        with open(stats_file, 'r') as f:
            for line in f:
                if "Nested Sampling Global Log-Evidence" in line:
                    vals = line.split()
                    return {'log_evidence': float(vals[5]),
                            'log_evidence_error': float(vals[7]),
                            'converged': True}
        return {'log_evidence': -np.inf, 'converged': False}

    # Compare Models
    def compare_models(self, max_states=None, n_live_points=500):
        if max_states is None:
            max_states = min(self.max_states, self.n_microstates)

        results, log_evidences = {}, {}
        for n_states in range(1, max_states + 1):
            print(f"Running {n_states}-state model...")
            res = self.run_nested_sampling(n_states, n_live_points)
            results[n_states] = res
            log_evidences[n_states] = res['log_evidence']
            print(f"{n_states}-state log_evidence = {res['log_evidence']:.3f}")

        # Determine best model
        valid_models = {k: v for k, v in log_evidences.items() if v > -1e9}
        best_n_states = max(valid_models.keys(), key=lambda k: valid_models[k])
        bayes_factors = {f"{best_n_states}_vs_{n}": log_evidences[best_n_states] - log_evidences[n]
                         for n in valid_models if n != best_n_states}

        return {'results': results,
                'best_n_states': best_n_states,
                'log_evidences': log_evidences,
                'bayes_factors': bayes_factors}

# Main
if __name__ == "__main__":
    peptide_list_file = "/Users/kehuang/Documents/projects/nsp2/analysis/hdx/nsp2_pep.txt"
    with open(peptide_list_file) as f:
        peptide_list = [line.strip() for line in f if line.strip()]

    microstates_file = "combined_output/microstates.json"
    representatives_file = "combined_output/representative_microstates.txt"

    microstates, representative_pdbs = load_microstates_and_representatives(
        microstates_file,
        representatives_file
    )

    hdx_likelihood = HDXMultiNestLikelihood(
        exp_csv="/Users/kehuang/Documents/projects/nsp2/analysis/hdx/nsp2_hdx.csv",
        time_points=[30,60,180,600,1800,3600,7200],
        deuterium_fraction=0.85,
        pH=7.0,
        temperature=300.0,
        peptide_list=peptide_list
    )

    sampler = HDXMultiNestSampler(
        microstates=microstates,
        representative_pdbs=representative_pdbs,
        hdx_likelihood_obj=hdx_likelihood,
        max_states=6,
        output_dir="./multinest_output/"
    )

    # Compare models
    results = sampler.compare_models(max_states=6, n_live_points=500)
    print("Comparison Results:", results)
