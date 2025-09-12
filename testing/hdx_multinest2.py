import numpy as np
import pymultinest
import json
import os
import logging

# Load Microstates & PDBs #
def load_microstates_and_representatives(microstate_json_path, representative_txt_path):
    """
    Load microstates.json and representative_microstates.txt
    and return two Python dicts:
    1. microstates: { "ms_0": [indices], "ms_1": [indices], ... }
    2. representative_pdbs: { "ms_0": "/path/to/file.pdb", ... }
    """
    # --- Load microstates.json ---
    with open(microstate_json_path, 'r') as f:
        microstates = json.load(f)  # Already in correct format: dict
    
    # --- Load representative_microstates.txt ---
    with open(representative_txt_path, 'r') as f:
        representative_paths = [line.strip() for line in f if line.strip()]
    
    # Ensure both files match
    if len(representative_paths) != len(microstates):
        raise ValueError(
            f"Mismatch: {len(representative_paths)} representative structures "
            f"but {len(microstates)} microstates."
        )
    
    # Build mapping: ms_id -> pdb_path
    representative_pdbs = {
        ms_id: representative_paths[i] for i, ms_id in enumerate(microstates.keys())
    }
    
    return microstates, representative_pdbs

# HDX MultiNest Sampler
class HDXMultiNestSampler:
    """
    Performs Bayesian inference of HDX multi-state models
    using PyMultiNest.
    """
    
    def __init__(self, microstates, representative_pdbs, hdx_likelihood_obj,
                 max_states=6, output_dir='./multinest_output/'):
        """
        Parameters
        ----------
        microstates : dict
            Mapping microstate_id -> list of structure indices
        representative_pdbs : dict
            Mapping microstate_id -> representative PDB path
        hdx_likelihood_obj : HDX likelihood object
            Must implement get_likelihood(structures, weights)
        max_states : int
            Maximum number of states to consider
        output_dir : str
            Directory to save PyMultiNest outputs
        """
        self.microstates = microstates
        self.hdx_likelihood = hdx_likelihood_obj
        self.max_states = max_states
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.microstate_ids = list(microstates.keys())
        self.n_microstates = len(self.microstate_ids)

        # Store representative PDB for each microstate
        self.microstate_structures = representative_pdbs

        self.logger = logging.getLogger(__name__)
    
    
    
    
    # Run Nested Sampling for a fixed number of states
    def run_nested_sampling(self, n_states, n_live_points=1000):
        self.current_n_states = n_states
        
        if n_states > self.n_microstates:
            return {'log_evidence': -np.inf, 'converged': False}
        
        n_dims = n_states + self.n_microstates
        output_prefix = os.path.join(self.output_dir, f'hdx_{n_states}states_')
        
        pymultinest.run(
            LogLikelihood=self._log_likelihood_wrapper,
            Prior=self._prior_transform_wrapper,
            n_dims=n_dims,
            outputfiles_basename=output_prefix,
            verbose=False,
            resume=False,
            n_live_points=n_live_points,
            evidence_tolerance=0.5,
            sampling_efficiency=0.8,
            multimodal=True
        )
        
        return self._load_results(output_prefix)
    
    
    
    #Likelihood Calculation
    def _log_likelihood_wrapper(self, params, n_dims, n_params):
        try:
            n_states = self.current_n_states
            
            weights = np.array(params[:n_states])
            indicators = np.array(params[n_states:n_states + self.n_microstates])
            
            # Determine which microstates are active
            active_mask = indicators > 0.5
            active_microstate_ids = [
                self.microstate_ids[i] for i in range(self.n_microstates) if active_mask[i]
            ]
            
            # Require exactly n_states active microstates
            if len(active_microstate_ids) != n_states:
                return -1e10
            
            # Get PDB paths for the active microstates
            active_structures = [self.microstate_structures[mid] for mid in active_microstate_ids]
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Compute likelihood using user-provided function
            log_likelihood = self.hdx_likelihood.get_likelihood(active_structures, weights)
            
            return log_likelihood
            
        except Exception:
            return -1e10
    
    
    
    # Transform Parameter Space?
    def _prior_transform_wrapper(self, params, n_dims, n_params):
        n_states = self.current_n_states
        
        # Transform to Dirichlet weights (uniform simplex)
        unit_weights = params[:n_states]
        gamma_samples = [-np.log(max(1e-10, u)) for u in unit_weights]
        gamma_sum = sum(gamma_samples)
        weights = [g / gamma_sum for g in gamma_samples] if gamma_sum > 0 else [1.0/n_states] * n_states
        
        # Transform to indicators (choose exactly n_states active microstates)
        unit_indicators = params[n_states:n_states + self.n_microstates]
        indicators = self._sample_indicators(unit_indicators, n_states)
        
        # Update params in-place
        for i, val in enumerate(weights + indicators):
            params[i] = val
    
    def _sample_indicators(self, unit_indicators, n_states):
        n_microstates = len(unit_indicators)
        
        if n_states >= n_microstates:
            return [1.0] * n_microstates
        
        sorted_indices = np.argsort(unit_indicators)
        indicators = [0.0] * n_microstates
        for i in range(n_states):
            idx = sorted_indices[-(i+1)]
            indicators[idx] = 1.0
        
        return indicators
    
    
    
    #Load Results
    def _load_results(self, output_prefix):
        with open(output_prefix + 'stats.dat', 'r') as f:
            lines = f.readlines()

        for line in lines:
            if "Nested Sampling Global Log-Evidence" in line:
                vals = line.split()
                log_evidence = float(vals[5])
                log_evidence_error = float(vals[7])
                break

        return {
            'log_evidence': log_evidence,
            'log_evidence_error': log_evidence_error,
            'converged': True
        }
    
    
    
    
    #Compare Models with 1..max_states
    def compare_models(self, max_states=None, n_live_points=2000):
        if max_states is None:
            max_states = min(self.max_states, self.n_microstates)
        
        self.logger.info(f"Comparing models with 1 to {max_states} states...")
        
        results = {}
        log_evidences = {}
        
        for n_states in range(1, max_states + 1):
            self.logger.info(f"Running {n_states}-state model...")
            result = self.run_nested_sampling(n_states, n_live_points)
            results[n_states] = result
            log_evidences[n_states] = result['log_evidence']
            self.logger.info(f"{n_states} states: log_evidence = {result['log_evidence']:.3f}")
        
        valid_models = {k: v for k, v in log_evidences.items() if v > -1e9}
        best_n_states = max(valid_models.keys(), key=lambda k: valid_models[k])
        best_log_evidence = log_evidences[best_n_states]
        
        # Bayes factors
        bayes_factors = {
            f"{best_n_states}_vs_{n_states}": best_log_evidence - log_evidence
            for n_states, log_evidence in valid_models.items() if n_states != best_n_states
        }
        
        comparison_results = {
            'model_results': results,
            'log_evidences': log_evidences,
            'best_n_states': best_n_states,
            'bayes_factors': bayes_factors
        }
        
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump({k: v for k, v in comparison_results.items() if k != 'model_results'}, f, indent=2)
        
        return comparison_results


    #Extract Weights for Best Model
    def get_best_weights(self, n_states=None):
        if n_states is None:
            results_file = os.path.join(self.output_dir, 'results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    comparison_results = json.load(f)
                n_states = comparison_results['best_n_states']
            else:
                raise ValueError("No comparison results found. Run compare_models() first.")
    
        output_prefix = os.path.join(self.output_dir, f'hdx_{n_states}states_')
    
        try:
            posterior_samples = np.loadtxt(output_prefix + 'post_equal_weights.dat')
            samples = posterior_samples[:, :-2]
            likelihoods = -0.5 * samples[:, -1]
    
            best_sample_idx = np.argmax(likelihoods)
            best_params = posterior_samples[best_sample_idx, :]
            
            weights = best_params[:n_states]
            weights = weights / np.sum(weights)
            
            indicators = best_params[n_states:n_states + self.n_microstates]
            active_mask = indicators > 0.5
            active_microstate_ids = [
                self.microstate_ids[i] for i in range(self.n_microstates) if active_mask[i]
            ]
            
            return {
                'n_states': n_states,
                'weights': weights,
                'active_microstate_ids': active_microstate_ids,
                'max_log_likelihood': posterior_samples[best_sample_idx, -1]
            }
        
        except Exception as e:
            print(f"Error loading results: {e}")
            return None


#Workflow
if __name__ == "__main__":
    # --- Paths to clustering outputs ---
    microstates_file = "combined_output/microstates.json"
    representatives_file = "combined_output/representative_microstates.txt"

    # --- Load microstates and representative structures ---
    microstates, representative_pdbs = load_microstates_and_representatives(
        microstates_file,
        representatives_file
    )

################# fake likelihood function (replace with real HDX) ---
    class DummyHDXLikelihood:
        def get_likelihood(self, pdb_paths, weights):
            print("Active PDBs:", pdb_paths)
            print("Weights:", weights)
            return -0.5  # placeholder log-likelihood

    hdx_likelihood = DummyHDXLikelihood()
###################################

    # --- Initialize sampler ---
    sampler = HDXMultiNestSampler(
        microstates=microstates,
        representative_pdbs=representative_pdbs,
        hdx_likelihood_obj=hdx_likelihood,
        max_states=6,
        output_dir="./multinest_output/"
    )

    # --- Run model comparison ---
    results = sampler.compare_models(max_states=6)
    print("Comparison Results:", results)

    # --- Extract best weights ---
    best_weights = sampler.get_best_weights()
    print("Best Model Details:", best_weights)