import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import itertools
from collections import defaultdict
import logging

class HDXMultiStateModeling:
    """
    Multi-state modeling for HDX data
    """
    
    def __init__(self, structures, hdx_data):
        """
        Initialize the multi-state modeling
        
        Parameters:
        -----------
        structures : list
            List of protein structures (all of them, table? )
        hdx_data : experimental data
        """
        self.structures = structures
        self.hdx_data = hdx_data
        
        
    def compute_rmsd_matrix(self, structures):
        """Compute pairwise RMSD matrix between structures"""
        n_structures = len(structures)
        rmsd_matrix = np.zeros((n_structures, n_structures))
        
        for i in range(n_structures):
            for j in range(i+1, n_structures):
                rmsd = self._calculate_rmsd(structures[i], structures[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
                
        return rmsd_matrix
    
    def _calculate_rmsd(self, struct1, struct2):
        """ Structures RMSD """
        return rmsd
    
    def cluster_structures(self, rmsd_cutoff):
        """Cluster structures using DBSCAN with given RMSD cutoff"""
        rmsd_matrix = self.compute_rmsd_matrix(self.structures)
        
        clustering = DBSCAN(eps=rmsd_cutoff, min_samples=3, metric='precomputed')
        labels = clustering.fit_predict(rmsd_matrix)
        
        # Organize clusters
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Ignore outliers
                clusters[label].append(i)
        
        return dict(clusters)
    
    def evaluate_clustering_quality(self, clusters, rmsd_matrix):
        """
        Evaluate clustering quality 
        """
        if len(clusters) < 1:
            return -1.0
        
        # Silhouette score for cluster quality
        labels = np.full(len(self.structures), -1)
        for cluster_id, structure_indices in clusters.items():
            for idx in structure_indices:
                labels[idx] = cluster_id
        
        valid_indices = labels != -1
        if np.sum(valid_indices) < 2 or len(clusters) < 2:
            return 0.0  # Can't compute silhouette with < 2 clusters
        
        try:
            silhouette = silhouette_score(rmsd_matrix[valid_indices][:, valid_indices], 
                                        labels[valid_indices], metric='precomputed')
        except:
            silhouette = 0.0
        
        # Cluster size balance (prefer similar-sized clusters)
        cluster_sizes = [len(indices) for indices in clusters.values()]
        size_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        
        # Coverage (fraction of structures in clusters vs noise)
        coverage = np.sum(valid_indices) / len(self.structures)
        
        # Maybe, combined  - maybe just one
        quality = 0.5 * silhouette + 0.3 * size_balance + 0.2 * coverage
        
        return quality
    
    def find_optimal_microstate_clustering(self, rmsd_range=(0.8, 3.5), n_points=15):
        """
        Stage 1: Find optimal microstate clustering using ONLY structural criteria
        """
        
        cutoffs = np.linspace(rmsd_range[0], rmsd_range[1], n_points)
        rmsd_matrix = self.compute_rmsd_matrix(self.structures)
        
        results = []
        
        for cutoff in cutoffs:
            clusters = self.cluster_structures(cutoff)
            
            # Skip bad cases
            if len(clusters) == 0:
                quality = -1.0
            elif len(clusters) > len(self.structures) // 2:  
                quality = -0.5
            else:
                quality = self.evaluate_clustering_quality(clusters, rmsd_matrix)
            
            results.append({
                'cutoff': cutoff,
                'n_clusters': len(clusters),
                'quality_score': quality,
                'clusters': clusters
            })
            
         
        
        # Find best clustering 
        best_result = max(results, key=lambda x: x['quality_score'])
        
        
        return best_result, results
    
    def compute_microstate_protection_factors(self, structures):
        """Compute averaged protection factors for a microstate -- MOVE TO ANOTHER CLASS"""
        
        
        all_pf = []
        for struct in strcutures:
            # Compute PF
            
        return 0.0

    def _precompute_macrostate_likelihood(self, structures):
        """ MOVE TO ANOTHER CLASS"""
        return likelihood

    def _generate_new_point(self):
        """
        Uses MCMC sampling
        """

        return new_point


    def nested_sampling_evidence(self, n_states, n_live_points=100, max_iterations=10000, 
                                tolerance=1e-5):
        """
        Nested sampling - Skilling algorithm
        
    
        """
        
        microstate_ids = list(self.microstates.keys())
        
        if n_states > len(microstate_ids):
            return {'log_evidence': -np.inf, 'converged': False, 'reason': 'n_states > n_microstates'}
        
        # Initialize live points
        live_points = self._initialize_live_points(n_states, n_live_points)
        
        # Compute initial likelihoods
        live_likelihoods = []
        for point in live_points:
            likelihood = self.compute_ensemble_likelihood(point['active_states'], point['weights'])
            live_likelihoods.append(likelihood)
        
        # Nested sampling 
        log_evidence = -np.inf
        log_width = np.log(1.0 - np.exp(-1.0 / n_live_points))  # Initial width
        
        evidence_contributions = []
        iteration = 0
        
        while iteration < max_iterations:
            # Find worst (lowest likelihood) point
            worst_idx = np.argmin(live_likelihoods)
            worst_likelihood = live_likelihoods[worst_idx]
            
            # Add contribution to evidence
            log_evidence_contrib = worst_likelihood + log_width
            log_evidence = logsumexp([log_evidence, log_evidence_contrib])
            evidence_contributions.append(log_evidence_contrib)
            
            # Check convergence
            if len(evidence_contributions) > 10:
                recent_contributions = evidence_contributions[-10:]
                max_recent = max(recent_contributions)
                if (log_evidence - max_recent) > abs(log_evidence) * tolerance:
                    break
            
            # Replace worst point with new sample
            new_point = self._generate_new_point(
                live_points, live_likelihoods, worst_likelihood, n_states
            )
            
            
            new_likelihood = self.compute_ensemble_likelihood(
                new_point['active_states'], new_point['weights']
            )
            
            # Update live points
            live_points[worst_idx] = new_point
            live_likelihoods[worst_idx] = new_likelihood
            
            # Update width
            log_width -= 1.0 / n_live_points
            
            iteration += 1
            
        
        # Add final live points
        for likelihood in live_likelihoods:
            log_evidence_contrib = likelihood + log_width - np.log(n_live_points)
            log_evidence = logsumexp([log_evidence, log_evidence_contrib])
        
        converged = iteration < max_iterations
        
        result = {
            'log_evidence': log_evidence,
            'converged': converged,
            'iterations': iteration,
            'n_live_points': n_live_points,
            'final_live_likelihoods': live_likelihoods,
            'reason': 'converged' if converged else 'max_iterations_reached'
        }
        
        
        return result
    
    def run_multistate_modeling(self):
        """
        1. Find microstates (structure-based)
        2. Select optimal number of macrostates
        """
        
        # Stage 1: Find optimal microstate clustering (structure-based)
        best_clustering, clustering_results = self.find_optimal_microstate_clustering()
        microstates = best_clustering['clusters']
        
        # Stage 2: Model selection (HDX-based)
        evidences = self.nested_sampling_model_selection(microstates)
        
        # Select best number of states
        if evidences:
            best_n_states = max(evidences.keys(), key=lambda k: evidences[k])
            best_evidence = evidences[best_n_states]
            
            # Calculate Bayes factors
            bayes_factors = {}
            for n_states, evidence in evidences.items():
                if n_states != best_n_states:
                    bayes_factors[f"{best_n_states}_vs_{n_states}"] = best_evidence - evidence

        else:
            best_n_states = 1
            bayes_factors = {}
        
        results = {
            'optimal_rmsd_cutoff': best_clustering['cutoff'],
            'microstates': microstates,
            'n_microstates': len(microstates),
            'optimal_n_macrostates': best_n_states,
            'model_evidences': evidences,
            'bayes_factors': bayes_factors,
            'clustering_results': clustering_results
        }
        
        return results