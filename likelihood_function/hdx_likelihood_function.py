#all from Saltzberg 2016
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import forward_model as fm
import scipy as sp 

def noise_model(d_exp:float, d_model:float, sigma:float, A: float, B: float):
    exp = (np.exp(np.power(d_exp - d_model, 2)/ (2 * np.power(sigma, 2))))/ (np.sqrt(2 * np.pi) * sigma)
    erf = 0.5*(sp.special.erf((A-d_exp)/(np.sqrt(2) * sigma))) - (sp.special.erf((B-d_exp)/(np.sqrt(2) * sigma)))
    return exp*erf
    

def calculate_sigma(replicates):
    # Convert input to numpy array (in case it's a list)
    replicates = np.array(replicates)    
    # Calculate the standard deviation using numpy's std function with ddof=1 for sample std dev
    std_dev = np.std(replicates, ddof=1)  # ddof=1 for sample standard deviation (Bessel's correction)
    return std_dev

#define a function that calculates a dictionary of sigma values for each peptide and time point

def total_likelihood(df:pd.DataFrame, a = -0.1, b = 1.2, phi = 0.85):
    list_of_peptides = set(df['peptide'])
    time_points = ['0s', '30s', '60s', '300s', '900s', '3600s', '14400s', '86400s']
    replicate = ['1', '2', '3']
    total_log_likelihood = 0
    results = []
    for peptide in list_of_peptides:
        for time in time_points:
            for rep in replicate:
                d_exp = df[df['peptide'] == peptide][f'{time}_noised_{rep}'].iloc[0]
                d_model = df[df['peptide'] == peptide][f'{time}'].iloc[0]
                # Define columns using f-strings
                columns = [f'{time}_noised_1', f'{time}_noised_2', f'{time}_noised_3']
                all_reps = np.array(df[df['peptide'] == peptide][columns]).flatten()
                sigma = calculate_sigma(all_reps)
                A = a*phi
                B = b*phi
                total_log_likelihood += np.log(noise_model(d_exp, d_model, sigma, A, B))
    return total_log_likelihood