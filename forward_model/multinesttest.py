import os
import sys
import pymultinest
import numpy as np

# Ensure output directory exists
output_dir = "chains"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

lib_path = "/Users/kehuang/miniconda3/envs/hdx_test/lib"
os.environ["DYLD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

def likelihood(cube, ndim, nparams):
    x = cube[0]
    return -0.5 * ((x - 2.0) / 0.5) ** 2

def prior(cube, ndim, nparams):
    cube[0] = cube[0] * 5.0

print("Starting PyMultiNest run...")
pymultinest.run(
    LogLikelihood=likelihood,
    Prior=prior,
    n_dims=1,
    importance_nested_sampling=False,
    resume=False,
    verbose=True,
    outputfiles_basename='chains/test_'
)
print("PyMultiNest run completed successfully.")

