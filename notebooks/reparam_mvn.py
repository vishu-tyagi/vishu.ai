import jax.numpy as jnp
from jax import random
from numpyro import distributions as dist

# Setup
key = random.PRNGKey(0)
num_samples = 1000  # N = 1_000
dim = 3             # d = 3

# Generate mean
key, sub = random.split(key)
mu = random.normal(sub, (dim,))

# Generate covariance
key, sub = random.split(key)
A = random.normal(sub, (dim, dim))
cov = A @ A.T   # ensure positive definite

# Generate samples directly
key, sub = random.split(key)
direct = dist.MultivariateNormal(mu, cov).sample(sub, (num_samples,))

# Lower Cholesky factor 
L = jnp.linalg.cholesky(cov)    # (d, d)

# Generate samples with reparametrization
key, subkey = random.split(key)
z = random.normal(subkey, (num_samples, dim))   # (N, d)
reparam = mu + jnp.einsum("ij,nj->ni", L, z)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util import custom_pairplot

fig, axes = custom_pairplot(direct, reparam)

import os
from constants import BUILD_DIR, BG_COLOR

BUILD_DIR = os.path.join(BUILD_DIR, "notebooks")
os.makedirs(BUILD_DIR, exist_ok=True)

site_bg = BG_COLOR
fig.patch.set_facecolor(site_bg)
for ax in fig.axes:
    ax.set_facecolor(site_bg)
output_path = os.path.join(BUILD_DIR, "reparam_mvn.png")
fig.savefig(output_path, dpi=300)
print(f"Saved to {output_path}")
plt.close(fig)
