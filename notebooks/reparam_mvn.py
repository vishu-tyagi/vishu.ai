import jax.numpy as jnp
from jax import random
from numpyro.distributions import MultivariateNormal

# Setup
key = random.PRNGKey(0)
num_samples = 1000
dim = 3

# Generate mean and covariance
key, subkey = random.split(key)
mu = random.normal(subkey, (dim,))
A = random.normal(subkey, (dim, dim))
cov = A @ A.T  # ensure positive-definite

# Sample directly
samples = MultivariateNormal(mu, cov).sample(subkey, (num_samples,))

L = jnp.linalg.cholesky(cov)

key, subkey = random.split(key)
z = random.normal(subkey, (num_samples, dim))
samples_reparam = mu + z @ L.T

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util import custom_pairplot

fig, axes = custom_pairplot(samples, samples_reparam)

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
plt.close(fig)
