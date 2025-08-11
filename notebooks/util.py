import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def custom_pairplot(
    direct, reparam, cols=None,
    figsize=6.5, bins=30, alpha=0.5, s=12,
    tick_labelsize=8, labelsize=10, legend_fontsize=10,
    corner=True,
):
    direct = np.asarray(direct)
    reparam = np.asarray(reparam)
    assert direct.shape[1] == reparam.shape[1], "dims must match"
    d = direct.shape[1]
    if cols is None:
        cols = [f"Dimension {i+1}" for i in range(d)]

    fig, axes = plt.subplots(d, d, figsize=(figsize, figsize), squeeze=False, constrained_layout=True)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            # Hide upper triangle 
            if corner and i < j:
                continue

            # Diagonal: hist
            if i == j:
                ax.hist(direct[:, j], bins=bins, density=False,
                        histtype="stepfilled", alpha=0.25, color="C0")
                ax.hist(reparam[:, j], bins=bins, density=False,
                        histtype="stepfilled", alpha=0.25, color="C1")

            # Lower triangle: scatter
            else:
                sns.scatterplot(x=direct[:, j], y=direct[:, i], s=s, alpha=alpha, color="C0", label="Direct samples", ax=ax)
                sns.scatterplot(x=reparam[:, j], y=reparam[:, i], s=s, alpha=alpha, color="C1", label="Reparameterized", ax=ax)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[1, -1].legend(
        handles, labels,
        loc="upper left",
        frameon=False,
        fontsize=legend_fontsize
    )
    
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if ax.get_legend():
                ax.get_legend().remove()
            if i == j:
                ax.tick_params(axis="both", left=False, labelleft=False)
            if i > j:
                ax.sharex(axes[-1, j])
                ax.sharey(axes[i, j])
            if i == d - 1:
                ax.set_xlabel(cols[j], fontsize=labelsize)
            else:
                ax.tick_params(axis="both", labelbottom=False)
            if j:
                ax.tick_params(axis="both", labelleft=False)
            if not j and i:
                ax.set_ylabel(cols[i], fontsize=labelsize)
            ax.tick_params(axis="both", labelsize=tick_labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    fig.align_xlabels()
    fig.align_ylabels()
    return fig, axes
