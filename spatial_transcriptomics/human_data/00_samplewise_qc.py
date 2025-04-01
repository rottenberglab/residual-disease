import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def density_scatter(x, y, s=3, cmap='viridis', ax=None):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    nx, ny, z = x[idx], np.array(y)[idx], z[idx]
    x_list = nx.tolist()
    y_list = ny.tolist()
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    return plt.scatter(x_list, y_list, alpha=1, c=z, s=s, zorder=2, cmap=cmap)

#%%
# manual QC per sample - selected cutoffs are stored in metadata table

meta_df = pd.read_csv('data/meta_df_human_filled.csv', index_col=0)

row = meta_df.iloc[6]

ad = sc.read_visium(row['sample_path'], count_file='raw_feature_bc_matrix.h5')
ad = ad[ad.obs['in_tissue'] == 1]
sc.pp.calculate_qc_metrics(ad, inplace=True)
sc.pp.filter_genes(ad, min_cells=10)

density_scatter(x=ad.obs['total_counts'], y=ad.obs['n_genes_by_counts'], cmap='viridis')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.distplot(ad.obs["total_counts"], kde=False, ax=axs[0])
sns.distplot(ad.obs["n_genes_by_counts"], kde=False, bins=40, ax=axs[1])
plt.show()

sc.pl.spatial(ad, img_key="hires", color=["total_counts", "n_genes_by_counts"], size=1.6, cmap='viridis', alpha=0.9)

sc.pp.filter_cells(ad, min_counts=500)

sc.pp.filter_cells(ad, max_counts=80000)

sc.pp.filter_cells(ad, min_genes=1000)
