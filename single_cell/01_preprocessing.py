import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from single_cell.functions import density_scatter

# Specific cutoff parameters for preprocessing can be found in the metadata table for each individual sample
path = 'cell_ranger_matrix'
min_cells = 0
min_counts = 0
max_counts = 0
min_genes = 0

adata = sc.read_10x_mtx(path)
adata.obs_names_make_unique()

adata.var['mt'] = adata.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

density_scatter(x=adata.obs['total_counts'], y=adata.obs['n_genes_by_counts'], cmap='viridis')
plt.show()

# Gene count distribution plots
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
plt.scatter(x=np.log1p(adata.var['mean_counts']), y=np.log(adata.var['n_cells_by_counts']), s=2)
plt.vlines(x=0, ymin=-0.1, ymax=np.log(10), color='black')
plt.hlines(np.log(10), 0, 4, color='black')
plt.show()

# Distribution histograms
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0], bins=100)
sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 2000], kde=False, bins=100, ax=axs[1])
sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=100, ax=axs[2])
sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 2000], kde=False, bins=100, ax=axs[3])
plt.show()

# Filtering steps
sc.pp.filter_genes(adata, min_cells=min_cells)
sc.pp.filter_cells(adata, min_counts=min_counts)
sc.pp.filter_cells(adata, max_counts=max_counts)
sc.pp.filter_cells(adata, min_genes=min_genes)
adata = adata[adata.obs.pct_counts_mt < 5, :]
