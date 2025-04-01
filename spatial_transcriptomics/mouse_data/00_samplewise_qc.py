import anndata
import numpy as np
import scanpy as sc
import pandas as pd
from glob import glob
import seaborn as sns
from tqdm import tqdm
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


names = {'primary_tumor no_treatment na': 'Primary tumor',
         'control no_treatment na': 'Control',
         'relapsed_tumor TAC_regimen 30_days' : 'TAC 30dpt',
         'relapsed_tumor cisplatin_6mg/kg 30_days': 'Cisplatin 30dpt',
         'residual_tumor TAC_regimen 12_days': 'TAC 12dpt',
         'residual_tumor TAC_regimen 7_days': 'TAC 7dpt',
         'residual_tumor cisplatin_6mg/kg 12_days': 'Cisplatin 12dpt',
         'residual_tumor cisplatin_6mg/kg 7_days': 'Cisplatin 7dpt',
         'primary_tumor cisplatin_6mg/kg 4_hours': 'Cisplatin 4hpt',
         'primary_tumor cisplatin_6mg/kg 24_hours': 'Cisplatin 24hpt',
         'primary_tumor cisplatin_12mg/kg 24_hours': 'Cisplatin 24hpt'}

#%%
# manual QC per sample - selected cutoffs are stored in metadata table

root_folder = '/mnt/c/Bern/Visium_processed/'
samples = glob(root_folder + 'mammary*/runs/*/')

meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)
meta_df['sample_path'] = samples

row = meta_df.iloc[0]  # Select sample

ad = sc.read_visium(row['sample_path'])
sc.pp.calculate_qc_metrics(ad, inplace=True)
sc.pp.filter_genes(ad, min_cells=10)

density_scatter(x=ad.obs['total_counts'], y=ad.obs['n_genes_by_counts'], cmap='viridis')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.distplot(ad.obs["total_counts"], kde=False, ax=axs[0])
sns.distplot(ad.obs["n_genes_by_counts"], kde=False, bins=40, ax=axs[1])
plt.show()

sc.pl.spatial(ad, img_key="hires", color=["total_counts", "n_genes_by_counts"], size=1.6, cmap='viridis', alpha=0.9)

sc.pp.filter_cells(ad, min_counts=2000)
sc.pp.filter_cells(ad, max_counts=55000)
sc.pp.filter_cells(ad, min_genes=2000)

#%%
# QC plots

# collect samples + segment tissue images
adatas = []
for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):

    ad = sc.read_visium(row['sample_path'])

    # replace spatial key
    spatial_key = list(ad.uns['spatial'].keys())[0]
    ad.uns['spatial'][row['sample_id']] = ad.uns['spatial'][spatial_key]
    del ad.uns['spatial'][spatial_key]

    # add metadata to adata
    row.pop('sample_path')
    for k, v in row.items():
        ad.obs[k] = v

    if ad.shape[0] != 0:
        # normalize
        sc.pp.calculate_qc_metrics(ad, inplace=True)
        sc.pp.filter_genes(ad, min_cells=10)

        if row['count_lower_cutoff'] != 0:
            sc.pp.filter_cells(ad, min_counts=row['count_lower_cutoff'])
        if row['count_upper_cutoff'] != 0:
            sc.pp.filter_cells(ad, max_counts=row['count_upper_cutoff'])
        if row['n_gene_cutoff'] != 0:
            sc.pp.filter_cells(ad, min_genes=row['n_gene_cutoff'])

        sc.pp.normalize_total(ad, inplace=True, target_sum=1e4, exclude_highly_expressed=True)
        sc.pp.log1p(ad)
        adatas.append(ad)

adata = anndata.concat(adatas, index_unique='-', uns_merge='unique', merge='unique')

sample_indexes = []
for idx, row in adata.obs.iterrows():
    sample_index = f"{row['condition']} {row['treatment']} {row['elapsed_time']}"
    sample_index = f"{row['sample_id']} | {names[sample_index]}"
    sample_indexes.append(sample_index)

adata.obs['label'] = sample_indexes

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(adata.obs, x='label', y='total_counts', ax=ax, errorbar="sd")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel('Total counts')
ax.set_xlabel('Sample')
ylim = list(ax.get_ylim())
ylim[0] = 0.0
ax.set_ylim(tuple(ylim))
ax.grid(axis='both')
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(adata.obs, x='label', y='n_genes_by_counts', ax=ax, errorbar="sd")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel('Genes expressed')
ax.set_xlabel('Sample')
ax.grid(axis='both')
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.barplot(adata.obs, x='label', y='total_counts', ax=ax[0], errorbar="sd")
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_ylabel('Total counts')
ax[0].set_xlabel(None)
ylim = list(ax[0].get_ylim())
ylim[0] = 0.0
ax[0].set_ylim(tuple(ylim))
ax[0].grid(axis='both')
ax[0].set_axisbelow(True)

sns.barplot(adata.obs, x='label', y='n_genes_by_counts', ax=ax[1], errorbar="sd")
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
ax[1].set_ylabel('Genes expressed')
ax[1].set_xlabel('Sample')
ax[1].grid(axis='both')
ax[1].set_axisbelow(True)

plt.tight_layout()
plt.show()
