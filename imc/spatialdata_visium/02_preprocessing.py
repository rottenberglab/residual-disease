import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
from spatial_transcriptomics.functions import sort_paths
from imc.spatialdata_visium.functions import swap_gene_ids


samples = glob('/mnt/c/Users/demeter_turos/PycharmProjects/hyperion/data/adatas/*.h5ad')
output_folder = 'data/imc_samples'

#%%
# read samples
meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)

# match sample names and folders
samples = sort_paths(samples, meta_df['sample_id'], pos=-1, suffix='.h5ad')
meta_df['sample_path'] = samples

#%%

metadata_columns = ['sample_id', 'batch', 'slide', 'condition', 'slide_name']  # columns to save to .obs from metadata

for idx, row in meta_df.iterrows():

    ad = sc.read_h5ad(row['sample_path'])
    swap_gene_ids(ad)  # swap gene symbols with gene IDs

    # add metadata as .obs columns
    assert set(metadata_columns).issubset(set(meta_df.columns))
    for c in metadata_columns:
        ad.obs[c] = row[c]

    # 3. filter genes
    sc.pp.calculate_qc_metrics(ad, inplace=True)
    sc.pp.filter_genes(ad, min_cells=10)

    if row['count_lower_cutoff'] != 0:
        sc.pp.filter_cells(ad, min_counts=row['count_lower_cutoff'])
    if row['count_upper_cutoff'] != 0:
        sc.pp.filter_cells(ad, max_counts=row['count_upper_cutoff'])
    if row['n_gene_cutoff'] != 0:
        sc.pp.filter_cells(ad, min_genes=row['n_gene_cutoff'])

    ad.raw = ad  # save raw counts

    # normalize count matrix
    sc.pp.normalize_total(ad, inplace=True, target_sum=1e4, exclude_highly_expressed=True)
    sc.pp.log1p(ad)

    print(ad.uns['spatial'].keys())

    ad.write_h5ad(f'{output_folder}/{ad.obs["sample_id"][0]}.h5ad')

#%%
# read and concat samples

h5ads =  glob(f'{output_folder}/*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
adata_spatial = anndata.concat(adatas, index_unique='-', uns_merge='unique', merge='first')

adata_spatial.obs['annotations'] = adata_spatial.obs['annotations'].astype('category')

#%%
# annotation plots

rows = 3
cols = 3

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
ax = ax.flatten()
for a in ax:
    a.axis('off')
for idx, s in enumerate(np.unique(adata_spatial.obs['sample_id'])):
    ad = adata_spatial[adata_spatial.obs['sample_id'] == s].copy()
    sc.pl.spatial(ad, color='annotations', size=1.5, alpha=1, library_id=s,
                  ax=ax[idx], show=False, cmap='viridis')
    ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{ad.obs["condition"][0]}\n')
plt.tight_layout()
plt.show()

#%%
# H&E with annotations

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
ax = ax.flatten()
for a in ax:
    a.axis('off')
for idx, s in enumerate(np.unique(adata_spatial.obs['sample_id'])):
    ad = adata_spatial[adata_spatial.obs['sample_id'] == s].copy()
    sc.pl.spatial(ad, color='annotations', size=1.5, alpha=1, library_id=s,
                  ax=ax[idx], show=False, cmap='viridis')
    ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}\n' + \
                      f'{treatment[ad.obs["treatment"][0]]}\n{time[ad.obs["elapsed_time"][0]]}')
plt.tight_layout()
plt.show()

#%%
# IMC data

print(adata_spatial.obsm['imc_intensity'].columns)

isotope = 'Yb172-BRCA1'
adata_spatial.obs[isotope] = adata_spatial.obsm['imc_intensity'][isotope]

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
ax = ax.flatten()
for a in ax:
    a.axis('off')
for idx, s in enumerate(np.unique(adata_spatial.obs['sample_id'])):
    ad = adata_spatial[adata_spatial.obs['sample_id'] == s].copy()
    sc.pl.spatial(ad, color=isotope, size=1.5, alpha=1, library_id=s,
                  ax=ax[idx], show=False, cmap='viridis', vmin=0)
    ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}\n' + \
                      f'{treatment[ad.obs["treatment"][0]]}\n{time[ad.obs["elapsed_time"][0]]}')
plt.tight_layout()
plt.show()

#%%
# IMC data Pt

print(adata_spatial.obsm['imc_intensity'].columns)

isotope = 'Platinum'
cols = [x for x in adata_spatial.obsm['imc_intensity'].columns if 'Pt' in x]
pt_df = adata_spatial.obsm['imc_intensity'][cols].sum(axis=1)
pt_df[pt_df == 0] = np.nan
adata_spatial.obs[isotope] = np.log1p(pt_df)

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
ax = ax.flatten()
for a in ax:
    a.axis('off')
for idx, s in enumerate(np.unique(adata_spatial.obs['sample_id'])):
    ad = adata_spatial[adata_spatial.obs['sample_id'] == s].copy()
    sc.pl.spatial(ad, color=isotope, size=1.5, alpha=1, library_id=s,
                  ax=ax[idx], show=False, cmap='viridis', vmin=0, vmax=4)
    ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}\n' + \
                      f'{treatment[ad.obs["treatment"][0]]}\n{time[ad.obs["elapsed_time"][0]]}')
plt.tight_layout()
plt.show()

#%%
# QC

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.violinplot(data=adata_spatial.obs, x='sample_id', y='n_genes_by_counts')
ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_ylabel("n_genes_by_counts")
ax.set_xlabel('samples')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.violinplot(data=adata_spatial.obs, x='sample_id', y='log1p_total_counts')
ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_ylabel("log1p_total_counts")
ax.set_xlabel('samples')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()
