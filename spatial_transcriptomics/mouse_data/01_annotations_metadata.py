import anndata
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from spatial_transcriptomics.functions import read_annotations, preprocess_visium, gene_symbol_to_ensembl_id


#%%
# Preprocess samples
root_folder = '/path/'
samples = glob(root_folder + 'mammary*/runs/*/')

# Read samples
meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)
meta_df['sample_path'] = samples

# Add annotations
annots, slides = read_annotations(meta_df, qupath_project='data/annotations/project.qpproj', show=False)

meta_df = meta_df.merge(annots, left_on='sample_id', right_index=True, how='left')
meta_df = meta_df.merge(slides, left_on='sample_id', right_index=True, how='left')

# Add annotations and metadata
adatas = preprocess_visium(meta_df)

# Save preprocessed samples
for ad in adatas:
    ad.write_h5ad(f'data/samples/{ad.obs["sample_id"][0]}.h5ad')

#%%
# Read the samples back
samples = glob('data/samples/*.h5ad')
adatas = []

for s in tqdm(samples):
    ad = sc.read_h5ad(s)
    ad = gene_symbol_to_ensembl_id(ad)

    ad.var_names_make_unique()

    sc.pp.calculate_qc_metrics(ad, inplace=True)

    sc.pp.filter_genes(ad, min_cells=10)
    sc.pp.filter_genes(ad, min_counts=10)
    sc.pp.filter_cells(ad, min_genes=10)

#%%
# H&E with annotations and labels

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor',
             'residual_tumor': 'Residual tumor', 'control': 'Control'}

treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg',
             'no_treatment': 'No treatment', 'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}

time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment', '7_days': '7 days post-treatment',
        'na': '-', '24_hours': '24 hours post-treatment', '4_hours': '4 hours post-treatment'}

cats = ['Tumor', 'Stroma', 'Necrosis', 'Immune cells', 'Epithelial cells', 'Muscle', 'Other', 'Ignore*']
for ad in adatas:
    ad.obs['annotations'] = ad.obs['annotations'].cat.set_categories(cats, ordered=True)
    ad.obs['annotations'] = ad.obs['annotations'].cat.reorder_categories(cats, ordered=True)

rows = 6
cols = 6

# adding annotation colors
hexes = ['#9d5', '#4d8', '#2cb', '#0bc', '#e69100', '#36b', '#a679d2', '#817']
hexes = hexes[::-1]
custom_colors_rgb = [mcolors.hex2color(color) for color in hexes]
palette = sns.color_palette(custom_colors_rgb)

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
ax = ax.flatten()
for a in ax:
    a.axis('off')
for idx, ad in enumerate(adatas):
    sc.pl.spatial(ad, color='annotations', size=1.5, alpha=1, library_id=ad.obs['sample_id'][0],
                  ax=ax[idx], show=False,
                  # palette=sns.color_palette("muted", 8),
                  palette=palette,
                  legend_loc=None)

    ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}\n' + \
                      f'{treatment[ad.obs["treatment"][0]]}\n{time[ad.obs["elapsed_time"][0]]}')
plt.tight_layout()
plt.savefig(f'annotations.svg')
plt.show()

# labels
unique_annotations = adatas[0].obs['annotations'].cat.categories
fig, ax = plt.subplots()
ax.axis('off')
handles = []
for annotation, color in zip(unique_annotations, palette):
    handle = ax.scatter([], [], label=annotation, color=color)
    handles.append(handle)
ax.legend(handles=handles, labels=unique_annotations.tolist(), loc='center',
          fontsize='small', title='Annotations')
plt.savefig(f'annotation_labels.svg')
plt.show()

#%%
# concat samples

adata_spatial = anndata.concat(adatas, index_unique='-', uns_merge='unique', merge='unique')
adata_spatial.obs['annotations'] = adata_spatial.obs['annotations'].astype('category')

#%%
# QC
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
sns.violinplot(data=adata_spatial.obs, x='sample_id', y='n_genes_by_counts')
ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_ylabel("n_genes_by_counts")
ax.set_xlabel('samples')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
sns.violinplot(data=adata_spatial.obs, x='sample_id', y='total_counts')
ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_ylabel("total_counts")
ax.set_xlabel('samples')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()

#%%
# 2D embedding

sc.pp.pca(adata_spatial)
sc.pp.neighbors(adata_spatial, use_rep="X_scanorama")
sc.tl.umap(adata_spatial)
sc.tl.leiden(adata_spatial, key_added="clusters")

sc.pl.umap(adata_spatial, color=["clusters", "condition"])
sc.pl.umap(adata_spatial, color=["sample_id", "treatment"])

sc.tl.rank_genes_groups(adata_spatial, "clusters", method="wilcoxon")
sc.pl.rank_genes_groups_heatmap(adata_spatial, groups=['26'], n_genes=10, groupby="clusters")

sc.pl.umap(adata_spatial, color=["Rack1"])

sc.pp.calculate_qc_metrics(adata_spatial, inplace=True)
sc.pl.highest_expr_genes(adata_spatial, n_top=20)
