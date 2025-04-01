import ast
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from spatial_transcriptomics.human_data.functions_human import (segment_spatial_image, remove_spots_not_under_tissue,
                                                                show_tissue_image, get_annotations, map_annotations,
                                                                show_annotations, swap_gene_ids)


root_folder = '/mnt/c/Bern/Visium_processed/'
samples = glob(root_folder + 'human*/runs/*/')
output_folder = 'data/human_samples'

meta_df = pd.read_csv('human_metadata.csv', index_col=0)
qupath_project = 'data/annotations/project.qpproj'
remove_spots = False  # remove spots outside the segmentation mask
metadata_columns = ['sample_id', 'batch', 'slide', 'condition', 'slide_name']  # columns to save to .obs from metadata

for idx, row in meta_df.iterrows():
    ad = sc.read_visium(row['sample_path'], count_file='raw_feature_bc_matrix.h5')
    ad = ad[ad.obs['in_tissue'] == 1]
    gene_names = [False if 'DEPRECATED_' in x else True for x in ad.var['gene_ids']]
    ad = ad[:, gene_names]
    swap_gene_ids(ad)  # swap gene symbols with gene IDs

    # add metadata as .obs columns
    assert set(metadata_columns).issubset(set(meta_df.columns))
    for c in metadata_columns:
        ad.obs[c] = row[c]

    # 1. segment the image and show the result
    show_tissue_image(ad, show_spots=False)
    plt.show()
    segment_spatial_image(ad, l=50, h=1)
    show_tissue_image(ad, show_spots=False)
    plt.show()
    if remove_spots:
        ad = remove_spots_not_under_tissue(ad)
        show_tissue_image(ad, show_spots=True)
        plt.show()

    # 2. map annotations to spots
    bbox = ast.literal_eval(row['bounding_box'])
    bbox = np.array(bbox)
    polys = get_annotations(bbox, row['slide_name'], qupath_project, rotate=row['rotate'], show=True)
    map_annotations(ad, polys, default_annot='Stroma')

    show_annotations(ad)
    plt.show()

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

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
adata_spatial = anndata.concat(adatas, index_unique='-', uns_merge='unique', merge='unique')
adata_spatial.obs['annotations'] = adata_spatial.obs['annotations'].astype('category')
# adata_spatial.write_h5ad('data/mouse_tumors.h5ad')

#%%
# annotation plots
rows = 2
cols = 3

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
# H&E plot

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}

treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}

time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}


cats = ['Tumor', 'Stroma', 'Tumor in situ', 'Infiltrative tumor', 'Transition?', 'Epithelial cells', 'Other']
for ad in adatas:
    ad.obs['annotations'] = ad.obs['annotations'].cat.set_categories(cats, ordered=True)
    ad.obs['annotations'][ad.obs['annotations']=='Solid tumor mass'] = 'Tumor'
    ad.obs['annotations'][ad.obs['annotations']=='Infiltrative/infiltrated tumor mass'] = 'Infiltrative tumor'
    ad.obs['annotations'] = ad.obs['annotations'].cat.reorder_categories(cats, ordered=True)

adata_spatial.obs['annotations'] = adata_spatial.obs['annotations'].cat.set_categories(cats, ordered=True)
adata_spatial.obs['annotations'][adata_spatial.obs['annotations'] == 'Solid tumor mass'] = 'Tumor'
adata_spatial.obs['annotations'][adata_spatial.obs['annotations'] == 'Infiltrative/infiltrated tumor mass'] \
    = 'Infiltrative tumor'

rows = 2
cols = 3

# adding annotation colors
hexes = ['#9d5', '#4d8', '#2cb', '#0bc', '#e69100', '#36b', '#a679d2']
hexes = hexes[::-1]
custom_colors_rgb = [mcolors.hex2color(color) for color in hexes]
palette = sns.color_palette(custom_colors_rgb)

plt.rcParams['svg.fonttype'] = 'none'
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)

fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
ax = ax.flatten()
for a in ax:
    a.axis('off')
for idx, s in enumerate(['T23_47', 'T23_48', 'T1212', 'T23_41', 'T23_42']):
    if idx >= 2:
        idx += 1
    ad = adata_spatial[adata_spatial.obs['sample_id'] == s].copy()
    ad.obs['annotations'] = ad.obs['annotations'].cat.set_categories(cats, ordered=True)
    sc.pl.spatial(ad, color='annotations', size=1.5, alpha=1, library_id=s,
                  ax=ax[idx], show=False, palette=palette,
                  legend_loc=None
                  )
    ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig4/human_annotations.svg')
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
plt.savefig(f'figs/manuscript/fig4/human_annotations_annots_labels.svg')
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
