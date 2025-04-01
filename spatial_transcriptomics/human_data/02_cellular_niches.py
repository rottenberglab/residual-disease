import os
import scanorama
import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
import seaborn as sns
import chrysalis as ch
import matplotlib.pyplot as plt
from chrysalis.utils import estimate_compartments
from spatial_transcriptomics.functions import proportion_plot
from spatial_transcriptomics.human_data.functions_human import show_annotations
from spatial_transcriptomics.chrysalis_functions_update import plot_samples, plot_heatmap, plot_weights, plot_rss


output_folder = 'data/human_samples'

# Read samples
h5ads =  glob(f'{output_folder}/*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]

# Chrysalis detects SVGs
show = False
if show:
    for ad in adatas:
        show_annotations(ad)
        plt.show()

for idx, ad in enumerate(adatas):
    adatas[idx] = ad[(ad.obs['annotations'] != 'Adipose tissue') &
                     (ad.obs['annotations'] != 'Skeletal muscle') &
                     (ad.obs['annotations'] != 'Large vein')]

for ad in adatas:
    ch.detect_svgs(ad, min_morans=0.025, min_spots=0.05)
    ch.plot_svgs(ad)
    plt.show()
    fname = ad.obs["sample_id"][0]
    ad.write_h5ad(f'{output_folder}/ch_{fname}.h5ad')

#%%
h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]

# Run chrysalis
sample_names = [x.obs['sample_id'][0] for x in adatas]
ch.plot_svg_matrix(adatas, figsize=(5, 4), obs_name='sample_id', cluster=True)
plt.show()
adata = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')

#%%
#scanorama integration
h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]

# scanorama seems unstable with the raw data included, add it back after
for ad in adatas:
    ad.raw = None
    try:
        ad.obs.drop(columns=['ch_sample_id'], inplace=True)
    except KeyError:
        pass

scr_adata = scanorama.correct_scanpy(adatas, return_dimred=True)
scr_adata = ch.integrate_adatas(scr_adata, sample_names=sample_names, sample_col='ch_sample_id')

scr_adata.write_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

#%%
adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

# replace ENSEMBL IDs with the gene symbols and make them unique
adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()

# run PCA
ch.pca(adata, n_pcs=50)
ch.plot_explained_variance(adata)
plt.show()

estimate_compartments(adata, range_archetypes=(6, 30), max_iter=10)
plot_rss(adata)

ch.aa(adata, n_pcs=20, n_archetypes=20)
ch.plot_samples(adata, rows=2, cols=3, dim=20, suptitle='Scanorama', sample_col='ch_sample_id', show_title=True)
plt.show()
ch.plot_heatmap(adata, reorder_comps=True, figsize=(6, 7))
plt.show()

# plot individual compartments
os.makedirs(f'{output_folder}/compartments/', exist_ok=True)
for x in range(20):
    ch.plot_samples(adata, rows=2, cols=3, dim=20, suptitle=f'Compartment {x}', sample_col='ch_sample_id',
                    show_title=True, seed=8, selected_comp=x)
    plt.savefig(f'{output_folder}/compartments/{x}.png')
    plt.close()

# save the adata
adata.write_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

#%%
# save final plots
adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

condition = {'primary_tumor': 'Primary tumor', 'residual_tumor': 'Residual tumor'}
# reorder sample_id for plotting
sample_order = ['T23_47', 'T23_48', 'T1212', 'T23_41', 'T23_42']
adata.obs['sample_id'] = adata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'condition']].drop_duplicates()
sample_name_df = sample_name_df.sort_values(by='sample_id', key=lambda x: x.map(lambda val: sample_order.index(val)))
sample_name_df['condition'] = sample_name_df['condition'].map(lambda x: condition[x])
sample_name_df.index = sample_name_df['sample_id']
sample_name_df['label'] = [x + ' | ' + y for x, y in
                           zip(sample_name_df['sample_id'].astype(str), sample_name_df['condition'])]
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

adata.obs['label'] = [x + ' | ' + condition[y] for x, y in
                      zip(adata.obs['sample_id'].astype(str), adata.obs['condition'])]
adata.obs['label'] = adata.obs['label'].astype('category')
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(list(sample_name_df.values()), ordered=True)

# add spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

plt.rcParams['svg.fonttype'] = 'none'
plot_samples(adata, rows=1, cols=5, dim=20, sample_col='label', show_title=True,
                seed=8, hspace=0.5, wspace=-0.5, spot_size=2, rasterized=True)
plt.savefig(f'figs/manuscript/fig4/chr_plot.svg')
plt.show()

#%%

plot_heatmap(adata, reorder_comps=True, figsize=(8, 5), seed=8, rasterized=True)
plt.savefig(f'figs/manuscript/fig4/chr_heatmap.svg')
plt.show()

selected_comps = [
    6, 10, 18,  # primary
    1, 8, 11, 16  # residual
]

plt.rcParams['svg.fonttype'] = 'none'
ch.plot_weights(adata, compartments=selected_comps, ncols=7, seed=8, w=0.8, h=0.9)
plt.savefig(f'figs/manuscript/fig4/chr_weights_main.svg')
plt.show()

rest_comps = [x for x in range(20) if x not in selected_comps]
plt.rcParams['svg.fonttype'] = 'none'
plot_weights(adata, ncols=7, seed=8, w=0.8, h=0.95)
plt.savefig(f'figs/manuscript/fig4/chr_weights.svg')
plt.show()

ch.plot_weights(adata, ncols=5, seed=8)
plt.savefig(f'{output_folder}/chr_weights.png')
plt.show()

# %%
# add raw values and save

adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]

adata_raw = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')
adata_raw.var_names = list(adata_raw.var['gene_symbols'])
adata_raw.var_names_make_unique()
var_dict = {k:v for k,v in zip(adata.var['gene_ids'], adata.var_names)}
adata_raw = adata_raw.raw.to_adata()
adata_raw.var_names = [var_dict[x] for x in adata_raw.var_names]

adata.raw = adata_raw

adata.write_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

#%%
# proportion plots
comps = 20
labels = []
spot_nr = []
prop_matrix = np.zeros((len(np.unique(adata.obs['sample'])), comps))
for idx, i in enumerate(np.unique(adata.obs['sample'])):
    ad = adata[adata.obs['sample'] == i]
    spot_nr.append(len(ad))
    compartments = ad.obsm['chr_aa']
    compartments_mean = compartments.sum(axis=0)
    compartments_prop = compartments_mean / np.sum(compartments_mean)
    prop_matrix[i] = compartments_prop
    label = adata.obs[adata.obs['sample'] == i]['label'][0]
    labels.append(label)

props_df = pd.DataFrame(data=prop_matrix,
                           index=labels)
spot_nr = pd.Series(data=spot_nr, index=labels, name='spot_nr')

# Define the custom order
custom_order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
                'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

props_df['order'] = props_df.index.str.split('|').str[1].str.strip()
props_df['sample'] = props_df.index
props_df['order'] = props_df['order'].astype('category')
props_df['order'] = props_df['order'].cat.reorder_categories(custom_order, ordered=True)
props_df = props_df.sort_values(by=['order', 'sample'])
props_df = props_df.drop(columns=['order', 'sample'])

spot_nr = pd.DataFrame(spot_nr)
spot_nr['order'] = spot_nr.index.str.split('|').str[1].str.strip()
spot_nr['sample'] = spot_nr.index
spot_nr['order'] = spot_nr['order'].astype('category')
spot_nr['order'] = spot_nr['order'].cat.reorder_categories(custom_order, ordered=True)
spot_nr = spot_nr.sort_values(by=['order', 'sample'])
spot_nr = spot_nr.drop(columns=['order', 'sample'])

hexcodes = ch.utils.generate_random_colors(num_colors=13, min_distance=1 / 13 * 0.5, seed=87,
                                  saturation=0.65, lightness=0.60)
cmap = sns.color_palette(hexcodes, 13)

proportion_plot(props_df[::-1], spot_nr['spot_nr'][::-1], palette=hexcodes)
plt.tight_layout()
plt.show()

# boxplots
for c in props_df.columns:
    sub_df = props_df[[c]].copy()
    sub_df['condition'] = sub_df.index.str.split('|').str[1].str.strip()
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5))
    sns.boxplot(sub_df, y=c, x='condition', ax=axs, color=hexcodes[c])
    # axs.set_ylim(0, 0.5)
    axs.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    axs.set_axisbelow(True)
    axs.set_ylabel('Fraction')
    axs.set_title(f'Compartment {c}', fontsize=14)
    axs.set_xlabel(None)
    axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(f'figs/compartments/boxplots/{c}.png')
    plt.close()

rows = 3
cols = 5

fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(props_df.columns):
    axs[idx].axis('on')
    sub_df = props_df[[c]].copy()
    sub_df['condition'] = sub_df.index.str.split('|').str[1].str.strip()
    sns.boxplot(sub_df, y=c, x='condition', ax=axs[idx], color=hexcodes[c])
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Proportion')
    axs[idx].set_title(f'Compartment {c}', fontsize=14)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
fig.supylabel(None)
plt.tight_layout()
plt.show()

# get the relative change
normalized_prop_df = props_df.multiply(spot_nr['spot_nr'], axis=0)

fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(normalized_prop_df.columns):
    axs[idx].axis('on')
    sub_df = normalized_prop_df[[c]].copy()
    sub_df['condition'] = sub_df.index.str.split('|').str[1].str.strip()
    sns.boxplot(sub_df, y=c, x='condition', ax=axs[idx], color=hexcodes[c])
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Compartment coverage')
    axs[idx].set_title(f'Compartment {c}', fontsize=14)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
fig.supylabel(None)
plt.tight_layout()
plt.show()
