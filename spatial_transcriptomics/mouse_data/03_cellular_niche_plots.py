import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.functions import proportion_plot, matrixplot, spatial_plot
from spatial_transcriptomics.chrysalis_functions_update import plot_samples, plot_weights, plot, plot_matrix


#%%
# 8 samples
# read sample
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
comps = 13

# subset anndata
selected_samples = [ 17, 14, 15, 16, 19, 18, 8, 21]
adata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_10', '20221010-3_11', '20221010-3_12',
                '20221010-4_13', '20221010-4_14', '20221010-1_3', '20221010-4_16']
adata.obs['sample_id'] = adata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

plt.rcParams['svg.fonttype'] = 'none'
plot_samples(adata, 2, 4, comps, sample_col='label', seed=87, spot_size=1.9, rasterized=True,
             hspace=0.5, wspace=-0.5)
plt.savefig(f'figs/manuscript/fig2/chrysalis_8.svg')
plt.show()

#%%
# MIP compartments for all sample
# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)
plt.rcParams['svg.fonttype'] = 'none'

plot_samples(adata, 5, 5, comps, sample_col='label', seed=87, spot_size=1.9, rasterized=True,
             hspace=0.5, wspace=0.5)
plt.savefig(f'figs/manuscript/fig2/chrysalis_5by5.svg')
plt.show()

#%%
# Matrixplot showing top genes for each compartment
selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]
sf = 0.845
plt.rcParams['svg.fonttype'] = 'none'
plot_matrix(adata, comps=selected_comps, figsize=(6*sf, 12*sf), flip=False, scaling=True, square=False,
            colorbar_shrink=0.2, colorbar_aspect=10, dendrogram_ratio=0.05, cbar_label='Z-scored gene contribution',
            xlabel='Tissue compartment', ylabel='Top contributing genes per compartment', rasterized=True, seed=87)
# plt.savefig(f'figs/manuscript/fig2/chr_matrixplot.svg')
plt.show()

#%%
# 7 sample row per compartment
comps = 13

# read sample
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
# subset anndata
selected_samples = [ 17,  14,  15,  16,  19,  8,  21]
adata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_10', '20221010-3_11', '20221010-3_12',
                '20221010-4_14', '20221010-1_3', '20221010-4_16']
adata.obs['sample_id'] = adata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

for i in range(comps):
    plot_samples(adata, 1, 7, comps, sample_col='label', seed=87, spot_size=2, rasterized=True,
                 hspace=0.5, wspace=-0.7, selected_comp=i)
    plt.savefig(f'figs/manuscript/fig2/chrysalis_comps_{i}.svg')
    plt.show()

#%%
# Single sample for panel a
comps = 13
# read sample
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
# subset anndata
selected_samples = [20]
ad = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()
ch.plot(ad, comps, sample_col='sample_id', sample_id='20221010-4_15', seed=87, figsize=(10, 10), spot_size=1.2)
plt.savefig(f'figs/manuscript/fig2/chrysalis_demo.png')
plt.show()

#%%
# chrysalis spatial plots with alpha blending

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)
for c in compartment_df.columns:
    adata.obs[f'C{c}'] = compartment_df[c]

hexcodes = ch.utils.get_hexcodes(None, 13, 87, 45543)

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

# lowres props for crop
for c in [2, 5, 10]:
    print(c)
    pc_cmap = ch.utils.color_to_color('black', hexcodes[c])
    spatial_plot(adata, 1, 7, f'C{c}', cmap=pc_cmap, sample_col='label', s=12, share_range=True,
                 title=label_cats, suptitle=c, alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.4, subplot_size=4, alpha_blend=True,
                 x0=0.2, suptitle_fontsize=15, topspace=0.8, bottomspace=0.05, leftspace=0.1, rightspace=0.9,
                 k=50, facecolor='black')
    plt.savefig(f'figs/manuscript/fig2/chrysalis_he_7/{c}_lowres.svg')
    plt.show()

    spatial_plot(adata, 1, 7, f'C{c}', cmap=pc_cmap, sample_col='label', s=9, share_range=False,
                 title=label_cats, suptitle=c, alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.4, subplot_size=25, alpha_blend=True,
                 x0=0.2, suptitle_fontsize=15, topspace=0.8, bottomspace=0.05, leftspace=0.1, rightspace=0.9,
                 k=50)
    plt.savefig(f'figs/manuscript/fig2/chrysalis_he_7/{c}_hires.svg')
    plt.show()


#%%
# Proportion plots and barplots
comps = 13
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

hexcodes = ch.utils.generate_random_colors(num_colors=13, min_distance=1 / 13 * 0.5, seed=87,
                                  saturation=0.65, lightness=0.60)
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
plt.savefig(f'figs/manuscript/fig2/chr_compartment_props.svg')
plt.show()

# barplots
rows = 2
cols = 5
plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate([0, 1, 3, 4, 6, 7, 8, 9, 11, 12]):
    axs[idx].axis('on')
    sub_df = props_df[[c]].copy()
    sub_df['condition'] = sub_df.index.str.split('|').str[1].str.strip()
    sns.barplot(sub_df, y=c, x='condition', ax=axs[idx], color=hexcodes[c])
    sns.stripplot(sub_df, y=c, x='condition', ax=axs[idx], color='#4C4C4C')
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Proportion')
    axs[idx].set_title(f'Compartment {c}', fontsize=14)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
fig.supylabel(None)
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig2/chr_compartments_bar.svg')
plt.show()

# barplots main fig
rows = 3
cols = 1
plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows), sharex=True)
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate([5, 2, 10]):
    axs[idx].axis('on')
    sub_df = props_df[[c]].copy()
    sub_df['condition'] = sub_df.index.str.split('|').str[1].str.strip()
    sns.barplot(sub_df, y=c, x='condition', ax=axs[idx], color=hexcodes[c])
    sns.stripplot(sub_df, y=c, x='condition', ax=axs[idx], color='#4C4C4C')
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Proportion')
    axs[idx].set_title(f'Compartment {c}', fontsize=14)
    axs[idx].set_xlabel(None)
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
axs[-1].set_xticklabels(axs[-1].get_xticklabels(), rotation=90)
fig.supylabel(None)
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig2/chr_compartments_bar_3.svg')
plt.show()

#%%
# per spot
compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)
compartment_df['label'] = adata.obs['label']
compartment_df['condition'] = compartment_df['label'].str.split('|').str[1].str.strip()

fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(range(comps)):
    axs[idx].axis('on')
    sns.boxplot(compartment_df, y=c, x='condition', ax=axs[idx], color=hexcodes[c], showfliers=False)
    # sns.stripplot(compartment_df, y=c, x='condition', ax=axs[idx], color='black')
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

#%%
# weight plots
plt.rcParams['svg.fonttype'] = 'none'
ch.plot_weights(adata, compartments=[0, 1, 3, 6, 8, 9, 12], ncols=4, seed=87, w=0.8, h=0.9)
# plt.savefig(f'figs/manuscript/fig2/chr_weights.svg')
plt.show()

plt.rcParams['svg.fonttype'] = 'none'
plot_weights(adata, compartments=[2, 4, 5, 7, 10, 11], ncols=3, seed=87, w=0.6, h=0.4, top_genes=6)
plt.savefig(f'figs/manuscript/fig2/chr_weights_main_small.svg')
plt.show()

plt.rcParams['svg.fonttype'] = 'none'
plot_weights(adata, ncols=7, compartments=[0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12], seed=87, w=0.8, h=0.95, top_genes=20)
plt.savefig(f'figs/manuscript/fig2/chr_weights_main_suppl.svg')
plt.show()

# compartments - not in use
ch.plot_heatmap(adata, seed=87, reorder_comps=True, figsize=(5.5, 5), rasterized=True)
plt.savefig(f'figs/manuscript/fig2/chr_heatmap.svg')
plt.show()

#%%
# one sample animation
adata = adata[adata.obs['ch_sample_id'].isin([16])].copy()

frames = [
    [8, 0, 5],
    [8, 0, 5, 10],
    [8, 0, 5, 10, 2],
    [8, 0, 5, 10, 2, 4, 7, 1, 9],
    [8, 0, 5, 10, 2, 4, 7, 1, 9, 11, 12]
]

for i, x in enumerate(frames):
    plot(adata, dim=13, sample_id=16, spot_size=1.4, seed=87, selected_comp=x)
    plt.savefig(f'figs/compartment_animation/{i}.png')
    plt.show()

#%%
cell_type_dict = {
    'APC': 'Antigen-presenting cell',
    'Bcell': 'B cell',
    'CD4T': 'T cell CD4+',
    'CD8T': 'T cell CD8+',
    'CHMacrophage': 'Macrophage CH',
    'Endothelialcell': 'Endothelial cell',
    'Fibroblast': 'Fibroblast',
    'InflMacrophage': 'Macrophage Inf.',
    'NKcell': 'NK cell',
    'Plasmacell': 'Plasma cell',
    'Spp1Macrophage': 'Macrophage SPP1+',
    'TB-EMT': 'Tumor basal-EMT',
    'TBasal1': 'Tumor basal',
    'TBasal2': 'Tumor basal hypoxic',
    'TFlike': 'Tumor fibroblast-like',
    'TL-Alv': 'Tumor luminal-alveolar',
    'TLA-EMT': 'Tumor luminal-alveolar-EMT',
    'TMlike': 'Tumor macrophage-like',
    'TProliferating': 'Tumor proliferating',
    'Tcell' : 'T cell',
    'Treg': 'T cell regulatory',
    'cDC': 'Dendritic cell',
    'pDC': 'Dendritic cell plasmacytoid',
}

# cell type correlations with the compartments
compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)
celltypes_df = adata.obsm['cell2loc']
celltypes_df.columns = [cell_type_dict[x] for x in celltypes_df.columns]

corr_matrix = np.empty((len(celltypes_df.columns), len(compartment_df.columns)))
for i, col1 in enumerate(celltypes_df.columns):
    for j, col2 in enumerate(compartment_df.columns):
        corr, _ = pearsonr(celltypes_df[col1], compartment_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     # index=[cell_type_dict[x] for x in celltypes_df.columns],
                     index=celltypes_df.columns,
                     columns=compartment_df.columns).T

hexcodes = ch.utils.get_hexcodes(None, comps, 87, len(adata))

# heatmap
z = linkage(corrs, method='ward')
order = leaves_list(z)
corrs = corrs.iloc[order, :]
hexcodes = [hexcodes[i] for i in order]

z = linkage(corrs.T, method='ward')
order = leaves_list(z)
corrs = corrs.iloc[:, order]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(corrs.T, square=True, center=0,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True), ax=ax)
for idx, t in enumerate(ax.get_xticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
plt.title('Cell type correlation')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
plt.tight_layout()
# plt.savefig(f'figs/manuscript/fig2/chr_comp_cell_type_heatmap.svg')
plt.show()

selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]
sf = 0.845

plt.rcParams['svg.fonttype'] = 'none'
matrixplot(corrs, comps=selected_comps, figsize=(8.2*sf, 12*sf), flip=True, scaling=False, square=True,
            colorbar_shrink=0.17, colorbar_aspect=10, title='Cell type contributions to tissue compartments',
            dendrogram_ratio=0.05, cbar_label="Pearsons's r", xlabel='Cell types',
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Tissue compartment', rasterized=True, seed=87, linewidths=0.0, xrot=45,
            reorder_obs=True)
plt.savefig(f'figs/manuscript/fig2/chr_comp_cell_type_heatmap_v2.svg')
plt.show()


#%%
# cell type correlation per condition

names = {'primary_tumor no_treatment na': 'Primary tumor',
         'relapsed_tumor TAC_regimen 30_days' : 'TAC 30dpt',
         'relapsed_tumor cisplatin_6mg/kg 30_days': 'Cisplatin 30dpt',
         'residual_tumor TAC_regimen 12_days': 'TAC 12dpt',
         'residual_tumor TAC_regimen 7_days': 'TAC 7dpt',
         'residual_tumor cisplatin_6mg/kg 12_days': 'Cisplatin 12dpt',
         'residual_tumor cisplatin_6mg/kg 7_days': 'Cisplatin 7dpt'}

order =['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
        'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

adata.obs['condition_cat'] = [x + ' ' + y + ' ' + z for x, y, z in
                              zip(adata.obs['condition'],
                                  adata.obs['treatment'],
                                  adata.obs['elapsed_time'])]
adata.obs['condition_cat'] = adata.obs['condition_cat'].astype('category')
adata.obs['condition_cat'] = [names[x] for x in list(adata.obs['condition_cat'])]

cell_type_dict = {
    'APC': 'Antigen-presenting cell',
    'Bcell': 'B cell',
    'CD4T': 'T cell CD4+',
    'CD8T': 'T cell CD8+',
    'CHMacrophage': 'Macrophage CH',
    'Endothelialcell': 'Endothelial cell',
    'Fibroblast': 'Fibroblast',
    'InflMacrophage': 'Macrophage Inf.',
    'NKcell': 'NK cell',
    'Plasmacell': 'Plasma cell',
    'Spp1Macrophage': 'Macrophage SPP1+',
    'TB-EMT': 'Tumor basal-EMT',
    'TBasal1': 'Tumor basal',
    'TBasal2': 'Tumor basal hypoxic',
    'TFlike': 'Tumor fibroblast-like',
    'TL-Alv': 'Tumor luminal-alveolar',
    'TLA-EMT': 'Tumor luminal-alveolar-EMT',
    'TMlike': 'Tumor macrophage-like',
    'TProliferating': 'Tumor proliferating',
    'Tcell' : 'T cell',
    'Treg': 'T cell regulatory',
    'cDC': 'Dendritic cell',
    'pDC': 'Dendritic cell plasmacytoid',
}

# cell type correlations with the compartments
conditions = np.unique(adata.obs['condition_cat'])
scomp = 10
comp_dict = {}
for c in order:
    ad = adata[adata.obs['condition_cat'] == c]
    compartments = ad.obsm['chr_aa']
    compartment_df = pd.DataFrame(data=compartments, index=ad.obs.index)
    celltypes_df = ad.obsm['cell2loc']
    # celltypes_df.columns = [cell_type_dict[x] for x in celltypes_df.columns]

    corr_matrix = np.empty((len(celltypes_df.columns), len(compartment_df.columns)))
    for i, col1 in enumerate(celltypes_df.columns):
        for j, col2 in enumerate(compartment_df.columns):
            corr, _ = pearsonr(celltypes_df[col1], compartment_df[col2])
            corr_matrix[i, j] = corr

    corrs = pd.DataFrame(data=corr_matrix,
                         # index=[cell_type_dict[x] for x in celltypes_df.columns],
                         index=celltypes_df.columns,
                         columns=compartment_df.columns).T
    comp_dict[c] = corrs.iloc[scomp]

comp_corr = pd.DataFrame(comp_dict)

plt.rcParams['svg.fonttype'] = 'none'
matrixplot(comp_corr, figsize=(8.2*sf, 12*sf), flip=True, scaling=False, square=True,
            colorbar_shrink=0.17, colorbar_aspect=10, title='Cell type correlation',
            dendrogram_ratio=0.05, cbar_label="Pearsons's r", xlabel='Cell types',
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Tissue compartment', rasterized=True, seed=87, reorder_obs=False,
            color_comps=False)
# plt.savefig(f'figs/manuscript/fig2/chr_comp_cell_type_heatmap_v2.svg')
plt.show()

#%%
# chrysalis dendrogram
genes_df = pd.DataFrame(data=adata.uns['chr_aa']['loadings'], columns=adata.uns['chr_pca']['features'])
hexcodes = ch.utils.get_hexcodes(None, comps, 87, len(adata))
# drop 3 and 6
drop_comps = [3, 6]
genes_df = genes_df.drop(drop_comps, axis=0)
hexcodes = [item for i, item in enumerate(hexcodes) if i not in drop_comps]
labels = list(genes_df.index)

z = linkage(genes_df, method='ward')
order = leaves_list(z)
genes_df = genes_df.iloc[order, :]
hexcodes = [hexcodes[i] for i in order]

hierarchy.set_link_color_palette(['#ed5ed1', '#ed5e5e'])

plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
dn = hierarchy.dendrogram(z, orientation='left', labels=labels, above_threshold_color='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticklabels('')
ax.xaxis.set_ticks([])
ax.set_yticklabels(ax.get_yticklabels())
for idx, t in enumerate(ax.get_yticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
plt.savefig(f'figs/manuscript/fig2/dendrogram.svg')
plt.show()
