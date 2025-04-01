import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from spatial_transcriptomics.functions import spatial_plot


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

celltype_dict = {
    'Tumor': ['TB-EMT', 'TBasal1', 'TBasal2', 'TFlike', 'TL-Alv', 'TLA-EMT', 'TMlike', 'TProliferating'],
    'T cell': ['CD4T', 'CD8T', 'Tcell', 'Treg'],
    'NK cell': ['NKcell'],
    'Dendritic cell': ['cDC', 'pDC', 'APC'],
    'Fibroblast': ['Fibroblast'],
    'B cell': ['Bcell', 'Plasmacell'],
    'Macrophage': ['CHMacrophage', 'Spp1Macrophage', 'InflMacrophage'],
    'Endothelial cell': ['Endothelialcell'],
}

#%%
# read sample
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

#%%
# All cell types with alpha blending

celltypes_df = adata.obsm['cell2loc']

celltypes_props = celltypes_df.values / celltypes_df.sum(axis=1).values.reshape(-1, 1)
celltypes_props_df = pd.DataFrame(celltypes_props, index=celltypes_df.index, columns=celltypes_df.columns)

props_df = pd.DataFrame()

celltypes_df = adata.obsm['cell2loc']
ct_props = celltypes_df.values / celltypes_df.sum(axis=1).values.reshape(-1, 1)
ct_props = pd.DataFrame(data=ct_props, index=celltypes_df.index, columns=celltypes_df.columns)
sample_props = {}
for k, v in celltype_dict.items():
    sample_props[k] = ct_props[v].sum(axis=1)
sample_prop_df = pd.DataFrame(data=sample_props)
sample_prop_df.to_csv('data/cell_type_props_lowres.csv')

for c in celltypes_props_df.columns:
    adata.obs[f'{c}_props'] = celltypes_props_df[c]

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

# lowres props for crop
for c in celltypes_df.columns:
    adata.obs[c] = celltypes_df[c]
    spatial_plot(adata, 1, 7, f'{c}_props', cmap='mako', sample_col='label', s=12,
                 title=label_cats, suptitle=cell_type_dict[c], alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.4, subplot_size=4, alpha_blend=True,
                 x0=0.3, suptitle_fontsize=15, topspace=0.8, bottomspace=0.05, leftspace=0.1, rightspace=0.9)
    plt.savefig(f'figs/manuscript/fig1/ctprops_lowres_7/{c}_props.svg')
    plt.show()

# hires props for crop
for c in celltypes_df.columns:
    adata.obs[c] = celltypes_df[c]
    spatial_plot(adata, 1, 7, f'{c}_props', cmap='mako', sample_col='label', s=9,
                 title=label_cats, suptitle=cell_type_dict[c], alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.4, subplot_size=15, alpha_blend=True,
                 x0=0.3, suptitle_fontsize=15, topspace=0.8, bottomspace=0.05, leftspace=0.1, rightspace=0.9)
    plt.savefig(f'figs/manuscript/fig1/ctprops_hires_7/{c}_props.svg')
    plt.show()
    plt.clf()
    plt.close()

    spatial_plot(adata, 1, 7, f'TProliferating_props', cmap='mako', sample_col='label', s=9,
                 title=label_cats, suptitle=cell_type_dict['TProliferating'], alpha_img=1,
                 colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.4, subplot_size=4, alpha_blend=True,
                 x0=1, suptitle_fontsize=15, topspace=0.8, bottomspace=0.05, leftspace=0.1, rightspace=0.9)
    plt.savefig(f'figs/manuscript/fig1/ctprops_lowres_7/he.svg')
    plt.show()

#%%
# Spatial plots with only three representative samples
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

# subset anndata
selected_samples = [17,  15,  16]
adata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_11', '20221010-3_12']
comps = 13
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

# cell types

celltypes_df = adata.obsm['cell2loc']

label_cats = ['Primary tumor | 20221010-3_9',
              'Residual tumor | 20221010-3_11',
              'Recurrent tumor | 20221010-3_12']

# 3 samples cell types crop
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

spatial_plot(adata, 1, 3, 'total_counts', cmap='rocket', sample_col='label', s=0,
             title=label_cats, suptitle=None, alpha_img=1, colorbar_label='Cell density',
             colorbar_aspect=5, colorbar_shrink=0.5, wspace=-0.2, subplot_size=10)
plt.savefig(f'figs/manuscript/fig1/3_histology_new.svg')
plt.show()

for c in celltypes_df.columns:
    adata.obs[c] = celltypes_df[c]
    spatial_plot(adata, 1, 3, c, cmap='rocket', sample_col='label', s=14,
                 title=label_cats, suptitle=cell_type_dict[c], alpha_img=0.5, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.5, wspace=-0.7)
    plt.show()

# hires props for crop
for c in celltypes_df.columns:
    adata.obs[c] = celltypes_df[c]
    spatial_plot(adata, 1, 3, f'{c}_props', cmap='mako', sample_col='label', s=9,
                 title=label_cats, suptitle=cell_type_dict[c], alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.5, wspace=-0.2, subplot_size=15, alpha_blend=True,
                 x0=0.3)
    plt.savefig(f'figs/manuscript/fig1/ctprops_hires/{c}_props.svg')
    plt.show()

# lowres props for crop
for c in celltypes_df.columns:
    adata.obs[c] = celltypes_df[c]
    spatial_plot(adata, 1, 3, f'{c}_props', cmap='mako', sample_col='label', s=9,
                 title=label_cats, suptitle=cell_type_dict[c], alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.2, subplot_size=4, alpha_blend=True,
                 x0=0.3, suptitle_fontsize=15)
    plt.savefig(f'figs/manuscript/fig1/ctprops_lowres/{c}_props.svg')
    plt.show()

# cell2loc porportions - 3 samples only
celltypes_df = adata.obsm['cell2loc']

ct_props = celltypes_df.values / celltypes_df.sum(axis=1).values.reshape(-1, 1)
ct_props = pd.DataFrame(data=ct_props, index=celltypes_df.index, columns=celltypes_df.columns)

tumor_names = ['TB-EMT', 'TBasal1', 'TBasal2', 'TFlike', 'TL-Alv', 'TLA-EMT', 'TMlike', 'TProliferating']
tumor_props = ct_props[tumor_names].sum(axis=1)
adata.obs['Tumor proportion'] = tumor_props

sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

spatial_plot(adata, 1, 3, 'Tumor proportion', cmap='mako', sample_col='label', s=14,
             title=label_cats, suptitle='Tumor proportion', alpha_img=0.5)
# plt.show()
plt.savefig(f'figs/manuscript/fig1/3_tumor_cell_props.svg')
plt.close()

# do it for the others
for k, v in celltype_dict.items():
    props = ct_props[v].sum(axis=1)
    adata.obs[f'{k} proportion'] = props

    sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
    plt.rcParams['svg.fonttype'] = 'none'

    spatial_plot(adata, 1, 3, f'{k} proportion', cmap='mako', sample_col='label', s=16,
                 title=label_cats, suptitle=f'{k} proportion', alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.9, subplot_size=4, alpha_blend=False,
                 x0=0.3, suptitle_fontsize=15)

    plt.savefig(f'figs/manuscript/fig1/lowres_props/3_{k}_cell_props.svg')
    plt.show()
    plt.close()

#%%
# cell2loc porportions - 7 samples
celltypes_df = adata.obsm['cell2loc']

ct_props = celltypes_df.values / celltypes_df.sum(axis=1).values.reshape(-1, 1)
ct_props = pd.DataFrame(data=ct_props, index=celltypes_df.index, columns=celltypes_df.columns)

tumor_names = ['TB-EMT', 'TBasal1', 'TBasal2', 'TFlike', 'TL-Alv', 'TLA-EMT', 'TMlike', 'TProliferating']
tumor_props = ct_props[tumor_names].sum(axis=1)
adata.obs['Tumor proportion'] = tumor_props

sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

spatial_plot(adata, 1, 7, 'Tumor proportion', cmap='mako', sample_col='label', s=14,
             title=label_cats, suptitle='Tumor proportion', alpha_img=0.5)
# plt.show()
plt.savefig(f'figs/manuscript/fig1/3_tumor_cell_props.svg')
plt.close()

# do it for the others

celltype_dict = {
    'Tumor': ['TB-EMT', 'TBasal1', 'TBasal2', 'TFlike', 'TL-Alv', 'TLA-EMT', 'TMlike', 'TProliferating'],
    'T cell': ['CD4T', 'CD8T', 'Tcell', 'Treg'],
    'NK cell': ['NKcell'],
    'Dendritic cell': ['cDC', 'pDC', 'APC'],
    'Fibroblast': ['Fibroblast'],
    'B cell': ['Bcell', 'Plasmacell'],
    'Macrophage': ['CHMacrophage', 'Spp1Macrophage', 'InflMacrophage'],
    'Endothelial cell': ['Endothelialcell'],
}

short_title = [x.split(' | ')[1] for x in label_cats]
for k, v in celltype_dict.items():
    props = ct_props[v].sum(axis=1)
    adata.obs[f'{k} proportion'] = props

    sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
    plt.rcParams['svg.fonttype'] = 'none'

    spatial_plot(adata, 1, 7, f'{k} proportion', cmap='mako', sample_col='label', s=12,
                 title=short_title, suptitle=f'{k} proportion', alpha_img=0.8, colorbar_label='Cell density',
                 colorbar_aspect=5, colorbar_shrink=0.25, subplot_size=4, alpha_blend=False,
                 x0=0.3, suptitle_fontsize=15, wscale=0.8)

    plt.tight_layout()
    plt.savefig(f'figs/manuscript/fig1/lowres_props/7_{k}_cell_props_mako.svg')
    plt.show()
    plt.close()

#%%
# proportion barplots - major cell types per sample

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

props_df = pd.DataFrame()
for sample in np.unique(adata.obs['sample_id']):
    ad = adata[adata.obs['sample_id'] == sample]
    celltypes_df = ad.obsm['cell2loc']
    ct_props = celltypes_df.values / celltypes_df.sum(axis=1).values.reshape(-1, 1)
    ct_props = pd.DataFrame(data=ct_props, index=celltypes_df.index, columns=celltypes_df.columns)
    ct_props_lowres = pd.DataFrame(index=celltypes_df.index, columns=celltypes_df.columns)
    sample_props = {}
    for k, v in celltype_dict.items():
        sample_props[k] = ct_props[v].sum(axis=1).sum()
    sample_prop_df = pd.DataFrame(data=sample_props, index=[sample])
    sample_prop_df = sample_prop_df.divide(sample_prop_df.sum(axis=1).sum())
    metadata = ['sample_id', 'batch', 'slide', 'condition', 'treatment', 'elapsed_time', 'label']
    sample_prop_df[metadata] = ad.obs.iloc[0, :][metadata]
    sample_prop_df['short_id'] = sample_prop_df['label'][0].split(' | ')[-1]
    props_df = pd.concat([props_df, sample_prop_df], axis=0)

condition_map = {'primary_tumor': 'Primary',
                 'relapsed_tumor': 'Recurrent',
                 'residual_tumor': 'Residual'}

props_df['condition'] = props_df['condition'].map(lambda x: condition_map[x])

plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(1, 8, figsize=(8 * 2, 1 * 2.5))
ax = ax.flatten()
for idx, ct in enumerate(celltype_dict.keys()):
    sns.barplot(props_df, y=ct, x='condition', ax=ax[idx], color='#57CBAD')
    sns.stripplot(props_df, y=ct, x='condition', ax=ax[idx], size=4, color=".3")
    ax[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    ax[idx].set_axisbelow(True)
    ax[idx].set_title(ct)
    ax[idx].set_ylabel('Proportion')
    ax[idx].set_xlabel(None)
    ax[idx].spines['top'].set_visible(False)
    ax[idx].spines['right'].set_visible(False)
    ax[idx].set_xticklabels(ax[idx].get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig1/cell_props_bar2.svg')
plt.show()

for idx, ct in enumerate(celltype_dict.keys()):

    fig, ax = plt.subplots(1, 1, figsize=(1.6, 2.5))
    sns.barplot(props_df, y=ct, x='condition', ax=ax, color='#57CBAD')
    sns.stripplot(props_df, y=ct, x='condition', ax=ax, size=4, color=".3")
    ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_title(ct)
    ax.set_ylabel('Fraction')
    ax.set_xlabel(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(f'figs/manuscript/fig1/cell_type_bar/cell_props_bar_{ct}.svg')
    plt.show()

# tumour types

celltype_dict = {
    'Tumor\nproliferating': ['TProliferating'],
    'Tumor\nluminal-alveolar': ['TL-Alv'],
    'Tumor\nluminal-alveolar-EMT': ['TLA-EMT'],
    'Tumor\nbasal': ['TBasal1'],
    'Tumor\nbasal hypoxic': ['TBasal2'],
    'Tumor\nbasal-EMT': ['TB-EMT'],
    'Tumor\nfibroblast-like': ['TFlike'],
    'Tumor\nmacrophage-like': ['TMlike'],

}

props_df = pd.DataFrame()
for sample in np.unique(adata.obs['sample_id']):
    ad = adata[adata.obs['sample_id'] == sample]
    celltypes_df = ad.obsm['cell2loc']
    ct_props = celltypes_df.values / celltypes_df.sum(axis=1).values.reshape(-1, 1)
    ct_props = pd.DataFrame(data=ct_props, index=celltypes_df.index, columns=celltypes_df.columns)
    ct_props_lowres = pd.DataFrame(index=celltypes_df.index, columns=celltypes_df.columns)
    sample_props = {}
    for k, v in celltype_dict.items():
        sample_props[k] = ct_props[v].sum(axis=1).sum()
    sample_prop_df = pd.DataFrame(data=sample_props, index=[sample])
    sample_prop_df = sample_prop_df.divide(sample_prop_df.sum(axis=1).sum())
    metadata = ['sample_id', 'batch', 'slide', 'condition', 'treatment', 'elapsed_time', 'label']
    sample_prop_df[metadata] = ad.obs.iloc[0, :][metadata]
    sample_prop_df['short_id'] = sample_prop_df['label'][0].split(' | ')[-1]
    props_df = pd.concat([props_df, sample_prop_df], axis=0)

condition_map = {'primary_tumor': 'Primary',
                 'relapsed_tumor': 'Recurrent',
                 'residual_tumor': 'Residual'}

props_df['condition'] = props_df['condition'].map(lambda x: condition_map[x])

plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(1, 8, figsize=(8 * 2, 1 * 2.5))
ax = ax.flatten()
for idx, ct in enumerate(celltype_dict.keys()):
    sns.barplot(props_df, y=ct, x='condition', ax=ax[idx], color='#57CBAD')
    sns.stripplot(props_df, y=ct, x='condition', ax=ax[idx], size=4, color=".3")
    ax[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    ax[idx].set_axisbelow(True)
    ax[idx].set_title(ct)
    ax[idx].set_ylabel('Proportion')
    ax[idx].set_xlabel(None)
    ax[idx].spines['top'].set_visible(False)
    ax[idx].spines['right'].set_visible(False)
    ax[idx].set_xticklabels(ax[idx].get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig1/cell_props_bar_tumours.svg')
plt.show()
