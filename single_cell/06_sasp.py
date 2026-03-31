import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import linregress


cell_type_dict = {
    'APC': 'Antigen-presenting cell', 'Bcell': 'B cell', 'CD4T': 'T cell CD4+',
    'CD8T': 'T cell CD8+', 'CHMacrophage': 'Macrophage CH', 'Endothelialcell': 'Endothelial cell',
    'Fibroblast': 'Fibroblast', 'InflMacrophage': 'Macrophage Inf.', 'NKcell': 'NK cell',
    'Plasmacell': 'Plasma cell', 'Spp1Macrophage': 'Macrophage SPP1+', 'TB-EMT': 'Tumor basal-EMT',
    'TBasal1': 'Tumor basal', 'TBasal2': 'Tumor basal hypoxic', 'TFlike': 'Tumor fibroblast-like',
    'TL-Alv': 'Tumor luminal-alveolar', 'TLA-EMT': 'Tumor luminal-alveolar-EMT',
    'TMlike': 'Tumor macrophage-like', 'TProliferating': 'Tumor proliferating', 'Tcell': 'T cell',
    'Treg': 'T cell regulatory', 'cDC': 'Dendritic cell', 'pDC': 'Dendritic cell plasmacytoid'
}

label_map = {
    'Tumor proliferating': '1', 'Tumor luminal-alveolar': '2', 'Tumor luminal-alveolar-EMT': '3',
    'Tumor basal': '4', 'Tumor basal hypoxic': '5', 'Tumor basal-EMT': '6', 'Tumor fibroblast-like': '7',
    'Tumor macrophage-like': '8', 'Macrophage CH': '9', 'Macrophage Inf.': '10', 'Macrophage SPP1+': '11',
    'NK cell': '12', 'Antigen-presenting cell': '13', 'Dendritic cell': '14', 'Dendritic cell plasmacytoid': '15',
    'B cell': '16', 'Plasma cell': '17', 'T cell': '18', 'T cell CD4+': '19', 'T cell CD8+': '20',
    'T cell regulatory': '21', 'Fibroblast': '22', 'Endothelial cell': '23'
}

color_map = {
    '1': '#19a1e6', '2': '#5ebeed', '3': '#5e5eed', '4': '#5aedbc', '5': '#6cda88', '6': '#3cdd51',
    '7': '#178280', '8': '#172982', '9': '#ed5e5e', '10': '#ed825e', '11': '#edb25e', '12': '#ed5e9c',
    '13': '#f28cb8', '14': '#e699c4', '15': '#d999e6', '16': '#7b53f3', '17': '#af75f5', '18': '#a947eb',
    '19': '#c788f2', '20': '#976ef7', '21': '#cbb6fb', '22': '#7fdd3c', '23': '#c32222'
}

#%%
# Main t-SNE plot
adata = sc.read_h5ad('data/sc_dataset.h5ad')

adata.obs['hires_map'] = [label_map[x] for x in adata.obs['cell_type_hires']]

#%%
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
sc.pl.tsne(adata, color=['hires_map'], legend_loc='on data', legend_fontoutline=1,
           palette=color_map, frameon=False, ax=ax, legend_fontsize=10,
           s=40, alpha=1, show=False)
plt.show()

#%%
# SASP
orthologs_df = pd.read_csv('data/resources/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

marker_df = pd.read_csv('data/resources/sasp_mayo.csv')
mouse_orths = []
for v in marker_df['Gene(human)'].values:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)
marker_df['genes_mouse'] = mouse_orths
gene_list = marker_df['genes_mouse']

#%%
gene_list_sc = adata.var_names[np.in1d(adata.var_names, gene_list)]
sc.tl.score_genes(adata, gene_list=gene_list_sc, score_name=f'SASP')

#%%
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.set_figure_params(vector_friendly=True)
sc.pl.tsne(adata, color='SASP', frameon=False, ax=ax,
           s=40, alpha=1, show=False, vmax=10)
plt.show()

sasp_df = adata.obs[['SASP', 'cell_type_lowres']].copy()

fig, ax = plt.subplots(1, 1, figsize=(4, 5))
sns.boxplot(sasp_df, y='SASP', x='cell_type_lowres', ax=ax, showfliers=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("Cell type")
plt.tight_layout()
plt.show()

#%%
# check the spatial samples for sasp

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
adata.obs_names = [f"{a}-1-{b}" for a, b in zip([x.split('-1-')[0] for x in adata.obs_names], adata.obs['sample_id'])]
adata.obs.index = adata.obs_names

gene_list_st = adata.var_names[np.in1d(adata.var_names, gene_list)]

sc.tl.score_genes(adata, gene_list=gene_list_st, score_name=f'SASP')

sasp_spatial_df = adata.obs[['SASP', 'ancestral_tumor', 'condition']].copy()

fig, ax = plt.subplots(1, 1, figsize=(4, 5))
sns.boxplot(sasp_spatial_df, y='SASP', x='condition', ax=ax, showfliers=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("Cell type")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(6, 5), gridspec_kw={'width_ratios': [3, 1]})
sns.boxplot(sasp_df, y='SASP', x='cell_type_lowres', ax=axs[0], showfliers=False, color='lightgray')
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
axs[0].set_xlabel("Cell type")
axs[0].set_ylabel("SASP score")
axs[0].set_title("Single cell")

sns.boxplot(sasp_spatial_df, y='SASP', x='ancestral_tumor', ax=axs[1], showfliers=False, color='lightgray')
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)
axs[1].set_xlabel("Parental\ntumour")
axs[1].set_ylabel("SASP score")
axs[1].set_title("Spatial")
plt.tight_layout()
plt.show()

#%%

# cell type correlations with the compartments
corr_df = pd.DataFrame()
for x in np.unique(adata.obs['ancestral_tumor']):
    ad = adata[adata.obs['ancestral_tumor'] == x]

    compartments = ad.obsm['chr_aa']
    compartment_df = pd.DataFrame(data=compartments, index=ad.obs.index)
    sasp_spatial_df = ad.obs[['SASP', 'ancestral_tumor', 'condition']].copy()

    corr_matrix = np.empty((1, len(compartment_df.columns)))
    for i, col1 in enumerate(sasp_spatial_df[['SASP']]):
        for j, col2 in enumerate(compartment_df.columns):
            corr, _ = pearsonr(sasp_spatial_df[col1], compartment_df[col2])
            corr_matrix[i, j] = corr

    corrs = pd.DataFrame(data=corr_matrix,
                         index=[x],
                         columns=compartment_df.columns).T
    corr_df[x] = corrs

cell_type_df = pd.read_csv("data/resources/mouse_cell_type_deconvolution.csv", index_col=0)
cell_type_df = cell_type_df.loc[adata.obs_names]
cell_type_df = cell_type_df.div(cell_type_df.sum(axis=1), axis=0)

cell_type_df['Tumor'] = cell_type_df[['TB-EMT', 'TBasal1', 'TBasal2', 'TFlike', 'TL-Alv', 'TLA-EMT', 'TMlike',
                                      'TProliferating']].sum(axis=1)

for x in np.unique(cell_type_df.columns):
    sasp_spatial_df = adata.obs[['SASP', 'ancestral_tumor', 'condition']].copy()

    corr_matrix = np.empty((1, len(cell_type_df.columns)))
    for i, col1 in enumerate(sasp_spatial_df[['SASP']]):
        for j, col2 in enumerate(cell_type_df.columns):
            corr, _ = pearsonr(sasp_spatial_df[col1], cell_type_df[col2])
            corr_matrix[i, j] = corr

    corrs = pd.DataFrame(data=corr_matrix,
                         index=['SASP'],
                         columns=cell_type_df.columns).T

# Extract x and y
y = cell_type_df['Tumor']
x = sasp_spatial_df['SASP']

plt.hist2d(x, y, bins=60, cmap='twilight_r', norm=LogNorm())

slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='red', linewidth=2)

plt.xlabel('Tumour fraction')
plt.ylabel('SASP score')
plt.colorbar(label='Counts')
plt.show()

#%%
# final plot

# load single cell
adata = sc.read_h5ad('data/sc_dataset.h5ad')

# SASP signature with orthologs
orthologs_df = pd.read_csv('data/resources/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

marker_df = pd.read_csv('data/resources/sasp_mayo.csv')
mouse_orths = []
for v in marker_df['Gene(human)'].values:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)
marker_df['genes_mouse'] = mouse_orths
gene_list = marker_df['genes_mouse']

gene_list_sc = adata.var_names[np.in1d(adata.var_names, gene_list)]
sc.tl.score_genes(adata, gene_list=gene_list_sc, score_name=f'SASP')

sasp_df = adata.obs[['SASP', 'cell_type_lowres']].copy()

# check the spatial samples for sasp
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
adata.obs_names = [f"{a}-1-{b}" for a, b in zip([x.split('-1-')[0] for x in adata.obs_names], adata.obs['sample_id'])]
adata.obs.index = adata.obs_names

gene_list_st = adata.var_names[np.in1d(adata.var_names, gene_list)]
sc.tl.score_genes(adata, gene_list=gene_list_st, score_name=f'SASP')
sasp_spatial_df = adata.obs[['SASP', 'ancestral_tumor', 'condition']].copy()

cell_type_df = pd.read_csv("data/resources/mouse_cell_type_deconvolution.csv", index_col=0)
cell_type_df = cell_type_df.loc[adata.obs_names]
cell_type_df = cell_type_df.div(cell_type_df.sum(axis=1), axis=0)

cell_type_df['Tumor'] = cell_type_df[['TB-EMT', 'TBasal1', 'TBasal2', 'TFlike', 'TL-Alv', 'TLA-EMT', 'TMlike',
                                      'TProliferating']].sum(axis=1)
# Extract x and y
y = cell_type_df['Tumor']
x = sasp_spatial_df['SASP']

# cell type pearson
for c in np.unique(cell_type_df.columns):
    sasp_spatial_df = adata.obs[['SASP', 'ancestral_tumor', 'condition']].copy()

    corr_matrix = np.empty((1, len(cell_type_df.columns)))
    for i, col1 in enumerate(sasp_spatial_df[['SASP']]):
        for j, col2 in enumerate(cell_type_df.columns):
            corr, _ = pearsonr(sasp_spatial_df[col1], cell_type_df[col2])
            corr_matrix[i, j] = corr

    corrs = pd.DataFrame(data=corr_matrix,
                         index=['SASP'],
                         columns=cell_type_df.columns).T

#%%

fig, axs = plt.subplots(1, 2, figsize=(5.5, 3.5), gridspec_kw={'width_ratios': [3, 3]}, constrained_layout=True)
sns.boxplot(sasp_df, y='SASP', x='cell_type_lowres', ax=axs[0], showfliers=False, color='lightgray')
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
axs[0].set_xlabel("Cell type")
axs[0].set_ylabel("SASP score")
axs[0].set_title("Single cell")
axs[0].grid(axis='y')

hist = axs[1].hist2d(x, y, bins=60, cmap='twilight_r', norm=LogNorm())

slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
axs[1].plot(x_line, y_line, color='red', linewidth=2)
axs[1].set_title(rf"Correlation ($r = {corrs.T['Tumor'].values[0]:.2f}$)")

axs[1].set_xlabel('Tumour fraction')
axs[1].set_ylabel('SASP score')
fig.colorbar(hist[3], ax=axs[1], label='Counts')
plt.show()

#%%

# cell type correlations with the compartments
corr_df = pd.DataFrame()
for x in np.unique(adata.obs['ancestral_tumor']):
    ad = adata[adata.obs['ancestral_tumor'] == x]

    compartments = ad.obsm['chr_aa']
    compartment_df = pd.DataFrame(data=compartments, index=ad.obs.index)
    sasp_spatial_df = ad.obs[['SASP', 'ancestral_tumor', 'condition']].copy()

    corr_matrix = np.empty((1, len(compartment_df.columns)))
    for i, col1 in enumerate(sasp_spatial_df[['SASP']]):
        for j, col2 in enumerate(compartment_df.columns):
            corr, _ = pearsonr(sasp_spatial_df[col1], compartment_df[col2])
            corr_matrix[i, j] = corr

    corrs = pd.DataFrame(data=corr_matrix,
                         index=[x],
                         columns=compartment_df.columns).T
    corr_df[x] = corrs
