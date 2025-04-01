import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
adata.obs['hires_map'] = [label_map[x] for x in adata.obs['hires']]

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
sc.pl.tsne(adata, color=['hires_map'], legend_loc='on data', legend_fontoutline=1,
           palette=color_map, frameon=False, ax=ax, legend_fontsize=10,
           s=40, alpha=1, show=False)
plt.show()

# Legends
color_list = ['#19a1e6','#5ebeed','#5e5eed','#5aedbc','#6cda88','#3cdd51','#178280','#172982','#ed5e5e','#ed825e',
              '#edb25e','#ed5e9c','#f28cb8','#e699c4','#d999e6','#7b53f3','#af75f5','#a947eb','#c788f2','#976ef7',
              '#cbb6fb','#7fdd3c','#c32222',]

long_names = [f'{n} | {l}' for n, l in zip(list(label_map.values()), list(label_map.keys()))]
label_dict = {k: v for k, v in zip(long_names, color_list)}
legend_patches = [mpatches.Patch(color=label_dict[key], label=key) for key in label_dict]

fig, ax = plt.subplots(figsize=(5, 10))
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=key, markersize=10,
          markerfacecolor=label_dict[key]) for key in label_dict],
          loc='center', title='Legend', frameon=False)
ax.axis('off')
plt.show()

# Latent time plot
adata.obs['latent_time'] = pd.read_csv('data/single_cell/latent_time.csv', index_col=0)
adata_latent = adata[~adata.obs['latent_time'].isna()].copy()

fig, ax = plt.subplots(1, 1, figsize=(5, 2.8))
sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
g = sc.pl.tsne(adata_latent, color=['latent_time'],
               legend_fontoutline=2, frameon=False, ax=ax, legend_fontsize=10,
               cmap='magma', s=60, alpha=1, show=False, colorbar_loc=None)
sc_img = ax.collections[0]
colorbar = fig.colorbar(sc_img, ax=ax, shrink=0.4, aspect=5)  # Shrink the colorbar here
colorbar.set_label('Latent time')
plt.tight_layout()
plt.show()

# Primary - residual labels
cp = ['#8b33ff', '#ff3363']

for batch in adata.obs['treatment'].cat.categories:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
    sc.pl.tsne(adata, color=['treatment'],
               palette=cp, frameon=False, groups=[batch],
               s=40, alpha=1, ax=ax, show=False)
    plt.show()

# Heatmap of some marker genes
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X

marker_genes_dict = {
    'B cell': ['Cd19', 'Jchain'],
    'Dendritic cell': ['Relb', 'Lsp1', 'Flt3'],
    'Endothelial cell': ['Pecam1', 'Cdh5'],
    'Fibroblast': ['Col1a1', 'Fn1'],
    'Macrophage': ['Cd68', 'C1qa', 'Spp1'],
    'NK cell': ['Ncr1', 'Klrb1c'],
    'T cell': ['Cd3e'],
    'Tumor': ['Krt18', 'Krt14']
}

fig, ax = plt.subplots(1, 1, figsize=(5, 7))
plt.rcParams['svg.fonttype'] = 'none'
sc.pl.matrixplot(adata, marker_genes_dict, 'lowres', dendrogram=False,
                 colorbar_title='mean z-score', layer='scaled', vmin=-2, vmax=2,
                 cmap=sns.diverging_palette(60, 304, l=63, s=87, center="dark", as_cmap=True),
                 swap_axes=True, ax=ax, show=False)
plt.tight_layout()
plt.show()
