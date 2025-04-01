import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import ausankey as sky
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram
from spatial_transcriptomics.functions import matrixplot
from scipy.cluster.hierarchy import linkage, leaves_list


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

names = {'primary_tumor no_treatment na': 'Primary tumor',
         'relapsed_tumor TAC_regimen 30_days' : 'TAC 30dpt',
         'relapsed_tumor cisplatin_6mg/kg 30_days': 'Cisplatin 30dpt',
         'residual_tumor TAC_regimen 12_days': 'TAC 12dpt',
         'residual_tumor TAC_regimen 7_days': 'TAC 7dpt',
         'residual_tumor cisplatin_6mg/kg 12_days': 'Cisplatin 12dpt',
         'residual_tumor cisplatin_6mg/kg 7_days': 'Cisplatin 7dpt'}

name_order =['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
             'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

#%%
# Cell type co-occurrence matrices and sankey plot
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

celltypes_df = adata.obsm['cell2loc']
celltypes_df.columns = [cell_type_dict[x] for x in celltypes_df.columns]

adata.obs['condition_cat'] = [x + ' ' + y + ' ' + z for x, y, z in
                              zip(adata.obs['condition'],
                                  adata.obs['treatment'],
                                  adata.obs['elapsed_time'])]
adata.obs['condition_cat'] = adata.obs['condition_cat'].astype('category')

cluster_conds = dict()
for idx, s in enumerate(np.unique(adata.obs['condition_cat'])):
    ad = adata[adata.obs['condition_cat'] == s].copy()
    celltypes_df = ad.obsm['cell2loc']
    # celltypes_df.columns = [cell_type_dict[x] for x in celltypes_df.columns]

    corr_matrix = np.empty((len(celltypes_df.columns), len(celltypes_df.columns)))
    for i, col1 in enumerate(celltypes_df.columns):
        for j, col2 in enumerate(celltypes_df.columns):
            corr, _ = pearsonr(celltypes_df[col1], celltypes_df[col2])
            corr_matrix[i, j] = corr

    corr_matrix[corr_matrix > 0.999] = 1

    corrs = pd.DataFrame(data=corr_matrix,
                         # index=[cell_type_dict[x] for x in celltypes_df.columns],
                         index=celltypes_df.columns,
                         columns=celltypes_df.columns).T
    sf = 0.65
    plt.rcParams['svg.fonttype'] = 'none'
    matrixplot(corrs, figsize=(18.2 * sf, 12 * sf), flip=False, scaling=False, square=True,
               colorbar_shrink=0.20, colorbar_aspect=10, title=s,
               dendrogram_ratio=0.1, cbar_label="Score", xlabel='Pathways', comps=None,
               cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
               ylabel='Compartments', rasterized=True, seed=87, reorder_obs=True, reorder_comps=True,
               color_comps=False, adata=adata, xrot=90, ha='center', linewidth=0.0, fill_diags=False)
    # plt.savefig(f'figs/manuscript/fig3/cell_co_corr_{idx}.svg')
    plt.show()

    # cluster cutoff
    threshold = 3.5
    z = linkage(corrs, method='ward')
    order = leaves_list(z)
    corrs = corrs.iloc[order, :]
    dendrogram(z, no_labels=True, color_threshold=0,
                        above_threshold_color='black')
    plt.axhline(threshold)
    plt.show()

    cluster_labels = fcluster(z, threshold, criterion='distance')
    corrs['cluster'] = cluster_labels
    cell_cluster_dict = {k:v for k, v in zip(corrs.index, corrs['cluster'])}
    cluster_conds[names[s]] = cell_cluster_dict
cluster_conds = pd.DataFrame(cluster_conds)

cluster_conds = cluster_conds[name_order]

sankey_df = cluster_conds.iloc[:, :4].copy()

new_columns = []
for col in sankey_df.columns:
    new_columns.append(sankey_df[col])
    new_columns.append(sankey_df[col].rename(f'{sankey_df[col].name}_dupl'))
df_duplicated = pd.DataFrame(new_columns).T

for idx, c in enumerate(df_duplicated.columns):
    if idx > 1:
        if idx % 2 == 0:
            print(idx)
            max_num = np.max(df_duplicated.iloc[:, idx-2])
            col = df_duplicated[c].copy()
            col += max_num
            df_duplicated[c] = col

plt.figure()

sky.sankey(
    df_duplicated,
    sort = "top",
    titles = sankey_df.columns[:int(8/2)],
    valign = "center",
    label_duplicate=True,
    colormap='Set3',
    node_width=0.1,
)
plt.show()
