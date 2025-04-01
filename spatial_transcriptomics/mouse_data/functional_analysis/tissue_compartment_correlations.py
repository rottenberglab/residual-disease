import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.functions import matrixplot


adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

#%%
# PROGENY
enrich_df = pd.read_csv('pathway_activity_scores.csv', index_col=0)

corr_m = pd.concat([compartment_df, enrich_df], axis=1).corr()
corr_m = corr_m.drop(index=enrich_df.columns, columns=compartment_df.columns)

corr_m.index = [x for x in range(13)]

hexcodes = ch.utils.get_hexcodes(None, 13, 87, len(adata))

selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]
plt.rcParams['svg.fonttype'] = 'none'
sf = 0.6
matrixplot(corr_m, figsize=(8.2*sf, 12*sf), flip=True, scaling=False, square=True,
            colorbar_shrink=0.15, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.1, cbar_label="Score", xlabel='Pathways', comps=selected_comps,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Compartments', rasterized=True, seed=87, reorder_obs=True,
            color_comps=True, adata=adata, xrot=90, ha='center')
# plt.savefig(f'figs/manuscript/fig3/tissue_comp_pathway_v2.svg')
plt.show()

#%%
# MSIGDB HALLMARKS
enrich_df = pd.read_csv('hallmarks_scores.csv', index_col=0)

hexcodes = ch.utils.get_hexcodes(None, 13, 87, len(adata))

pvals_df = pd.read_csv('hallmarks_pvals.csv', index_col=0)
signif_df = 0.05 > pvals_df
signif_df = signif_df.sum(axis=0)
signif_df = pd.DataFrame(data=signif_df, columns=['num_significant'])
signif_sets = signif_df[signif_df['num_significant'] > 10000].index.tolist()
enrich_df = enrich_df[signif_sets]
enrich_df.columns = [' '.join(x.split('_')) for x in enrich_df.columns]

corr_m = pd.concat([compartment_df, enrich_df], axis=1).corr()
corr_m = corr_m.drop(index=enrich_df.columns, columns=compartment_df.columns)
corr_m.index = ['' + str(x) for x in range(13)]

z = linkage(corr_m, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[order, :]
hexcodes = [hexcodes[i] for i in order]

z = linkage(corr_m.T, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[:, order]

plt.rcParams['svg.fonttype'] = 'none'
corr_m.columns = [x.split('HALLMARK ')[-1].title() for x in corr_m.columns]

fig, ax = plt.subplots(1, 1, figsize=(11, 12))
sns.heatmap(corr_m.T, square=True, cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            center=0, rasterized=True)
for idx, t in enumerate(ax.get_xticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
plt.title('Tissue compartment \nhallmark correlation')
# plt.savefig(f'figs/manuscript/fig3/hallmark_correlations.svg')
plt.tight_layout()
plt.show()

#%%
# HALLMARKS OF CANCER

orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

# get cancer hallmark signatures
marker_df = pd.read_csv('cancer_hallmark_signatures.csv', index_col=0)

# define a cancer hallmark df with the mouse orthologs
hm_dict = {}
hm_names = []
for hm in np.unique(marker_df['hallmark']):
    hm_df = marker_df[marker_df['hallmark'] == hm]
    gene_set = set()
    for c in hm_df['gene_list']:
        genes = c.split(', ')
        for g in genes:
            gene_set.add(g)

    # search for orthologs
    mouse_orths = []
    for v in gene_set:
        try:
            g = orthologs_dict[v]
        except:
            pass
        mouse_orths.append(g)
    hm_dict[hm] = mouse_orths
    hm_names.append(hm_df['hallmark_name'].iloc[0])
hm_mouse = pd.Series(hm_dict, name='gene_set')
hm_mouse_df = pd.DataFrame(hm_mouse)
hm_mouse_df['hallmark_name'] = hm_names

for idx, row in hm_mouse_df.iterrows():
    gene_list = row['gene_set']
    gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]
    sc.tl.score_genes(adata, gene_list=gene_list, score_name=row['hallmark_name'])

columns = hm_mouse_df['hallmark_name']

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

corr_m = pd.concat([compartment_df, adata.obs[columns]], axis=1).corr(method='pearson')
corr_m = corr_m.drop(index=columns, columns=compartment_df.columns)
corr_m = corr_m.rename(columns={'sasp': 'SASP'})
corr_m.index = [str(x) for x in corr_m.index]

corr_m.index = [x for x in range(13)]

selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]
plt.rcParams['svg.fonttype'] = 'none'
sf = 0.6
matrixplot(corr_m, figsize=(11.2*sf, 12*sf), flip=False, scaling=False, square=True,
            colorbar_shrink=0.15, colorbar_aspect=10, title='Hallmarks of cancer',
            dendrogram_ratio=0.1, cbar_label="Score", ylabel='Cancer hallmark', comps=selected_comps,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            xlabel='Compartments', rasterized=True, seed=87, reorder_obs=True,
            color_comps=True, adata=adata, xrot=0, ha='center')
# plt.savefig(f'figs/manuscript/fig3/cancer_hallmarks_v3.svg')
plt.show()

sf = 0.45
plt.rcParams['svg.fonttype'] = 'none'
matrixplot(corr_m, figsize=(18.2*sf, 12*sf), flip=False, scaling=False, square=True,
            colorbar_shrink=0.20, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.05, cbar_label="Score", xlabel='Pathways', comps=selected_comps,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Compartments', rasterized=True, seed=87, reorder_obs=False, reorder_comps=True,
            color_comps=True, adata=adata, xrot=0, ha='center', linewidth=0.5)
# plt.savefig(f'figs/manuscript/fig3/go_terms.svg')
plt.show()
