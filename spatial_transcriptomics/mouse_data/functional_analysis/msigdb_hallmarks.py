import time
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import decoupler as dc
from scipy import stats
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.functions import rank_sources_groups, matrixplot, spatial_plot_old


adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)

gene_sets_df = pd.read_csv('msigdb_hallmark_mouse.csv', index_col=0)
dc.run_ora(mat=adata, net=gene_sets_df, source='geneset', target='genesymbol', verbose=True, use_raw=False)
acts = dc.get_acts(adata, obsm_key='ora_estimate')
acts_df = acts.to_df()

acts_v = acts_df.values.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
acts_df.values[~np.isfinite(acts_df.values)] = max_e
adata.obsm['ora_estimate'] = acts_df

pvals = dc.get_acts(adata, obsm_key='ora_pvals')
pvals_df = pvals.to_df()

signif_df = pd.DataFrame({'num_significant': (0.05 > pvals_df).sum(axis=0),
                          'score_sum': acts_df.sum(axis=0)})

signif_sets = signif_df[signif_df['num_significant'] > 40000].index.tolist()

names = {'primary_tumor no_treatment na': 'Primary tumor',
         'relapsed_tumor TAC_regimen 30_days': 'TAC 30dpt',
         'relapsed_tumor cisplatin_6mg/kg 30_days': 'Cisplatin 30dpt',
         'residual_tumor TAC_regimen 12_days': 'TAC 12dpt',
         'residual_tumor TAC_regimen 7_days': 'TAC 7dpt',
         'residual_tumor cisplatin_6mg/kg 12_days': 'Cisplatin 12dpt',
         'residual_tumor cisplatin_6mg/kg 7_days': 'Cisplatin 7dpt'}

adata.obs['condition_cat'] = [' '.join(x) for x in zip(adata.obs['condition'],
                                                       adata.obs['treatment'],
                                                       adata.obs['elapsed_time'])]

pathways_matrix = np.array([adata[adata.obs['condition_cat'] == s].obsm['ora_estimate'].mean(axis=0)
                             for s in np.unique(adata.obs['condition_cat'])])

sample_indexes = [names[x] for x in np.unique(adata.obs['condition_cat'])]
pathways_df = pd.DataFrame(data=pathways_matrix, index=sample_indexes, columns=acts_df.columns)[signif_sets]

pathways_df.columns = [x.capitalize() for x in [' '.join(x.split('_')[1:]) for x in pathways_df.columns]]

df = rank_sources_groups(acts, groupby='sample_id', reference='rest', method='t-test_overestim_var')

order = [0, 4, 3, 1, 6, 5, 2]
pathways_df = pathways_df.iloc[order, :]

z = linkage(pathways_df.T, method='ward')
pathways_df = pathways_df.iloc[:, leaves_list(z)[::-1]]

z_score_df = pathways_df.apply(stats.zscore)

plt.rcParams['svg.fonttype'] = 'none'
sf = 0.6
matrixplot(z_score_df.T, figsize=(12*sf, 12*sf), flip=False, scaling=False, square=True,
            colorbar_shrink=0.2, colorbar_aspect=10, title='Hallmarks',
            dendrogram_ratio=0.1, cbar_label="Z-scaled\nscore", xlabel=None,
            cmap=sns.diverging_palette(325, 145, l=60, s=80, center="dark", as_cmap=True),
            ylabel=None, rasterized=True, seed=87, reorder_obs=False,
            color_comps=False, adata=adata, xrot=90, ha='center')
plt.savefig(f'figs/manuscript/fig3/hallmarks_v2.svg')
plt.show()

#%%
# Spatial plots for representative samples
condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

adata.obs['condition'] = adata.obs['condition'].map(lambda x: condition[x])
adata.obs['treatment'] = adata.obs['treatment'].map(lambda x: treatment[x])
adata.obs['elapsed_time'] = adata.obs['elapsed_time'].map(lambda x: time[x])

selected_samples = [17,  15,  16]
subdata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_11', '20221010-3_12']
comps = 13
subdata.obs['sample_id'] = subdata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = subdata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(subdata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        subdata.uns['spatial'][sample_name_df[k]] = subdata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in subdata.obs['sample_id'].cat.categories]
subdata.obs['label'] = subdata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

label_cats = ['Primary tumor | 20221010-3_9',
              'Residual tumor | 20221010-3_11',
              'Recurrent tumor | 20221010-3_12']

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

i = 0
for c in signif_sets:
    subdata.obs[c] = subdata.obsm['ora_estimate'][c]

    spatial_plot_old(subdata, 1, 3, c, cmap='Spectral_r', title=label_cats,
                 suptitle=c.split('HALLMARK_')[-1].capitalize().replace("_", " "), alpha_img=0.5)
    plt.tight_layout()
    plt.savefig(f"figs/manuscript/fig3/hallmarks_new/{c}.svg")
    plt.show()
    i += 1
