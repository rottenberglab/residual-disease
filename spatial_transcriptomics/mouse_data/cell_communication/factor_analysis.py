import numpy as np
import liana as li
import pandas as pd
import scanpy as sc
import seaborn as sns
from mudata import MuData
from itertools import product
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


#%%
# read the data
lr_mean_df = pd.read_csv("data/liana/ligand-receptor_scores.csv", index_col=0)
lrdataset = sc.read_h5ad("data/cell_comm_cosine.h5ad")

lrdataset.obs['sample_cat'] = [x.split(' | ')[-1] for x in lrdataset.obs['label']]
order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt', 'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']
lrdataset.obs['sample_cat'] = lrdataset.obs['sample_cat'].astype('category').cat.reorder_categories(order)


def cap_last(x):
    if len(x) > 1:
        x[-1] = x[-1].capitalize()
    return x


new_names  = ['→'.join(map(str.capitalize, x.split('^'))) for x in lrdataset.var_names]
new_names  = ['/'.join(cap_last(x.split('_'))) for x in new_names]

lrdataset.var_names = new_names
lrdataset.var.index = new_names

nmfdataset = sc.read_h5ad("data/cell_comm_nmf.h5ad")

#%%
# get chrysalis compartments
compartments = lrdataset.obsm['chr_aa']
lrdataset.obsm['chr_aa'] = pd.DataFrame(data=compartments, index=lrdataset.obs.index)
comps = li.ut.obsm_to_adata(lrdataset, 'chr_aa')

# get combination of compartments and factors
factors = nmfdataset.var_names
interactions = list(product(comps.var.index, factors))

mdata = MuData({"factors":nmfdataset, "comps":comps})
mdata.obsp = lrdataset.obsp
mdata.uns = nmfdataset.uns
mdata.obsm = nmfdataset.obsm

bdata = li.mt.bivariate(mdata,
                        x_mod="comps",
                        y_mod="factors",
                        local_name="cosine",
                        global_name="morans",
                        mask_negatives=True,
                        add_categories=True,
                        interactions=interactions,
                        x_use_raw=False,
                        y_use_raw=False,
                        xy_sep="<->",
                        x_name='niche',
                        y_name='factor'
                        )

#%%
# get top hits for every compartment

nhits = 6
var = 'mean'

var_df = bdata.var.copy()
top_hits = var_df.groupby('niche', group_keys=False).apply(lambda x: x.nlargest(nhits, var))

hit_matrix = np.zeros([len(top_hits), len(np.unique(top_hits['niche']))])

hit_matrix = []
hit_list = []
for n in np.unique(top_hits['niche']):
    niche_df = top_hits[top_hits['niche'] == str(n)]
    inter = niche_df['factor'].sort_values()
    interactions_df = var_df[var_df['factor'].isin(inter)]
    interactions_df = interactions_df.sort_values(by=['niche', 'factor'])
    means = np.reshape(interactions_df[var], (13, nhits)).T
    hit_matrix.append(means)
    hit_list.extend(inter)

hit_matrix = np.concatenate(hit_matrix, axis=0)
hit_df = pd.DataFrame(data=hit_matrix, index=hit_list)
hit_df = hit_df.drop_duplicates()

sns.heatmap(hit_df, cmap='Spectral_r', square=True)
plt.show()

#%%
# plots


def plot_dataset(nmfdataset, color, sample_col):
    condition = {'primary_tumor': 'Primary tumor',
                 'relapsed_tumor': 'Relapsed tumor',
                 'residual_tumor': 'Residual tumor',
                 'control': 'Control'}

    treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg',
                 'no_treatment': 'No treatment', 'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}

    time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment', '7_days': '7 days post-treatment',
            'na': '-', '24_hours': '24 hours post-treatment', '4_hours': '4 hours post-treatment'}

    rows = 5
    cols = 5

    sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)

    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    ax = ax.flatten()
    for a in ax:
        a.axis('off')

    for idx, sample_id in enumerate(np.unique(nmfdataset.obs[sample_col])):
        ad = nmfdataset[nmfdataset.obs[sample_col] == sample_id].copy()
        sc.pl.spatial(ad, color=color, size=1.5, alpha=1, library_id=sample_id,
                      ax=ax[idx], show=False, legend_loc=None)
        ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}\n' + \
                          f'{treatment[ad.obs["treatment"][0]]}\n{time[ad.obs["elapsed_time"][0]]}')
    plt.tight_layout()
    plt.show()


for x in [f'Factor{x+1}' for x in range(6)]:
    plot_dataset(nmfdataset, x, sample_col='sample_id')

#%%

# cell type correlations with the compartments
corr_df = pd.DataFrame()

compartments = nmfdataset.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=nmfdataset.obs.index)
factor_df = nmfdataset.to_df()

corr_matrix = np.empty((len(factor_df.columns), len(compartment_df.columns)))
for i, col1 in enumerate(factor_df.columns):
    for j, col2 in enumerate(compartment_df.columns):
        corr, _ = pearsonr(factor_df[col1], compartment_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     index=factor_df.columns,
                     columns=compartment_df.columns).T

sns.heatmap(corrs, cmap='coolwarm', square=True)
plt.show()
