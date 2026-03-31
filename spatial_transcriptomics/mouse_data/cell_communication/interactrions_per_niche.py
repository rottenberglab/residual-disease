import numpy as np
import liana as li
import pandas as pd
import scanpy as sc
import seaborn as sns
from mudata import MuData
from itertools import product
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from spatial_transcriptomics.functions import spatial_plot


def plot_spatial_data(adata, colors):

    # subset anndata
    selected_samples = [17,  15,  19,  16,  21]
    ad = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

    # reorder sample_id for plotting
    sample_order = ['20221010-3_9', '20221010-3_11', '20221010-4_14', '20221010-3_12', '20221010-4_16']
    ad.obs['sample_id'] = ad.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

    # add a dict quickly to replace titles
    sample_name_df = ad.obs[['sample_id', 'label']].drop_duplicates()
    sample_name_df.index = sample_name_df['sample_id']
    sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

    for k in list(ad.uns['spatial'].keys()):
        if k in list(sample_name_df.keys()):
            ad.uns['spatial'][sample_name_df[k]] = ad.uns['spatial'][k]

    label_cats = [sample_name_df[x] for x in ad.obs['sample_id'].cat.categories]
    ad.obs['label'] = ad.obs['label'].cat.reorder_categories(label_cats, ordered=True)

    print(adata.var.loc['5<->Fgf1→Egfr'])

    for c in colors:
        spatial_plot(ad, 5, 1, c,
                     cmap=sns.color_palette("Spectral_r", as_cmap=True), sample_col='label', s=15,
                     title=None, suptitle=f'{c} proportion', alpha_img=0.0, colorbar_label='Score',
                     colorbar_aspect=5, colorbar_shrink=0.15, hspace=-0.7, subplot_size=4, alpha_blend=False,
                     x0=0.3, suptitle_fontsize=15)

        plt.tight_layout()
        # plt.savefig(f'figs/manuscript/fig3/{c}_5_vertical_r.svg')
        plt.show()
        plt.close()


def cap_last(x):
    if len(x) > 1:
        x[-1] = x[-1].capitalize()
    return x


#%%
# read the ligand-receptor interaction data
lr_mean_df = pd.read_csv("data/liana/ligand-receptor_scores.csv", index_col=0)

lrdataset = sc.read_h5ad("data/cell_comm_morans.h5ad")
lrdataset.obs['sample_cat'] = [x.split(' | ')[-1] for x in lrdataset.obs['label']]
order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt', 'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

lrdataset.obs['sample_cat'] = lrdataset.obs['sample_cat'].astype('category').cat.reorder_categories(order)

new_names  = ['→'.join(map(str.capitalize, x.split('^'))) for x in lrdataset.var_names]
new_names  = ['/'.join(cap_last(x.split('_'))) for x in new_names]

lrdataset.var_names = new_names
lrdataset.var.index = new_names

#%%
# get correlations between the compartments

corr_df = pd.DataFrame()

compartments = lrdataset.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=lrdataset.obs.index)
factor_df = lrdataset.to_df()

corr_matrix = np.empty((len(factor_df.columns), len(compartment_df.columns)))
for i, col1 in enumerate(factor_df.columns):
    for j, col2 in enumerate(compartment_df.columns):
        corr, _ = pearsonr(factor_df[col1], compartment_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     index=factor_df.columns,
                     columns=compartment_df.columns)

sns.heatmap(corrs, cmap='coolwarm', square=False)
plt.show()

#%%
# get the top hits

nhits = 5

dfs = []
for c in corrs.columns:
    top_hits = corrs.nlargest(nhits, c)
    dfs.append(top_hits)
df = pd.concat(dfs)

sns.heatmap(df, cmap='coolwarm', square=False)
plt.tight_layout()
plt.show()

#%%
# select top interactions
lrdataset.var['cv'] = (np.std(np.asarray(lrdataset.X.todense()), axis=0) /
                       np.mean(np.asarray(lrdataset.X.todense()), axis=0))

# get chrysalis compartments
compartments = lrdataset.obsm['chr_aa']
lrdataset.obsm['chr_aa'] = pd.DataFrame(data=compartments, index=lrdataset.obs.index)

comps = li.ut.obsm_to_adata(lrdataset, 'chr_aa')

# get combination of compartments and interactions
# top_lrs = lrdataset.var.sort_values('cv', ascending=False, key=abs).head(50).index
top_lrs = lrdataset.var_names
interactions = list(product(comps.var.index, top_lrs))


mdata = MuData({"lr":lrdataset, "comps":comps})
mdata.obsp = lrdataset.obsp
mdata.uns = lrdataset.uns
mdata.obsm = lrdataset.obsm

bdata = li.mt.bivariate(mdata,
                        x_mod="comps",
                        y_mod="lr",
                        local_name="cosine",
                        global_name="morans",
                        mask_negatives=True,
                        add_categories=True,
                        interactions=interactions,
                        x_use_raw=False,
                        y_use_raw=False,
                        xy_sep="<->",
                        x_name='niche',
                        y_name='lr'
                        )

#%%
# get top hits for every compartment

nhits = 3

var_df = bdata.var.copy()
top_hits = var_df.groupby('niche', group_keys=False).apply(lambda x: x.nlargest(nhits, 'morans'))

hit_matrix = np.zeros([len(top_hits), len(np.unique(top_hits['niche']))])

hit_matrix = []
hit_list = []
for n in np.unique(top_hits['niche']):
    niche_df = top_hits[top_hits['niche'] == str(n)]
    inter = niche_df['lr'].sort_values()
    interactions_df = var_df[var_df['lr'].isin(inter)]
    interactions_df = interactions_df.sort_values(by=['niche', 'lr'])
    means = np.reshape(interactions_df['morans'], (13, nhits)).T
    hit_matrix.append(means)
    hit_list.extend(inter)

hit_matrix = np.concatenate(hit_matrix, axis=0)
hit_df = pd.DataFrame(data=hit_matrix, index=hit_list)

# hit_df = hit_df.drop_duplicates()

sns.heatmap(hit_df, cmap='Spectral_r')
plt.show()

#%%
# rank interactions with wilcoxon
sc.tl.rank_genes_groups(lrdataset, 'sample_cat', method='wilcoxon', key_added="wilcoxon")
stat_df = sc.get.rank_genes_groups_df(lrdataset, group='Cisplatin 12dpt', key='wilcoxon')

sc.pl.rank_genes_groups(lrdataset, n_genes=25, sharey=False, key="wilcoxon")

sf = 0.8

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(14*sf, 6.5*sf))
sc.pl.rank_genes_groups_dotplot(lrdataset, n_genes=5, key="wilcoxon", values_to_plot="logfoldchanges", dendrogram=False,
                                # cmap=sns.diverging_palette(260, 320, s=80, l=50, as_cmap=True),
                                cmap='RdYlBu_r',
                                show=False, ax=ax)
fig.subplots_adjust(left=0.125, bottom=0.335, right=0.90, top=0.80)
plt.savefig('data/liana/interactions_per_condition.svg')
plt.show()
