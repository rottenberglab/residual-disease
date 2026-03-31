import liana as li
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


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

#%%
# factorize LR interactions
li.multi.nmf(lrdataset, n_components=None, inplace=True, random_state=0, max_iter=200, verbose=True,
             k_range=range(1, 30))

# extract the variable loadings
lr_loadings = li.ut.get_variable_loadings(lrdataset, varm_key='NMF_H').set_index('index')

# extract the factor scores
factor_scores = li.ut.get_factor_scores(lrdataset, obsm_key='NMF_W')

nmf = sc.AnnData(X=lrdataset.obsm['NMF_W'],
                 obs=lrdataset.obs,
                 var=pd.DataFrame(index=lr_loadings.columns),
                 uns=lrdataset.uns,
                 obsm=lrdataset.obsm)

nmf.write_h5ad("data/cell_comm_nmf.h5ad")

#%%

lr_loadings.sort_values("Factor2", ascending=False).head(10)
nfactors = len(nmf.var)
c = '20220401-1_1'
fig, axs = plt.subplots(1, nfactors, figsize=(3 * nfactors, 3))
nmf_ad = nmf[nmf.obs['sample_id'] == c]
for ax, f in zip(axs, nmf.var_names):
    sc.pl.spatial(nmf_ad, color=f, size=1.4, cmap='coolwarm', show=False, ax=ax,
                  library_id=c)
plt.tight_layout()
plt.show()
