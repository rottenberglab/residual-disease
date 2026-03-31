import anndata
import numpy as np
import liana as li
import pandas as pd
import scanpy as sc
from glob import glob
import scipy.sparse as sp


#%%
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

print(li.mt.bivariate.show_functions())

#%%
# ortholog mapping
orthologs_df = pd.read_csv('data/resources/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

orthologs_dict = {v: k for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

human_orths = []
for v in adata.var_names:
    try:
        g = orthologs_dict[v]
    except KeyError:
        g = np.nan
    human_orths.append(g)

adata.var_names = human_orths
adata.var.index = human_orths
adata = adata[:, ~adata.var.index.isna()].copy()

#%%
# define parameters
# check bandwidth on one sample
s_id = np.unique(adata.obs['sample_id'])[0]
ad = adata[adata.obs['sample_id'] == s_id].copy()

plot, _ = li.ut.query_bandwidth(coordinates=ad.obsm['spatial'], start=0, end=250, interval_n=5)
plot.show()

# set and plot connectivity
li.ut.spatial_neighbors(ad, bandwidth=200, cutoff=0.1, kernel='gaussian', set_diag=True)

plot = li.pl.connectivity(ad, idx=0, size=1.3, figure_size=(4, 3), return_fig=True)
plot.show()

#%%
# calculate LR interactions per sample


def run_liana(adata, local_name='cosine', global_name='morans'):

    for c in (np.unique(adata.obs['sample_id'])):
        ad = adata[adata.obs['sample_id'] == c].copy()

        li.ut.spatial_neighbors(ad, bandwidth=200, cutoff=0.1, kernel='gaussian', set_diag=True)

        lrdata = li.mt.bivariate(ad, resource_name='consensus', local_name=local_name, global_name=global_name,
                                 n_perms=1000, mask_negatives=False, add_categories=True, nz_prop=0.01,
                                 use_raw=False, verbose=True)

        lrdata.write_h5ad(f"data/liana/lrdata_{c}_{local_name}.h5ad")

    # concat the calculated adatas and save
    lr_files = glob(f'data/liana/lrdata_*_{local_name}*.h5ad')

    adatas = []
    for lr in lr_files:
        ad = sc.read_h5ad(lr)
        adatas.append(ad)

    lrdataset = anndata.concat(adatas, index_unique='-', uns_merge='unique', merge='unique')

    # add connnectivity map
    lrdataset.obsp['spatial_connectivities'] = sp.block_diag(
        [ad.obsp['spatial_connectivities'] for ad in adatas], format='csr'
    )

    lrdataset.write_h5ad(f'data/cell_comm_{local_name}.h5ad')


run_liana(adata, local_name='cosine')

run_liana(adata, local_name='morans')
