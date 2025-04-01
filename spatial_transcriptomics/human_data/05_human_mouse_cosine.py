import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
import seaborn as sns
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from spatial_transcriptomics.functions import matrixplot


# read sample
output_folder = 'data/human_samples'

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]
adata = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')

meta_df = pd.read_csv('data/meta_df_human_filled.csv', index_col=0)
scr_adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')
sample_names = adata.obs['ch_sample_id'].cat.categories

adata.obsm['chr_aa'] = scr_adata.obsm['chr_aa']
human_signature_df = ch.get_compartment_df(scr_adata)
human_signature_df.to_csv('tables/human_cellular_niche_weights.csv')

adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()

signature_df = pd.read_csv('data/compartment_signatures.csv', index_col=0)
orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']
orthologs_dict = {k: v for v, k in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

mouse_orths = []
for v in signature_df.index:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)

signature_df['human'] = mouse_orths
signature_df = signature_df.dropna()
signature_df.index = signature_df['human']
signature_df = signature_df.drop(columns='human')

common_genes = set(signature_df.index).intersection(set(human_signature_df.index))

filtered_signature_df = signature_df.loc[list(common_genes)]
filtered_human_signature_df = human_signature_df.loc[list(common_genes)]

#%%
# cosine similarity
corr_matrix = np.empty((len(filtered_signature_df.columns), len(filtered_human_signature_df.columns)))
for i, col1 in enumerate(filtered_signature_df.columns):
    for j, col2 in enumerate(filtered_human_signature_df.columns):
        A = (filtered_signature_df[col1].values).reshape(1, -1)
        B = (filtered_human_signature_df[col2].values).reshape(1, -1)
        corr = cosine_similarity(A, B)
        corr_matrix[i, j] = corr[0][0]

corrs = pd.DataFrame(data=corr_matrix,
                     # index=[cell_type_dict[x] for x in celltypes_df.columns],
                     index=filtered_signature_df.columns,
                     columns=filtered_human_signature_df.columns).T

corrs.columns = [int(x.split('_')[-1]) for x in corrs.columns]
corrs.index = [int(x.split('_')[-1])for x in corrs.index]

selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]

sf = 0.45
plt.rcParams['svg.fonttype'] = 'none'
matrixplot(corrs.T, figsize=(8 * sf, 12 * sf), flip=False, scaling=False, square=True,
           colorbar_shrink=0.20, colorbar_aspect=10, title='Tissue compartment similarities',
           dendrogram_ratio=0.1, cbar_label="Cosine similarity", xlabel='Mouse compartments', comps=selected_comps,
           cmap='RdYlGn_r',
           ylabel='Human compartments', rasterized=True, seed=87, reorder_obs=True, reorder_comps=True,
           color_comps=True, adata=adata, xrot=0, ha='center', linewidth=0.0, fill_diags=False)
plt.savefig(f'figs/manuscript/fig4/mouse_human_cosine.svg')
plt.show()
