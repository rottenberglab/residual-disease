import scanpy as sc
import matplotlib.pyplot as plt
from single_cell.functions import density_scatter


def load_data(samples, treatment, origin):
    adatas = []
    for s, t in zip(samples, treatment):
        path = f'data/{s}.h5ad'
        adata = sc.read_h5ad(path)
        adata.obs['treatment'] = t
        adata.obs['origin'] = origin
        adatas.append(adata)
    return adatas

samples_mic = ['183_3_mic', '183_3_mi_r', '185_2_mic', '183_1_mic', '185_5_mic', '187_1_mic']
treatment_mic = ['non-treated', 'non-treated', 'non-treated', 'treated', 'treated', 'treated']

samples_tum = ['183_1_tum', '187_1_tum', '183_3_tum', '185_2_tu_r', '188_4_tum']
treatment_tum = ['treated', 'treated', 'non-treated', 'non-treated', 'non-treated']

adatas = load_data(samples_mic, treatment_mic, 'microenvironment')
adatas += load_data(samples_tum, treatment_tum, 'tumor')

adata = sc.concat(adatas, label='batch')
adata.obs_names_make_unique()

# Scatter plot
density_scatter(x=adata.obs['total_counts'], y=adata.obs['n_genes_by_counts'], cmap='viridis')
plt.show()

adata.rename_categories('batch', samples_mic + samples_tum)

# Preprocessing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor='seurat')
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.tsne(adata)
sc.tl.leiden(adata, key_added="clusters", resolution=1.0)

# Cell type annotations were created with iterative subclustering, examining marker gene signatures calculated with
# sc.tl.rank_genes_groups(adata, groupby="clusters", method='wilcoxon'). For T-cells, see tilpred.R.
