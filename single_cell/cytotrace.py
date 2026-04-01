from cytotrace2_py.cytotrace2_py import *


#%%
# read single cell data and save the matrices
sc_adata = sc.read_h5ad("data/sc_dataset_zenodo.h5ad")
sc_adata = sc_adata[~sc_adata.obs['latent_time'].isna()].copy()

sc_adata_t = sc_adata.T

matrix_df = sc_adata_t.to_df().astype(float)
annotation_df = sc_adata_t.var["cell_type_hires"]
matrix_df.to_csv("data/matrix_df.txt", sep='\t')
annotation_df.to_csv("data/annotation_df.txt", sep='\t')

#%%
# run cytotrace and save the results
species = "mouse"
results =  cytotrace2("data/matrix_df.txt",
                      annotation_path="data/annotation_df.txt",
                      species=species)
results.to_csv("data/cytosig.csv")

#%%
# plot potencies for the tumor cells

order = [
    'Totipotent',
    'Pluripotent',
    'Multipotent',
    'Oligopotent',
    'Unipotent',
    'Differentiated',
]

sc_adata = sc.read_h5ad("data/sc_dataset_zenodo.h5ad")
sc_adata = sc_adata[~sc_adata.obs['latent_time'].isna()].copy()

results = pd.read_csv("data/cytosig.csv", index_col=0)
sc_adata.obs[results.columns] = results

sc_adata = sc_adata[sc_adata.obs['cell_type_hires'].str.contains('Tumor')]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
sc.pl.tsne(sc_adata, color=['CytoTRACE2_Potency'],
           palette=sns.color_palette('deep'), frameon=False,
           s=80, alpha=1, show=False, ax=ax)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
sc.pl.tsne(sc_adata, color=['CytoTRACE2_Score'],
           palette=sns.color_palette('deep'), frameon=False,
           s=80, alpha=1, show=False, ax=ax)
plt.tight_layout()
plt.show()