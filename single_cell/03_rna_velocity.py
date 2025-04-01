import anndata
import scanpy as sc
import scvelo as scv
from glob import glob


# Load reference data
reference = sc.read_h5ad("path/sc_dataset.h5ad")
reference.obs_names_make_unique()
ref_names = set(reference.obs_names)

# Load and concatenate .loom files
looms = [scv.read(f, cache=True) for f in glob("path/*.loom")]
for adata in looms:
    adata.var_names_make_unique()
adata = anndata.concat(looms, join='outer')

# Filter genes and format cell names
sc.pp.filter_genes(adata, min_cells=10)
adata.obs_names = [x.split(':')[-1][:-1] + '-1' for x in adata.obs_names]
adata.obs_names_make_unique()

# Subset data to intersecting cell names
intersecting_names = ref_names.intersection(adata.obs_names)
adata = adata[adata.obs.index.isin(intersecting_names)]
reference = reference[reference.obs.index.isin(intersecting_names)]

# Transfer annotations from reference
adata.obs[['lowres', 'hires']] = reference.obs[['lowres', 'hires']]
scv.pl.proportions(adata, groupby='lowres')

# Filter for tumor cells and remove specific subtypes
adata = adata[adata.obs['lowres'] == 'Tumor']
adata = adata[~adata.obs['hires'].isin(['TMlike', 'TFlike'])]

# Preprocessing
sc.pp.calculate_qc_metrics(adata)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='hires')

# Velocity preprocessing
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

# Compute velocity and visualization
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(adata, basis='umap', color='hires', size=70, alpha=0.5)

# Cell cycle analysis
scv.tl.score_genes_cell_cycle(adata)
scv.pl.scatter(adata, color_gradients=['S_score', 'G2M_score'], smooth=True, perc=[5, 95])
scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120)

# Velocity confidence
scv.tl.velocity_confidence(adata)
scv.pl.scatter(adata, c=['velocity_length', 'velocity_confidence'], cmap='coolwarm', perc=[5, 95])
scv.pl.velocity_graph(adata, threshold=0.1)

# Pseudotime and PAGA analysis
scv.tl.velocity_pseudotime(adata, root_key='TProliferating', use_velocity_graph=False)
scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')
scv.tl.paga(adata, groups='hires')
scv.pl.paga(adata, basis='umap', size=50, alpha=0.1, min_edge_width=2, node_size_scale=1.5)
scv.tl.recover_dynamics(adata)

# Latent time analysis
scv.tl.latent_time(adata)
scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=80)

# Save latent time results
adata.obs['latent_time'].to_csv("C:/Users/demeter_turos/PycharmProjects/persistance/data/single_cell/latent_time.csv")

# Generate heatmap for top genes
top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index[:300]
scv.pl.heatmap(adata, var_names=top_genes, sortby='latent_time', col_color='hires', n_convolve=100)
