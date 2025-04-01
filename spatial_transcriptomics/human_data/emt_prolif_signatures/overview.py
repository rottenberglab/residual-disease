import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
import seaborn as sns
import chrysalis as ch
import matplotlib as mpl
import matplotlib.pyplot as plt


# Overview on EMT signature expression in humans
# read sample
output_folder = 'data/human_samples'

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]
adata = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')

meta_df = pd.read_csv('human_metadata.csv', index_col=0)
scr_adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')
sample_names = adata.obs['ch_sample_id'].cat.categories

adata.obsm['chr_aa'] = scr_adata.obsm['chr_aa']

adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()
adata = adata[~adata.obs['annotations'].isin(['Other', 'Stroma'])]

signature_df = pd.read_csv('data/compartment_signatures.csv', index_col=0)
orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']
orthologs_dict = {k: v for v, k in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

gene_exp_df = adata.var.sort_values(by='log1p_mean_counts', ascending=False).copy()
gene_exp_df['rank'] = [x for x in range(len(gene_exp_df))]

expression_prog_df = pd.DataFrame()
for cn in signature_df.columns:
    c = signature_df[cn]
    m = np.mean(c)
    std = np.std(c)
    threshold = m + (2 * std)
    genes = c > threshold
    expression_prog_df[cn] = genes

mouse_orths = []
for v in signature_df.index:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)

signature_df['human'] = mouse_orths
expression_prog_df['human'] = mouse_orths

# Save signatures
signature_df.to_csv('tables/cellular_niche_weights.csv')
expression_prog_df.to_csv('tables/cellular_niche_signatures.csv')

signature_df = signature_df.dropna()
signature_df['mouse'] = signature_df.index
signature_df.index = signature_df['human']

#%%

# select signatures
selected_comp = 5  # 2 emt 5 prolif
m = np.mean(signature_df[f'compartment_{selected_comp}'])
std = np.std(signature_df[f'compartment_{selected_comp}'])
threshold = m + (2 * std)
gene_list = signature_df[f'compartment_{selected_comp}'][signature_df[f'compartment_{selected_comp}'] > threshold]
gene_list = [g for g in gene_list.index if g in adata.var_names]
prolif_gene_list = gene_list
signature_name = 'Proliferating'
sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'{signature_name}_signature', use_raw=False)

# select signatures
selected_comp = 2  # 2 emt 5 prolif
m = np.mean(signature_df[f'compartment_{selected_comp}'])
std = np.std(signature_df[f'compartment_{selected_comp}'])
threshold = m + (2 * std)
gene_list = signature_df[f'compartment_{selected_comp}'][signature_df[f'compartment_{selected_comp}'] > threshold]
gene_list = gene_list.sort_values(ascending=False)
gene_list = [g for g in gene_list.index if g in adata.var_names]
emt_gene_list = gene_list
signature_name = 'EMT'
sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'{signature_name}_signature', use_raw=False)

gene_exp_df = adata.var.sort_values(by='log1p_mean_counts', ascending=False).copy()
gene_exp_df['rank'] = [x for x in range(len(gene_exp_df))]

subset_df = gene_exp_df[gene_exp_df.index.isin(emt_gene_list[:])]

#%%
# EMT signature rankplot
# Initialize a JointGrid
plt.rcParams['svg.fonttype'] = 'none'
g = sns.JointGrid(data=subset_df, x="rank", y="log1p_mean_counts", height=3, marginal_ticks=True)

# Add histogram to the marginal axes
g.plot_marginals(sns.histplot, bins=10, kde=False, color="#caa167", edgecolor=None)
# Add line plot to the main axes with lower zorder
sns.lineplot(x=range(len(gene_exp_df)), y=gene_exp_df["log1p_mean_counts"], ax=g.ax_joint, color="gray", zorder=1,
             linewidth=2)
# Add scatter plot on top with higher zorder
g.plot_joint(sns.scatterplot, zorder=2, color="#caa167", edgecolor=None, alpha=0.5, rasterized=True, s=50)
# Remove the y-axis marginal plot
g.ax_marg_y.remove()
g.ax_joint.set_ylabel('Gene expression')
g.ax_joint.grid(axis='y')
plt.title('EMT genes in human cancer')
# Show the plot
plt.savefig('figs/manuscript/fig4/rank_plot.svg', dpi=200)
plt.show()

#%%
# EMT signature top genes
sc.tl.rank_genes_groups(adata, 'condition', method='wilcoxon', use_raw=False)

fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sc.pl.rank_genes_groups_violin(adata, n_genes=10, jitter=False, gene_names=list(subset_df.index[:10]),
                               groups='primary_tumor', ax=ax, show=False)
for child in ax.get_children():
    if isinstance(child, mpl.collections.PathCollection):  # Points are typically PathCollections
        child.set_rasterized(True)
    elif isinstance(child, mpl.patches.Patch):  # Ensure violins remain vectorized
        child.set_rasterized(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figs/manuscript/fig4/top_genes.svg', dpi=150)
plt.show()
