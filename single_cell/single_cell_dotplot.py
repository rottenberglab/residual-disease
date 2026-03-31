import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt


#%%
sc_adata = sc.read_h5ad('data/sc_dataset.h5ad')

with plt.rc_context({'figure.figsize': (10, 10)}):
    sc.pl.tsne(sc_adata, color=['cell_type_hires'], legend_loc='on data', legend_fontoutline=3,
               palette=sns.color_palette('deep'), frameon=False,
               s=80, alpha=1)

sc.pp.normalize_total(sc_adata, inplace=True, target_sum=1e4, exclude_highly_expressed=True)
sc.pp.log1p(sc_adata)

sc.tl.rank_genes_groups(
    sc_adata,
    groupby='cell_type_hires',
    method='wilcoxon',
    n_genes=50,
    pts=True
)

plt.rcParams['svg.fonttype'] = 'none'

scale = 1.5
fig, ax = plt.subplots(figsize=(17*scale, 5*scale))
sc.pl.rank_genes_groups_dotplot(
    sc_adata,
    n_genes=5,
    groupby='cell_type_hires',
    standard_scale='var',
    ax=ax,
    show=False,
)

plt.tight_layout()
plt.savefig(f'figures/single_cell_dotplot.svg')
plt.show()