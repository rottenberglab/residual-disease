import scanpy as sc
import decoupler as dc
import matplotlib.pyplot as plt
from pydeseq2.ds import DeseqStats
from pydeseq2.dds import DeseqDataSet, DefaultInference
from spatial_transcriptomics.functions import plot_volcano_df


#%% Pseudobulk DGEA between primary and residual tumour spots
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony_with_raw.h5ad')

# get raw file
adata.layers['counts'] = adata.raw.X
adata = adata[adata.obs['condition'] != 'relapsed_tumor']

# Get pseudo-bulk profile
pdata = dc.get_pseudobulk(
    adata,
    sample_col='sample_id',
    groups_col='condition',
    layer='counts',
    mode='sum',
    min_cells=10,
    min_counts=1000,
)

#%%
dc.plot_psbulk_samples(pdata, groupby=['sample_id', 'condition'], figsize=(12, 4))
plt.show()

dc.plot_filter_by_expr(pdata, group='condition', min_count=500, min_total_count=10000)
plt.show()
genes = dc.filter_by_expr(pdata, group='condition', min_count=500, min_total_count=10000)
pdata = pdata[:, genes].copy()

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    adata=pdata,
    design_factors=["condition"],
    refit_cooks=True,
    inference=inference,
)

dds.deseq2()
stat_res = DeseqStats(dds, contrast=["condition", "residual-tumor", "primary-tumor"],
                      inference=inference, independent_filter=True)
stat_res.summary()

results_df = stat_res.results_df

scale = 0.85
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(3*scale, 3.2*scale))
plot_volcano_df(results_df, x='log2FoldChange', y='padj', top=10, ax=ax,
                   color_neg='#3573a1', color_pos='#50c6ad', s=4, sign_limit=None)
scatter = ax.collections[0]
scatter.set_rasterized(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('-log10(p-value)')
ax.set_xlabel('logFC')
ax.set_title(f'Pseudo-bulk DEA\n<--Primary | Residual-->')
plt.tight_layout()
plt.savefig('figs/manuscript/fig1/primary_residual_spatial_volcano.svg', dpi=300)
plt.show()

results_df.to_csv('data/residual_vs_primary_tumor_spatial_dea.csv')
