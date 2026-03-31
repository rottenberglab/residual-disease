import scanpy as sc
import decoupler as dc
import matplotlib.pyplot as plt
from pydeseq2.ds import DeseqStats
from single_cell.functions import plot_volcano_df
from pydeseq2.dds import DeseqDataSet, DefaultInference
import pandas as pd
import numpy as np


# Read file
adata = sc.read_h5ad('data/sc_dataset.h5ad')

# Get pseudo-bulk profile
pdata = dc.get_pseudobulk(
    adata,
    sample_col='batch',
    groups_col='lowres',
    layer='counts',
    mode='sum',
    min_cells=10,
    min_counts=1000,
)

tumor_pdata = pdata[pdata.obs['lowres'] == 'Tumor'].copy()

dc.plot_psbulk_samples(tumor_pdata, groupby=['batch', 'lowres'], figsize=(12, 4))
plt.show()
dc.plot_filter_by_expr(tumor_pdata, group='treatment', min_count=10, min_total_count=200)
plt.show()

genes = dc.filter_by_expr(tumor_pdata, group='treatment', min_count=10, min_total_count=200)
tumor_pdata = tumor_pdata[:, genes].copy()

# Run pydeseq2
inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    adata=tumor_pdata,
    design_factors=["treatment"],
    refit_cooks=True,
    inference=inference,
)

dds.deseq2()
stat_res = DeseqStats(dds, contrast=["treatment", "treated", "non-treated"],
                      inference=inference, independent_filter=True)
stat_res.summary()

results_df = stat_res.results_df
results_df.to_csv('data/single_cell/residual_vs_primary_tumor_dea.csv')

results_df = pd.read_csv('data/single_cell/residual_vs_primary_tumor_dea.csv', index_col=0)

# Plot volcano
scale = 1  # 0.85
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(3*scale, 3.2*scale))
plot_volcano_df(results_df, x='log2FoldChange', y='padj', top=10, ax=ax,
                   color_neg='#8b33ff', color_pos='#ff3363', color_null='#d3d3d3', s=4, sign_limit=None)
scatter = ax.collections[0]
scatter.set_rasterized(True)
ax.set_ylabel('-log10(p-value)')
ax.set_xlabel('log2FC')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.axhline(-np.log10(0.05), color='r', linestyle='--', alpha=0.5)
ax.grid('xy', c='lightgray')
ax.set_axisbelow(True)
ax.set_title(f'Pseudo-bulk DEA\n<--Primary | Residual-->')
plt.tight_layout()
plt.savefig('data/primary_residual_sc_volcano.svg', dpi=300)
plt.show()
