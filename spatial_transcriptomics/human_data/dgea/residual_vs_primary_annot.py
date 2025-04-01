from func.preprocessing.plotting import spatial_plot
import os
import chrysalis as ch
import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
import decoupler as dc
import matplotlib.pyplot as plt
from pydeseq2.ds import DeseqStats
from pydeseq2.dds import DeseqDataSet, DefaultInference
import adjustText as at


def plot_volcano_df(data, x, y, top=5, sign_thr=0.05, lFCs_thr=0.5, sign_limit=None, lFCs_limit=None,
                    color_pos='#D62728',color_neg='#1F77B4', color_null='gray', figsize=(7, 5),
                    dpi=100, ax=None, return_fig=False, s=1):

    def filter_limits(df, sign_limit=None, lFCs_limit=None):
        # Define limits if not defined
        if sign_limit is None:
            sign_limit = np.inf
        if lFCs_limit is None:
            lFCs_limit = np.inf
        # Filter by absolute value limits
        msk_sign = df['pvals'] < np.abs(sign_limit)
        msk_lFCs = np.abs(df['logFCs']) < np.abs(lFCs_limit)
        df = df.loc[msk_sign & msk_lFCs]
        return df

    # Transform sign_thr
    sign_thr = -np.log10(sign_thr)

    # Extract df
    df = data.copy()
    df['logFCs'] = df[x]
    df['pvals'] = -np.log10(df[y])

    # Filter by limits
    df = filter_limits(df, sign_limit=sign_limit, lFCs_limit=lFCs_limit)
    # Define color by up or down regulation and significance
    df['weight'] = color_null
    up_msk = (df['logFCs'] >= lFCs_thr) & (df['pvals'] >= sign_thr)
    dw_msk = (df['logFCs'] <= -lFCs_thr) & (df['pvals'] >= sign_thr)
    df.loc[up_msk, 'weight'] = color_pos
    df.loc[dw_msk, 'weight'] = color_neg
    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    df.plot.scatter(x='logFCs', y='pvals', c='weight', sharex=False, ax=ax, s=s)
    ax.set_axisbelow(True)
    # Draw sign lines
    # ax.axhline(y=sign_thr, linestyle='--', color="grey")
    # ax.axvline(x=lFCs_thr, linestyle='--', color="grey")
    # ax.axvline(x=-lFCs_thr, linestyle='--', color="grey")
    ax.axvline(x=0, linestyle='--', color="grey")
    # Plot top sign features
    # signs = df[up_msk | dw_msk].sort_values('pvals', ascending=False)
    # signs = signs.iloc[:top]

    up_signs = df[up_msk].sort_values('pvals', ascending=False)
    dw_signs = df[dw_msk].sort_values('pvals', ascending=False)
    up_signs = up_signs.iloc[:top]
    dw_signs = dw_signs.iloc[:top]
    signs = pd.concat([up_signs, dw_signs], axis=0)
    # Add labels
    ax.set_ylabel('-log10(pvals)')
    texts = []
    for x, y, s in zip(signs['logFCs'], signs['pvals'], signs.index):
        texts.append(ax.text(x, y, s))
    if len(texts) > 0:
        at.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ax=ax)
    if return_fig:
        return fig


output_folder = 'data/human_samples'

# read sample

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]
adata = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')

meta_df = pd.read_csv('data/meta_df_human_filled.csv', index_col=0)
scr_adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')
sample_names = adata.obs['ch_sample_id'].cat.categories

adata.obsm['chr_aa'] = scr_adata.obsm['chr_aa']

adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()

#%%
# select spots with high tissue compartments scores

label_1 = 'residual'
label_2 = 'primary'

diffexp_folder = output_folder + f'/diffexp/{label_1}_vs_{label_2}_human_annnot_based/'
os.makedirs(diffexp_folder, exist_ok=True)

spatial_plot(adata, 2, 3, 'annotations', share_range=False)
plt.show()


#%%
# pseudobulk

adata = adata[~adata.obs['annotations'].isin(['Stroma', 'Other'])].copy()
adata.obs['diffexp'] = adata.obs['condition']
adata = adata[adata.obs['diffexp'] != 'None']
adata.obs['sample'] = adata.obs['sample_id'].astype(str)

pdata = dc.get_pseudobulk(adata, sample_col='sample', groups_col='diffexp', use_raw=True,
                          mode='sum', min_cells=0, min_counts=0)

dc.plot_psbulk_samples(pdata, groupby=['slide', 'diffexp'], figsize=(11, 5))
plt.show()

pp_pdata = pdata.copy()
sc.pp.normalize_total(pp_pdata, target_sum=1e6)
sc.pp.log1p(pp_pdata)
# sc.pp.scale(pp_pdata, max_value=10)
sc.tl.pca(pp_pdata, n_comps=4)
sc.pl.pca(pp_pdata, color=['batch', 'slide', 'condition',
                           'diffexp'],
          ncols=2, show=False, size=300)
plt.show()

dc.get_metadata_associations(
    pp_pdata,
    # metadata columns to associate to PCs
    obs_keys = ['batch', 'slide', 'condition',
                'diffexp', 'psbulk_n_cells', 'psbulk_counts'],
    obsm_key='X_pca',  # where the PCs are stored
    uns_key='pca_anova',  # where the results are stored
    inplace=True
)
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
dc.plot_associations(
    pp_pdata,
    uns_key='pca_anova',  # summary statistics from the anova tests
    obsm_key='X_pca',  # where the PCs are stored
    stat_col='p_adj',  # which summary statistic to plot
    obs_annotation_cols = ['batch', 'slide', 'condition',
                'diffexp'], # which sample annotations to plot
    titles=['Adjusted p-values from ANOVA', 'Principle component scores']
)
plt.subplots_adjust(bottom=0.15, right=0.80)
plt.show()

dc.plot_filter_by_expr(pdata, group='diffexp', min_count=200, min_total_count=1500)
plt.show()

#%%
genes = dc.filter_by_expr(pdata, group='diffexp', min_count=200, min_total_count=1500)
pdata = pdata[:, genes].copy()

# deseq2
# modeling
# dfactors = ['batch', 'slide', 'condition', 'treatment', 'ancestral-tumor', 'diffexp']
dfactors = ['batch', 'diffexp']
dds = DeseqDataSet(adata=pdata, design_factors=dfactors, refit_cooks=True)

#%%

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    adata=pdata,
    design_factors=['batch', 'diffexp'],
    refit_cooks=True,
    inference=inference,
)

dds.deseq2()
stat_res = DeseqStats(dds, contrast=["diffexp", "residual-tumor", "primary-tumor"],
                      inference=inference, independent_filter=True)
stat_res.summary()

results_df = stat_res.results_df
results_df.to_csv(output_folder + f'/diffexp/{label_1}_vs_{label_2}_human_annnot_based/residual_vs_primary_tumor_dea.csv')

scale = 0.85  # 0.85
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(3*scale, 3.2*scale))
plot_volcano_df(results_df, x='log2FoldChange', y='padj', top=10, ax=ax,
                   color_neg='#eda6a9', color_pos='#a07eb4', s=4, sign_limit=None)
scatter = ax.collections[0]
scatter.set_rasterized(True)
# ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('-log10(p-value)')
ax.set_xlabel('logFC')
ax.set_title(f'Pseudo-bulk DEA\n<--Primary | Residual-->')
plt.tight_layout()
plt.savefig('figs/manuscript/fig4/primary_residual_volcano.svg', dpi=300)
plt.show()

sc.pl.violin(adata, 'FAM3B', groupby="condition", use_raw=False)

#%%
