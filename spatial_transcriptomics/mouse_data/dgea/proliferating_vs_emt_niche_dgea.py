import numpy as np
import scanpy as sc
import pandas as pd
import decoupler as dc
import matplotlib.pyplot as plt
from pydeseq2.ds import DeseqStats
from pydeseq2.dds import DeseqDataSet
from spatial_transcriptomics.functions import spatial_plot


def make_obs_categories(adata, obs_cols):
    assert len(obs_cols) >= 2
    # convert strings back to bools
    df = adata.obs[obs_cols].applymap(lambda x: True if x == 'True' else False)
    assert df.values.sum() <= len(df)
    # find col with True value
    out = df.idxmax(axis=1)
    # set default value to None
    out[df.sum(axis=1) == 0] = 'None'
    return out

def process_string(input_string):
    processed_string = input_string.replace('_', ' ')
    words = processed_string.split()
    if len(words) > 1:
        words = words[1:]
    result = ' '.join(words)
    return result


#%% Pseudobulk DGEA between proliferating and EMT spots selected based on a threshold
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony_with_raw.h5ad')

# select spots with high tissue compartments scores
threshold = 0.80
show_plot = False

# EMT
emt = adata.obsm['chr_aa'][:, 2] + adata.obsm['chr_aa'][:, 4]
positive = emt > threshold
print(np.sum(positive))
adata.obs['EMT'] = positive.astype(str)
if show_plot:
    spatial_plot(adata, 5, 5, 'EMT', share_range=False)
    plt.show()

# proliferating
prolif = adata.obsm['chr_aa'][:, 5]
positive = prolif > threshold
print(np.sum(positive))
adata.obs['proliferating'] = positive.astype(str)
if show_plot:
    spatial_plot(adata, 5, 5, 'proliferating', share_range=False)
    plt.show()

#%%
# create new column for compartments of interest
cats = make_obs_categories(adata, ['EMT', 'proliferating'])

adata.obs['diffexp'] = cats
adata = adata[adata.obs['diffexp'] != 'None']
adata.obs['sample'] = adata.obs['sample'].astype(str)

pdata = dc.get_pseudobulk(adata, sample_col='sample', groups_col='diffexp',  use_raw=True,
                          mode='sum', min_cells=0, min_counts=0)

if show_plot:
    dc.plot_psbulk_samples(pdata, groupby=['slide', 'diffexp'], figsize=(11, 5))
    plt.show()

pp_pdata = pdata.copy()
sc.pp.normalize_total(pp_pdata, target_sum=1e6)
sc.pp.log1p(pp_pdata)
# sc.pp.scale(pp_pdata, max_value=10)
sc.tl.pca(pp_pdata, n_comps=10)
if show_plot:
    sc.pl.pca(pp_pdata, color=['batch', 'slide', 'condition', 'treatment', 'elapsed_time',
                               'ancestral_tumor', 'diffexp'],
              ncols=2, show=False, size=300)
    plt.show()

dc.get_metadata_associations(
    pp_pdata,
    # metadata columns to associate to PCs
    obs_keys = ['batch', 'slide', 'condition', 'treatment', 'elapsed_time', 'ancestral_tumor',
                'diffexp', 'psbulk_n_cells', 'psbulk_counts'],
    obsm_key='X_pca',  # where the PCs are stored
    uns_key='pca_anova',  # where the results are stored
    inplace=True
)

if show_plot:
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    dc.plot_associations(
        pp_pdata,
        uns_key='pca_anova',  # summary statistics from the anova tests
        obsm_key='X_pca',  # where the PCs are stored
        stat_col='p_adj',  # which summary statistic to plot
        obs_annotation_cols = ['batch', 'slide', 'condition', 'treatment', 'elapsed_time', 'ancestral_tumor',
                    'diffexp'], # which sample annotations to plot
        titles=['Adjusted p-values from ANOVA', 'Principle component scores']
    )
    plt.subplots_adjust(bottom=0.15, right=0.80)
    plt.show()

if show_plot:
    dc.plot_filter_by_expr(pdata, group='diffexp', min_count=10, min_total_count=200)
    plt.show()

#%%
genes = dc.filter_by_expr(pdata, group='diffexp', min_count=10, min_total_count=200)
pdata = pdata[:, genes].copy()

dfactors = ['batch', 'condition', 'diffexp']
dds = DeseqDataSet(adata=pdata, design_factors=dfactors, refit_cooks=True)

dds.deseq2()
stat_res = DeseqStats(dds, contrast=['diffexp', 'proliferating', 'EMT'])

# invert the compartments in the design matrix, seems like the 'contrast' method doesn't work as intended
stat_res.design_matrix['diffexp_proliferating_vs_EMT'] = (
    np.abs(stat_res.design_matrix['diffexp_proliferating_vs_EMT'] - 1))
stat_res.summary()

# lfc shrinking
stat_res.lfc_shrink(coeff='diffexp_proliferating_vs_EMT')
results_df = stat_res.results_df

# stat column is also reversed
results_df['stat'] = results_df['stat'] * -1

if show_plot:
    label_1 = 'EMT'
    label_2 = 'proliferating'

    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    dc.plot_volcano_df(results_df, x='log2FoldChange', y='padj', top=20, ax=ax,
                       color_neg='#4e68d8', color_pos='#d95847')
    scatter = ax.collections[0]
    scatter.set_rasterized(True)
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('-log10(p-value)')
    ax.set_xlabel('logFC')
    ax.set_title(f'{label_1} vs. {label_2} tumor')
    fig.savefig(f'figs/manuscript/fig3/{label_1}_{label_2}_volcano.svg')
    plt.show()

    results_df['-log10padj'] = np.log10(results_df['padj']) * -1
    results_df.to_csv('figs/diffexp/emt_prolif/ '+ 'results.csv')

#%%
# functional analysis
results_df = pd.read_csv('figs/diffexp/emt_prolif/ '+ 'results.csv', index_col=0)

mat = results_df[['stat']].T.rename(index={'stat': 'EMT'})

# transcription factor
tri = pd.read_csv('data/decoupler/tri_mouse_tfs.csv', index_col=0)
tf_acts, tf_pvals = dc.run_ulm(mat=mat, net=tri)
fig = dc.plot_barplot(tf_acts, 'EMT', top=25, vertical=True, figsize=(5, 5), return_fig=True)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].grid(axis='x', linestyle='-', linewidth='0.5', color='grey')
fig.axes[0].axvline(0, color='black')
fig.axes[0].set_axisbelow(True)
fig.axes[0].set_title('Transcription factor activity')
cbar = fig.axes[-1]
cbar.set_frame_on(False)
fig.savefig(f'figs/manuscript/fig3/prolif_emt_tf.svg')
plt.show()

# tf scatters
logFCs = results_df[['log2FoldChange']].T.rename(index={'log2FoldChange': 'EMT'})
pvals = results_df[['padj']].T.rename(index={'padj': 'EMT'})

top_tfs = np.abs(tf_acts.T)
top_tfs = top_tfs.sort_values(by='EMT', ascending=False)
top_tfs = top_tfs.iloc[:25, 0]

for c in top_tfs.index:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    dc.plot_volcano(logFCs, pvals, 'EMT', name=c, net=tri, top=10, sign_thr=0.05, lFCs_thr=0.5,
                    ax=ax)
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_title(c)
    fig.savefig(f'figs/diffexp/emt_prolif/tf_scatter/{c}.png')
    plt.clf()
    plt.close()

# progeny
progeny = pd.read_csv('data/decoupler/progeny_mouse_500.csv', index_col=0)
pathway_acts, pathway_pvals = dc.run_mlm(mat=mat, net=progeny)
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
fig = dc.plot_barplot(pathway_acts, 'EMT', top=25, vertical=False, return_fig=True, figsize=(5, 5))
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
fig.axes[0].axhline(0, color='black')
fig.axes[0].set_axisbelow(True)
fig.axes[0].set_title('Pathway activity')
cbar = fig.axes[-1]
cbar.set_frame_on(False)
plt.tight_layout()
fig.savefig(f'figs/manuscript/fig3/prolif_emt_progeny.svg')
plt.show()


for c in pathway_acts.columns:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig = dc.plot_targets(results_df, stat='stat', source_name=c, net=progeny, top=15, return_fig=True)
    fig.savefig(f'figs/diffexp/emt_prolif/progeny_scatter/{c}.png')
    plt.clf()
    plt.close()

# hallmarks
gene_sets_df = pd.read_csv(f'msigdb_hallmark_mouse.csv', index_col=0)

gene_sets_df['geneset'] = gene_sets_df['geneset'].apply(process_string)

# upregulated
top_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] > 0.5)]
enr_pvals = dc.get_ora_df(df=top_genes, net=gene_sets_df, source='geneset', target='genesymbol')
enr_pvals = enr_pvals.sort_values(by='Combined score', ascending=False)
plt.rcParams['svg.fonttype'] = 'none'
fig = dc.plot_dotplot(enr_pvals, x='Combined score', y='Term', s='Odds ratio', c='FDR p-value',
                      scale=0.4, figsize=(6, 9), return_fig=True, cmap='rocket')
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
# fig.axes[0].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
fig.axes[0].set_axisbelow(True)
fig.axes[0].set_title('Upregulated terms')
cbar = fig.axes[-1]
cbar.set_frame_on(False)
plt.tight_layout()
fig.savefig(f'figs/manuscript/fig3/prolif_emt_upreg.svg')
plt.show()

for c in enr_pvals['Term'][:5]:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig, _ = dc.plot_running_score(results_df, stat='stat', net=gene_sets_df, source='geneset', target='genesymbol',
                                   set_name=c, return_fig=True)
    fig.savefig(f'figs/manuscript/fig3/UP_{c}.svg')
    plt.show()

# downregulated
top_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] < 0.5)]
enr_pvals = dc.get_ora_df(df=top_genes, net=gene_sets_df, source='geneset', target='genesymbol')
enr_pvals = enr_pvals.sort_values(by='Combined score', ascending=False)
plt.rcParams['svg.fonttype'] = 'none'
fig = dc.plot_dotplot(enr_pvals, x='Combined score', y='Term', s='Odds ratio', c='FDR p-value',
                      scale=0.2, figsize=(6, 9), return_fig=True, cmap='rocket')
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
# fig.axes[0].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
fig.axes[0].set_axisbelow(True)
fig.axes[0].set_title('Downregulated terms')
cbar = fig.axes[-1]
cbar.set_frame_on(False)
plt.tight_layout()
fig.savefig(f'figs/manuscript/fig3/prolif_emt_downreg.svg')
plt.show()

for c in enr_pvals['Term'][:5]:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig, _ = dc.plot_running_score(results_df, stat='stat', net=gene_sets_df, source='geneset', target='genesymbol',
                                   set_name=c, return_fig=True)
    fig.savefig(f'figs/manuscript/fig3/DOWN_{c}.svg')
    plt.show()
