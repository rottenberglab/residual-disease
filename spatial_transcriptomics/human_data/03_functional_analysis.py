import os
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from glob import glob
import decoupler as dc
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list


output_folder = 'data/human_samples'

# read samples
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

#%%
# PROGENY pathway activites
progeny = pd.read_csv('data/decoupler/progeny_human_500.csv', index_col=0)
# run model
dc.run_mlm(mat=adata, net=progeny, source='source', target='target', weight='weight', verbose=True, use_raw=False)

# heatmap per sample
sample_col = 'ch_sample_id'
sample_names = adata.obs[sample_col].cat.categories

pathways_matrix = np.zeros((len(np.unique(adata.obs[sample_col])), len(np.unique(progeny['source']))))
for idx, s in enumerate(sample_names):
    ad = adata[adata.obs[sample_col] == s].copy()
    pathways = ad.obsm['mlm_estimate']
    ad.obs[pathways.columns] = pathways
    pathways_matrix[idx] = pathways.mean(axis=0)

pathways_df = pd.DataFrame(data=pathways_matrix,
                           index=sample_names,
                           columns=pathways.columns)

z = linkage(pathways_df, method='ward')
order = leaves_list(z)
pathways_df = pathways_df.iloc[order, :]

z = linkage(pathways_df.T, method='ward')
order = leaves_list(z)
pathways_df = pathways_df.iloc[:, order]

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(pathways_df, square=True, center=0,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True))
plt.title('Mean pathway\nactivity score')
plt.tight_layout()
plt.show()

#%%
# heatmap per condition
condition_col = 'condition'
conditions = np.unique(adata.obs[condition_col])

pathways_matrix = np.zeros((len(conditions), len(np.unique(progeny['source']))))
for idx, s in enumerate(conditions):
    ad = adata[adata.obs[condition_col] == s].copy()
    pathways = ad.obsm['mlm_estimate']
    ad.obs[pathways.columns] = pathways
    pathways_matrix[idx] = pathways.mean(axis=0)

pathways_df = pd.DataFrame(data=pathways_matrix,
                           index=conditions,
                           columns=pathways.columns)

z = linkage(pathways_df.T, method='ward')
order = leaves_list(z)
pathways_df = pathways_df.iloc[:, order]

fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))
sns.heatmap(pathways_df, square=True, center=0, cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True))
plt.title('Mean pathway\nactivity score')
plt.tight_layout()
plt.show()

#%%
# plot violins
acts = dc.get_acts(adata, obsm_key='mlm_estimate')
acts_df = acts.to_df()

condition_col = 'condition'
conditions = np.unique(adata.obs[condition_col])

rows = 3
cols = 5
with sns.axes_style("darkgrid", {"axes.facecolor": ".95"}):
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axs = axs.flatten()
    for a in axs:
        a.axis('off')
    for idx, pw in enumerate(pathways_df.columns):
            pw_df = pd.DataFrame(data=acts[:, pw].X, columns=[pw])
            pw_df[condition_col] = [x for x in list(acts.obs[condition_col])]

            axs[idx].axis('on')

            axs[idx].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
            axs[idx].set_axisbelow(True)
            sns.violinplot(data=pw_df, x=condition_col, y=pw, scale='width',
                           palette=['#92BD6D', '#E9C46B', '#F2A360', '#E66F4F', '#4BC9F1', '#4993F0', '#435FEE'],
                           ax=axs[idx])
            axs[idx].set_ylabel(None)
            axs[idx].set_title(f'{pw} activity', fontsize=14)
            axs[idx].set_xlabel(None)
            axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
            legend_labels = ['False', 'True']
            handles, _ = axs[idx].get_legend_handles_labels()
    fig.supylabel('Pathway activity score')
    plt.tight_layout()
    # plt.savefig('/mnt/c/Bern/Docs/chrysalis/figs/svgs/germ_center_violin.svg')
    plt.show()

#%%
# spatial plots
acts = dc.get_acts(adata, obsm_key='mlm_estimate')
acts_df = acts.to_df()

sample_col = 'ch_sample_id'
sample_names = adata.obs[sample_col].cat.categories
condition = {'primary_tumor': 'Primary tumor', 'residual_tumor': 'Residual tumor'}
os.makedirs(f'{output_folder}/progeny/spatial_plots/', exist_ok=True)

rows = 2  # number of samples
cols = 3

plt.rcParams['svg.fonttype'] = 'none'

sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)

for pw in pathways_df.columns:

    vmin = np.percentile(acts[:, pw].X, 0.2)
    vmax = np.percentile(acts[:, pw].X, 99.8)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, s in enumerate(sample_names):
        ad = acts[acts.obs[sample_col] == s].copy()
        sc.pl.spatial(ad, color=pw, size=1.5, alpha=1, library_id=s,
                      ax=ax[idx], show=False, cmap='Spectral_r',
                      vcenter=None, vmin=vmin, vmax=vmax, alpha_img=0.75)
        ax[idx].set_title(f'{s}\n{condition[ad.obs["condition"][0]]}\n')
        cbar = fig.axes[-1]
        cbar.set_frame_on(False)
    plt.suptitle(f'{pw} pathway', fontsize=20, y=0.99)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/progeny/spatial_plots/{pw}.png')
    plt.close()

#%%
# compartment correlations

acts = dc.get_acts(adata, obsm_key='mlm_estimate')
acts_df = acts.to_df()
compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

corr_m = pd.concat([compartment_df, acts_df], axis=1).corr()
corr_m = corr_m.drop(index=acts_df.columns, columns=compartment_df.columns)

corr_m.index = ['' + str(x) for x in range(compartments.shape[1])]

z = linkage(corr_m, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[order, :]

z = linkage(corr_m.T, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[:, order]

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.heatmap(corr_m.T, square=True, cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True), center=0)
plt.title('Pathway correlation')
plt.tight_layout()
plt.show()
