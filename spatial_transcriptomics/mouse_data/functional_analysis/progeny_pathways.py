import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import decoupler as dc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.functions import matrixplot


names = {'primary_tumor no_treatment na': 'Primary tumor',
         'relapsed_tumor TAC_regimen 30_days' : 'TAC 30dpt',
         'relapsed_tumor cisplatin_6mg/kg 30_days': 'Cisplatin 30dpt',
         'residual_tumor TAC_regimen 12_days': 'TAC 12dpt',
         'residual_tumor TAC_regimen 7_days': 'TAC 7dpt',
         'residual_tumor cisplatin_6mg/kg 12_days': 'Cisplatin 12dpt',
         'residual_tumor cisplatin_6mg/kg 7_days': 'Cisplatin 7dpt'}

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}

treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}

time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

# run progeny on top 500 genes
meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
progeny = pd.read_csv(f'data/decoupler/progeny_mouse_500.csv', index_col=0)

dc.run_mlm(mat=adata, net=progeny, source='source', target='target', weight='weight', verbose=True, use_raw=False)

#%%
# pathway activity mean per condition

meta_df_sub = meta_df[meta_df['sample_id'].isin(np.unique(adata.obs['sample_id']))]
cond = [x + ' ' + y + ' ' + z for x, y, z in zip(meta_df_sub['condition'], meta_df_sub['treatment'],
                                                           meta_df_sub['elapsed_time'])]

adata.obs['condition_cat'] = [x + ' ' + y + ' ' + z for x, y, z in zip(adata.obs['condition'], adata.obs['treatment'],
                                                           adata.obs['elapsed_time'])]


pathways_matrix = np.zeros((len(np.unique(adata.obs['condition_cat'])), len(np.unique(progeny['source']))))
for idx, s in enumerate(np.unique(adata.obs['condition_cat'])):
    ad = adata[adata.obs['condition_cat'] == s].copy()
    pathways = ad.obsm['mlm_estimate']
    ad.obs[pathways.columns] = pathways
    pathways_matrix[idx] = pathways.mean(axis=0)

sample_indexes = [names[x] for x in np.unique(adata.obs['condition_cat'])]

pathways_df = pd.DataFrame(data=pathways_matrix,
                           index=sample_indexes,
                           columns=pathways.columns)

order = [0, 4, 3, 1, 6, 5, 2]
pathways_df = pathways_df.iloc[order, :]

z = linkage(pathways_df.T, method='ward')
order = leaves_list(z)
pathways_df = pathways_df.iloc[:, order]

plt.rcParams['svg.fonttype'] = 'none'
sf = 0.6
matrixplot(pathways_df.T, figsize=(8.2*sf, 12*sf), flip=False, scaling=False, square=True,
            colorbar_shrink=0.15, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.1, cbar_label="Score", xlabel='Pathways',
            cmap=sns.diverging_palette(325, 145, l=60, s=80, center="dark", as_cmap=True),
            ylabel=None, rasterized=True, seed=87, reorder_obs=False,
            color_comps=False, adata=adata, xrot=90, ha='center')
# plt.savefig(f'figs/manuscript/fig3/pathway_activity_heatmap_main_v3.svg')
plt.show()

#%%
# Violin plots

acts = dc.get_acts(adata, obsm_key='mlm_estimate')

acts_df = acts.to_df()
adata.obsm['mlm_pvals'].to_csv(f'data/decoupler/functional_enrichment/pathway_activity_pvals.csv')

order =['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
        'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

rows = 3
cols = 5

fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, pw in enumerate(pathways_df.columns):
        pw_df = pd.DataFrame(data=acts[:, pw].X, columns=[pw])
        pw_df['condition_cat'] = [names[x] for x in list(acts.obs['condition_cat'])]

        axs[idx].axis('on')

        axs[idx].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
        axs[idx].set_axisbelow(True)
        sns.violinplot(data=pw_df, x='condition_cat', y=pw, scale='width',
                       palette=['#92BD6D', '#E9C46B', '#F2A360', '#E66F4F', '#4BC9F1', '#4993F0', '#435FEE'],
                       order=order, ax=axs[idx])
        axs[idx].set_ylabel(None)
        axs[idx].set_title(f'{pw} activity', fontsize=14)
        axs[idx].set_xlabel(None)
        axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        legend_labels = ['False', 'True']
        handles, _ = axs[idx].get_legend_handles_labels()
fig.supylabel('Pathway activity score')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/pathway_violin.svg')
plt.show()

#%%
# spatial plots

rows = 5
cols = 5

for pw in pathways_df.columns:

    vmin = np.percentile(acts[:, pw].X, 0.2)
    vmax = np.percentile(acts[:, pw].X, 99.8)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, s in enumerate(np.unique(acts.obs['sample_id'])):
        ad = acts[acts.obs['sample_id'] == s].copy()
        sc.pl.spatial(ad, color=pw, size=1.5, alpha=1, library_id=s,
                      ax=ax[idx], show=False, cmap='Spectral_r',
                      vcenter=0, vmin=vmin, vmax=vmax, alpha_img=0.75)
        ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}\n' + \
                          f'{treatment[ad.obs["treatment"][0]]}\n{time[ad.obs["elapsed_time"][0]]}')
        cbar = fig.axes[-1]
        cbar.set_frame_on(False)
    plt.suptitle(f'{pw} pathway', fontsize=20, y=0.99)
    plt.tight_layout()
    plt.savefig(f'figs/pathway_activities/{pw}.png')
    plt.close()

#%%
# spatial plots

rows = 5
cols = 5

sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'
for pw in pathways_df.columns:

    vmin = np.percentile(acts[:, pw].X, 0.2)
    vmax = np.percentile(acts[:, pw].X, 99.8)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, s in enumerate(np.unique(acts.obs['sample_id'])):
        ad = acts[acts.obs['sample_id'] == s].copy()
        sc.pl.spatial(ad, color=pw, size=1.5, alpha=1, library_id=s,
                      ax=ax[idx], show=False, cmap='Spectral_r',
                      vcenter=0, vmin=vmin, vmax=vmax, alpha_img=0.75)
        ax[idx].set_title(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}\n' + \
                          f'{treatment[ad.obs["treatment"][0]]}\n{time[ad.obs["elapsed_time"][0]]}')
        cbar = fig.axes[-1]
        cbar.set_frame_on(False)
    plt.suptitle(f'{pw} pathway', fontsize=20, y=0.99)
    plt.tight_layout()
    plt.savefig(f'figs/pathway_activities/{pw}.svg')
    plt.close()
