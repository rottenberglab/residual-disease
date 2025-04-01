import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from glob import glob
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.functions import spatial_plot, matrixplot


# Read AnnData and prepare ortholog table
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}

treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}

time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

names = {'primary_tumor no_treatment na': 'Primary tumor',
         'relapsed_tumor TAC_regimen 30_days' : 'TAC 30dpt',
         'relapsed_tumor cisplatin_6mg/kg 30_days': 'Cisplatin 30dpt',
         'residual_tumor TAC_regimen 12_days': 'TAC 12dpt',
         'residual_tumor TAC_regimen 7_days': 'TAC 7dpt',
         'residual_tumor cisplatin_6mg/kg 12_days': 'Cisplatin 12dpt',
         'residual_tumor cisplatin_6mg/kg 7_days': 'Cisplatin 7dpt'}

order =['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
        'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

adata.obs['condition_cat'] = [x + ' ' + y + ' ' + z for x, y, z in
                              zip(adata.obs['condition'],
                                  adata.obs['treatment'],
                                  adata.obs['elapsed_time'])]
adata.obs['condition_cat'] = adata.obs['condition_cat'].astype('category')

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}
marker_df = pd.read_csv('data/cell_cycle/cell_cycle_markers.csv')
for c in marker_df.columns:
    mouse_orths = []
    for v in marker_df[c].values:
        try:
            g = orthologs_dict[v]
        except:
            g = np.nan
        mouse_orths.append(g)
    marker_df[c] = mouse_orths

#%%
# cell cycle
def cell_cycle_score(adata, marker_df):
    for c in marker_df.columns:
        gene_list = marker_df[c]
        gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]
        sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'{c}_score')
    obs_names = [f'{c}_score' for c in marker_df.columns]
    adata.obs['mean_cc_score'] = adata.obs[obs_names].mean(axis=1)

cell_cycle_score(adata, marker_df)
adata.obs = adata.obs.rename(columns={'mean_cc_score': 'Cell cycle activity'})

#%%
# SASP
marker_df = pd.read_csv('data/sasp_mayo.csv')
mouse_orths = []
for v in marker_df['Gene(human)'].values:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)
marker_df['genes_mouse'] = mouse_orths
gene_list = marker_df['genes_mouse']
gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]
sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'SASP')

#%%
# Adaptive mutability
genes_pres = ['Mlh1', 'Msh2', 'Msh6', 'Pms2', 'Exo1', 'Brca1',
              'Brca2', 'Rad51', 'Pola1', 'Polg', 'Pole', 'Pole2']
genes_pres = adata.var_names[np.in1d(adata.var_names, genes_pres)]

sc.tl.score_genes(adata, gene_list=genes_pres, score_name=f'Adaptive mutability')

#%% optional part for plotting
# subset anndata
selected_samples = [ 17,  14,  15,  16,  19,  8,  21]
adata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_10', '20221010-3_11', '20221010-3_12',
                '20221010-4_14', '20221010-1_3', '20221010-4_16']
adata.obs['sample_id'] = adata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

spatial_plot(adata, 1, 7, 'Adaptive mutability', cmap='Spectral_r', sample_col='label',
             title=label_cats, suptitle='Adaptive mutability', alpha_img=0.5, vcenter=0)
plt.savefig(f'figs/manuscript/fig3/adaptive_mutability_spatial.svg')
plt.show()

#%%
# Matrixplots
obs_names = ['Cell cycle activity', 'SASP', 'Adaptive mutability']

pathways_matrix = np.zeros((len(np.unique(adata.obs['condition_cat'])), len(obs_names)))
for idx, s in enumerate(np.unique(adata.obs['condition_cat'])):
    ad = adata[adata.obs['condition_cat'] == s].copy()
    pathways = ad.obs[obs_names]
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
pathways_df = pathways_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

plt.rcParams['svg.fonttype'] = 'none'
sf = 0.32
matrixplot(pathways_df.T, figsize=(8.2*sf, 12*sf), flip=False, scaling=False, square=True,
            colorbar_shrink=0.45, colorbar_aspect=10, title=None,
            dendrogram_ratio=0.1, cbar_label="Z-scaled\nscore", xlabel='Cancer hallmarks',
            cmap=sns.diverging_palette(325, 145, l=60, s=80, center="dark", as_cmap=True),
            ylabel=None, rasterized=True, seed=87, reorder_obs=False,
            color_comps=False, adata=adata, xrot=90, ha='center')
# plt.savefig(f'figs/manuscript/fig3/cell_cycle_sasp_am.svg')
plt.show()

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

corr_m = pd.concat([compartment_df, pathways], axis=1).corr(method='pearson')
corr_m = corr_m.drop(index=pathways.columns, columns=compartment_df.columns)
corr_m = corr_m.rename(columns={'sasp': 'SASP'})
corr_m.index = [str(x) for x in corr_m.index]

corr_m.index = [x for x in range(13)]

selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]
plt.rcParams['svg.fonttype'] = 'none'
sf = 0.365
matrixplot(corr_m, figsize=(8.2*sf, 12*sf), flip=True, scaling=False, square=True,
            colorbar_shrink=0.45, colorbar_aspect=10, title=None,
            dendrogram_ratio=0.35, cbar_label="Score", ylabel='Cancer hallmark', comps=selected_comps,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            xlabel='Compartments', rasterized=True, seed=87, reorder_obs=True,
            color_comps=True, adata=adata, xrot=90, ha='center')
# plt.savefig(f'figs/manuscript/fig3/cc_sasp_am_corr.svg')
plt.show()

#%%
# Spatial plots (Fig.3c)

enrichments = glob('data/decoupler/functional_enrichment/*scores.csv')
enrich_df = pd.read_csv(enrichments[0], index_col=0)
for c in enrich_df.columns:
    adata.obs[c] = enrich_df[c]

# subset anndata
selected_samples = [17,  15,  19,  16,  21]
ad = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_11', '20221010-4_14', '20221010-3_12', '20221010-4_16']
ad.obs['sample_id'] = ad.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = ad.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

for k in list(ad.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        ad.uns['spatial'][sample_name_df[k]] = ad.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in ad.obs['sample_id'].cat.categories]
ad.obs['label'] = ad.obs['label'].cat.reorder_categories(label_cats, ordered=True)


for c in ['Cell cycle activity', 'JAK-STAT', 'SASP', 'MAPK', 'Trail']:
    spatial_plot(ad, 5, 1, c,
                 cmap=sns.color_palette("Spectral_r", as_cmap=True), sample_col='label', s=15,
                 title=None, suptitle=f'{c} proportion', alpha_img=0.0, colorbar_label='Score',
                 colorbar_aspect=5, colorbar_shrink=0.15, hspace=-0.7, subplot_size=4, alpha_blend=False,
                 x0=0.3, suptitle_fontsize=15, vcenter=0)

    plt.tight_layout()
    # plt.savefig(f'figs/manuscript/fig3/{c}_5_vertical_r.svg')
    plt.show()
    plt.close()

spatial_plot(ad, 5, 1, c,
             cmap=sns.color_palette("Spectral_r", as_cmap=True), sample_col='label', s=15,
             title=None, suptitle=f'{c} proportion', alpha_img=1.0, colorbar_label='Score',
             colorbar_aspect=5, colorbar_shrink=0.15, hspace=-0.7, subplot_size=4, alpha_blend=False,
             x0=0.3, suptitle_fontsize=15, vcenter=0, alpha=0)
plt.tight_layout()
# plt.savefig(f'figs/manuscript/fig3/he_5_vertical_r.svg')
plt.show()


#%%

# read sample
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

# subset anndata
selected_samples = [ 17,  14,  15,  16,  19,  8,  21]
adata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_10', '20221010-3_11', '20221010-3_12',
                '20221010-4_14', '20221010-1_3', '20221010-4_16']
adata.obs['sample_id'] = adata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

cell_cycle_score(adata, marker_df)

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

spatial_plot(adata, 1, 7, 'mean_cc_score', cmap='Spectral_r', sample_col='label',
             title=label_cats, suptitle='Cell cycle activity', alpha_img=0.5, vcenter=0)
plt.savefig(f'figs/manuscript/fig3/cell_cycle_spatial.svg')
plt.show()

#%%
# read sample
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

cell_cycle_score(adata, marker_df)

# subset anndata
selected_samples = [20]
ad = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

spatial_plot(ad, 1, 1, 'mean_cc_score', cmap='Spectral_r', sample_col='label',
             title=['Mean CC score'], alpha_img=0.5, vcenter=0)
plt.savefig(f'figs/manuscript/fig3/cell_cycle_spatial_main.svg')
plt.show()

#%%
# Violin plots cell cycle

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

rows = 2
cols = 3

columns_dict = {'G1/S_score': 'G1/S score', 'S_score': 'S score', 'G2/M_score': 'G2/M score',
                'M_score': 'M score', 'M/G1_score': 'M/G1 score', 'mean_cc_score': 'Mean CC score'}

columns = [x for x in adata.obs.columns if '_score' in x]

plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), sharex=True)
axs = axs.flatten()
# for a in axs:
#     a.axis('off')
for idx, pw in enumerate(adata.obs[columns]):
        pw_df = pd.DataFrame(data=adata.obs[pw], columns=[pw])
        pw_df['condition_cat'] = [names[x] for x in list(adata.obs['condition_cat'])]

        axs[idx].axis('on')

        axs[idx].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
        axs[idx].set_axisbelow(True)
        sns.violinplot(data=pw_df, x='condition_cat', y=pw, scale='width',
                       palette=['#92BD6D', '#E9C46B', '#F2A360', '#E66F4F', '#4BC9F1', '#4993F0', '#435FEE'],
                       order=order, ax=axs[idx])
        # sns.stripplot(data=pw_df, x='condition_cat', y=pw, jitter=True,
        #                order=order, color='black', size=2, alpha=.1)
        axs[idx].set_ylabel(None)
        axs[idx].set_title(columns_dict[pw], fontsize=14)
        axs[idx].set_xlabel(None)
        if idx >= cols * (rows - 1):  # check if it's in the last row
            axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        legend_labels = ['False', 'True']
        handles, _ = axs[idx].get_legend_handles_labels()
fig.supylabel('Gene set score')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/cell_cycle_violin.svg')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(3 * 1, 3.5 * 1))
pw = 'mean_cc_score'
pw_df = pd.DataFrame(data=adata.obs[pw], columns=[pw])
pw_df['condition_cat'] = [names[x] for x in list(adata.obs['condition_cat'])]

ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
sns.violinplot(data=pw_df, x='condition_cat', y=pw, scale='width',
               palette=['#92BD6D', '#E9C46B', '#F2A360', '#E66F4F', '#4BC9F1', '#4993F0', '#435FEE'],
               order=order,)
# sns.stripplot(data=pw_df, x='condition_cat', y=pw, jitter=True,
#                order=order, color='black', size=2, alpha=.1)
ax.set_ylabel('Gene set score')
ax.set_title(columns_dict[pw], fontsize=14)
ax.set_xlabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_labels = ['False', 'True']
handles, _ = ax.get_legend_handles_labels()
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/mean_cell_cycle_violin.svg')
plt.show()

#%%
# Heatmap of cell cycle signatures vs cellular niches

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

hexcodes = ch.utils.get_hexcodes(None, 13, 87, len(adata))

columns = [x for x in adata.obs.columns if '_score' in x]

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

corr_m = pd.concat([compartment_df, adata.obs[columns]], axis=1).corr()
corr_m = corr_m.drop(index=columns, columns=compartment_df.columns)
corr_m.index = ['' + str(x) for x in range(13)]

z = linkage(corr_m, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[order, :]
hexcodes = [hexcodes[i] for i in order]

z = linkage(corr_m.T, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[:, order]

corr_m.columns = [columns_dict[x] for x in corr_m.columns]

fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sns.heatmap(corr_m.T, square=True, center=0, rasterized=True,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True), ax=ax)
plt.title('Tissue compartment \ncell cycle activity')
for idx, t in enumerate(ax.get_xticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/cell_cycle_heatmap.svg')
plt.show()

#%%
# SASP spatial plots
# subset anndata
selected_samples = [ 17,  14,  15,  16,  19,  8,  21]
adata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_10', '20221010-3_11', '20221010-3_12',
                '20221010-4_14', '20221010-1_3', '20221010-4_16']
adata.obs['sample_id'] = adata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = adata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(adata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        adata.uns['spatial'][sample_name_df[k]] = adata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in adata.obs['sample_id'].cat.categories]
adata.obs['label'] = adata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'sasp')

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

spatial_plot(adata, 1, 7, 'sasp', cmap='Spectral_r', sample_col='label',
             title=label_cats, suptitle='SASP', alpha_img=0.5, vcenter=0)
plt.savefig(f'figs/manuscript/fig3/sasp_spatial_7.svg')
plt.show()
