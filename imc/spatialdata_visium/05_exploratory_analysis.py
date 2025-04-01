import anndata
import numpy as np
import scanpy as sc
import pandas as pd
from glob import glob
import seaborn as sns
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from imc.imc_functions import spatial_plot,  spatial_plot_v2, matrixplot
from scipy.cluster.hierarchy import linkage, leaves_list


def gene_correltations(adata, key):
    correlation_dict = {}  # Initialize inside function to match original behavior
    exp_data = adata.to_df()

    for col in exp_data.columns:
        pearson_corr, pearson_pval = pearsonr(adata.obs[key], exp_data[col].values)
        spearman_corr, spearman_pval = spearmanr(adata.obs[key], exp_data[col].values)

        correlation_dict[col] = {
            'pearson': pearson_corr,
            'pearson_p': pearson_pval,
            'spearman': spearman_corr,
            'spearman_p': spearman_pval
        }
    return correlation_dict


#%%
# get new imc adatas

ad_files = glob('data/adatas/sdata/*.h5ad')
adatas = []
for adf in ad_files:
    adf_name = adf.split('/')[-1][:-5]
    ad = sc.read_h5ad(adf)
    ad.obs_names = [x + f'-{adf_name}' for x in ad.obs_names]
    adatas.append(ad)
imc_adata = anndata.concat(adatas, index_unique='-', uns_merge='unique', merge='first', join='outer')
imc_adata.obs_names = [x[:-2] for x in imc_adata.obs_names]

tumor_cell_types = ['Tumor mod. Pt', 'Tumor low Pt', 'Tumor high Pt', 'Tumor necrotic margin']
imc_adata.obs[tumor_cell_types] = imc_adata.obsm['cell_type_params'][tumor_cell_types]
cell_types = imc_adata.obsm['cell_type_params'].columns
imc_adata.obs[cell_types] = imc_adata.obsm['cell_type_params'][cell_types]
# read the old ones
output_folder = '/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/imc_samples'
adata = sc.read_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')

# add variables to adata
obs_keys = ['n_cell', 'sum_pt', 'mean_pt', 'tumor_sum_pt', 'tumor_mean_pt', 'n_tumor_cell'] + tumor_cell_types
adata.obs[obs_keys] = imc_adata.obs[obs_keys]
adata.obs[cell_types] = imc_adata.obsm['cell_type_params'][cell_types]
adata.obs['log1p_sum_pt'] = np.log1p(adata.obs['sum_pt'])
adata = adata[~adata.obs['n_cell'].isna()]
adata.obs['Tumor necrotic margin'] = adata.obs['Tumor necrotic margin'].fillna(0)
adata.obs['sample_id'] = adata.obs['sample_id'].cat.set_categories(['20230607-2_6', '20230607-1_3',
                                                                    '20230607-2_8', '20230607-2_5'])

# subset adata to IMC spots
title_dict = {'cisplatin_6mg/kg_4_hours': 'Cisplatin 6mg/kg 4hpt',
              'cisplatin_6mg/kg_24_hours': 'Cisplatin 6mg/kg 24hpt',
              'cisplatin_6mg/kg_12_days': 'Cisplatin 6mg/kg 12dpt',
              'no_treatment_na': 'Primary tumor',}


#%%
# cell types and pt corr

cell_type_dict = {
    'APC': 'Antigen-presenting cell',
    'Bcell': 'B cell',
    'CD4T': 'T cell CD4+',
    'CD8T': 'T cell CD8+',
    'CHMacrophage': 'Macrophage CH',
    'Endothelialcell': 'Endothelial cell',
    'Fibroblast': 'Fibroblast',
    'InflMacrophage': 'Macrophage Inf.',
    'NKcell': 'NK cell',
    'Plasmacell': 'Plasma cell',
    'Spp1Macrophage': 'Macrophage SPP1+',
    'TB-EMT': 'Tumor basal-EMT',
    'TBasal1': 'Tumor basal',
    'TBasal2': 'Tumor basal hypoxic',
    'TFlike': 'Tumor fibroblast-like',
    'TL-Alv': 'Tumor luminal-alveolar',
    'TLA-EMT': 'Tumor luminal-alveolar-EMT',
    'TMlike': 'Tumor macrophage-like',
    'TProliferating': 'Tumor proliferating',
    'Tcell' : 'T cell',
    'Treg': 'T cell regulatory',
    'cDC': 'Dendritic cell',
    'pDC': 'Dendritic cell plasmacytoid',
}

adata.obs['treatment_time'] = [title_dict[f'{x}_{y}'] for x, y
                               in zip(list(adata.obs['treatment']), list(adata.obs['elapsed_time']))]

sample_dict = {}
for c in np.unique(adata.obs['treatment_time']):
    ad = adata[adata.obs['treatment_time'] == c]
    # cell type correlations with platinum
    celltypes_df = ad.obsm['cell2loc'].copy()
    celltypes_df.columns = [cell_type_dict[x] for x in celltypes_df.columns]

    corr_matrix = np.empty((len(celltypes_df.columns), 1))
    for i, col1 in enumerate(celltypes_df.columns):
        corr, _ = pearsonr(celltypes_df[col1], ad.obs['mean_pt'])
        corr_matrix[i] = corr
    sample_dict[c] = corr_matrix.flatten()

corrs_df = pd.DataFrame(sample_dict, index=celltypes_df.columns).T
corrs_df = corrs_df.reindex(['Primary tumor', 'Cisplatin 6mg/kg 4hpt',
                             'Cisplatin 6mg/kg 24hpt', 'Cisplatin 6mg/kg 12dpt'])


fig, ax = plt.subplots(1, 1, figsize=(8, 5.2))
sns.heatmap(corrs_df, square=True, center=0,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            rasterized=True, cbar_kws={'label': "Pearson's r", "shrink": 0.25})
plt.title('Correlation of Pt with cell types')
ax.set_ylabel(None)
plt.tight_layout()
# plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/cell_types_plat_hm.svg')
plt.show()

corrs_df = corrs_df.T
corrs_df = corrs_df[['Cisplatin 6mg/kg 4hpt', 'Cisplatin 6mg/kg 24hpt', 'Cisplatin 6mg/kg 12dpt', 'Primary tumor']]

plt.rcParams['svg.fonttype'] = 'none'
matrixplot(corrs_df, figsize=(8, 7), flip=True, scaling=False, square=True, reorder_comps=True, reorder_obs=False,
            colorbar_shrink=0.2, colorbar_aspect=10, cmap=sns.diverging_palette(267, 20, as_cmap=True, center="dark"),
            dendrogram_ratio=0.35, cbar_label="Pearson's r", xlabel='Cell types',
            title='Correlation of cell types\nwith tissue Pt content', color_comps=False,
            ylabel='Tissue sample', rasterized=True, seed=87, xrot=90)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/cell_type_pt_corr2.svg')
# plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/cell_type_pt_corr.png')
plt.show()

sample_dict = {}
for c in np.unique(adata.obs['treatment_time']):
    ad = adata[adata.obs['treatment_time'] == c]
    # cell type correlations with platinum
    celltypes_df = ad.obsm['cell2loc'].copy()
    celltypes_df.columns = [cell_type_dict[x] for x in celltypes_df.columns]
    tumor_cts = [x for x in celltypes_df.columns if 'Tumor' in x]
    celltypes_df = celltypes_df[tumor_cts]

    corr_matrix = np.empty((len(celltypes_df.columns), 1))
    for i, col1 in enumerate(celltypes_df.columns):
        corr, _ = pearsonr(celltypes_df[col1], ad.obs['tumor_mean_pt'])
        corr_matrix[i] = corr
    sample_dict[c] = corr_matrix.flatten()

corrs_df = pd.DataFrame(sample_dict, index=celltypes_df.columns).T
corrs_df = corrs_df.reindex(['Primary tumor', 'Cisplatin 6mg/kg 4hpt',
                             'Cisplatin 6mg/kg 24hpt', 'Cisplatin 6mg/kg 12dpt'])


fig, ax = plt.subplots(1, 1, figsize=(7, 3))
sns.heatmap(corrs_df, square=True, center=0,
            cmap=sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True),
            rasterized=True, cbar_kws={'label': "Pearson's r", "shrink": 0.5}, ax=ax)
plt.title('Correlation of Pt with cell types')
ax.set_ylabel(None)
plt.tight_layout()
plt.show()
# plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/cell_types_plat_hm.svg')

#%%
correlation_dict = {}
for f in np.unique(adata.obs['sample_id']):
    ad = adata[adata.obs['sample_id'] == f].copy()
    ad = ad[ad.obs['n_tumor_cell'] > 2]
    if len(ad) > 0:
        condition = ad.obs['treatment'][0] + '_' + ad.obs['elapsed_time'][0]
        condition = title_dict[condition]
        sc.pl.spatial(ad, color=['tumor_sum_pt', 'tumor_mean_pt', 'n_tumor_cell'], s=12, library_id=f)
        plt.show()
        # correlations
        correlation_dict = gene_correltations(ad, 'tumor_mean_pt')

correlation_df = pd.DataFrame({k: v['pearson'].values for k, v in correlation_dict.items()},
                              index=adata.var_names)

# spatial plots
spatial_plot_v2(adata, 2, 2, 'tumor_mean_pt',
             cmap='Reds', sample_col='sample_id', s=14,
             title=list(title_dict.values()), suptitle='Pt density in tumor cells',
                alpha_img=0.5, colorbar_label='mean Pt content',
             colorbar_aspect=5, colorbar_shrink=0.15, hspace=-0.7, subplot_size=4, alpha_blend=False,
             x0=0.3, suptitle_fontsize=15, share_range=True)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/visium_avg_pt_v3.svg')
plt.show()

spatial_plot(adata, 1, 4, 'tumor_mean_pt', cmap='mako_r', sample_col='sample_id', s=14,
             title=list(title_dict.values()), suptitle='Pt density in tumor cells (Pt / µm²)', alpha_img=0.5,
             share_range=True)
# plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/visium_avg_pt_v2.svg')
plt.show()

spatial_plot(adata, 1, 4, 'tumor_sum_pt', cmap='PuRd', sample_col='sample_id', s=14,
             title=list(title_dict.values()), suptitle='Summed Pt content in tumor cells', alpha_img=0.5,
             share_range=False)
# plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/visium_sum_pt.svg')
plt.show()

spatial_plot(adata, 1, 4, 'Tumor low Pt', cmap='mako', sample_col='sample_id', s=14,
             title=list(title_dict.values()), suptitle='Tumor low Pt', alpha_img=0.5,
             share_range=True)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/visium_low.svg')
plt.show()
spatial_plot(adata, 1, 4, 'Tumor mod. Pt', cmap='mako', sample_col='sample_id', s=14,
             title=list(title_dict.values()), suptitle='Tumor mod. Pt', alpha_img=0.5,
             share_range=True)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/visium_mod.svg')
plt.show()
spatial_plot(adata, 1, 4, 'Tumor high Pt', cmap='mako', sample_col='sample_id', s=14,
             title=list(title_dict.values()), suptitle='Tumor high Pt', alpha_img=0.5,
             share_range=True)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/visium_high.svg')
plt.show()
spatial_plot(adata, 1, 4, 'Tumor necrotic margin', cmap='mako', sample_col='sample_id', s=14,
             title=list(title_dict.values()), suptitle='Tumor necrotic margin', alpha_img=0.5,
             share_range=True)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/visium_margin.svg')
plt.show()

sns.pairplot(correlation_df, kind='reg', vars=list(title_dict.values()), corner=False, diag_kind='kde',
             height=2,
             plot_kws={'line_kws':{'color':'red'},
                       'scatter_kws': {'alpha': 0.5, 's': 1}})
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(8, 10))
# heatmap top correlations
heatmap_df = pd.DataFrame()
for c in list(title_dict.values()):
    sorted_df = correlation_df.sort_values(by=c, ascending=False)
    heatmap_df = heatmap_df.append(sorted_df.iloc[:10, :])
heatmap_df = heatmap_df[list(title_dict.values())]
plt.rcParams['svg.fonttype'] = 'none'
sns.heatmap(heatmap_df, center=0, cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            square=False, rasterized=True, cbar_kws={'label': f"Pearson's r", "shrink": 0.55},
            ax=ax[0])
ax[0].set_title('Top positively\ncorrelating genes')
# plt.tight_layout()
# plt.show()

# heatmap bottom correlations
heatmap_df = pd.DataFrame()
for c in list(title_dict.values()):
    sorted_df = correlation_df.sort_values(by=c, ascending=True)
    heatmap_df = heatmap_df.append(sorted_df.iloc[:10, :])
heatmap_df = heatmap_df[list(title_dict.values())]
plt.rcParams['svg.fonttype'] = 'none'
# fig, ax = plt.subplots(1, 1, figsize=(4, 10))
sns.heatmap(heatmap_df, center=0, cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            square=False, rasterized=True, cbar_kws={'label': f"Pearson's r", "shrink": 0.55},
            ax=ax[1])
ax[1].set_title('Top negatively\ncorrelating genes')
plt.suptitle('Total Pt content')
plt.tight_layout()
plt.show()

for f in np.unique(adata.obs['sample_id']):
    ad = adata[adata.obs['sample_id'] == f].copy()
    ad = ad[ad.obs['n_tumor_cell'] > 2]
    if len(ad) > 0:
        sc.pl.spatial(ad, color=['Vegfa', 'Ccn2'], s=12, library_id=f, use_raw=False)
        plt.show()

# treated
treated_df = correlation_df[['Cisplatin 6mg/kg 4hpt', 'Cisplatin 6mg/kg 24hpt']]
pval_df = pd.DataFrame({k: v['pearson_p'].values for k, v in correlation_dict.items()},
                              index=adata.var_names)
pval_df = pval_df[['Cisplatin 6mg/kg 4hpt', 'Cisplatin 6mg/kg 24hpt']]
treated_df['significant'] = list((pval_df < 0.05).all(axis=1))

for b in [True, False]:
    cdf = treated_df[treated_df['significant'] == b]
    plt.scatter(cdf['Cisplatin 6mg/kg 4hpt'], cdf['Cisplatin 6mg/kg 24hpt'],
                s=1)
plt.show()

#%%
# pt tumor cell abundance
keys = ['Tumor low Pt', 'Tumor mod. Pt', 'Tumor high Pt', 'Tumor necrotic margin']
for key in keys:
    correlation_dict = {}
    for f in np.unique(adata.obs['sample_id']):
        ad = adata[adata.obs['sample_id'] == f].copy()
        ad = ad[ad.obs['n_tumor_cell'] > 2]
        if len(ad) > 0:
            condition = ad.obs['treatment'][0] + '_' + ad.obs['elapsed_time'][0]
            condition = title_dict[condition]
            # correlations
            correlation_dict = gene_correltations(ad, key)

    correlation_df = pd.DataFrame({k: v['pearson'].values for k, v in correlation_dict.items()},
                                  index=adata.var_names)

    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 2, figsize=(8, 10))
    # heatmap top correlations
    heatmap_df = pd.DataFrame()
    for c in list(title_dict.values()):
        sorted_df = correlation_df.sort_values(by=c, ascending=False)
        print(c)
        print(sorted_df.iloc[:10, :].index)
        heatmap_df = heatmap_df.append(sorted_df.iloc[:10, :])
    heatmap_df = heatmap_df[list(title_dict.values())]
    plt.rcParams['svg.fonttype'] = 'none'
    sns.heatmap(heatmap_df, center=0, cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
                square=False, rasterized=True, cbar_kws={'label': f"Pearson's r", "shrink": 0.55},
                ax=ax[0])
    ax[0].set_title('Top positively\ncorrelating genes')
    # heatmap bottom correlations
    heatmap_df = pd.DataFrame()
    for c in list(title_dict.values()):
        sorted_df = correlation_df.sort_values(by=c, ascending=True)
        heatmap_df = heatmap_df.append(sorted_df.iloc[:10, :])
    heatmap_df = heatmap_df[list(title_dict.values())]
    plt.rcParams['svg.fonttype'] = 'none'
    # fig, ax = plt.subplots(1, 1, figsize=(4, 10))
    sns.heatmap(heatmap_df, center=0, cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
                square=False, rasterized=True, cbar_kws={'label': f"Pearson's r", "shrink": 0.55},
                ax=ax[1])
    ax[1].set_title('Top negatively\ncorrelating genes')
    plt.suptitle(key)
    plt.tight_layout()
    plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/{key}_corr_hm.svg')
    plt.show()

key = '20230607-1_3'
ad = adata[adata.obs['sample_id'] == key]
ad.uns['sample_id_colors'] = ['#50c6ad']
image_shape = ad.uns['spatial'][key]['images']['hires'].shape
x_max, y_max = image_shape[1], image_shape[0]
crop_coord = [0, 8000, 0, 8000]  # (x_min, y_min, x_max, y_max)
sc.set_figure_params(vector_friendly=True, dpi_save=200, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sc.pl.spatial(ad, color='sample_id', library_id=key, alpha_img=0.8, ax=ax, show=False, crop_coord=crop_coord)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig1/visium_imc_demo.svg')
plt.show()

#%%
# compartment correlations with pt
adata = sc.read_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')

# cell type correlations with the compartments
compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)
pt_df = adata.obs[['Tumor low Pt', 'Tumor mod. Pt', 'Tumor high Pt', 'Tumor necrotic margin', 'tumor_mean_pt']]

pt_df = pt_df.rename(columns={'tumor_mean_pt': "Pt/\u00B5m\u00B2"})

corr_matrix = np.empty((len(pt_df.columns), len(compartment_df.columns)))
for i, col1 in enumerate(pt_df.columns):
    for j, col2 in enumerate(compartment_df.columns):
        corr, _ = pearsonr(pt_df[col1], compartment_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     index=pt_df.columns,
                     columns=compartment_df.columns).T

hexcodes = ch.utils.get_hexcodes(None, 12, 65, len(adata))

# heatmap
z = linkage(corrs, method='ward')
order = leaves_list(z)
corrs = corrs.iloc[order, :]
hexcodes = [hexcodes[i] for i in order]

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.heatmap(corrs.T, square=True, center=0, rasterized=True,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True), ax=ax,
            cbar_kws={'label': "Pearson's r", "shrink": 0.45})
for idx, t in enumerate(ax.get_xticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
plt.title('Pt correlation')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/chr_comp_pt_heatmap.svg')
plt.show()

#%%
# tumor cell categories by pt
adata = sc.read_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')

ch_df = pd.DataFrame(data=adata.obsm['chr_aa'], index=adata.obs_names,
                     columns=[f'C{x}' for x in range(adata.obsm['chr_aa'].shape[1])])

adata.obsm['ch_df'] = ch_df
ch_df_to_csv =  ch.get_compartment_df(adata)
ch_df_to_csv.to_csv(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/tables/cellular_niche_weights_pt.csv')
