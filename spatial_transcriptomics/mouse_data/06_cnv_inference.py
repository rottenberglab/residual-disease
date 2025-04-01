import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
import decoupler as dc
import chrysalis as ch
import infercnvpy as cnv
import matplotlib.pyplot as plt
from spatial_transcriptomics.functions import spatial_plot, chromosome_heatmap, chromosome_heatmap_summary


# labels for figures
condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor',
             'residual_tumor': 'Residual tumor', 'control': 'Control'}

treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg',
             'no_treatment': 'No treatment', 'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}

time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment', '7_days': '7 days post-treatment',
        'na': '-', '24_hours': '24 hours post-treatment', '4_hours': '4 hours post-treatment'}

names = {'primary_tumor no_treatment na': 'Primary tumor',
         'control no_treatment na': 'Control',
         'relapsed_tumor TAC_regimen 30_days' : 'TAC 30dpt',
         'relapsed_tumor cisplatin_6mg/kg 30_days': 'Cisplatin 30dpt',
         'residual_tumor TAC_regimen 12_days': 'TAC 12dpt',
         'residual_tumor TAC_regimen 7_days': 'TAC 7dpt',
         'residual_tumor cisplatin_6mg/kg 12_days': 'Cisplatin 12dpt',
         'residual_tumor cisplatin_6mg/kg 7_days': 'Cisplatin 7dpt',
         'primary_tumor cisplatin_6mg/kg 4_hours': 'Cisplatin 4hpt',
         'primary_tumor cisplatin_6mg/kg 24_hours': 'Cisplatin 24hpt',
         'primary_tumor cisplatin_12mg/kg 24_hours': 'Cisplatin 24hpt'}

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

#%%

# collect samples + segment tissue images
meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)

adatas = []
for idx, s in enumerate(glob('data/samples/*.h5ad')):
    # subset for tumor tissue - discard other samples / spots
    ad = sc.read_h5ad(s)
    ad.var_names_make_unique()

    if ad.shape[0] != 0:
        # normalize
        sc.pp.calculate_qc_metrics(ad, inplace=True)
        sc.pp.filter_genes(ad, min_cells=10)

        if ad.obs['count_lower_cutoff'][0] != 0:
            sc.pp.filter_cells(ad, min_counts=ad.obs['count_lower_cutoff'][0])
        if ad.obs['count_upper_cutoff'][0] != 0:
            sc.pp.filter_cells(ad, max_counts=ad.obs['count_upper_cutoff'][0])
        if ad.obs['n_gene_cutoff'][0] != 0:
            sc.pp.filter_cells(ad, min_genes=ad.obs['n_gene_cutoff'][0])

        sc.pp.normalize_total(ad, inplace=True, target_sum=1e4, exclude_highly_expressed=True)
        sc.pp.log1p(ad)
        print(ad.obs['sample_id'][0])
        adatas.append(ad)

adata = anndata.concat(adatas, index_unique='-', uns_merge='unique', merge='unique')

# Remove spots marked as necrosis or ignore
adata = adata[~adata.obs['annotations'].isin(['Necrosis', 'Ignore*'])]

#%%
# Map genomic positions
cnv.io.genomic_position_from_gtf('data/mouse_genome/gencode_vM10_GRCm38_annotation.gtf', adata)
adata = adata[:, ~adata.var['chromosome'].isna()]
adata.write_h5ad('data/infercnv.h5ad')

#%%
# Infer CNV
adata = sc.read_h5ad('data/infercnv.h5ad')

cnv.tl.infercnv(
    adata,
    reference_key="annotations",
    reference_cat=['Epithelial cells', 'Immune cells', 'Muscle', 'Other', 'Stroma'],
    window_size=100,
    step=10,
    chunksize=100,
)

adata.write_h5ad('data/infercnv.h5ad')

#%%
# calculate CNV clusters
adata = sc.read_h5ad('data/infercnv.h5ad')
cnv.tl.pca(adata)
cnv.pp.neighbors(adata)
cnv.tl.leiden(adata)
cnv.tl.umap(adata)
cnv.tl.cnv_score(adata)
adata.write_h5ad('data/infercnv.h5ad')

#%% update parental tumor annots
adata = sc.read_h5ad('data/infercnv.h5ad')

meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)
act_dict = {k: v for k, v in zip(meta_df['sample_id'], meta_df['ancestral_tumor'])}
adata.obs['ancestral_tumor'] = [act_dict[x] for x in adata.obs['sample_id']]

adata.write_h5ad('data/infercnv.h5ad')

#%%
# add chrysalis to tumor spots

adata = sc.read_h5ad('data/infercnv.h5ad')
# add sample id for plots
sample_indexes = [x + ' ' + y + ' ' + z for x, y, z in
                  zip(adata.obs['condition'], adata.obs['treatment'], adata.obs['elapsed_time'])]
sample_indexes = [names[x] for x in sample_indexes]
sample_indexes = [x + ' | ' + y for x, y in zip(adata.obs['sample_id'], sample_indexes)]
adata.obs['sample_id_fig'] = sample_indexes

adata.obs_names = [f'{x[:16]}_{y}' for x, y in zip(adata.obs_names, adata.obs['sample_id'])]

adata_chr = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
adata_chr.obs_names = [f'{x[:16]}_{y}' for x, y in zip(adata_chr.obs_names, adata_chr.obs['sample_id'])]

adata_tumor = adata[adata.obs_names.isin(adata_chr.obs_names)]
adata_tumor.obsm['chr_aa'] = adata_chr.obsm['chr_aa']
adata_tumor.obsm['cell2loc'] = adata_chr.obsm['cell2loc']
adata_tumor.obs['ch_sample_id'] = adata_chr.obs['ch_sample_id']

adata_tumor.write_h5ad('data/infercnv_tumor_chr.h5ad')

#%%
# we need to run a heatmap to get the color mapping list saved, save the object with it later

adata = adata[adata.obs['annotations'] == 'Tumor']

plt.rcParams['svg.fonttype'] = 'none'
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
spatial_plot(adata, 5, 6, 'cnv_leiden', title=np.unique(adata.obs['sample_id_fig']),
             share_range=False, legend_loc=None, alpha_img=0.5)
plt.savefig(f'figs/manuscript/fig3/cnv_spatial_all.svg')
plt.show()

# labels
colors = adata.uns['cnv_leiden_colors']
unique_annotations = adata.obs['cnv_leiden'].cat.categories
fig, ax = plt.subplots()
ax.axis('off')
handles = []
for annotation, color in zip(unique_annotations, colors):
    handle = ax.scatter([], [], label=annotation, color=color)
    handles.append(handle)
ax.legend(handles=handles, labels=unique_annotations.tolist(), loc='center',
          fontsize='small', title='Cluster')
plt.savefig(f'figs/manuscript/fig3/cnv_spatial_annots_labels.svg')
plt.show()

#%%
# Heatmaps, stacked violinplots

adata = sc.read_h5ad('data/infercnv.h5ad')
adata_tumor = sc.read_h5ad('data/infercnv_tumor_chr.h5ad')

compartments = adata_tumor.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata_tumor.obs.index)
adata_tumor.obsm['chr_aa'] = compartment_df

acts = dc.get_acts(adata_tumor, obsm_key='chr_aa')

# stacked violin plot
plt.rcParams['svg.fonttype'] = 'none'
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
fig, ax = plt.subplots(1, 1, figsize=(5*1.2, 4.5))
hexcodes = ch.utils.get_hexcodes(None, 13, 87, len(adata_tumor))
dp = sc.pl.stacked_violin(acts, [str(x) for x in range(13)], groupby='cnv_leiden', swap_axes=False, dendrogram=True,
                     ax=ax, show=False, return_fig=True)
dp.style(linewidth=0, cmap='mako_r').legend(title='Median\ncompartment score')
axs = dp.get_axes()
mainax = axs['mainplot_ax']
mainax.set_ylabel('CNV cluster')
mainax.set_title('CNV cluster\ntissue compartment composition')
for idx, t in enumerate(mainax.get_xticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
plt.savefig(f'figs/manuscript/fig3/cnv_tissue_comp.svg')
plt.show()

# cell types
ct = adata_tumor.obsm['cell2loc']
ct = ct.values / ct.sum(axis=1).values.reshape(-1, 1)
ct_df = pd.DataFrame(data=ct, index=adata_tumor.obsm['cell2loc'].index, columns=adata_tumor.obsm['cell2loc'].columns)
ct_df.columns = [cell_type_dict[x] for x in adata_tumor.obsm['cell2loc']]
adata_tumor.obsm['cell2loc'] = ct_df

acts = dc.get_acts(adata_tumor, obsm_key='cell2loc')

# stacked violin plot
plt.rcParams['svg.fonttype'] = 'none'
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hexcodes = ch.utils.get_hexcodes(None, 13, 87, len(adata_tumor))
dp = sc.pl.stacked_violin(acts, [str(x) for x in ct_df.columns], groupby='cnv_leiden',
                          swap_axes=False, dendrogram=True,
                          ax=ax, show=False, return_fig=True, figsize=(8, 6))
dp.style(linewidth=0, cmap='mako_r').legend(title='Median\ncell type proportion')
axs = dp.get_axes()
mainax = axs['mainplot_ax']
mainax.set_ylabel('CNV cluster')
mainax.set_title('CNV cluster\ncell type composition')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/cnv_cell_types.svg')
plt.show()

# Chromosome heatmaps
plt.rcParams['svg.fonttype'] = 'none'
chromosome_heatmap(adata, groupby="cnv_leiden", dendrogram=True,
                               cmap='twilight', show=False, figsize=(10, 8))
# plt.savefig(f'figs/manuscript/fig3/cnv_heatmap.svg')
plt.show()

plt.rcParams['svg.fonttype'] = 'none'
cnv.pl.chromosome_heatmap_summary(adata, groupby="cnv_leiden", dendrogram=True,
                               cmap='twilight', show=False, figsize=(10, 8))
# plt.savefig(f'figs/manuscript/fig3/cnv_heatmap.svg')
plt.show()

plt.rcParams['svg.fonttype'] = 'none'
chromosome_heatmap(adata, groupby="ancestral_tumor", dendrogram=True,
                               cmap='twilight', show=False, figsize=(10, 8), y_label='Parental tumour')
# plt.savefig(f'figs/manuscript/fig3/ cnv_heatmap_parental_tumor.svg')
plt.show()

#%%
# Summary heatmaps for parental tumors and conditions
adata = sc.read_h5ad('data/infercnv.h5ad')
adata_sub = adata[adata.obs['ancestral_tumor']!='na'].copy()
adata_sub.obs['condition'] = [condition[x] for x in adata_sub.obs['condition']]
adata_sub.obs['condition'] = adata_sub.obs['condition'].astype('category')
adata_sub.obs['condition'] = adata_sub.obs['condition'].cat.reorder_categories()
cond_order = ['Primary tumor', 'Residual tumor', 'Relapsed tumor']

plt.rcParams['svg.fonttype'] = 'none'
chromosome_heatmap_summary(adata_sub, groupby="ancestral_tumor", dendrogram=False,
                               cmap='twilight', show=False, figsize=(6, 1.25))
plt.tight_layout()
# plt.savefig(f'figs/manuscript/fig3/cnv_heatmap_parental_tumor_v2.svg')
plt.show()

plt.rcParams['svg.fonttype'] = 'none'
chromosome_heatmap_summary(adata_sub, groupby="condition", dendrogram=False,
                               cmap='twilight', show=False, figsize=(6, 1.25), groups=cond_order)
# plt.savefig(f'figs/manuscript/fig3/cnv_heatmap_conditions_v2.svg')
plt.show()

# plot clusters for main
plt.rcParams['svg.fonttype'] = 'none'
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
# subset anndata
selected_samples = [ 17,  14,  15,  16,  18, 19,  8,  21]
adata_tumor = adata_tumor[adata_tumor.obs['ch_sample_id'].isin(selected_samples)].copy()
# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-4_13', '20221010-3_10', '20221010-4_14',
                '20221010-3_11', '20221010-1_3', '20221010-3_12', '20221010-4_16']
adata_tumor.obs['sample_id'] = adata_tumor.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

titles = []
for s in sample_order:
    result_series = adata_tumor.obs[adata_tumor.obs['sample_id_fig'].str.contains(s, na=False)]['sample_id_fig']
    titles.append(result_series.iloc[0])

spatial_plot(adata_tumor, 4, 2, 'cnv_leiden', title=titles,
             share_range=False, legend_loc=None, alpha_img=0.5)
plt.savefig(f'figs/manuscript/fig3/cnv_spatial_main_8.svg')
plt.show()
