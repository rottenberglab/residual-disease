import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import chrysalis as ch
import matplotlib.pyplot as plt
from spatial_transcriptomics.functions import spatial_plot
from spatial_transcriptomics.chrysalis_functions_update import plot_weights


def def_thershold(vector):
    m = np.mean(vector)
    sd = np.std(vector)
    th = m + (sd * 2)
    selected_vals = [x for x in vector if x > th]
    num_inc_genes = len(selected_vals)
    return num_inc_genes

#%%
# EMT and proliferating tumor niche signatures
orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

output_folder = 'data/human_samples'

#%%
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
chr_signatues = ch.get_compartment_df(adata)

adata = sc.read_h5ad(f'data/human_samples/human_samples_scanorama.h5ad')
orthologs_dict = {k: v for v, k in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

mouse_orths = []
for v in chr_signatues.index:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)

chr_signatues['human'] = mouse_orths
chr_signatues = chr_signatues.dropna()
chr_signatues['mouse'] = chr_signatues.index
chr_signatues.index = chr_signatues['human']

#%%
plt.rcParams['svg.fonttype'] = 'none'
plot_weights(adata, compartments=[6, 18, 1, 8, 16, 11, 7], ncols=3, seed=8, w=0.6, h=0.4, top_genes=6)
plt.savefig(f'figs/manuscript/fig4/chr_weights_human_small.svg')
plt.show()

# compartments
ch.plot_weights(adata, seed=8, ncols=6)
# plt.savefig(f'figs/manuscript/fig2/chr_heatmap.svg')
plt.show()

#%%
cats = ['T23_47', 'T23_48', 'T1212', 'T23_41', 'T23_42']
adata.obs['sample_id'] = adata.obs['sample_id'].cat.reorder_categories(cats, ordered=True)

condition = {'primary_tumor': 'Primary tumor', 'residual_tumor': 'Residual tumor'}
val_vars = {'EMT_signature': 'EMT signature', 'prolif_signature': 'Proliferating signature'}

titles = []
for idx, s in enumerate(adata.obs['sample_id'].cat.categories):
    ad = adata[adata.obs['sample_id'] == s].copy()
    titles.append(f'{ad.obs["sample_id"][0]}\n{condition[ad.obs["condition"][0]]}')

#%%
# Spatial plots with the signature scores
save = False
emt = chr_signatues['compartment_2']
emt = emt.sort_values(ascending=False)

th = def_thershold(emt)

gene_list = emt[:th].index

gene_list = [g for g in gene_list if g in adata.var_names]
sc.tl.score_genes(adata, gene_list=gene_list, score_name='EMT_signature', use_raw=False)

if save:
    plt.rcParams['svg.fonttype'] = 'none'
    sc.set_figure_params(vector_friendly=True, dpi_save=200, fontsize=12)
    spatial_plot(adata, 1, 5, 'EMT_signature', cmap='Spectral_r', title=titles, suptitle='EMT signature',
                 alpha_img=0.0, colorbar_label='Score', wspace=0.25, colorbar_shrink=0.65, spot_size=85)
    plt.tight_layout()
    plt.savefig(f'figs/manuscript/fig4/emt_sig_spatial2.svg')
    plt.show()

prolif = chr_signatues['compartment_5']
prolif = prolif.sort_values(ascending=False)

th = def_thershold(prolif)

gene_list = prolif[:th].index
gene_list = [g for g in gene_list if g in adata.var_names]
sc.tl.score_genes(adata, gene_list=gene_list, score_name='prolif_signature', use_raw=False)

if save:
    sc.set_figure_params(vector_friendly=True, dpi_save=200, fontsize=12)
    plt.rcParams['svg.fonttype'] = 'none'
    spatial_plot(adata, 1, 5, 'prolif_signature', cmap='Spectral_r', title=titles, suptitle='Proliferating signature',
                 alpha_img=0.0, colorbar_label='Score', wspace=0.25, colorbar_shrink=0.65, spot_size=85)
    plt.tight_layout()
    plt.savefig(f'figs/manuscript/fig4/prolif_sig_spatial2.svg')
    plt.show()

    sc.set_figure_params(vector_friendly=True, dpi_save=200, fontsize=12)
    plt.rcParams['svg.fonttype'] = 'none'
    spatial_plot(adata, 1, 5, 'prolif_signature', cmap='Spectral_r', title=titles, suptitle='HE',
                 alpha_img=1.0, colorbar_label='Score', wspace=0.25, colorbar_shrink=0.65, spot_size=85,
                 alpha=0)
    plt.tight_layout()
    plt.savefig(f'figs/manuscript/fig4/prolif_sig_he.svg')
    plt.show()

#%%
# Barplot with the distributions
val_vars = {'EMT_signature': 'EMT', 'prolif_signature': 'Proliferating'}

long_df = pd.melt(adata.obs,
                  id_vars=['annotations', 'condition'],
                  value_vars=['EMT_signature', 'prolif_signature'])

long_df = long_df[~long_df['annotations'].isin(['Other', 'Stroma'])]
long_df['condition'] = long_df['condition'].map(lambda x: condition[x])
long_df['variable'] = long_df['variable'].map(lambda x: val_vars[x])

plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(1, 1, figsize=(2.25, 4), dpi=200)
custom_palette = {"EMT": "#DBA456", "Proliferating": "#56B5DB"}
flierprops = dict(markerfacecolor='0', markersize=2.5, markeredgecolor='none',
                  linestyle='none', rasterized=True)
sns.boxplot(long_df, x='condition', y='value', hue='variable', ax=ax, palette=custom_palette,
            flierprops=flierprops, gap=.1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel('')
ax.set_title(f'Tissue compartment scores')
ax.set_ylabel('Gene set score')
ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = ax.get_legend()
legend.set_title('')
plt.tight_layout()
# plt.savefig(f'figs/manuscript/fig4/tissue_comp_score_barplot_v2.svg')
plt.show()

residual_df = long_df[long_df['condition'] == 'primary_tumor']
residual_df = residual_df[residual_df['variable'] == 'EMT_signature']
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.violinplot(residual_df, x='annotations', y='value', hue='variable', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()

adata.obs['tumor_spots'] = adata.obs['annotations'].isin(['Other', 'Stroma'])
adata.obs['tumor_spots'] = ['False' if x is True else 'True' for x in adata.obs['tumor_spots']]

# show selected spots
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'
spatial_plot(adata, 1, 5, 'tumor_spots', cmap='Spectral_r', title=titles, suptitle='Proliferating signature',
             alpha_img=0.0, share_range=False, palette=['#cccccc', 'black'], spot_size=85)
# plt.savefig(f'figs/manuscript/fig4/selected_spots_human.svg')
plt.show()
