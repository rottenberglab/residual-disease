import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
import seaborn as sns
import chrysalis as ch
from pycirclize import Circos
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.functions import spatial_plot, proportion_plot
from spatial_transcriptomics.chrysalis_functions_update import (plot_svg_matrix, plot, plot_rss,
                                                                plot_heatmap, plot_weights)


# labels for figures
condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor',
             'residual_tumor': 'Residual tumor', 'control': 'Control'}

treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg',
             'no_treatment': 'No treatment', 'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}

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

#%%
# Main dataset of tumor spots
root_folder = '/path/'

# collect samples + segment tissue images
adatas = []
for idx, s in enumerate(glob(root_folder + '/*.h5ad')):
    # subset for tumor tissue - discard other samples / spots
    ad = sc.read_h5ad(s)
    # we can optionally use ENSEMBL gene IDs instead of gene symbols to deal with non-unique gene identifiers
    ad.var['gene_symbols'] = ad.var_names
    ad.var_names = ad.var['gene_ids']
    # subset for tumor tissue - discard other samples / spots
    ad = ad[ad.obs['condition'] != 'control']
    # remove non-tumor spots
    ad = ad[ad.obs['annotations'] == 'Tumor']
    if len(ad) == 0:
        print(f'Control sample popped.')
    elif ad.obs["elapsed_time"][0] in ['4_hours', '24_hours']:
        print(f'Sample popped: {ad.obs["sample_id"][0]}')
    else:
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

            ch.detect_svgs(ad, min_morans=0.025, min_spots=0.05)
            ch.plot_svgs(ad)
            plt.show()
            ad.write_h5ad(f'data/chrysalis/tumors/{ad.obs["sample_id"][0]}.h5ad')

#%%
# Save RAW counts
# collect samples + segment tissue images
adatas = []
for idx, s in enumerate(glob(root_folder + '/*.h5ad')):
    # subset for tumor tissue - discard other samples / spots
    ad = sc.read_h5ad(s)
    # we can optionally use ENSEMBL gene IDs instead of gene symbols to deal with non-unique gene identifiers
    ad.var['gene_symbols'] = ad.var_names
    ad.var_names = ad.var['gene_ids']
    # subset for tumor tissue - discard other samples / spots
    ad = ad[ad.obs['condition'] != 'control']
    # remove non-tumor spots
    ad = ad[ad.obs['annotations'] == 'Tumor']
    if len(ad) == 0:
        print(f'Control sample popped.')
    elif ad.obs["elapsed_time"][0] in ['4_hours', '24_hours']:
        print(f'Sample popped: {ad.obs["sample_id"][0]}')
    else:
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

            ad.write_h5ad(f'data/chrysalis/tumors/raw/{ad.obs["sample_id"][0]}_raw.h5ad')

#%%
# Combine AnnDatas
adatas = []
sample_names = []
for p in glob('data/chrysalis/tumors/*.h5ad'):

    ad = sc.read_h5ad(p)
    ad.var_names_make_unique()

    # add labels
    sample_index = f"{ad.obs['condition'][0]} {ad.obs['treatment'][0]} {ad.obs['elapsed_time'][0]}"
    sample_index = f"{ad.obs['sample_id'][0]} | {names[sample_index]}"
    ad.obs['label'] = sample_index
    sample_names.append(sample_index)
    adatas.append(ad)

plot_svg_matrix(adatas, figsize=(14, 11), obs_name='label', cluster=True)
plt.show()

adata = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')

#%%
# Look at some qc metrics
rows = 5
cols = 5

spatial_plot(adata, rows, cols, 'total_counts')
plt.show()

spatial_plot(adata, rows, cols, 'log1p_total_counts')
plt.show()

plt.hist(adata.obs['log1p_total_counts'], bins=30)
plt.show()

spatial_plot(adata, rows, cols, 'n_genes_by_counts')
plt.show()

plt.hist(adata.obs['n_genes_by_counts'], bins=100)
plt.show()

plt.scatter(x=adata.obs['total_counts'], y=adata.obs['n_genes_by_counts'], s=1)
plt.show()

#%%
# Perform PCA and HARMONY integration

adata.var.index = adata.var['gene_symbols']
adata.var_names = adata.var['gene_symbols']

ch.pca(adata, n_pcs=50)

# save uncorrected PCs and plot
adata.obsm['chr_X_pca_uncorrected'] = adata.obsm['chr_X_pca'].copy()

adata.obs['label'] = adata.obs['label'].astype('category')

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plt.scatter(x=adata.obsm['chr_X_pca_uncorrected'][:, 0], y=adata.obsm['chr_X_pca_uncorrected'][:, 1],
            rasterized=True, s=4, c=adata.obs['label'].cat.codes)
plt.tight_layout()
plt.show()

ch.harmony_integration(adata, 'ch_sample_id', random_state=42, block_size=0.05)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plt.scatter(x=adata.obsm['chr_X_pca'][:, 0], y=adata.obsm['chr_X_pca'][:, 1],
            rasterized=True, s=4, c=adata.obs['label'].cat.codes)
plt.tight_layout()
plt.show()

adata.write_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

#%%
# Run chrysalis AA
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

ch.utils.estimate_compartments(adata, range_archetypes=(6, 20), max_iter=10)
plot_rss(adata)

ch.plot_explained_variance(adata)
plt.show()

ch.aa(adata, n_pcs=20, n_archetypes=13)

adata.obs['sample'] = adata.obs['ch_sample_id']

adata.write_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

#%%
# Spatial plots
comps = 13
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

compartment_signatures = ch.get_compartment_df(adata)
compartment_signatures.to_csv('data/mouse_compartment_signatures.csv')

# Chrysalis MIP
rows = 5
cols = 5
fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
ax = ax.flatten()
for a in ax:
    a.axis('off')
for idx, i in enumerate(np.unique(adata.obs['sample'])):
    plot(adata, dim=comps, sample_id=i, ax=ax[idx], rasterized=True, spot_size=4, seed=87)
    label = adata.obs[adata.obs['sample'] == i]['label'][0]
    ax[idx].set_title(label)
plt.tight_layout()
plt.show()

# Individual compartment
for x in range(comps):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, i in enumerate(np.unique(adata.obs['sample'])):
        plot(adata, dim=comps, sample_id=i, ax=ax[idx], rasterized=True, spot_size=4.0, selected_comp=x, seed=87)
        label = adata.obs[adata.obs['sample'] == i]['label'][0]
        ax[idx].set_title(label)
    plt.suptitle(f'Compartment {x}', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'data/chrysalis/compartments/compartment_{x}.png')
    plt.close()
    plt.show()

plot_weights(adata, ncols=5, seed=87)
plt.show()

plot_heatmap(adata, reorder_comps=True, seed=87, figsize=(6, 7))
plt.show()

#%%
# Spatial plots - selected samples for plots
comps = 13
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

selected_samples = [ 17,  14,  15,  16,  19,  8,  21]

for x in range(comps):
    fig, ax = plt.subplots(1, 7, figsize=(7 * 3.5, 1 * 4))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    plt.suptitle(f'Compartment {x}', fontsize=15, y=0.99)
    plt.tight_layout()
    for idx, i in enumerate(selected_samples):
        plot(adata, dim=comps, sample_id=i, ax=ax[idx], rasterized=True, spot_size=1.1, selected_comp=x, seed=87,
             figsize=(7 * 3.5, 1 * 4))
        label = adata.obs[adata.obs['sample'] == i]['label'][0]
        ax[idx].set_title(label.split(' | ')[-1])
    plt.savefig(f'data/chrysalis/compartments/demo/compartment_{x}.png')
    plt.show()

#%%
# proportion plots

labels = []
spot_nr = []
prop_matrix = np.zeros((len(np.unique(adata.obs['sample'])), comps))
for idx, i in enumerate(np.unique(adata.obs['sample'])):
    ad = adata[adata.obs['sample'] == i]
    spot_nr.append(len(ad))
    compartments = ad.obsm['chr_aa']
    compartments_mean = compartments.sum(axis=0)
    compartments_prop = compartments_mean / np.sum(compartments_mean)
    prop_matrix[i] = compartments_prop
    label = adata.obs[adata.obs['sample'] == i]['label'][0]
    labels.append(label)

props_df = pd.DataFrame(data=prop_matrix,
                           index=labels)
spot_nr = pd.Series(data=spot_nr, index=labels, name='spot_nr')

# Define the custom order
custom_order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
                'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

props_df['order'] = props_df.index.str.split('|').str[1].str.strip()
props_df['sample'] = props_df.index
props_df['order'] = props_df['order'].astype('category')
props_df['order'] = props_df['order'].cat.reorder_categories(custom_order, ordered=True)
props_df = props_df.sort_values(by=['order', 'sample'])
props_df = props_df.drop(columns=['order', 'sample'])

spot_nr = pd.DataFrame(spot_nr)
spot_nr['order'] = spot_nr.index.str.split('|').str[1].str.strip()
spot_nr['sample'] = spot_nr.index
spot_nr['order'] = spot_nr['order'].astype('category')
spot_nr['order'] = spot_nr['order'].cat.reorder_categories(custom_order, ordered=True)
spot_nr = spot_nr.sort_values(by=['order', 'sample'])
spot_nr = spot_nr.drop(columns=['order', 'sample'])

hexcodes = ch.utils.generate_random_colors(num_colors=13, min_distance=1 / 13 * 0.5, seed=87,
                                  saturation=0.65, lightness=0.60)
cmap = sns.color_palette(hexcodes, 13)

proportion_plot(props_df[::-1], spot_nr['spot_nr'][::-1], palette=hexcodes)
plt.tight_layout()
plt.show()

# boxplots
rows = 3
cols = 5

fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(props_df.columns):
    axs[idx].axis('on')
    sub_df = props_df[[c]].copy()
    sub_df['condition'] = sub_df.index.str.split('|').str[1].str.strip()
    sns.boxplot(sub_df, y=c, x='condition', ax=axs[idx], color=hexcodes[c])
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Proportion')
    axs[idx].set_title(f'Compartment {c}', fontsize=14)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
fig.supylabel(None)
plt.tight_layout()
plt.show()

# get the relative change
normalized_prop_df = props_df.multiply(spot_nr['spot_nr'], axis=0)
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(normalized_prop_df.columns):
    axs[idx].axis('on')
    sub_df = normalized_prop_df[[c]].copy()
    sub_df['condition'] = sub_df.index.str.split('|').str[1].str.strip()
    sns.boxplot(sub_df, y=c, x='condition', ax=axs[idx], color=hexcodes[c])
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Compartment coverage')
    axs[idx].set_title(f'Compartment {c}', fontsize=14)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
fig.supylabel(None)
plt.tight_layout()
plt.show()

#%%
# Misc plots
# compartment co-localization and diversity
# calculate Gini-Simpson index
def ginisimpson(y):
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    D = 1 - np.sum(t ** 2)
    return D

adata.obs['gini_simpson_index'] = np.apply_along_axis(ginisimpson, 1, adata.obsm['chr_aa'])
spatial_plot(adata, 5, 5, 'gini_simpson_index', cmap='viridis', title=True, suptitle='Gini-Simpson index',
             alpha_img=0.75)
plt.show()

# violin plot
columns = ['gini_simpson_index']

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
# palette 80, 60, 50 lightness

rows = 1
cols = 1

columns_dict = {'gini_simpson_index': 'Gini-Simpson index'}

with sns.axes_style("darkgrid", {"axes.facecolor": ".95"}):
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axs.axis('off')
    for idx, pw in enumerate(adata.obs[columns]):
        pw_df = pd.DataFrame(data=adata.obs[pw], columns=[pw])
        pw_df['condition_cat'] = [names[x] for x in list(adata.obs['condition_cat'])]

        axs.axis('on')

        axs.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
        axs.set_axisbelow(True)
        sns.violinplot(data=pw_df, x='condition_cat', y=pw, scale='width',
                       palette=['#92BD6D', '#E9C46B', '#F2A360', '#E66F4F', '#4BC9F1', '#4993F0', '#435FEE'],
                       order=order, ax=axs)
        # sns.stripplot(data=pw_df, x='condition_cat', y=pw, jitter=True,
        #                order=order, color='black', size=2, alpha=.1)
        axs.set_ylabel('Index')
        axs.set_title(columns_dict[pw], fontsize=14)
        axs.set_xlabel(None)
        axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
        # axs[idx].spines['top'].set_visible(False)
        # axs[idx].spines['right'].set_visible(False)
        legend_labels = ['False', 'True']
        handles, _ = axs.get_legend_handles_labels()
    plt.tight_layout()
    plt.show()

# chord plots and interaction matrices

# iterate over conditions
fig, axs = plt.subplots(3, 3, figsize=(12, 10))
axs = axs.flatten()
for a in axs:
    a.axis('off')

comp_order =['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
        'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']
rev_names = {v: k for k, v in names.items()}
comp_order = [rev_names[n] for n in comp_order]
for idx, c in enumerate(comp_order):

    ad = adata[adata.obs['condition_cat'] == c].copy()

    # get compartments and multiply the fractional values by 10 to ease interpretability
    comp_df = pd.DataFrame(data=ad.obsm['chr_aa'])
    comp_vals = comp_df.values * 1
    n_comps = comp_vals.shape[1]

    # calculate pairwise interaction matrix
    sums = np.zeros((n_comps, n_comps))
    for i in range(n_comps):
        for k in range(n_comps):
            m = comp_vals[:, i] * comp_vals[:, k]
            m = np.mean(m)
            sums[i, k] = m

    # plot them
    # comp_names = ['Compartment ' + str(x) for x in range(n_comps)]
    comp_names = [str(x) for x in range(n_comps)]
    sums_df = pd.DataFrame(sums, index=comp_names, columns=comp_names)

    z = linkage(sums_df, method='ward')
    order = leaves_list(z)
    sums_df = sums_df.iloc[order, order]

    # sns.heatmap(np.log1p(sums_df))
    # plt.title(names[c])
    # plt.show()

    axs[idx].axis('on')
    g = sns.heatmap(sums_df, cmap='Spectral_r', ax=axs[idx], square=True)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    axs[idx].set_title(names[c])
plt.suptitle('Co-occurance')
plt.tight_layout()
plt.show()

# do the same for the chord plots
# iterate over conditions

for idx, c in enumerate(comp_order):

    ad = adata[adata.obs['condition_cat'] == c].copy()

    # get compartments and multiply the fractional values by 10 to ease interpretability
    comp_df = pd.DataFrame(data=ad.obsm['chr_aa'])
    comp_vals = comp_df.values * 1
    n_comps = comp_vals.shape[1]

    # calculate pairwise interaction matrix
    sums = np.zeros((n_comps, n_comps))
    for i in range(n_comps):
        for k in range(n_comps):
            m = comp_vals[:, i] * comp_vals[:, k]
            m = np.sum(m)
            sums[i, k] = m

    # plot them
    comp_names = [str(x) for x in range(n_comps)]
    # comp_names = ['Compartment ' + str(x) for x in range(n_comps)]
    sums_df = pd.DataFrame(sums, index=comp_names, columns=comp_names)

    hexcodes = ch.utils.generate_random_colors(num_colors=13, min_distance=1 / 13 * 0.5, seed=87,
                                               saturation=0.65, lightness=0.60)
    colors = {k: v for k, v in zip(comp_names, hexcodes)}
    circos = Circos.initialize_from_matrix((sums_df.astype(int)), space=3, cmap=colors)
    ax = circos.plotfig(dpi=100)
    ax.set_figwidth(5)
    ax.set_figheight(5)
    plt.title(names[c])
    plt.show()

# show integrated umap
sc.pp.neighbors(adata, use_rep='chr_X_pca')
sc.tl.umap(adata)
fix, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.umap(adata, color=['ch_sample_id'], show=False)
plt.tight_layout()
plt.show()

colors = ch.get_color_vector(adata, dim=comps)
plt.scatter(x=adata.obsm['X_umap'][:, 0], y=adata.obsm['X_umap'][:, 1], c=colors, s=2)
plt.show()

# show the uncorrected one
sc.pp.pca(adata)
sc.pp.neighbors(adata, use_rep='X_pca', key_added='uncorrected_nh')
sc.tl.umap(adata, neighbors_key='uncorrected_nh')
fix, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.umap(adata, color=['ch_sample_id'], show=False)
plt.tight_layout()
plt.show()

# compare PCAs
colors = ch.get_color_vector(adata, dim=comps)
plt.scatter(x=adata.obsm['chr_X_pca'][:, 0], y=adata.obsm['chr_X_pca'][:, 1], c=colors, s=1)
plt.show()

colors = ch.get_color_vector(adata, dim=comps)
plt.scatter(x=adata.obsm['X_pca'][:, 0], y=adata.obsm['X_pca'][:, 1], c=colors, s=8)
plt.show()
