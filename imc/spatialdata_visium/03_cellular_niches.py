import scanorama
import numpy as np
import scanpy as sc
from glob import glob
import chrysalis as ch
import matplotlib.pyplot as plt
from imc.spatialdata_visium.functions import show_annotations
from imc.spatialdata_visium.chrysalis_plot_fix import plot_samples


output_folder = 'data/imc_samples'

#%%
# read samples
h5ads =  glob(f'{output_folder}/*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]

# chrysalis save samples
show = False
if show:
    for ad in adatas:
        show_annotations(ad)
        plt.show()

for ad in adatas:
    ch.detect_svgs(ad, min_morans=0.025, min_spots=0.05)
    ch.plot_svgs(ad)
    plt.show()
    fname = ad.obs["sample_id"][0]
    ad.write_h5ad(f'{output_folder}/ch_{fname}.h5ad')

#%%
#scanorama integration

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]

ch.plot_svg_matrix(adatas, figsize=(5, 4), obs_name='sample_id', cluster=True)
plt.show()

# scanorama seems unstable with the raw data included, add it back after
for ad in adatas:
    ad.raw = None
    try:
        ad.obs.drop(columns=['ch_sample_id'], inplace=True)
    except KeyError:
        pass

scr_adata = scanorama.correct_scanpy(adatas, return_dimred=True)
scr_adata = ch.integrate_adatas(scr_adata, sample_names=sample_names, sample_col='ch_sample_id')
scr_adata.obs = scr_adata.obs.drop(columns=['pt_intensity', 'log1p_pt_intensity'])

scr_adata.write_h5ad(f'{output_folder}/imc_samples_scanorama.h5ad')

#%%
adata = sc.read_h5ad(f'{output_folder}/imc_samples_scanorama.h5ad')

# replace ENSEMBL IDs with the gene symbols and make them unique
adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()

# run PCA
ch.pca(adata, n_pcs=50)
ch.plot_explained_variance(adata)
plt.show()

ch.utils.estimate_compartments(adata, range_archetypes=(6, 25), max_iter=10)
ch.plot_rss(adata)

ch.aa(adata, n_pcs=20, n_archetypes=20)

ch.plot_samples(adata, rows=3, cols=3, dim=12, suptitle='Scanorama', sample_col='ch_sample_id', show_title=True)
plt.show()
ch.plot_heatmap(adata, reorder_comps=True, figsize=(6, 7))
plt.show()

# save plots
ch.plot_samples(adata, rows=2, cols=3, dim=20, suptitle='Scanorama', sample_col='ch_sample_id', show_title=True,
                seed=8)
plt.savefig(f'{output_folder}/chr_plot.png')
plt.close()

ch.plot_heatmap(adata, reorder_comps=True, figsize=(10, 10), seed=8)
plt.savefig(f'{output_folder}/chr_heatmap.png')
plt.close()

ch.plot_weights(adata, ncols=5, seed=8)
plt.savefig(f'{output_folder}/chr_weights.png')
plt.close()

# save the adata with the selected comps
adata.write_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

# %%
# add raw values and save

adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]

adata_raw = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')
adata_raw.var_names = list(adata_raw.var['gene_symbols'])
adata_raw.var_names_make_unique()
var_dict = {k:v for k,v in zip(adata.var['gene_ids'], adata.var_names)}
adata_raw = adata_raw.raw.to_adata()
adata_raw.var_names = [var_dict[x] for x in adata_raw.var_names]

adata.raw = adata_raw

adata.write_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

#%%
# scanorama 4 samples used for IMC

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
selected_samples = ['20230607-2_6', '20230607-1_3', '20230607-2_8', '20230607-2_5']
selected_h5ads = []
for s in h5ads:
    result = any(substring in s for substring in selected_samples)
    if result:
        selected_h5ads.append(s)

adatas = [sc.read_h5ad(x) for x in selected_h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]

# scanorama seems unstable with the raw data included, add it back after
for ad in adatas:
    ad.raw = None
    try:
        ad.obs.drop(columns=['ch_sample_id'], inplace=True)
    except KeyError:
        pass

scr_adata = scanorama.correct_scanpy(adatas, return_dimred=True)
scr_adata = ch.integrate_adatas(scr_adata, sample_names=sample_names, sample_col='ch_sample_id')
scr_adata.obs = scr_adata.obs.drop(columns=['pt_intensity', 'log1p_pt_intensity'])

scr_adata.write_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')

# %%
# add raw values and save

adata = sc.read_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
selected_samples = ['20230607-2_6', '20230607-1_3', '20230607-2_8', '20230607-2_5']
selected_h5ads = []
for s in h5ads:
    result = any(substring in s for substring in selected_samples)
    if result:
        selected_h5ads.append(s)

adatas = [sc.read_h5ad(x) for x in selected_h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]

adata_raw = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')
adata_raw.var_names = list(adata_raw.var['gene_symbols'])
adata_raw.var_names_make_unique()
var_dict = {k:v for k,v in zip(adata.var['gene_ids'], adata.var_names)}
adata_raw = adata_raw.raw.to_adata()
adata_raw.var_names = [var_dict[x] for x in adata_raw.var_names]

adata.raw = adata_raw

adata.write_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')

#%%
adata = sc.read_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')

# replace ENSEMBL IDs with the gene symbols and make them unique
adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()

adata = adata[adata.obs['annotations'].isin(['Tumor', 'Necrosis'])]

# run PCA
ch.pca(adata, n_pcs=50)
ch.plot_explained_variance(adata)
plt.show()

ch.utils.estimate_compartments(adata, range_archetypes=(6, 25), max_iter=10)
ch.plot_rss(adata)

ch.aa(adata, n_pcs=12, n_archetypes=20)

ch.plot_samples(adata, rows=3, cols=3, dim=12, suptitle='Scanorama', sample_col='ch_sample_id', show_title=True)
plt.show()
ch.plot_heatmap(adata, reorder_comps=True, figsize=(6, 7))
plt.show()

# save final plots
ch.plot_samples(adata, rows=2, cols=3, dim=12, suptitle='Scanorama', sample_col='ch_sample_id', show_title=True,
                seed=65)
plt.savefig(f'{output_folder}/chr_plot_4.png')
plt.close()

ch.plot_heatmap(adata, reorder_comps=True, figsize=(10, 10), seed=8)
plt.savefig(f'{output_folder}/chr_heatmap_4.png')
plt.close()

ch.plot_weights(adata, ncols=5, seed=8)
plt.savefig(f'{output_folder}/chr_weights_4.png')
plt.close()

# save the adata with the selected comps-
adata.write_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')

#%%
# chrysalis plots
# subset adata to IMC spots
title_dict = {'cisplatin_6mg/kg_4_hours': 'Cisplatin 6mg/kg 4hpt',
              'cisplatin_6mg/kg_24_hours': 'Cisplatin 6mg/kg 24hpt',
              'cisplatin_6mg/kg_12_days': 'Cisplatin 6mg/kg 12dpt',
              'no_treatment_na': 'Primary tumor',}

adata.obs['title'] = [title_dict[x + '_' + y] for x, y in zip(adata.obs['treatment'], adata.obs['elapsed_time'])]
replace_dict = {}
for s in np.unique(adata.obs['sample_id']):
    ad = adata[adata.obs['sample_id'] == s]
    replace_dict[s] = ad.obs['title'][0]

for k, v in replace_dict.items():
    adata.uns['spatial'][v] = adata.uns['spatial'][k]

adata.obs['title'] = adata.obs['title'].astype('category')
adata.obs["title"] = adata.obs['title'].cat.set_categories(list(title_dict.values()))

plt.rcParams['svg.fonttype'] = 'none'
plot_samples(adata, rows=1, cols=4, dim=12, seed=65, sample_col='title', show_title=True, rasterized=True,
             wspace=-0.7)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/chr_comps_main.svg')
plt.show()

for i in range(12):
    plt.rcParams['svg.fonttype'] = 'none'
    plot_samples(adata, rows=1, cols=4, dim=12, seed=65, sample_col='title', show_title=True, selected_comp=i,
                 wspace=-0.7, rasterized=True)
    plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/chr_comp_{i}.svg')
    plt.show()

plt.rcParams['axes.grid'] = False  # Turn off global grid
plt.rcParams['svg.fonttype'] = 'none'
ch.plot_weights(adata, ncols=7, seed=65, w=0.8, h=0.95)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/chr_weights_main2.svg')
plt.show()

ch.plot_heatmap(adata, reorder_comps=True, figsize=(6, 7), rasterized=True, seed=65)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/chr_heatmap_main.svg')
plt.show()
