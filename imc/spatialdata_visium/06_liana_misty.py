import anndata
import liana as li
import numpy as np
import scanpy as sc
import pandas as pd
import mudata as mu
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D


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
# get imc adatas

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
adata = sc.read_h5ad(f'{output_folder}/imc_samples_scanorama_4.h5ad')
adata_depr = sc.read_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')  # this contains relevant metadata

# add variables to adata
adata_depr = adata_depr[adata_depr.obs.index.isin(adata.obs.index)]

cc_cols = ['G1/S_score', 'S_score', 'G2/M_score', 'M_score', 'M/G1_score', 'mean_cc_score']
adata.obs[cc_cols] = adata_depr.obs[cc_cols]
adata.obs['sasp'] = adata_depr.obs[['sasp']]

adata.obsm['cell2loc'] = adata_depr.obsm['cell2loc']
adata.obsm['mlm_estimate'] = adata_depr.obsm['mlm_estimate']

obs_keys = ['n_cell', 'sum_pt', 'mean_pt', 'tumor_sum_pt', 'tumor_mean_pt', 'n_tumor_cell'] + tumor_cell_types
adata.obs[tumor_cell_types] = imc_adata.obsm['cell_type_params'][tumor_cell_types]
adata.obsm['pt_tumor'] = adata.obs[tumor_cell_types]
adata.obs[obs_keys] = imc_adata.obs[obs_keys]
adata.obs[cell_types] = imc_adata.obsm['cell_type_params'][cell_types]
adata.obs['log1p_sum_pt'] = np.log1p(adata.obs['sum_pt'])
adata = adata[~adata.obs['n_cell'].isna()]
adata.obs['Tumor necrotic margin'] = adata.obs['Tumor necrotic margin'].fillna(0)
adata.obs['sample_id'] = adata.obs['sample_id'].cat.set_categories(['20230607-2_6', '20230607-1_3',
                                                                    '20230607-2_8', '20230607-2_5'])

adata.obsm['cell_cycle'] = adata.obs[['G1/S_score', 'S_score', 'G2/M_score', 'M_score',
                                      'M/G1_score', 'mean_cc_score']]
rename_dict = {'G1/S_score': 'G1/S', 'S_score': 'S', 'G2/M_score': 'G2/M', 'M_score': 'M',
               'M/G1_score': 'M/G1', 'mean_cc_score': 'CC activity'}
adata.obsm['cell_cycle'] = adata.obsm['cell_cycle'].rename(columns=rename_dict)
adata.obsm['sasp'] = adata.obs[['sasp']]
adata.obsm['sasp'] = adata.obsm['sasp'].rename(columns={'sasp': 'SASP'})
adata.obsm['mlm_estimate'].columns = [x + '' for x in adata.obsm['mlm_estimate'].columns]
adata.obsm['cell2loc'].columns = [cell_type_dict[x] for x in adata.obsm['cell2loc'].columns]

# add variables as .obsm and use liana's function to create separate anndatas
title_dict = {'cisplatin_6mg/kg_4_hours': 'Cisplatin 6mg/kg 4hpt',
              'cisplatin_6mg/kg_24_hours': 'Cisplatin 6mg/kg 24hpt',
              'cisplatin_6mg/kg_12_days': 'Cisplatin 6mg/kg 12dpt',
              'no_treatment_na': 'Primary tumor',}

#%%
# calculate results for each sample
metrics_df = pd.DataFrame()
interactions_df = pd.DataFrame()
for s in np.unique(adata.obs['sample_id']):
    adata_sample = adata[adata.obs['sample_id'] == s]

    # define feature sets
    adata_sample.obsm['pt_tumor'] = adata_sample.obsm['pt_tumor'].dropna(axis=1)
    adata_sample.obsm['pt_tumor']['Tumor Pt content'] = adata.obs['tumor_mean_pt']

    target = li.ut.obsm_to_adata(adata_sample, 'pt_tumor')
    ct = li.ut.obsm_to_adata(adata_sample, 'cell2loc')
    pg = li.ut.obsm_to_adata(adata_sample, 'mlm_estimate')
    tc = li.ut.obsm_to_adata(adata_sample, 'ch_df')
    cc = li.ut.obsm_to_adata(adata_sample, 'cell_cycle')
    ss = li.ut.obsm_to_adata(adata_sample, 'sasp')

    # add them to mudata
    mods = {'pt_tumor': target, 'Cell types': ct, 'Pathways': pg, 'Chrysalis': tc, 'Cell cycle': cc, 'SASP': ss}
    mdata = mu.MuData(mods)
    reference = mdata.mod["pt_tumor"].obsm["spatial"]

    # define spatial neighborhoods
    bandwidth = 500
    cutoff = 0.1
    li.ut.spatial_neighbors(ct, bandwidth=bandwidth, cutoff=cutoff, spatial_key="spatial",
                            set_diag=False, standardize=False, reference=reference)
    li.ut.spatial_neighbors(pg, bandwidth=bandwidth, cutoff=cutoff, spatial_key="spatial",
                            set_diag=False, standardize=False, reference=reference)
    li.ut.spatial_neighbors(tc, bandwidth=bandwidth, cutoff=cutoff, spatial_key="spatial",
                            set_diag=False, standardize=False, reference=reference)
    li.ut.spatial_neighbors(cc, bandwidth=bandwidth, cutoff=cutoff, spatial_key="spatial",
                            set_diag=False, standardize=False, reference=reference)
    li.ut.spatial_neighbors(ss, bandwidth=bandwidth, cutoff=cutoff, spatial_key="spatial",
                            set_diag=False, standardize=False, reference=reference)

    # train model
    mdata.update_obs()
    misty = li.mt.MistyData({"intra": target, 'Cell types': ct, 'Pathways': pg, 'Chrysalis': tc,
                             'Cell cycle': cc, 'SASP': ss},
                            enforce_obs=False, obs=mdata.obs)
    # RandomForestModel, LinearModel, RobustLinearModel
    misty(model=li.mt.sp.RandomForestModel, verbose=True, bypass_intra=True)

    # collect results
    targets = misty.uns['target_metrics']
    targets['sample'] = s
    targets['condition'] = title_dict[adata_sample.obs['treatment'][0] + '_' + adata_sample.obs['elapsed_time'][0]]
    interactions = misty.uns['interactions']
    interactions['sample'] = s
    interactions['condition'] = title_dict[adata_sample.obs['treatment'][0] + '_' + adata_sample.obs['elapsed_time'][0]]
    interactions['rank'] = interactions['importances'].rank(ascending=False)

    metrics_df = pd.concat([metrics_df, targets], axis=0, ignore_index=True)
    interactions_df = pd.concat([interactions_df, interactions], axis=0, ignore_index=True)
metrics_df['id'] = metrics_df.index

li.pl.contributions(misty, return_fig=False, figure_size=(7, 7))
plt.tight_layout()
plt.savefig('contributions.svg')

li.pl.interactions(misty, view='Cell types', top_n=20)

#%%
# R squared barplot main figure

all_sample_vals = []
for k in list(title_dict.values()):
    condition_df = metrics_df[metrics_df['condition'] == k].copy()

    if 'Tumor necrotic margin' in condition_df['target'].values:
        order = ['Tumor Pt content', 'Tumor low Pt', 'Tumor mod. Pt', 'Tumor high Pt', 'Tumor necrotic margin']
    else:
        order = ['Tumor Pt content', 'Tumor low Pt', 'Tumor mod. Pt', 'Tumor high Pt']
    condition_df['target'] = pd.Categorical(condition_df['target'], categories=order, ordered=True)
    condition_df = condition_df.sort_values('target')

    colors = ['#32B166', '#6cd598', '#6cd5d4', '#34b7b5', '#9f34b7']
    mod_names = list(mods.keys())[1:]
    vals_list = []
    ans = np.zeros(condition_df.shape[0])
    for m, c in zip(mod_names, colors):
        vals = condition_df[m] * condition_df['multi_R2']
        print(vals)
        vals_added = vals + ans
        ans += vals
        vals_list.append(vals_added)
        print(vals_added)

    # plot stuff in reverse order to organize the overlaps
    mod_names.reverse()
    colors.reverse()
    vals_list.reverse()

    # select first column from every sample
    selected_val = []
    for val in vals_list:
            selected_val.append(val.values[0])
    all_sample_vals.append(selected_val)

val_array = np.array(all_sample_vals).T[:, :-1].tolist()

mod_names[mod_names.index('Chrysalis')] = 'Tissue\ncompartments'

plt.rcParams['svg.fonttype'] = 'none'
sf = 0.9
fig, ax = plt.subplots(1, 1, figsize=(4*sf, 5*sf), sharey='all')
sample_names = list(title_dict.values())[:-1]
for m, c, v in zip(mod_names, colors, val_array):
    sns.barplot(y=v, x=sample_names, ax=ax, color=c, label=m)
for m, c, v in zip(mod_names, colors, val_array):
    ax.scatter(y=v, x=sample_names, color='black')

ax.tick_params(axis='x', rotation=90)
ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_ylabel('R\u00b2')
ax.set_xlabel(None)
ax.set_title('Tumor Pt content')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/misty_bar_main2.svg')
plt.show()

#%%
# misty rank main
plt.rcParams['svg.fonttype'] = 'none'

sf = 0.9
fig, axs = plt.subplots(1, 3, figsize=(10*sf, 3.5*sf))
for ax in axs:
    ax.axis('off')

for ax, k in zip(axs, list(title_dict.values())[:3]):
    mod_names = list(mods.keys())[1:]
    mod_names.reverse()
    condition_df = interactions_df[interactions_df['condition'] == k].copy()

    met_df = metrics_df[metrics_df['condition'] == k].copy()
    met_r2 = {k:v for k, v in zip(met_df['target'], met_df['multi_R2'])}

    # scale importance for random forest
    rel_importance = []
    for idx, row in condition_df.iterrows():
        p = row['view']
        t = row['target']
        r2 = met_df[met_df['target'] == t][p]
        val = row['importances'] * r2.values
        rel_importance.append(val[0])

    condition_df['relative_importance'] = rel_importance

    if 'Tumor necrotic margin' in condition_df['target'].values:
        order = ['Tumor Pt content', 'Tumor low Pt', 'Tumor mod. Pt', 'Tumor high Pt', 'Tumor necrotic margin']
    else:
        order = ['Tumor Pt content', 'Tumor low Pt', 'Tumor mod. Pt', 'Tumor high Pt']

    plt.rcParams['svg.fonttype'] = 'none'
    t = 'Tumor Pt content'
    ax.axis('on')
    target_df = condition_df[condition_df['target'] == t]
    target_df['rank'] = target_df['relative_importance'].rank(ascending=False)
    target_df = target_df.sort_values(by='rank')
    scatter = ax.scatter(target_df['rank'] - 1, target_df['relative_importance'], s=20,
                c=target_df['view'].map({x:y for x, y in zip(mod_names, colors)}), zorder=2)

    ax.plot(list(target_df['relative_importance']), zorder=1, c='#bfbfbf')
    texts = []
    top_n = target_df[target_df['rank'] <= 8]
    for i, row in top_n.iterrows():
        tl = row['predictor']

        texts.append(ax.text(row['rank'], row['relative_importance'], tl, fontsize=10, multialignment='left'))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=1), ax=ax, min_arrow_len=0, pull_threshold=5)
    ax.set_axisbelow(True)
    ax.set_title(f"{k}\nR\u00b2: {round(met_r2[t], 2)}")
    ax.set_ylabel('Importance')
    ax.set_xlabel('Rank')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

mod_names[mod_names.index('Chrysalis')] = 'Tissue\ncompartments'
patches = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=15, label=name)
           for name, color in zip(mod_names, colors)]
plt.legend(handles=patches, loc='upper right', ncol=1)
plt.suptitle('Feature importances\nTotal Pt content')
plt.subplots_adjust(bottom=0.45)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/misty_features_main2.svg')
plt.show()
