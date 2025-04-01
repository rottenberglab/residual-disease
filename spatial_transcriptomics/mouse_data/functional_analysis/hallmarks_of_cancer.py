import numpy as np
import scanpy as sc
import pandas as pd
import pymart as pm
import seaborn as sns
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.functions import spatial_plot


#%%
# get human-mouse orthologs
# fetch human ensembl IDs from BioMart
human_data = pm.fetch_data(dataset_name="hsapiens_gene_ensembl", attributes=["ensembl_gene_id", "external_gene_name"])

# fetch mouse orthologs
dataset_name = "hsapiens_gene_ensembl"
hom_species = ["mmusculus"]
hom_query = ["ensembl_gene", "associated_gene_name", "orthology_type", "orthology_confidence", "perc_id"]
data = pm.fetch_data(dataset_name=dataset_name, hom_species=hom_species, hom_query=hom_query)

data = data[['Gene stable ID', 'Mouse gene stable ID', 'Mouse gene name', 'Mouse homology type',
       '%id. target Mouse gene identical to query gene',
       'Mouse orthology confidence [0 low, 1 high]']]

data = data.drop_duplicates()
data = data.dropna()

# combine dataframes
merged_df = data.merge(human_data, left_on='Gene stable ID', right_on='Gene stable ID', how='left')
merged_df.to_csv('data/human_mouse_orthologs.csv')

#%%

orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

#%%
condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

# get cancer hallmark signatures
marker_df = pd.read_csv('data/cancer_hallmark_signatures.csv', index_col=0)

# define a cancer hallmark df with the mouse orthologs
hm_dict = {}
hm_names = []
for hm in np.unique(marker_df['hallmark']):
    hm_df = marker_df[marker_df['hallmark'] == hm]
    gene_set = set()
    for c in hm_df['gene_list']:
        genes = c.split(', ')
        for g in genes:
            gene_set.add(g)

    # search for orthologs
    mouse_orths = []
    for v in gene_set:
        try:
            g = orthologs_dict[v]
        except:
            pass
        mouse_orths.append(g)
    hm_dict[hm] = mouse_orths
    hm_names.append(hm_df['hallmark_name'].iloc[0])
hm_mouse = pd.Series(hm_dict, name='gene_set')
hm_mouse_df = pd.DataFrame(hm_mouse)
hm_mouse_df['hallmark_name'] = hm_names

#%%
# Hallmarks of cancer spatial plots
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

adata.obs['condition'] = adata.obs['condition'].map(lambda x: condition[x])
adata.obs['treatment'] = adata.obs['treatment'].map(lambda x: treatment[x])
adata.obs['elapsed_time'] = adata.obs['elapsed_time'].map(lambda x: time[x])

selected_samples = [17,  15,  16]
subdata = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

# reorder sample_id for plotting
sample_order = ['20221010-3_9', '20221010-3_11', '20221010-3_12']
comps = 13
subdata.obs['sample_id'] = subdata.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

# add a dict quickly to replace titles
sample_name_df = subdata.obs[['sample_id', 'label']].drop_duplicates()
sample_name_df.index = sample_name_df['sample_id']
sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

# add ne spatial uns for chrysalis plot - need to replace later
for k in list(subdata.uns['spatial'].keys()):
    if k in list(sample_name_df.keys()):
        subdata.uns['spatial'][sample_name_df[k]] = subdata.uns['spatial'][k]

label_cats = [sample_name_df[x] for x in subdata.obs['sample_id'].cat.categories]
subdata.obs['label'] = subdata.obs['label'].cat.reorder_categories(label_cats, ordered=True)

label_cats = ['Primary tumor | 20221010-3_9',
              'Residual tumor | 20221010-3_11',
              'Recurrent tumor | 20221010-3_12']

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

i = 0
for idx, row in hm_mouse_df.iterrows():
    gene_list = row['gene_set']
    gene_list = subdata.var_names[np.in1d(subdata.var_names, gene_list)]
    sc.tl.score_genes(subdata, gene_list=gene_list, score_name=row['hallmark_name'])

    spatial_plot(subdata, 1, 3, row['hallmark_name'], cmap='Spectral_r', title=label_cats,
                 suptitle=row['hallmark_name'], alpha_img=0.5)
    plt.savefig(f"figs/manuscript/fig3/cancer_hallmarks/{row['hallmark_name']}.svg")
    plt.show()
    i += 1

#%%
# show everything on one sample
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

adata.obs['condition'] = adata.obs['condition'].map(lambda x: condition[x])
adata.obs['treatment'] = adata.obs['treatment'].map(lambda x: treatment[x])
adata.obs['elapsed_time'] = adata.obs['elapsed_time'].map(lambda x: time[x])

selected_samples = [20]

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(4, 4, figsize=(4 * 4, 3 * 4))
ax = ax.flatten()
for a in ax:
    a.axis('off')
i = 0
for idx, row in hm_mouse_df.iterrows():
    gene_list = row['gene_set']
    gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]
    ad = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()
    sc.tl.score_genes(ad, gene_list=gene_list, score_name=row['hallmark_name'])

    sc.pl.spatial(ad, color=row['hallmark_name'], size=1.5, alpha=1, library_id=ad.obs['sample_id'][0],
                  ax=ax[i], show=False, cmap='Spectral_r', alpha_img=0.5,
                  vcenter=0)
    i += 1
plt.tight_layout()
plt.show()

#%%
# Violin plots

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

for idx, row in hm_mouse_df.iterrows():
    gene_list = row['gene_set']
    gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]
    sc.tl.score_genes(adata, gene_list=gene_list, score_name=row['hallmark_name'])

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

rows = 4
cols = 4

sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)
plt.rcParams['svg.fonttype'] = 'none'

fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, pw in enumerate(hm_mouse_df['hallmark_name']):
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
        if len(pw.split(' ')) > 2:
            title_ws = pw.split(' ')
            title = title_ws[0] + ' ' + title_ws[1] + '\n'
            title = title + ' '.join(title_ws[2:])
            axs[idx].set_title(title, fontsize=12)
        else:
            axs[idx].set_title(pw, fontsize=12)
        axs[idx].set_xlabel(None)
        axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        legend_labels = ['False', 'True']
        handles, _ = axs[idx].get_legend_handles_labels()
fig.supylabel('Hallmarks of cancer')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/cancer_hallmark_violin.svg')
plt.show()

#%%
# Correlation with tissue compartments
columns = hm_mouse_df['hallmark_name']

hexcodes = ch.utils.get_hexcodes(None, 13, 87, len(adata))

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

corr_m = pd.concat([compartment_df, adata.obs[columns]], axis=1).corr(method='pearson')
corr_m = corr_m.drop(index=columns, columns=compartment_df.columns)
corr_m = corr_m.rename(columns={'sasp': 'SASP'})
corr_m.index = [str(x) for x in corr_m.index]

z = linkage(corr_m, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[order, :]
hexcodes = [hexcodes[i] for i in order]

z = linkage(corr_m.T, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[:, order]

plt.rcParams['svg.fonttype'] = 'none'

fig, axs = plt.subplots(1, 1, figsize=(5, 8))
sns.heatmap(corr_m, square=True, center=0,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True), rasterized=True,
            )
for idx, t in enumerate(axs.get_yticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
axs.set_yticklabels(axs.get_yticklabels(), rotation=0)
plt.title('Hallmarks of cancer')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/hallmark_corrs.svg')
plt.show()
