import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from glob import glob
from anndata import AnnData
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
from sklearn.mixture import GaussianMixture
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts


#%%
# preprocess data
samples = glob('data/tcga_brca/*/*.tsv')

# get metadata
metadata = pd.read_csv('data/tcga_brca/clinical.tsv', sep='\t', skiprows=0)
to_drop = []
for c in metadata.columns:
    u = np.unique(metadata[c])
    if len(u) == 1:
        if u[0] == "'--":
            to_drop.append(c)
metadata = metadata.drop(columns=to_drop)
metadata.index = metadata['case_submitter_id']
metadata = metadata.drop_duplicates(subset=['case_id'])
metadata = metadata.drop(columns=['case_submitter_id', 'treatment_or_therapy', 'treatment_type'])

# get sample sheet because the file IDs don't match in the metadata unfortunately
sample_sheet = pd.read_csv('data/tcga_brca/sample_sheet.tsv', sep='\t', skiprows=0)
sample_sheet = sample_sheet[sample_sheet['Sample Type'] == 'Primary Tumor']
sample_sheet = sample_sheet[sample_sheet['Project ID'] == 'TCGA-BRCA']
sample_sheet = sample_sheet.drop_duplicates(subset='Sample ID')
len(np.unique(sample_sheet['Sample ID']))
len(np.unique(sample_sheet['File ID']))
sh_dict = {k:v for k, v in zip(sample_sheet['File ID'], sample_sheet['Sample ID'])}
sample_sheet.index = sample_sheet['Sample ID']

#%%
# construct an anndata object
counts_df = pd.DataFrame()
for s in tqdm(samples):
    sample_id = s.split('/')[2]
    if sample_id in list(sample_sheet['File ID']):
        sample_id = sh_dict[sample_id]
        df = pd.read_csv(s, sep='\t', skiprows=1)
        df.index = df['gene_id']
        df = df['tpm_unstranded'].dropna(axis=0)
        df = df.rename(sample_id)
        counts_df = pd.concat([counts_df, df], axis=1)

counts_df = counts_df.T

adata = AnnData(X=counts_df.values,
                obs=pd.DataFrame(index=counts_df.index),
                var=pd.DataFrame(index=counts_df.columns))
adata.obs['case_submitter_id'] = sample_sheet['Case ID']

# add metadata to anndata
adata.obs = pd.merge(left=adata.obs, right=metadata, left_on='case_submitter_id', right_index=True, how='left')

# add gene names
df = pd.read_csv(samples[0], sep='\t', skiprows=1)
df = df.dropna(axis=0)
df.index = df['gene_id']
adata.var[['gene_name', 'gene_type']] = df[['gene_name', 'gene_type']]

len(np.unique(adata.obs.index))
adata.write_h5ad('data/tcga_brca/dataset.h5ad')

#%%
signature_df = pd.read_csv('data/compartment_signatures.csv', index_col=0)
orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']
orthologs_dict = {k: v for v, k in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

mouse_orths = []
for v in signature_df.index:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)

signature_df['human'] = mouse_orths
signature_df = signature_df.dropna()
signature_df['mouse'] = signature_df.index
signature_df.index = signature_df['human']

#%%

adata = sc.read_h5ad('data/tcga_brca/dataset.h5ad')
# get rid of some very lowly expressed genes
adata.var.index = adata.var['gene_name'].astype(str)
sc.pp.filter_genes(adata, min_cells=100)
adata.var_names_make_unique()
# look at mt content
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

adata = adata[adata.obs.pct_counts_mt < 50, :]
adata = adata[:, adata.var['mt'] == False]

sc.pl.highest_expr_genes(adata, n_top=20)

adata.raw = adata
# sc.pp.normalize_total(adata, target_sum=1e4)  # already normalized so we don't need to do it
sc.pp.log1p(adata)

# select signatures
selected_comp = 5  # 2 emt 5 prolif
m = np.mean(signature_df[f'compartment_{selected_comp}'])
std = np.std(signature_df[f'compartment_{selected_comp}'])
threshold = m + (2 * std)
gene_list = signature_df[f'compartment_{selected_comp}'][signature_df[f'compartment_{selected_comp}'] > threshold]
gene_list = [g for g in gene_list.index if g in adata.var_names]
prolif_gene_list = gene_list
signature_name = 'Proliferating'
sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'{signature_name}_signature', use_raw=False)

# select signatures
selected_comp = 2  # 2 emt 5 prolif
m = np.mean(signature_df[f'compartment_{selected_comp}'])
std = np.std(signature_df[f'compartment_{selected_comp}'])
threshold = m + (2 * std)
gene_list = signature_df[f'compartment_{selected_comp}'][signature_df[f'compartment_{selected_comp}'] > threshold]
gene_list = [g for g in gene_list.index if g in adata.var_names]
emt_gene_list = gene_list
signature_name = 'EMT'
sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'{signature_name}_signature', use_raw=False)

sns.histplot(data=adata.obs, x=f'{signature_name}_signature', kde=True, bins=100)
plt.show()

plt.scatter(adata.obs[f'{signature_name}_signature'], adata.obs['pct_counts_mt'], marker='x', s=7)
plt.show()

# hormone receptor expression

def get_trehold_bimodal(col):
    gm = GaussianMixture(n_components=2, random_state=42).fit(col.values.reshape(-1, 1))
    pred = gm.predict(col.values.reshape(-1, 1))
    # ensure that + cells are always labelled with 1
    if not gm.means_[1][0] > gm.means_[0][0]:
        pred = [np.abs(x - 1) for x in pred]
    df = pd.DataFrame(col)
    df['pred'] = pred
    # threshold for gaussian mixture
    th = np.max(df[df['pred'] == 0][col.name])
    return th, pred


adata_x_df = adata.to_df()
fig, axs = plt.subplots(3, 1, figsize=(5, 7))
sns.histplot(data=adata_x_df, x='ERBB2', kde=True, bins=100, ax=axs[0])
sns.histplot(data=adata_x_df, x='ESR1', kde=True, bins=100, ax=axs[1])
sns.histplot(data=adata_x_df, x='PGR', kde=True, bins=100, ax=axs[2])
her2_th, her2_pred = get_trehold_bimodal(adata_x_df['ERBB2'])
axs[0].axvline(x=her2_th, c='red')
er_th, er_pred = get_trehold_bimodal(adata_x_df['ESR1'])
axs[1].axvline(x=er_th, c='red')
pr_th, pr_pred = get_trehold_bimodal(adata_x_df['PGR'])
axs[2].axvline(x=pr_th, c='red')
axs[0].set_title('HER2 (ERBB2)')
axs[1].set_title('ER (ESR1)')
axs[2].set_title('PR (PGR)')
plt.tight_layout()
plt.show()

tnbc_df = pd.DataFrame(data={'HER2 (ERBB2)': her2_pred, 'ER (ESR1)': er_pred, 'PR (PGR)': pr_pred},
                       index=adata_x_df.index)
tnbc_df['TNBC'] = [1 if x == False else 0 for x in tnbc_df.any(axis=1)]

# get new metadata from cell genomics paper
annots_df = pd.read_csv('data/tcga_brca/thennavan_cell_genomics_2021.csv', index_col=0)
tnbc_df['TNBC_thennavan'] = annots_df['Triple Negative Status']
# this removes nans so only use it if we take the intersection of the two TNBC sets
tnbc_df['TNBC_thennavan'] = [1 if x == 'Yes' else 0 for x in tnbc_df['TNBC_thennavan']]
tnbc_df['TNBC_combined'] =  [0 if x == False else 1 for x in tnbc_df[['TNBC', 'TNBC_thennavan']].any(axis=1)]
adata.obs['TNBC_combined'] = tnbc_df['TNBC_combined']
adata.obs['TNBC_combined'] = tnbc_df['TNBC_thennavan']

plt.scatter(adata.obs[f'{signature_name}_signature'], adata.obs['pct_counts_mt'], marker='x', s=7,
            c=adata.obs['TNBC_combined'])
plt.show()

sns.histplot(data=adata.obs[adata.obs['TNBC_combined'] == 1], x=f'{signature_name}_signature', kde=True, bins=100)
sns.histplot(data=adata.obs[adata.obs['TNBC_combined'] == 0], x=f'{signature_name}_signature', kde=True, bins=100)
plt.show()

#%%
# K-means-based approach

tnbc_adata = adata[adata.obs['TNBC_combined'] == 1]  # subset to TNBC

plt.scatter(tnbc_adata.obs[f'EMT_signature'], tnbc_adata.obs[f'Proliferating_signature'], marker='x', s=7)
plt.show()

for signature_name, gene_list in zip(['EMT', 'Proliferating'], [emt_gene_list, prolif_gene_list]):
    adata_s = tnbc_adata[:, tnbc_adata.var_names.isin(gene_list)].copy()
    sc.pp.pca(adata_s)
    sc.pp.neighbors(adata_s)
    sc.tl.umap(adata_s)

    # extract pca coordinates
    X_pca = adata_s.obsm['X_pca']

    # kmeans
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X_pca)
    adata_s.obs['kmeans'] = kmeans.labels_.astype(str)
    sc.pl.pca(adata_s, color=['kmeans', f'{signature_name}_signature'])
    sc.pl.umap(adata_s, color=['kmeans', f'{signature_name}_signature'])

    # add signature labels
    tnbc_adata.obs['kmeans'] = kmeans.labels_.astype(str)
    if (tnbc_adata.obs[tnbc_adata.obs['kmeans'] == '1'][f'{signature_name}_signature'].mean() >
            tnbc_adata.obs[tnbc_adata.obs['kmeans'] == '0'][f'{signature_name}_signature'].mean()):
        tnbc_adata.obs[f'{signature_name}_high'] = tnbc_adata.obs.apply(lambda row: True if row['kmeans'] == '1'
        else False, axis=1)
        tnbc_adata.obs[f'{signature_name}_low'] = tnbc_adata.obs.apply(lambda row: True if row['kmeans'] == '0'
        else False, axis=1)
    else:
        tnbc_adata.obs[f'{signature_name}_high'] = tnbc_adata.obs.apply(lambda row: True if row['kmeans'] == '0'
        else False, axis=1)
        tnbc_adata.obs[f'{signature_name}_low'] = tnbc_adata.obs.apply(lambda row: True if row['kmeans'] == '1'
        else False, axis=1)

def categorize_row(row, Signature1='EMT', Signature2='Proliferating'):
    if row[f'{Signature1}_high'] and row[f'{Signature2}_high']:
        return r'EMT$_{High}$-Prolif.$_{High}$'
    elif row[f'{Signature1}_high'] and row[f'{Signature2}_low']:
        return r'EMT$_{High}$-Prolif.$_{Low}$'
    elif row[f'{Signature1}_low'] and row[f'{Signature2}_high']:
        return r'EMT$_{Low}$-Prolif.$_{High}$'
    elif row[f'{Signature1}_low'] and row[f'{Signature2}_low']:
        return r'EMT$_{Low}$-Prolif.$_{Low}$'
    else:
        return 'Unknown'  # If none of the conditions match

tnbc_adata.obs['emt_prol'] = tnbc_adata.obs.apply(categorize_row, axis=1)

# scatter plot with 4 categories
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.75), dpi=200)
sns.scatterplot(tnbc_adata.obs, x='EMT_signature', y='Proliferating_signature', hue='emt_prol', ax=ax,
                palette=['#67aeca', '#b8b8b8', '#737373', '#caa167'], s=50, edgecolor="none", rasterized=True)
ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_title(f'TCGA\nTNBC patients')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('EMT score')
ax.set_ylabel('Proliferating score')
legend = ax.legend(loc='lower left', bbox_to_anchor=(0, 0))
plt.tight_layout()
# plt.savefig(f'figs/manuscript/fig4/tnbc_scatter_v2.svg')
plt.show()

#%%
# stratify patients
obs_df = tnbc_adata.obs.copy()
obs_df = obs_df.replace(to_replace="'--", value=np.nan)

# construct censor status and survival time
obs_df['censor_status'] = 1 - (obs_df['vital_status'] == 'Alive').astype(int)
obs_df['survival_time'] = obs_df['days_to_death'].astype(float)

obs_df['survival_time'] = obs_df.apply(lambda row: row['days_to_last_follow_up'] if row['censor_status'] == 0
                          else row['survival_time'], axis=1)

obs_df['survival_time'] = obs_df['survival_time'].astype(float)

obs_df['survival_time'] = obs_df['survival_time'] / 365.25

obs_df = obs_df.dropna(subset='survival_time')
obs_df = obs_df[obs_df['survival_time'] > 0]
# obs_df = obs_df[obs_df['censor_status'] == 0]

# plot stuff
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(4, 5))

high_low_df = obs_df[obs_df[f'emt_prol'] == r'EMT$_{High}$-Prolif.$_{Low}$']
high_low = KaplanMeierFitter(label=r'EMT$_{High}$-Prolif.$_{Low}$')
high_low.fit(high_low_df['survival_time'], event_observed=high_low_df['censor_status'])
high_low.plot_survival_function(ax=ax, color='#caa167')

low_high_df = obs_df[obs_df[f'emt_prol'] == r'EMT$_{Low}$-Prolif.$_{High}$']
low_high = KaplanMeierFitter(label=r'EMT$_{Low}$-Prolif.$_{High}$')
low_high.fit(low_high_df['survival_time'], event_observed=low_high_df['censor_status'])
low_high.plot_survival_function(ax=ax, color='#67aeca')

ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_title(f'TCGA survival curve')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Timeline (year)')

legend = ax.legend(loc='lower right')

add_at_risk_counts(high_low, low_high, ax=ax)
ax.set_ylabel('Survival probability')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig4/tnbc_kaplanmeier_3.svg')
plt.show()

results = logrank_test(high_low_df['survival_time'], low_high_df['survival_time'],
                       high_low_df['censor_status'], low_high_df['censor_status'])
results.print_summary()
