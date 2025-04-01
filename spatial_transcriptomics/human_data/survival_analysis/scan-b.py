import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from anndata import AnnData
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.mixture import GaussianMixture
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts


#%%
data = pd.read_csv("data/SCAN-B/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv",
                   index_col=0)

metadata1 = pd.read_csv('data/SCAN-B/GSE96058-GPL11154_series_matrix.txt', sep="\t", skiprows=63, index_col=0)
metadata2 = pd.read_csv('data/SCAN-B/GSE96058-GPL18573_series_matrix.txt', sep="\t", skiprows=63, index_col=0)

def clean_metadata(metadata2):
    term = '!Sample_characteristics_ch1'
    rows = [True if x == term else False for x in metadata2.index]
    metadata2 = metadata2.iloc[rows, :]

    new_index = [x.split(': ')[0] for x in metadata2.iloc[:, 0].values]
    df = pd.DataFrame(columns=metadata2.columns, index=new_index)
    for c in metadata2.columns:
        col = metadata2[c]
        new_vals = [x.split(': ')[-1] for x in col.values]
        df[c] = new_vals
    return df

metadata1 = clean_metadata(metadata1)
metadata2 = clean_metadata(metadata2)

metadata = pd.concat([metadata1, metadata2], axis=1)
metadata = metadata.T
metadata = metadata.loc[data.columns]

data = data.T
adata = AnnData(X=data.values,
                obs=pd.DataFrame(index=data.index),
                var=pd.DataFrame(index=data.columns))

adata.obs = pd.concat([adata.obs, metadata], axis=1)
for col in adata.obs.columns:
    adata.obs[col] = pd.to_numeric(adata.obs[col], errors='ignore')

adata.write_h5ad('data/SCAN-B/dataset.h5ad')

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
adata = sc.read_h5ad('data/SCAN-B/dataset.h5ad')
# get rid of some very lowly expressed genes
sc.pp.filter_genes(adata, min_cells=100)
adata.var_names_make_unique()
# look at mt content
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.pct_counts_mt < 50, :]
adata = adata[:, adata.var['mt'] == False]
replicates = [False if 'repl' in x else True for x in adata.obs.index]
adata = adata[replicates]

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
sns.histplot(data=adata.obs, x=f'{signature_name}_signature', kde=True, bins=100)
plt.show()

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

tnbc_df = pd.DataFrame(data={'HER2 (ERBB2)': her2_pred, 'ER (ESR1)': er_pred, 'PR (PGR)': pr_pred},
                       index=adata_x_df.index)
tnbc_df['TNBC'] = [1 if x == False else 0 for x in tnbc_df.any(axis=1)]

# tnbc using the metadata
tnbc_meta_df = adata.obs[['er prediction mgc', 'pgr prediction mgc', 'her2 prediction mgc']].copy()
tnbc_meta_df['TNBC'] = [1 if x == False else 0 for x in tnbc_meta_df.any(axis=1)]
tnbc_df['TNBC_mgc'] = tnbc_meta_df['TNBC']
tnbc_meta_df = adata.obs[['er prediction sgc', 'pgr prediction sgc', 'her2 prediction sgc']].copy()
tnbc_meta_df['TNBC'] = [1 if x == False else 0 for x in tnbc_meta_df.any(axis=1)]
tnbc_df['TNBC_sgc'] = tnbc_meta_df['TNBC']

tnbc_df['TNBC_combined'] =  [0 if x == False else 1 for x in tnbc_df[['TNBC', 'TNBC_mgc', 'TNBC_sgc']].any(axis=1)]
adata.obs['TNBC_combined'] = tnbc_df['TNBC_combined']

#%%

# K-means-based approach
from sklearn.cluster import KMeans

tnbc_adata = adata[adata.obs['TNBC_combined'] == 1]  # subset to TNBC
# tnbc_adata = tnbc_adata[tnbc_adata.obs['endocrine treated'] == '0']
tnbc_adata = tnbc_adata[tnbc_adata.obs['chemo treated'] == '1']

plt.scatter(tnbc_adata.obs[f'EMT_signature'], tnbc_adata.obs[f'Proliferating_signature'], marker='x', s=7)
plt.show()

sns.histplot(data=tnbc_adata.obs, x=f'EMT_signature', kde=True, bins=100)
plt.show()
sns.histplot(data=tnbc_adata.obs, x=f'Proliferating_signature', kde=True, bins=100)
plt.show()

#%%
# stratify patients
obs_df = tnbc_adata.obs.copy()

# construct censor status and survival time
obs_df['censor_status'] = obs_df['overall survival event']
obs_df['survival_time'] = obs_df['overall survival days'].astype(float)
obs_df['survival_time'] = obs_df['survival_time'] / 365.25

obs_df = obs_df.dropna(subset='survival_time')
obs_df = obs_df[obs_df['survival_time'] > 0]

# run kmf
kmf = KaplanMeierFitter()
kmf.fit(obs_df['survival_time'], event_observed=obs_df['censor_status'])

# plot stuff
signature_name = 'EMT'
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(1, 1, figsize=(4, 5))

high_df = obs_df[obs_df[f'{signature_name}_high'] == True]
kmf_high = KaplanMeierFitter(label=f'{signature_name} high')
kmf_high.fit(high_df['survival_time'], event_observed=high_df['censor_status'])
kmf_high.plot_survival_function(ax=ax, color='#fa3')

low_df = obs_df[obs_df[f'{signature_name}_low'] == True]
kmf_low = KaplanMeierFitter(label=f'{signature_name} low')
kmf_low.fit(low_df['survival_time'], event_observed=low_df['censor_status'])
kmf_low.plot_survival_function(ax=ax, color='#999')

ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_title(f'SCAN-B\n{signature_name} signature')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Timeline (year)')

add_at_risk_counts(kmf_high, kmf_low, ax=ax)
ax.set_ylabel('Survival probability')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig4/scan-b-emt.svg')
plt.show()

results = logrank_test(high_df['survival_time'], low_df['survival_time'],
                       high_df['censor_status'], low_df['censor_status'])
results.print_summary()
