import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
from lifelines.utils import survival_table_from_events


def score_signature(adata, signature_df, selected_comp, signature_name):
    m = np.mean(signature_df[f'compartment_{selected_comp}'])
    std = np.std(signature_df[f'compartment_{selected_comp}'])
    threshold = m + (2 * std)
    gene_list = signature_df[f'compartment_{selected_comp}'][signature_df[f'compartment_{selected_comp}'] > threshold]
    gene_list = [g for g in gene_list.index if g in adata.var_names]
    sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'{signature_name.lower()}_signature', use_raw=False)
    sns.histplot(data=adata.obs, x=f'{signature_name.lower()}_signature', kde=True, bins=100)
    plt.show()
    return gene_list


#%%
# load matrix
data = pd.read_csv("data/metabric/data_mrna_illumina_microarray.txt", index_col=0, sep='\t')
# entrez gene id
gene_id = data.iloc[:, 0]
# construct adata
data = data.iloc[:, 1:].T
adata = AnnData(X=data.values,
                obs=pd.DataFrame(index=data.index),
                var=pd.DataFrame(index=data.columns))
adata.var['entrez_id'] = gene_id

# add metadata
clinical_metadata = pd.read_csv('data/metabric/data_clinical_sample.txt', sep="\t", index_col=0, comment="#")
patient_metadata = pd.read_csv('data/metabric/data_clinical_patient.txt', sep="\t", index_col=0, comment="#")

adata.obs[clinical_metadata.columns] = clinical_metadata
adata.obs[patient_metadata.columns] = patient_metadata

# see if there are nans in recurrence status
print(adata.obs[['RFS_STATUS', 'RFS_MONTHS']].isna().sum())
adata = adata[adata.obs['RFS_STATUS'].notna()].copy()

# check if there are no duplicate patient entries
print("Unique samples")
print(len(np.unique(adata.obs["SAMPLE_ID"])))

# define recurrence as ints
adata.obs['EVENT'] = adata.obs['RFS_STATUS'].map({'1:Recurred':1, '0:Not Recurred':0})

# get triple-negative samples
adata = adata[
    (adata.obs['ER_STATUS'] == "Negative") &
    (adata.obs['PR_STATUS'] == "Negative") &
    (adata.obs['HER2_STATUS'] == "Negative")
].copy()

adata.write_h5ad('data/metabric/dataset.h5ad')

#%%
# get signatures
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
adata = sc.read_h5ad('data/metabric/dataset.h5ad')
adata.var_names_make_unique()

# select signatures
# these add the signatures to .obs and return the genes used for calculating it
prolif_genes = score_signature(adata, signature_df, 5, 'proliferating')
emt_genes = score_signature(adata, signature_df, 2, 'emt')

#%%
# K-means-based approach
adata_emt = adata[:, adata.var_names.isin(emt_genes)].copy()
sc.pp.pca(adata_emt)
sc.pp.neighbors(adata_emt)
sc.tl.umap(adata_emt)

# extract pca coordinates
X_pca = adata_emt.obsm['X_pca']

# plot K-means results on UMAP
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_pca)
adata_emt.obs['kmeans'] = kmeans.labels_.astype(str)
sc.pl.pca(adata_emt, color=['kmeans', f'emt_signature'])
sc.pl.umap(adata_emt, color=['kmeans', f'emt_signature'])

# add signature labels
# adata.obs['kmeans'] = kmeans.labels_.astype(str)
# adata.obs[f'emt_high'] = adata.obs.apply(lambda row: True if row['kmeans'] == '0' else False, axis=1)
# adata.obs[f'emt_low'] = adata.obs.apply(lambda row: True if row['kmeans'] == '1' else False, axis=1)

median = np.median(adata.obs['emt_signature'])
adata.obs['emt_high'] = adata.obs.apply(
    lambda row: True if row['emt_signature'] > np.median(adata.obs['emt_signature']) else False, axis=1)
adata.obs['emt_low'] = adata.obs.apply(
    lambda row: True if row['emt_signature'] <= np.median(adata.obs['emt_signature']) else False, axis=1)

# construct censor status and survival time
obs_df = adata.obs.copy()
obs_df['survival_time'] = obs_df['RFS_MONTHS'] / 12
obs_df = obs_df[obs_df['survival_time'] > 0]

# truncate df to have at least 10 patients remaining
# build the at-risk table
st = survival_table_from_events(obs_df['survival_time'] , obs_df['EVENT'])
threshold = 20

mask = st['at_risk'] < threshold
if mask.any():
    cutoff_time = float(mask[mask==True].index[0])
else:
    cutoff_time = st['time'].max()

print(f"Cutoff time (days) where at_risk < {threshold}: {cutoff_time:.1f}")
print(f"Cutoff time (years): {cutoff_time}")

# truncate the unstable part
# obs_df = obs_df[obs_df['survival_time'] <= cutoff_time].copy()
# clip times at cutoff
obs_df['survival_time'] = obs_df['survival_time'].clip(upper=cutoff_time)
# if original time > cutoff then censored at cutoff
obs_df['EVENT'] = obs_df['EVENT'].where(obs_df['survival_time'] <= cutoff_time, 0).astype(int)

#%%
# plot stuff
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(4, 5))

high_df = obs_df[obs_df[f'emt_high'] == True]
kmf_high = KaplanMeierFitter(label=f'EMT high')
kmf_high.fit(high_df['survival_time'], event_observed=high_df['EVENT'])
kmf_high.plot_survival_function(ax=ax, color='#CAA167')

low_df = obs_df[obs_df[f'emt_low'] == True]
kmf_low = KaplanMeierFitter(label=f'EMT low')
kmf_low.fit(low_df['survival_time'], event_observed=low_df['EVENT'])
kmf_low.plot_survival_function(ax=ax, color='#999')

df_high = pd.concat([kmf_high.survival_function_, kmf_high.confidence_interval_], axis=1)
df_high.columns = ['emt-high_survival_probability', 'emt-high_CI_lower', 'emt-high_CI_upper']
df_low = pd.concat([kmf_low.survival_function_, kmf_low.confidence_interval_], axis=1)
df_low.columns = ['emt-low_survival_probability', 'emt-low_CI_lower', 'emt-low_CI_upper']
source_data = pd.concat([df_high, df_low], axis=1)
source_data = source_data.sort_index()
source_data = source_data.ffill()
source_data.to_csv('data/fig5j.csv')

ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_title(f'METABRIC\nRecurrence-free survival')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.set_xlabel('Timeline (year)')

high_5yr = kmf_high.predict(5)
low_5yr  = kmf_low.predict(5)

ax.plot(5, high_5yr, 'o', color='#CAA167', markersize=6)
ax.plot(5, low_5yr, 'o', color='#999', markersize=6)

add_at_risk_counts(kmf_high, kmf_low, ax=ax)
ax.set_ylabel('RFS probability')

print(f"5-year RFS:")
print(f"EMT high: {high_5yr:.2f}")
print(f"EMT low:  {low_5yr:.2f}")

results = survival_difference_at_fixed_point_in_time_test(5, kmf_high, kmf_low)
results.print_summary()

fig.text(0.325, 0.505, "5-year RFS\n" rf"$\chi^2$ = {results.p_value:.4f}", transform=fig.transFigure)

plt.tight_layout()
plt.savefig(f'figures/metabric.svg')
plt.show()


#%%
results = logrank_test(high_df['survival_time'], low_df['survival_time'],
                       high_df['EVENT'], low_df['EVENT'])
results.print_summary()
