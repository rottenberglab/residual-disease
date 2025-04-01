import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from spatial_transcriptomics.human_data.functions_human import make_obs_categories, spatial_plot


#%%
# write the raw expression matrix as a csv
output_folder = 'data/human_samples'

adata = sc.read_h5ad(f'{output_folder}/human_samples_scanorama.h5ad')

estimate_scores = pd.read_csv(f'{output_folder}/estimate_matrix.gct', sep='\t', index_col=0)
estimate_scores = estimate_scores.T
estimate_scores = estimate_scores.iloc[1:, :]
estimate_scores.index = adata.obs_names
adata.obs['ESTIMATE_score'] = estimate_scores['ESTIMATEScore'].astype(float)

spatial_plot(adata, 2, 3, var='ESTIMATE_score', share_range=False, cmap='Spectral')
plt.show()

#%%
# tissue compartments
columns = ['ESTIMATE_score']

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

corr_m = pd.concat([compartment_df, adata.obs[columns]], axis=1).corr(method='pearson')
corr_m = corr_m.drop(index=columns, columns=compartment_df.columns)
corr_m = corr_m.rename(columns={'ESTIMATE_score': 'ESTIMATE'})
corr_m.index = ['' + str(x) for x in corr_m.index]

z = linkage(corr_m, method='ward')
order = leaves_list(z)
corr_m = corr_m.iloc[order, :]

fig, axs = plt.subplots(1, 1, figsize=(5, 3))
sns.heatmap(corr_m.T, square=True, center=0,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True))
plt.title('ESTIMATE correlation')
plt.tight_layout()
plt.show()

#%%
# Pure tissue compartments

threshold = 0.65

for i in range(adata.obsm['chr_aa'].shape[1]):
    comp = adata.obsm['chr_aa'][:, i]
    positive = comp > threshold
    print(np.sum(positive))
    adata.obs[f'comp_{i}'] = positive.astype(str)

cats = make_obs_categories(adata, [f'comp_{x}' for x in range(adata.obsm['chr_aa'].shape[1])])

adata.obs['comps'] = cats.astype('category')

long_df = pd.melt(adata.obs, id_vars=['comps'], value_vars=['ESTIMATE_score'])
long_df = long_df[long_df['comps'] != 'None']
long_df['comps'] = long_df['comps'].cat.remove_unused_categories()

means = {}
for c in long_df['comps'].cat.categories:
    means[c] = np.mean(long_df[long_df['comps'] == c]['value'])
mean_df = pd.DataFrame(means, index=['mean']).T
mean_df = mean_df.sort_values(by='mean', ascending=False)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.violinplot(long_df, x='comps', y='value', scale='width', order=mean_df.index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()
